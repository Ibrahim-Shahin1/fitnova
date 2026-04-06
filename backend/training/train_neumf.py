"""
FitNova — NeuMF Training Pipeline
==================================
Implements He et al. (2017) "Neural Collaborative Filtering," WWW 2017.

Architecture:
  - GMF  : Generalized Matrix Factorization (element-wise product of embeddings)
  - MLP  : Multi-Layer Perceptron (concatenation of embeddings through dense layers)
  - NeuMF: Fusion of GMF + MLP, initialized from pre-trained weights (alpha=0.5)

Training protocol:
  1. Pre-train GMF  with Adam  (binary cross-entropy)
  2. Pre-train MLP  with Adam  (binary cross-entropy)
  3. Initialize NeuMF from pre-trained weights, fine-tune with SGD

Evaluation: HR@10 and NDCG@10 (leave-one-out protocol)
"""

import os
import sys
import random
import pickle
from collections import defaultdict

# Force UTF-8 output on Windows (avoids cp1252 encoding errors for Unicode chars)
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

# Generated interaction data lives in backend/data/ by default.
# Override with FITNOVA_DATA_DIR env var if needed.
_REPO_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
DATA_DIR = os.environ.get("FITNOVA_DATA_DIR", _REPO_DATA_DIR)

# Output goes inside backend/models/ relative to this script
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Dataset dimensions (must match generated data)
NUM_USERS = 973
NUM_ITEMS = 2598

# Model hyperparameters — following He et al. (2017)
GMF_FACTORS = 16          # embedding dim for GMF pathway
MLP_FACTORS = 32          # embedding dim per side for MLP (concat input = 64)
MLP_LAYERS  = [64, 32, 16, 8]  # decreasing tower; last dim feeds into NeuMF fusion
ALPHA       = 0.5         # weight for GMF vs MLP in NeuMF output layer init

# Training hyperparameters
BATCH_SIZE     = 256
EPOCHS_GMF     = 20
EPOCHS_MLP     = 20
EPOCHS_NEUMF   = 10
LR_PRETRAIN    = 1e-3     # Adam learning rate
LR_FINETUNE    = 1e-3     # SGD learning rate

# Evaluation
TOP_K        = 10
NUM_NEG_TEST = 99         # 99 negatives + 1 positive = 100 candidates per user

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_data():
    print("Loading data...")
    interactions    = pd.read_csv(os.path.join(DATA_DIR, "interactions.csv"))
    user_features   = pd.read_csv(os.path.join(DATA_DIR, "user_features.csv"))
    program_features = pd.read_csv(os.path.join(DATA_DIR, "program_features.csv"))

    print(f"  interactions   : {len(interactions):>10,} rows")
    print(f"  user_features  : {len(user_features):>10,} rows")
    print(f"  program_features:{len(program_features):>9,} rows")

    positives = interactions[interactions['interaction'] == 1].copy()
    negatives = interactions[interactions['interaction'] == 0].copy()
    print(f"  positives: {len(positives):,}  |  negatives: {len(negatives):,}")

    return positives, negatives


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN / TEST SPLIT  (leave-one-out per user)
# ─────────────────────────────────────────────────────────────────────────────

def split_data(positives, negatives):
    """
    For each user, hold out exactly 1 positive interaction as the test item.
    All remaining positives + all negatives form the training set.
    """
    print("\nSplitting train/test (leave-one-out)...")
    np.random.seed(SEED)

    test_interactions = {}   # user_id -> held-out program_id
    train_pos_indices = []

    for user_id, group in positives.groupby('user_id'):
        idxs = group.index.tolist()
        held_out_idx  = np.random.choice(idxs)
        held_out_item = int(positives.loc[held_out_idx, 'program_id'])
        test_interactions[int(user_id)] = held_out_item
        train_pos_indices.extend(i for i in idxs if i != held_out_idx)

    train_pos  = positives.loc[train_pos_indices]
    train_data = (
        pd.concat([train_pos, negatives], ignore_index=True)
        .sample(frac=1, random_state=SEED)
    )

    print(f"  training rows : {len(train_data):,}")
    print(f"  test users    : {len(test_interactions):,}")

    # Build full positive-item set per user (used to exclude known positives
    # when sampling evaluation negatives)
    user_pos_items = defaultdict(set)
    for _, row in positives.iterrows():
        user_pos_items[int(row['user_id'])].add(int(row['program_id']))

    return train_data, test_interactions, user_pos_items


# ─────────────────────────────────────────────────────────────────────────────
# tf.data DATASET
# ─────────────────────────────────────────────────────────────────────────────

def make_tf_dataset(df, batch_size, shuffle=True):
    users  = df['user_id'].values.astype(np.int32)
    items  = df['program_id'].values.astype(np.int32)
    labels = df['interaction'].values.astype(np.float32)

    ds = tf.data.Dataset.from_tensor_slices(
        ({'user_input': users, 'item_input': items}, labels)
    )
    if shuffle:
        ds = ds.shuffle(buffer_size=200_000, seed=SEED)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

def build_gmf(num_users, num_items, factors):
    """
    GMF: σ( h^T · (p_u ⊙ q_i) )
    p_u and q_i are user/item embeddings; h is the output projection vector.
    """
    user_input = layers.Input(shape=(1,), dtype='int32', name='user_input')
    item_input = layers.Input(shape=(1,), dtype='int32', name='item_input')

    user_embed = layers.Embedding(
        num_users, factors,
        embeddings_initializer='normal',
        name='gmf_user_embed'
    )(user_input)
    item_embed = layers.Embedding(
        num_items, factors,
        embeddings_initializer='normal',
        name='gmf_item_embed'
    )(item_input)

    user_vec = layers.Flatten(name='gmf_user_flat')(user_embed)
    item_vec = layers.Flatten(name='gmf_item_flat')(item_embed)

    gmf_vec = layers.Multiply(name='gmf_multiply')([user_vec, item_vec])

    output = layers.Dense(
        1, activation='sigmoid',
        kernel_initializer='lecun_uniform',
        name='gmf_output'
    )(gmf_vec)

    return keras.Model([user_input, item_input], output, name='GMF')


def build_mlp(num_users, num_items, factors, hidden_layers):
    """
    MLP: σ( h^T · φ_L([p_u; q_i]) )
    Concatenated embeddings through a decreasing tower of dense ReLU layers.
    """
    user_input = layers.Input(shape=(1,), dtype='int32', name='user_input')
    item_input = layers.Input(shape=(1,), dtype='int32', name='item_input')

    user_embed = layers.Embedding(
        num_users, factors,
        embeddings_initializer='normal',
        name='mlp_user_embed'
    )(user_input)
    item_embed = layers.Embedding(
        num_items, factors,
        embeddings_initializer='normal',
        name='mlp_item_embed'
    )(item_input)

    user_vec = layers.Flatten(name='mlp_user_flat')(user_embed)
    item_vec = layers.Flatten(name='mlp_item_flat')(item_embed)

    x = layers.Concatenate(name='mlp_concat')([user_vec, item_vec])

    for i, dim in enumerate(hidden_layers):
        x = layers.Dense(
            dim, activation='relu',
            kernel_initializer='lecun_uniform',
            name=f'mlp_layer_{i}'
        )(x)

    output = layers.Dense(
        1, activation='sigmoid',
        kernel_initializer='lecun_uniform',
        name='mlp_output'
    )(x)

    return keras.Model([user_input, item_input], output, name='MLP')


def build_neumf(num_users, num_items, gmf_factors, mlp_factors, hidden_layers,
                gmf_model=None, mlp_model=None, alpha=0.5):
    """
    NeuMF: fuses GMF and MLP pathways.

    Output layer input = [gmf_multiply (gmf_factors), mlp_last_hidden (hidden_layers[-1])]
    Output layer weights initialized as:
        W_neumf = concat(alpha * W_gmf, (1-alpha) * W_mlp)
        b_neumf = alpha * b_gmf + (1-alpha) * b_mlp
    """
    user_input = layers.Input(shape=(1,), dtype='int32', name='user_input')
    item_input = layers.Input(shape=(1,), dtype='int32', name='item_input')

    # ── GMF pathway ──────────────────────────────────────────────────────────
    gmf_user_embed = layers.Embedding(
        num_users, gmf_factors,
        embeddings_initializer='normal',
        name='neumf_gmf_user_embed'
    )(user_input)
    gmf_item_embed = layers.Embedding(
        num_items, gmf_factors,
        embeddings_initializer='normal',
        name='neumf_gmf_item_embed'
    )(item_input)

    gmf_user_vec = layers.Flatten(name='neumf_gmf_user_flat')(gmf_user_embed)
    gmf_item_vec = layers.Flatten(name='neumf_gmf_item_flat')(gmf_item_embed)
    gmf_out = layers.Multiply(name='neumf_gmf_multiply')([gmf_user_vec, gmf_item_vec])

    # ── MLP pathway ──────────────────────────────────────────────────────────
    mlp_user_embed = layers.Embedding(
        num_users, mlp_factors,
        embeddings_initializer='normal',
        name='neumf_mlp_user_embed'
    )(user_input)
    mlp_item_embed = layers.Embedding(
        num_items, mlp_factors,
        embeddings_initializer='normal',
        name='neumf_mlp_item_embed'
    )(item_input)

    mlp_user_vec = layers.Flatten(name='neumf_mlp_user_flat')(mlp_user_embed)
    mlp_item_vec = layers.Flatten(name='neumf_mlp_item_flat')(mlp_item_embed)

    x = layers.Concatenate(name='neumf_mlp_concat')([mlp_user_vec, mlp_item_vec])
    for i, dim in enumerate(hidden_layers):
        x = layers.Dense(
            dim, activation='relu',
            kernel_initializer='lecun_uniform',
            name=f'neumf_mlp_layer_{i}'
        )(x)

    # ── Fusion ───────────────────────────────────────────────────────────────
    # Concat GMF element-wise product + last MLP hidden layer
    neumf_vec = layers.Concatenate(name='neumf_concat')([gmf_out, x])

    output = layers.Dense(
        1, activation='sigmoid',
        kernel_initializer='lecun_uniform',
        name='neumf_output'
    )(neumf_vec)

    model = keras.Model([user_input, item_input], output, name='NeuMF')

    # ── Weight initialization from pre-trained models (He et al. Section 3.3) ──
    if gmf_model is not None and mlp_model is not None:
        print(f"  Initializing NeuMF from pre-trained weights (alpha={alpha})...")

        # GMF embeddings
        model.get_layer('neumf_gmf_user_embed').set_weights(
            gmf_model.get_layer('gmf_user_embed').get_weights())
        model.get_layer('neumf_gmf_item_embed').set_weights(
            gmf_model.get_layer('gmf_item_embed').get_weights())

        # MLP embeddings
        model.get_layer('neumf_mlp_user_embed').set_weights(
            mlp_model.get_layer('mlp_user_embed').get_weights())
        model.get_layer('neumf_mlp_item_embed').set_weights(
            mlp_model.get_layer('mlp_item_embed').get_weights())

        # MLP hidden layers
        for i in range(len(hidden_layers)):
            model.get_layer(f'neumf_mlp_layer_{i}').set_weights(
                mlp_model.get_layer(f'mlp_layer_{i}').get_weights())

        # NeuMF output layer: alpha-weighted combination
        # W_gmf shape: (gmf_factors, 1)  —  from pre-trained GMF output Dense
        # W_mlp shape: (hidden_layers[-1], 1) — from pre-trained MLP output Dense
        # W_neumf shape: (gmf_factors + hidden_layers[-1], 1)
        W_gmf, b_gmf = gmf_model.get_layer('gmf_output').get_weights()
        W_mlp, b_mlp = mlp_model.get_layer('mlp_output').get_weights()

        W_neumf = np.concatenate([alpha * W_gmf, (1.0 - alpha) * W_mlp], axis=0)
        b_neumf = alpha * b_gmf + (1.0 - alpha) * b_mlp

        model.get_layer('neumf_output').set_weights([W_neumf, b_neumf])

    return model


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION  —  HR@K and NDCG@K (leave-one-out)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(model, test_interactions, user_pos_items, num_neg=NUM_NEG_TEST, top_k=TOP_K):
    """
    For each user:
      - Rank their 1 held-out positive against `num_neg` randomly sampled negatives.
      - HR@K   = 1 if the positive is in the top-K ranked candidates.
      - NDCG@K = 1/log2(rank+1) if positive is in top-K, else 0.

    Returns (mean HR@K, mean NDCG@K) across all test users.
    """
    all_items = set(range(NUM_ITEMS))
    hits, ndcgs = [], []

    for user_id, pos_item in test_interactions.items():
        # Sample negatives: items with no positive interaction for this user
        neg_pool  = list(all_items - user_pos_items[user_id])
        neg_items = random.sample(neg_pool, min(num_neg, len(neg_pool)))

        # Positive is always at index 0
        candidate_items = [pos_item] + neg_items
        n_candidates    = len(candidate_items)

        users_arr = np.full(n_candidates, user_id, dtype=np.int32)
        items_arr = np.array(candidate_items, dtype=np.int32)

        preds = model.predict(
            {'user_input': users_arr, 'item_input': items_arr},
            batch_size=n_candidates,
            verbose=0
        ).flatten()

        # Position of the positive item in descending-score ranking (1-based)
        ranked   = np.argsort(-preds)
        pos_rank = int(np.where(ranked == 0)[0][0]) + 1

        hits.append(1 if pos_rank <= top_k else 0)
        ndcgs.append(1.0 / np.log2(pos_rank + 1) if pos_rank <= top_k else 0.0)

    return float(np.mean(hits)), float(np.mean(ndcgs))


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("FitNova — NeuMF Training Pipeline")
    print("He et al. (2017) Neural Collaborative Filtering")
    print("=" * 60)

    # ── Load & split ─────────────────────────────────────────────────────────
    positives, negatives = load_data()
    train_data, test_interactions, user_pos_items = split_data(positives, negatives)
    train_ds = make_tf_dataset(train_data, BATCH_SIZE, shuffle=True)

    print(f"\nHyperparameters:")
    print(f"  GMF factors : {GMF_FACTORS}")
    print(f"  MLP factors : {MLP_FACTORS} per side  (concat = {2*MLP_FACTORS})")
    print(f"  MLP layers  : {MLP_LAYERS}")
    print(f"  Alpha       : {ALPHA}")
    print(f"  Batch size  : {BATCH_SIZE}")
    print(f"  Eval top-K  : {TOP_K}")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 1 — Pre-train GMF
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 1 — Pre-training GMF  (Adam, BCE)")
    print("=" * 60)

    gmf_model = build_gmf(NUM_USERS, NUM_ITEMS, GMF_FACTORS)
    gmf_model.compile(
        optimizer=optimizers.Adam(learning_rate=LR_PRETRAIN),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    gmf_model.summary()

    gmf_model.fit(train_ds, epochs=EPOCHS_GMF, verbose=1)

    hr_gmf, ndcg_gmf = evaluate(gmf_model, test_interactions, user_pos_items)
    print(f"\nGMF — HR@{TOP_K}: {hr_gmf:.4f}  |  NDCG@{TOP_K}: {ndcg_gmf:.4f}")

    gmf_save_path = os.path.join(OUTPUT_DIR, "gmf_pretrained.keras")
    gmf_model.save(gmf_save_path)
    print(f"GMF saved -> {gmf_save_path}")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 2 — Pre-train MLP
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2 — Pre-training MLP  (Adam, BCE)")
    print("=" * 60)

    mlp_model = build_mlp(NUM_USERS, NUM_ITEMS, MLP_FACTORS, MLP_LAYERS)
    mlp_model.compile(
        optimizer=optimizers.Adam(learning_rate=LR_PRETRAIN),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    mlp_model.summary()

    mlp_model.fit(train_ds, epochs=EPOCHS_MLP, verbose=1)

    hr_mlp, ndcg_mlp = evaluate(mlp_model, test_interactions, user_pos_items)
    print(f"\nMLP — HR@{TOP_K}: {hr_mlp:.4f}  |  NDCG@{TOP_K}: {ndcg_mlp:.4f}")

    mlp_save_path = os.path.join(OUTPUT_DIR, "mlp_pretrained.keras")
    mlp_model.save(mlp_save_path)
    print(f"MLP saved -> {mlp_save_path}")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 3 — Build and fine-tune NeuMF
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3 — NeuMF  (SGD fine-tune from pre-trained weights)")
    print("=" * 60)

    neumf_model = build_neumf(
        NUM_USERS, NUM_ITEMS,
        GMF_FACTORS, MLP_FACTORS, MLP_LAYERS,
        gmf_model=gmf_model, mlp_model=mlp_model,
        alpha=ALPHA
    )
    neumf_model.compile(
        optimizer=optimizers.SGD(learning_rate=LR_FINETUNE, momentum=0.0),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    neumf_model.summary()

    best_hr      = 0.0
    best_ndcg    = 0.0
    best_weights = None

    for epoch in range(1, EPOCHS_NEUMF + 1):
        print(f"\n--- Epoch {epoch}/{EPOCHS_NEUMF} ---")
        neumf_model.fit(train_ds, epochs=1, verbose=1)

        hr, ndcg = evaluate(neumf_model, test_interactions, user_pos_items)
        marker   = "  ← best" if hr > best_hr else ""
        print(f"NeuMF Epoch {epoch:02d} — HR@{TOP_K}: {hr:.4f}  |  NDCG@{TOP_K}: {ndcg:.4f}{marker}")

        if hr > best_hr:
            best_hr      = hr
            best_ndcg    = ndcg
            best_weights = neumf_model.get_weights()

    # Restore best checkpoint
    neumf_model.set_weights(best_weights)

    # ─────────────────────────────────────────────────────────────────────────
    # SAVE
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Saving models and metadata")
    print("=" * 60)

    neumf_save_path = os.path.join(OUTPUT_DIR, "neumf_final.keras")
    neumf_model.save(neumf_save_path)
    print(f"NeuMF saved -> {neumf_save_path}")

    # Metadata needed at inference time
    metadata = {
        'num_users'   : NUM_USERS,
        'num_items'   : NUM_ITEMS,
        'gmf_factors' : GMF_FACTORS,
        'mlp_factors' : MLP_FACTORS,
        'mlp_layers'  : MLP_LAYERS,
        'alpha'       : ALPHA,
        'top_k'       : TOP_K,
        'best_hr'     : best_hr,
        'best_ndcg'   : best_ndcg,
    }
    meta_path = os.path.join(OUTPUT_DIR, "neumf_metadata.pkl")
    with open(meta_path, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"Metadata saved -> {meta_path}")

    # Save user positive-item history (used at inference to filter seen items)
    history_path = os.path.join(OUTPUT_DIR, "user_pos_items.pkl")
    with open(history_path, 'wb') as f:
        pickle.dump(dict(user_pos_items), f)
    print(f"User history saved -> {history_path}")

    # ─────────────────────────────────────────────────────────────────────────
    # FINAL SUMMARY
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"  GMF   — HR@{TOP_K}: {hr_gmf:.4f}  |  NDCG@{TOP_K}: {ndcg_gmf:.4f}")
    print(f"  MLP   — HR@{TOP_K}: {hr_mlp:.4f}  |  NDCG@{TOP_K}: {ndcg_mlp:.4f}")
    print(f"  NeuMF — HR@{TOP_K}: {best_hr:.4f}  |  NDCG@{TOP_K}: {best_ndcg:.4f}")
    print("=" * 60)
    print("Training complete.")


if __name__ == '__main__':
    main()
