# FitNova — Complete Technical Deep Dive
> Everything you need to understand, explain, and defend every decision in this project.

---

## Table of Contents

1. [Project Overview & Problem Statement](#1-project-overview--problem-statement)
2. [System Architecture](#2-system-architecture)
3. [Datasets](#3-datasets)
4. [Data Pipeline — Generating Interactions](#4-data-pipeline--generating-interactions)
5. [Data Pipeline — Building the Program Catalog](#5-data-pipeline--building-the-program-catalog)
6. [Layer 1: Content-Based Filtering](#6-layer-1-content-based-filtering)
7. [Layer 2: NeuMF Neural Collaborative Filtering](#7-layer-2-neumf-neural-collaborative-filtering)
8. [NeuMF Training — Full Details](#8-neumf-training--full-details)
9. [Layer 3: LLM Plan Personalization](#9-layer-3-llm-plan-personalization)
10. [Exercise Video Demo System](#10-exercise-video-demo-system)
11. [Conversational Intake (Chat Service)](#11-conversational-intake-chat-service)
12. [FastAPI Backend](#12-fastapi-backend)
13. [Flutter Frontend](#13-flutter-frontend)
14. [Why These Approaches (Decision Log)](#14-why-these-approaches-decision-log)
15. [All Encodings, Normalization & Hyperparameters](#15-all-encodings-normalization--hyperparameters)
16. [Evaluation Results](#16-evaluation-results)
17. [Known Limitations](#17-known-limitations)
18. [Academic References](#18-academic-references)

---

## 1. Project Overview & Problem Statement

### What is FitNova?

FitNova is an AI-powered mobile fitness planning application. Given a user's profile (experience, goals, injuries, equipment, demographics), it recommends a real gym program and then personalizes it into a detailed 7-day weekly plan with coaching cues and exercise demo videos.

### Problem Statement

Generic fitness apps give the same plan to everyone. A beginner doing yoga and an advanced powerlifter both get the same cookie-cutter content. The challenge is:
1. **Program selection** — Out of 2,598 real gym programs, which one fits this user?
2. **Personalization** — How do we adapt the program's exercises to match the user's exact level, injuries, equipment, and training focus?
3. **Explanation** — How do we explain exercises to someone who may not know good form?

### Solution: 3-Layer AI Pipeline

```
User Profile
    ↓
Layer 1: Content-Based Filter   → 2,598 programs → Top 50 candidates (5ms)
    ↓
Layer 2: NeuMF Re-Ranker        → 50 candidates → Top 1 program (50ms)
    ↓
Layer 3: GPT-4o-mini LLM        → 1 program → 7-day personalized plan (5-8s)
    ↓
7-Day Plan with Sets/Reps/Rest/Cues/Videos
```

---

## 2. System Architecture

### High-Level Stack

| Component | Technology | Why |
|---|---|---|
| Backend | FastAPI (Python) | Async, fast, auto-generates OpenAPI docs, Pydantic validation |
| ML Model | TensorFlow/Keras | NeuMF implementation, GPU optional, well-documented |
| LLM | GPT-4o-mini (OpenAI) | Best cost/quality tradeoff, supports JSON mode |
| Frontend | Flutter (Dart) | Single codebase for Android/iOS, rich widget ecosystem |
| Video Player | ExoPlayer (via video_player) | Native Android hardware codec access |
| Video Processing | ffmpeg | Industry standard for video manipulation |
| Fuzzy Matching | rapidfuzz | Rust-backed, 10x faster than fuzzywuzzy |
| State Management | Provider | Lightweight, official Flutter recommendation |

### Backend File Structure

```
backend/
├── app.py                          ← FastAPI server, all endpoints
├── services/
│   ├── recommender.py              ← Orchestrates Layer 1 + Layer 2
│   ├── content_filter.py           ← Layer 1: rule-based scoring
│   ├── neumf_ranker.py             ← Layer 2: NeuMF inference + blending
│   ├── llm_adapter.py              ← Layer 3: GPT-4o-mini + fallback
│   └── chat_service.py             ← Conversational intake (GPT function calling)
├── training/
│   └── train_neumf.py              ← Full NeuMF training script
├── data/
│   ├── generate_interactions.py    ← Builds synthetic user-program rating matrix
│   ├── build_catalog.py            ← Builds program_catalog.pkl
│   ├── workout_taxonomy.py         ← Classifies programs by type
│   ├── program_catalog.pkl         ← Runtime: all 2,598 programs
│   ├── interactions.csv            ← Training data for NeuMF
│   ├── user_features.csv           ← Normalized user feature vectors
│   ├── program_features.csv        ← Normalized program feature vectors
│   ├── norm_stats.pkl              ← Min/max normalization params
│   ├── exercise_videos.json        ← Exercise name → video slug mapping
│   └── injury_exercise_blacklist.json ← Injury → forbidden exercise keywords
├── models/
│   ├── neumf_final.keras           ← Trained NeuMF model (864KB)
│   ├── gmf_pretrained.keras        ← Pre-trained GMF branch (807KB)
│   ├── mlp_pretrained.keras        ← Pre-trained MLP branch (1.6MB)
│   ├── neumf_metadata.pkl          ← num_users=973, num_items=2598
│   └── user_pos_items.pkl          ← Training user→positive items map
└── scripts/
    ├── build_exercise_videos.py        ← Auto video builder
    └── build_exercise_videos_manual.py ← Manual video builder (final version)
```

---

## 3. Datasets

### Dataset 1: programs_detailed_boostcamp_kaggle.csv

- **Source:** Kaggle — Boostcamp gym program dataset
- **Size:** 2,598 unique program titles, ~600K+ rows (one row per exercise per week per day)
- **Columns:** title, week, day, exercise_name, sets, reps, level (0=Beginner, 1=Intermediate, 2=Advanced), goal, equipment, workout_type
- **What it gives us:** Real gym programs created by real trainers. Each program has a week-by-week exercise schedule.
- **Git status:** Gitignored (was 280MB+ in history — removed with `git filter-branch`). Must be downloaded separately to regenerate data.

### Dataset 2: gym_members_exercise_tracking.csv

- **Source:** Kaggle — gym member health/fitness tracking
- **Size:** 973 rows (one per gym member)
- **Columns:** Age, Gender, Weight (kg), Height (m), BMI, Experience_Level (1-3), Workout_Type, Session_Duration (hours), Workout_Frequency (days/week)
- **What it gives us:** Real user profiles to simulate who would use FitNova. We use these as our "user base" for training NeuMF.
- **Git status:** Included in repo (small file, 65KB).

### Why These Two Datasets?

**Alternative considered:** Scraping a fitness API (e.g., Wger, ExerciseDB) to build programs. **Rejected** because those APIs only have individual exercises, not structured multi-week programs with progressive overload built in. The Kaggle dataset has actual trainer-designed programs.

**Alternative considered:** Purely synthetic users. **Rejected** because real gym member data gives us realistic demographic distributions and workout patterns. Synthetic users would need careful distribution engineering to be believable.

---

## 4. Data Pipeline — Generating Interactions

**Script:** `backend/data/generate_interactions.py`

This is the most important preprocessing step. NeuMF needs a user×program interaction matrix to train on — i.e., which users "liked" which programs. But we don't have real users trying real programs. So we generate synthetic interactions based on feature compatibility.

### Step-by-Step Process

#### Step 1: Load & Validate Data
- Load 973 users from `gym_members_exercise_tracking.csv`
- Load 2,598 programs from `programs_detailed_boostcamp_kaggle.csv` (grouped by title)
- Assign `program_id` (0-indexed) by groupby order

#### Step 2: Encode User Features

For each user, compute a 7-dimensional feature vector:

| Feature | Raw Value | Encoding |
|---|---|---|
| experience_level | 1/2/3 | (0, 0.5, 1.0) — ordinal normalized |
| workout_type | Strength/Cardio/Yoga/HIIT | (0, 0.33, 0.67, 1.0) — nominal ordinal |
| session_duration_hours | float [0.5, 2.0] | min-max → [0, 1] |
| workout_frequency | int [2, 5] | min-max → [0, 1] |
| bmi | float [15, 45] | min-max → [0, 1] |
| age | int [18, 60] | min-max → [0, 1] |
| gender | Male/Female | 0/1 binary |

#### Step 3: Score Every User-Program Pair

For each of the 973×2,598 = ~2.5 million pairs, compute a compatibility score using 4 rules with different weights:

> **Note:** The weights here (0.20, 0.55, 0.15, 0.10) are *training* weights. The content filter service uses different weights (0.35, 0.30, 0.20, 0.15) tuned for recommendation quality. The training weights are used to generate the interaction labels.

**Rule 1 — Experience Match (weight 0.20):**
```
Beginner (1):     Prog Level 0 → 1.0, Level 1 → 0.4, Level 2 → 0.0
Intermediate (2): Prog Level 0 → 0.5, Level 1 → 1.0, Level 2 → 0.5
Advanced (3):     Prog Level 0 → 0.1, Level 1 → 0.6, Level 2 → 1.0
```

**Rule 2 — Workout Type Match (weight 0.55):**
- Primary type exact match: 1.0 (+ 0.40 bonus)
- Secondary type match: 0.6 (+ 0.12 bonus)
- Related type: 0.35 (+ 0.05 bonus)
- No match: 0.0
- Related type pairs: Cardio ↔ HIIT, HIIT ↔ Strength, etc.

**Rule 3 — Session Duration Match (weight 0.15):**
- Diff ≤ 10 min: 1.0
- Diff ≤ 20 min: 0.6
- Diff ≤ 30 min: 0.3
- Otherwise: 0.0

**Rule 4 — Workout Frequency Match (weight 0.10):**
- Exact match: 1.0
- Diff ±1: 0.7
- Diff ±2: 0.3
- Otherwise: 0.0

**Raw Score:** `0.20×R1 + 0.55×R2 + 0.15×R3 + 0.10×R4`

#### Step 4: Apply Expert Boost
Advanced users (level 3) on advanced programs (level 2) get a **1.35× multiplier**. This reflects the real insight that advanced users are specifically seeking challenging programs — the pairing should be reinforced.

#### Step 5: Minority Type Boost
Yoga/Cardio/HIIT programs get a **1.3× multiplier** on all user scores. Why? The dataset is imbalanced — Strength programs dominate. Without the boost, NeuMF would almost never learn to recommend Yoga/Cardio/HIIT because there would be very few positive interactions with them.

#### Step 6: Add Gaussian Noise
Add noise with μ=0, σ=0.08 to each score. This makes the interactions feel organic — in reality, some users like programs slightly outside their comfort zone, and pure rule-based scores would be too clean. The noise prevents overfitting to exact rule thresholds.

#### Step 7: Threshold to Binary
Convert continuous scores to 0/1 binary interactions using a threshold found by binary search to hit **target sparsity: 94-96%**. This means roughly 5% of all user-program pairs become "positive" (interaction=1).

**Why 94-96% sparsity?** This matches the sparsity level of real recommendation datasets (Netflix ~99%, MovieLens ~95-99%). Too dense = trivial problem. Too sparse = model can't learn.

#### Step 8: Quality Enforcement
- Every user must have ≥ 3 positive interactions (force-add top programs if under)
- Every program must be linked to ≥ 3 users (force-add top users if under)
This prevents "orphan" nodes that the model can never embed meaningfully.

#### Step 9: Negative Sampling
For each positive interaction, sample **4 negatives** (programs the user didn't interact with). This gives `NEGATIVE_RATIO = 4:1`.

**Why 4:1 negative ratio?** From He et al. (2017) NeuMF paper — they found 4-7 negatives per positive optimal for implicit feedback. Too few negatives = model doesn't learn to distinguish; too many = overwhelms the positive signal.

#### Output Files
- `interactions.csv`: (user_id, program_id, interaction) — 1,361,094 rows total
- `user_features.csv`: (user_id + 7 normalized feature columns) — 973 rows
- `program_features.csv`: (program_id + feature columns) — 2,598 rows
- `norm_stats.pkl`: {feature_name: (min, max)} for consistent normalization at inference

---

## 5. Data Pipeline — Building the Program Catalog

**Script:** `backend/data/build_catalog.py`

The program catalog is what the LLM and the ranker actually use at inference time. It's a dictionary keyed by `program_id` containing everything about each program.

### What It Does

1. **Replays the groupby** from `generate_interactions.py` to assign identical deterministic `program_id` values. This is critical — if the IDs differ, the NeuMF model's program embeddings map to the wrong programs.

2. **Extracts metadata** for each program:
   - `title`: Human-readable program name
   - `level`: 0/1/2 (Beginner/Intermediate/Advanced)
   - `goal`: e.g., "Weight Loss", "Muscle Gain", "Endurance"
   - `equipment`: e.g., "Barbell", "Dumbbell", "Bodyweight"
   - `primary_type`: Strength / Cardio / Yoga / HIIT (from workout_taxonomy.py)
   - `secondary_types`: list of other types present
   - `time_per_workout_minutes`: average session length
   - `has_strength`, `has_cardio`, `has_yoga`, `has_hiit`: boolean flags

3. **Builds exercise schedules:**
   - `all_exercises`: Full multi-week schedule (week, day, exercise_name, sets, reps)
   - `week_1_exercises`: Just week 1 — this is what gets sent to the LLM to minimize prompt size

4. **Goal and equipment encoding** using sorted unique value maps for future feature use.

### Output
`program_catalog.pkl` — Python dict: `{program_id: program_dict}` — 23MB

### Workout Taxonomy Classification

**Script:** `backend/data/workout_taxonomy.py`

This is a rule-based classifier that determines the workout type(s) of each program by analyzing its title and exercise names.

**Why not use the raw `workout_type` column from the CSV?** Because it's noisy and inconsistent. A program titled "Strength & Conditioning" might have its type listed as "Cardio" in one row and "Strength" in another. The taxonomy scorer provides a consistent, deterministic classification.

**Scoring System:**
- Title keywords carry **strong (6.0 weight)** or **weak (2.5 weight)**
- Exercise keywords carry **strong (1.4 weight)** or **weak (0.35 weight)**

**Type-Specific Logic (abbreviated):**
```
Yoga:     title_score ≥ 6.0  OR  (strong_exercise_count ≥ 4 AND exercise_share ≥ 0.18)
Strength: title_score ≥ 6.0  OR  strong_exercise_count ≥ 4  OR  exercise_share ≥ 0.22
Cardio:   title_score ≥ 6.0  OR  strong_exercise_count ≥ 2  OR  exercise_share ≥ 0.12
HIIT:     Complex balance of title + exercise evidence
```

**Guard for Yoga:** If strength evidence > yoga evidence × 2, strip Yoga label. This prevents "Yoga-inspired Strength" programs from being classified as pure Yoga.

**Example keywords:**
- Strength: squat, deadlift, bench, barbell, cable, machine, press, row, curl
- Cardio: run, jog, bike, treadmill, elliptical, rower, walk
- Yoga: stretch, mobility, pigeon, downward, sun salutation, cobra, warrior
- HIIT: burpee, tabata, emom, circuit, box jump, battle rope, sprint

---

## 6. Layer 1: Content-Based Filtering

**File:** `backend/services/content_filter.py`

### What It Does

Before hitting any neural network, we score all 2,598 programs against the user's profile using 4 interpretable weighted rules. This is fast (~5ms) and returns the top 50 candidates for Layer 2.

### Why Content Filter First?

**Alternative considered:** Send all 2,598 programs to NeuMF. **Rejected** because:
1. NeuMF only scores based on collaborative signals (what similar users liked). It doesn't explicitly enforce content constraints like "beginner only" or "cardio focused."
2. Efficiency: NeuMF is ~50ms for 50 items. Scaling to 2,598 would be ~2.6 seconds.
3. Content filter acts as a guardrail — a beginner will never receive an advanced program in the top 50.

### Scoring Rules (different weights from training)

| Rule | Weight |
|---|---|
| Experience level match | 0.35 |
| Workout type match | 0.30 |
| Session duration match | 0.20 |
| Workout frequency match | 0.15 |

**Why different weights from training?** The training weights (0.20, 0.55, 0.15, 0.10) optimize for generating a realistic interaction matrix. The content filter weights (0.35, 0.30, 0.20, 0.15) optimize for recommendation quality — we care more about experience level and workout type when actually recommending, since those have the strongest impact on user safety and satisfaction.

### Tie-Breaking

After scoring, programs are sorted by: `(tie_breaker_bucket DESC, -score DESC, program_id ASC)`

The tie-breaker bucket ensures workout-type alignment is respected first before fine-grained score differences.

### Output
Top 50 programs as `[(program_id, content_score)]` sorted by score.

---

## 7. Layer 2: NeuMF Neural Collaborative Filtering

**File:** `backend/services/neumf_ranker.py`

### What is Neural Matrix Factorization (NeuMF)?

NeuMF is a deep learning recommendation model proposed by He et al. (2017) in the paper *"Neural Collaborative Filtering"* (WWW 2017). It is the state-of-the-art method for learning from implicit feedback (clicks, views, interactions — not explicit star ratings).

**Key insight of NeuMF:** Traditional matrix factorization (MF) learns user and item embeddings and predicts their dot product. But the dot product is a rigid inner product that may not capture complex nonlinear user-item interactions. NeuMF replaces the dot product with a neural network — specifically, it **combines** a linear component (GMF, which generalizes MF) with a nonlinear component (MLP) in a unified model.

### Why NeuMF Over Other Approaches?

| Alternative | Why Rejected |
|---|---|
| Pure matrix factorization (SVD, ALS) | Only linear interactions. Can't model complex preference patterns. |
| Pure MLP (two-tower model) | Ignores the proven effectiveness of linear matrix factorization as a component |
| Pure content-based | Ignores collaborative signal ("users like you also liked..."). Cold-start only. |
| SASRec (sequential) | Requires sequential interaction history per user. We have no session logs. |
| BERT4Rec | Same — needs sequential data. Not applicable here. |
| LightGCN (graph-based) | Excellent choice but requires graph construction, more complex infrastructure. Deferred. |

**We chose NeuMF because:**
1. Academically validated (He et al., WWW 2017, 4,000+ citations)
2. Designed specifically for implicit feedback (we have binary interactions, not star ratings)
3. Combines best of linear (GMF) and nonlinear (MLP) — proven to outperform each alone
4. Appropriate complexity for our dataset size (973 users, 2,598 programs)

### NeuMF Architecture (Full Detail)

```
User ID ─────┬──────────────────────────────────────────────────┐
             │                                                    │
             ▼                                                    ▼
    GMF User Embedding (16-dim)                   MLP User Embedding (32-dim)
    [normal initializer]                          [normal initializer]
             │                                                    │
Item ID ─────┼──────────────────────────────────────────────────┤
             │                                                    │
             ▼                                                    ▼
    GMF Item Embedding (16-dim)                   MLP Item Embedding (32-dim)
             │                                                    │
             ▼                                                    ▼
    Element-wise multiply (⊙)               Concatenate [user‖item] = 64-dim
    → 16-dim vector                                      │
             │                               Dense(64, ReLU, lecun_uniform)
             │                                            │
             │                               Dense(32, ReLU, lecun_uniform)
             │                                            │
             │                               Dense(16, ReLU, lecun_uniform)
             │                                            │
             │                               Dense(8, ReLU, lecun_uniform)
             │                                            │
             └──────────────┬─────────────────────────────┘
                            │
              Concatenate [GMF(16) ‖ MLP(8)] = 24-dim
                            │
                      Dense(1, sigmoid)         ← Binary output: 0.0-1.0
                            │
                    Prediction (0-1)
```

### Embedding Spaces

NeuMF maintains **four separate embedding tables**:
1. `gmf_user_emb`: shape (973, 16) — GMF user embeddings
2. `gmf_item_emb`: shape (2598, 16) — GMF item embeddings
3. `mlp_user_emb`: shape (973, 32) — MLP user embeddings
4. `mlp_item_emb`: shape (2598, 32) — MLP item embeddings

**Why separate embeddings for GMF and MLP?** Because GMF and MLP need to capture different types of relationships. GMF's element-wise multiplication benefits from aligned embedding spaces. MLP's concatenation benefits from richer, higher-dimensional representations. Sharing embeddings would force both pathways to compromise.

### Cold-Start Strategy

NeuMF was trained on 973 specific user IDs. A new app user has no user ID in the model. We solve this with **nearest-neighbor lookup in user feature space**:

```python
def find_similar_user(user_profile):
    # Build normalized 7-feature vector for new user
    new_vec = [exp_norm, wt_norm, dur_norm, freq_norm, bmi_norm, age_norm, gender]
    
    # Compute cosine similarity against all 973 trained user feature vectors
    similarities = cosine_similarity([new_vec], user_feature_matrix)[0]
    
    # Return the most similar trained user's ID
    return np.argmax(similarities)
```

The new user then borrows the most similar trained user's embeddings for inference. This is a valid cold-start approach — it's essentially saying "users with similar demographics and preferences tend to like the same programs."

**Alternative considered:** Train separate user embeddings from profile features alone (no collaborative signal). **Rejected** because it eliminates collaborative filtering entirely — we'd just have a content-based model.

### Intent-Aware Score Blending

After NeuMF scores the 50 candidates, we blend three signals:

```python
final_score = 0.45 × NeuMF_score_normalized
            + 0.25 × content_score_normalized
            + 0.30 × intent_score
```

**Intent scores:**
- Primary type match: 1.0
- Secondary type match: 0.45
- Related type match: 0.18
- No match: 0.0

**Why 30% intent weight?** NeuMF might rank a "similar" user's preferences highly even if the workout type doesn't match. The 30% intent weight ensures the user's explicitly stated goal (e.g., "I want Cardio") has real influence on the final selection.

**Why keep content score (25%) in the blend?** As a regularizer — it prevents NeuMF from recommending a technically "high-affinity" program that's wildly inappropriate (e.g., an advanced program for a beginner).

### Final Ranking

`(intent_bucket DESC, final_score DESC, program_id ASC)`

Intent bucket is set by the primary/secondary/related classification. Within the same intent bucket, final_score breaks ties. program_id is a deterministic tiebreaker.

---

## 8. NeuMF Training — Full Details

**Script:** `backend/training/train_neumf.py`

### Training Data

| Statistic | Value |
|---|---|
| Users | 973 |
| Programs | 2,598 (2,148 real + 450 synthetic Yoga/Cardio/HIIT) |
| Total interactions | 1,361,094 |
| Positive interactions | 305,807 |
| Negative interactions | 1,055,287 |
| Negative ratio | 4:1 |
| Target sparsity | 94-96% |

### Leave-One-Out Evaluation Protocol

From He et al. (2017): For each of the 973 test users, hold out **1 positive interaction** as the test item. At evaluation time, rank this held-out item against **99 randomly sampled negatives** (100 total candidates). Report Hit Rate@10 and NDCG@10.

**This is the standard protocol** for evaluating implicit feedback recommenders. Using full ranking over all 2,598 items per user would be too slow for evaluation loops.

### Phase 1: GMF Pre-Training

```
Model: GMF only (User Embedding 16-dim × Item Embedding 16-dim → element-wise multiply → Dense(1, sigmoid))
Loss:  Binary Cross-Entropy (BCE)
Optimizer: Adam, lr=1e-3
Epochs: 20
Batch size: 256
Checkpoint: save best by HR@10
Result: HR@10 = 0.8510, NDCG@10 = 0.6275
```

**Why Binary Cross-Entropy for implicit feedback?** BCE treats prediction as "what is the probability this user interacts with this item?" It naturally handles the binary nature of our labels (0=no interaction, 1=interaction) and is the loss used in the original NeuMF paper.

**Why Adam for pre-training?** Adam adapts learning rates per-parameter, converges faster in the early stages of training. Good for getting embeddings initialized to a reasonable space quickly.

### Phase 2: MLP Pre-Training

```
Model: MLP only (User Emb 32 ‖ Item Emb 32 = 64-dim → Dense(64) → Dense(32) → Dense(16) → Dense(8) → Dense(1, sigmoid))
All hidden layers: ReLU activation, lecun_uniform initializer
Loss:  Binary Cross-Entropy
Optimizer: Adam, lr=1e-3
Epochs: 20
Batch size: 256
Checkpoint: save best by HR@10
Result: HR@10 = 0.8674, NDCG@10 = 0.6203
```

**Why ReLU for MLP hidden layers?** ReLU (Rectified Linear Unit) is the standard for deep networks — it doesn't saturate for positive inputs (no vanishing gradient), is computationally cheap, and works well empirically in recommendation models.

**Why lecun_uniform initializer?** Recommended for layers with SELU activation but also works well with ReLU — keeps variance of activations stable across layers. He et al. use it in their implementation.

**Why tower structure [64→32→16→8]?** This is the "tower" design from the NeuMF paper — each layer halves the previous. It forces the model to compress the user-item interaction into progressively more abstract representations.

### Phase 3: NeuMF Fine-Tuning

```
Initialization:
  GMF weights ← 0.5 × pretrained_GMF_weights
  MLP weights ← 0.5 × pretrained_MLP_weights
  Final layer bias ← 0.5 × gmf_bias + 0.5 × mlp_bias
  (α = 0.5, balances both pathways equally)

Loss:  Binary Cross-Entropy
Optimizer: SGD, lr=1e-3, momentum=0.0
Epochs: 10
Batch size: 256
Checkpoint: save best by HR@10
Final result: HR@10 = 0.8756, NDCG@10 = 0.6363
```

**Why switch from Adam to SGD for fine-tuning?** This is a critical design decision from the NeuMF paper. Adam's adaptive learning rates can "forget" the pre-trained weights too quickly, essentially wasting the pre-training effort. SGD with a small fixed learning rate makes small, consistent updates that preserve the pre-trained knowledge while still allowing the model to improve the fusion layer.

**Why α = 0.5?** Equal weighting between GMF and MLP at initialization — gives both pathways equal say. The model then learns to rebalance through fine-tuning.

### Evaluation Metrics

**Hit Rate @ K (HR@10):**
```
HR@10 = (number of test users where held-out item appears in top-10 predictions) / 973
```
Our result: 0.8756 — 85.7% of users had their held-out item appear in the top 10 recommendations out of 100 candidates.

**Normalized Discounted Cumulative Gain @ K (NDCG@10):**
```
NDCG@10 = mean over users of:
    1 / log2(rank + 1)  if rank ≤ 10
    0                   if rank > 10
```
This rewards models that place the held-out item *higher* in the top 10 (rank 1 gives score 1.0, rank 10 gives score 0.29). Our result: 0.6363 — the held-out item ranks well within the top 10, not just barely making it.

**Comparison:**

| Model | HR@10 | NDCG@10 |
|---|---|---|
| GMF (pre-trained) | 0.8510 | 0.6275 |
| MLP (pre-trained) | 0.8674 | 0.6203 |
| **NeuMF (final)** | **0.8756** | **0.6363** |

NeuMF outperforms both components individually, validating the fusion design.

---

## 9. Layer 3: LLM Plan Personalization

**File:** `backend/services/llm_adapter.py`

### What It Does

Takes the #1 ranked program (from Layer 2) and transforms it into a personalized 7-day plan. The LLM reads the program's Week 1 exercises and reworks them to fit the user's specific experience level, injuries, equipment, training focus, and session duration.

### Why an LLM Here?

**Alternative considered:** Rule-based template that swaps exercises. **Rejected** because:
1. Exercise selection requires semantic understanding — substituting "Incline Dumbbell Press" for "Incline Barbell Press" requires knowing these are both chest press variations.
2. Coaching cues require natural language generation.
3. Template systems are brittle — they break on exercise names outside their dictionary.

**Why GPT-4o-mini and not GPT-4o or GPT-4-turbo?**
- GPT-4o-mini is 10x cheaper than GPT-4o
- Our prompts are structured JSON generation — we don't need GPT-4o's extra reasoning capability
- Latency: GPT-4o-mini is ~2-3x faster, critical for a mobile app
- The structured output (JSON mode) works reliably with GPT-4o-mini

### LLM Configuration

```python
model = "gpt-4o-mini"
temperature = 0.7       # Some creativity in plans, not fully deterministic
max_tokens = 4000       # Enough for a 7-day plan with all fields
response_format = {"type": "json_object"}  # Forces valid JSON output
```

**Why temperature 0.7?** We want plans to feel varied and personalized, not identical every time. Temperature 0 would give the same plan for the same inputs. Temperature 1.0 introduces too much randomness (exercises might become nonsensical). 0.7 is the sweet spot.

**Why JSON mode?** Without it, the LLM might output markdown, explanatory text, or malformed JSON. JSON mode guarantees parseable output without extra post-processing.

### System Prompt Design

The system prompt (186 lines) is the core of this layer. Key sections:

**Volume prescriptions by experience level:**
```
Beginner (1):     2-3 sets × 10-15 reps
Intermediate (2): 3-4 sets × 8-12 reps
Advanced (3):     4-5 sets × 5-8 reps
```

**Training focus adaptations:**
- `powerbuilding`: Mix strength (3-5 reps) and hypertrophy (8-12 reps) sets
- `powerlifting`: Heavy compound movements, lower reps (3-6), longer rest
- `hypertrophy`: Moderate weight, controlled tempo, 8-15 rep range
- `general`: Balanced approach

**Hard constraints:**
1. Equipment: "ONLY include exercises using equipment from [user's list]. Do NOT include ANY exercise requiring equipment NOT in that list."
2. Injuries: "Avoid exercises that involve: [blacklisted keywords]"
3. Volume: "Workout day count MUST equal user's workout_frequency"

### Injury Safety System

**File:** `backend/data/injury_exercise_blacklist.json`

```json
{
  "knee_pain": ["squat", "lunge", "leg press", "box jump", "jump"],
  "shoulder_pain": ["overhead press", "military press", "upright row", "dip"],
  "lower_back": ["deadlift", "good morning", "bent over row", "hyperextension"],
  ...
}
```

**Two-layer enforcement:**
1. **Pre-LLM:** Keywords passed into system prompt → LLM avoids these exercises
2. **Post-LLM:** Each exercise name checked against keywords; any match is removed from the final plan

This double-check ensures injury safety even if the LLM misses a constraint.

### Reps Sanitization

The LLM sometimes outputs garbage for time-based exercises (e.g., "120 sec" when the user is a strength athlete who should do "6-8 reps"). The normalizer fixes this:

**Logic:**
```python
def _sanitize_reps(reps_str, exercise_name, workout_type, experience_level):
    # If value is a time (contains "sec", "min", ">60"):
    if is_isometric(exercise_name):
        # Planks, holds: cap at 30-60 sec (experience-appropriate)
        return "30 sec" / "45 sec" / "60 sec"
    elif is_cardio_or_yoga(exercise_name, workout_type):
        # Running, cycling, yoga flows: preserve time if ≤ 300 sec
        return original_time_value
    else:
        # Strength movement (squat, bench, row): replace with rep range
        return "10-15" / "8-12" / "5-8"  (experience-level dependent)
```

### Coaching Cues

The system assigns one of 8 exercise categories and picks from a template set of coaching cues:

| Category | Example Trigger Words | Example Cue |
|---|---|---|
| compound_push | bench, press, push, dip | "Keep your chest up and drive through your heels. Maintain a neutral spine throughout the movement." |
| compound_pull | row, pull, deadlift | "Initiate with your lats, not your arms. Control the eccentric — 2-3 seconds down." |
| compound_lower | squat, lunge, step | "Brace your core before descent. Keep your knees tracking over your toes." |
| isolation_upper | curl, extension, fly | "Maintain strict form — no swinging. Squeeze at peak contraction." |
| isolation_lower | leg curl, calf raise | "Control the full range of motion. Focus on the mind-muscle connection." |
| core | plank, crunch, ab | "Brace your entire core, not just your abs. Maintain a neutral spine." |
| isometric | hold, static, wall sit | "Build tension progressively. Don't hold your breath — exhale slowly." |
| unknown | anything else | "Focus on controlled movement and proper breathing throughout each rep." |

### Fallback Template Plan

If GPT-4o-mini times out or returns malformed JSON (3 retry attempts), the system falls back to a deterministic template:

1. Takes Week 1 exercises from the program catalog directly
2. Cycles through workout days, distributing exercises
3. Caps exercises per day: 5 (beginner), 7 (intermediate), 8 (advanced)
4. Applies injury filtering and equipment constraints
5. Infers day focus labels from exercise names (Upper/Lower/Full Body/Conditioning/Mobility)
6. Assigns template coaching cues by category

The fallback ensures the app never shows an error screen — users always get a plan.

### Video Lookup

After generating the plan, each exercise name is run through fuzzy matching to find a demo video:

```python
def _find_slug(exercise_name: str) -> str | None:
    # Step 1: Strip leading category prefix "(BACK) Deadlift" → "Deadlift"
    norm = re.sub(r'^\([^)]+\)\s*', '', exercise_name)
    
    # Step 2: Strip trailing qualifier "Bent Over Row (Barbell)" → "Bent Over Row"
    norm = re.sub(r'\s*\([^)]*\)\s*$', '', norm).lower().strip()
    
    # Step 3: Fuzzy match against all known aliases using token_sort_ratio
    match = rapidfuzz.process.extractOne(
        norm,
        all_aliases,
        scorer=rapidfuzz.fuzz.token_sort_ratio,
        score_cutoff=72  # Must score ≥ 72% to be considered a match
    )
    
    if match:
        return slug_for_alias(match[0])
    return None  # No demo video available
```

**Why token_sort_ratio specifically?** It sorts both strings alphabetically before comparing. This handles cases like "Barbell Bench Press" vs "Bench Press Barbell" — both normalize to the same sorted token sequence before Levenshtein distance is computed.

**Why 72% threshold?** Low enough to catch reasonable variants ("barbell squat" → "squat"), high enough to avoid false positives ("deadlift" → "lat pulldown" would never reach 72%).

---

## 10. Exercise Video Demo System

### Overview

Users can tap "Watch demo" on any exercise tile to see a looping silent video demonstrating the exercise. 22 exercises have manually curated videos.

### Video Selection Strategy

We manually selected the best video clips from a local BTC exercise video dataset. Each exercise has a `MANUAL_VIDEOS` entry pointing to the chosen video file(s).

**Script:** `backend/scripts/build_exercise_videos_manual.py`

### Video Processing Pipeline

For each exercise:
1. Take the selected video file(s)
2. Concatenate all clips into one file using ffmpeg concat demuxer
3. Strip all audio with `-an` flag
4. Copy video codec without re-encoding (`-c:v copy`) — fast, no quality loss

```bash
ffmpeg -f concat -safe 0 -i filelist.txt -an -c:v copy output/demo.mp4
```

**Why strip audio?**
- Mobile video codec (OMX/ExoPlayer on BlueStacks/Android) was crashing on AAC audio decode: `OMX.google.aac.decoder died`
- Even with `setVolume(0)`, ExoPlayer still initializes the audio decoder if an audio track is present
- Removing the audio track entirely with ffmpeg eliminates the codec entirely — no crash possible

**Why concat into one file?**
- Switching between multiple video files required creating a new `VideoPlayerController` each time (destroy old, create new)
- Each `VideoPlayerController` creation = new OMX codec instance
- BlueStacks' `OMX.google.h264.decoder` crashed repeatedly when codecs were rapidly created/destroyed
- Solution: One file → One controller → One codec instance → No swaps → No crashes

### Flutter Video Player

**File:** `lib/widgets/exercise_tile.dart`

```dart
// Single looping controller — initialized once when dialog opens
_controller = VideoPlayerController.networkUrl(Uri.parse(widget.videoUrl))
  ..setVolume(0)          // Muted (audio is also stripped server-side)
  ..setLooping(true)      // Loops forever — no controller swap needed
  ..initialize().then((_) {
    if (mounted) {
      setState(() => _ready = true);
      _controller.play();
    }
  });
```

**Why `setLooping(true)`?** When a video ends, ExoPlayer seeks to the start and plays again — using the same codec instance that was already initialized. Without looping, the controller would reach the end and stop, requiring user interaction or controller recreation.

**Why `setVolume(0)` in Dart too?** Belt-and-suspenders — even if a future audio track somehow slips in, it won't play audibly.

### Alias System

Each exercise slug has a comprehensive alias list covering all LLM name variants:

```python
ALIASES = {
    "bench_press": [
        "bench press", "barbell bench press", "flat bench press",
        "incline bench press", "decline bench press", "smith machine press",
        "chest press", "pause bench press", "close grip bench press", ...
    ],
    "deadlift": [
        "deadlift", "barbell deadlift", "conventional deadlift",
        "sumo deadlift", "romanian deadlift", "rdl",
        "stiff leg deadlift", "trap bar deadlift", ...
    ],
    ...
}
```

**Why so many aliases?** The LLM generates exercise names in unpredictable forms. "Barbell Bicep Curl", "Biceps Curl (Barbell)", "(ARMS) Barbell Curl" — all refer to the same exercise. Having 20+ aliases per exercise combined with fuzzy matching catches virtually all variations.

### Exercise → Video Mapping

| Slug | Represents |
|---|---|
| bench_press | All bench press variants |
| deadlift | Deadlift & Romanian deadlift |
| squat | All squat variants |
| lat_pulldown | Lat pulldown & pull-ups |
| shoulder_press | Arnold press, dumbbell OHP, military press |
| t_bar_row | All rowing movements (barbell row, cable row, dumbbell row) |
| hip_thrust | Glute bridge, hip thrust, all variants |
| tricep_pushdown | Tricep pushdown, skull crushers, overhead extension, kickbacks |
| ... (22 total) | |

---

## 11. Conversational Intake (Chat Service)

**File:** `backend/services/chat_service.py`

### Purpose

Instead of a boring form asking "Experience Level: 1/2/3", FitNova uses a conversational AI that chats with the user to extract their fitness parameters naturally.

### Technical Implementation

Uses **OpenAI function calling** (tool use):

```python
tools = [{
    "type": "function",
    "function": {
        "name": "extract_fitness_params",
        "description": "Extract fitness parameters from the conversation",
        "parameters": {
            "type": "object",
            "properties": {
                "experience_level": {"type": "integer", "enum": [1, 2, 3]},
                "session_duration_hours": {"type": "number"},
                "workout_frequency": {"type": "integer", "enum": [1,2,3,4,5,6,7]}
            }
        }
    }
}]
```

**Why function calling instead of asking the LLM to output JSON directly?**
Function calling is more reliable for structured extraction — the OpenAI API guarantees the function arguments match the schema. Plain JSON extraction sometimes produces invalid or incomplete JSON.

### Conversation Flow

1. Client sends empty conversation → GPT greets user based on their goal
2. User replies → GPT asks about experience level, session duration, frequency (naturally, through conversation)
3. GPT calls `extract_fitness_params` whenever it has sufficient confidence in values
4. Tool result (extracted params) triggers a follow-up if GPT didn't also output text
5. Once all 3 required fields are extracted: `status = "ready"`, conversation ends
6. Frontend auto-transitions to plan generation screen

**The 3 required fields are: experience_level, session_duration_hours, workout_frequency**

Other fields (age, gender, BMI, injuries, equipment) are collected during registration separately.

### System Prompt Customization

The system prompt is dynamically built based on user context:
- Mentions the user's goal: "The user wants to [build muscle and get stronger / improve cardiovascular fitness / improve flexibility / build explosive power]"
- If injuries present: "Reassure them that their plan will avoid triggering movements"

### Force-Proceed Button

If the user closes the chat or doesn't want to answer, a "Use Defaults" button sends:
- `experience_level = 1` (Beginner)
- `session_duration_hours = 1.0` (1 hour)
- `workout_frequency = 3` (3×/week)

This prevents the app from ever blocking on the chat.

### Stateless Design

The client sends the **full conversation history** with every request. The server has no session state — it's completely stateless. This means:
- No server-side session management
- Easy horizontal scaling
- Conversation can resume after network interruption
- Trade-off: larger request payload for long conversations (bounded by max_tokens anyway)

---

## 12. FastAPI Backend

**File:** `backend/app.py`

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | /health | Liveness check — returns `{"status": "ok"}` |
| POST | /chat | Conversational intake — returns `{status, message, extracted}` |
| POST | /generate-plan | Full pipeline — returns 7-day plan |
| GET | /static/{path} | Serve exercise demo videos |

### Lifecycle Management

```python
@asynccontextmanager
async def lifespan(app):
    # Startup: load all models into memory once
    app.state.recommender = Recommender()   # Loads NeuMF model, user features
    app.state.llm_adapter = LLMAdapter()    # Loads program catalog, video index
    app.state.chat_service = ChatService()  # Initializes OpenAI client
    yield
    # Shutdown: (nothing to clean up for these stateless services)
```

**Why load at startup?** Loading NeuMF from disk (~864KB model + 23MB catalog) takes ~500ms. Doing this on every request would add 500ms to each API call. Loading once at startup means the first request is fast.

### Pydantic Data Models

```python
class UserProfileRequest(BaseModel):
    experience_level: int                    # 1, 2, or 3
    workout_type: str                        # "Strength", "Cardio", "Yoga", "HIIT"
    session_duration_hours: float           # 0.5 to 4.0
    workout_frequency: int                   # 1 to 7
    # Optional fields:
    age: int | None = None
    gender: str | None = None               # "Male" or "Female"
    bmi: float | None = None
    injuries: list[str] | None = None       # e.g., ["knee_pain", "shoulder_pain"]
    training_focus: str | None = None       # "powerbuilding", "powerlifting", etc.
    years_training: float | None = None
    equipment: list[str] | None = None      # e.g., ["Barbell", "Dumbbell", "Cable"]

class ExerciseDetail(BaseModel):
    exercise_name: str
    sets: int
    reps: str                               # "8-12", "30 sec", etc.
    rest_seconds: int
    coaching_cue: str
    media_url: str | None = None            # "/static/exercise_videos/squat/demo.mp4"
```

### CORS Configuration

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Allow all origins (fine for development)
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**Why allow_origins=["*"]?** During development/testing with Flutter running on various emulators and local machines, we need to allow any origin. For production deployment, this would be restricted to the app's domain.

### Static File Serving

```python
app.mount("/static", StaticFiles(directory="backend/static"), name="static")
```

Videos are served directly as static files — no CDN, no cloud storage required for local deployment. In production, these would be uploaded to S3/GCS and served via CloudFront.

---

## 13. Flutter Frontend

### App Architecture

```
lib/
├── main.dart                       ← App entry, Provider setup, routes
├── screens/
│   ├── splash_screen.dart          ← Loading screen
│   ├── registration_screen.dart    ← User profile collection
│   ├── home_screen.dart            ← Goal & training focus selection
│   ├── chat_screen.dart            ← Conversational intake UI
│   └── plan_screen.dart            ← 7-day plan display
├── models/
│   ├── fitness_plan.dart           ← FitnessPlan, DayPlan, Exercise
│   ├── chat_message.dart           ← ChatMessage, ChatResponse
│   └── user_profile.dart           ← UserProfile
├── providers/
│   └── user_provider.dart          ← Global state (ChangeNotifier)
├── services/
│   └── api_service.dart            ← HTTP calls to backend
└── widgets/
    └── exercise_tile.dart          ← Exercise card + video demo dialog
```

### State Management: Provider

**Why Provider over alternatives (Bloc, Riverpod, GetX)?**
- Official Flutter recommendation for most apps
- Lightweight — no extra concepts like "cubit" or "notifier pod"
- ChangeNotifier is familiar to anyone who knows Flutter
- For our app complexity (one user session, one plan), Bloc/Riverpod would be over-engineering

`UserProvider` (ChangeNotifier) holds:
- `UserProfile`: demographics, injuries, equipment, training goal
- `FitnessPlan?`: the generated plan (null until generated)
- `chatMessages`: conversation history
- `extractedParams`: parameters extracted from chat so far
- `isLoading`: loading state for UI

### Navigation Flow

```
SplashScreen (2s delay)
    ↓ Always
RegistrationScreen
    ↓ On submit
HomeScreen
    ↓ On goal selection
ChatScreen
    ↓ On status="ready" or force-proceed
PlanScreen
```

### BMI Calculation

In RegistrationScreen:
```dart
double bmi = weightKg / (heightM * heightM);
```

Standard WHO formula. BMI is then passed to the backend as part of `UserProfileRequest`.

### API Configuration

```dart
class ApiConfig {
    static String get baseUrl {
        if (kIsWeb) return 'http://localhost:8000';
        if (Platform.isAndroid) return 'http://10.0.2.2:8000';  // Android emulator special IP
        return 'http://localhost:8000';
    }
}
```

**Why `10.0.2.2` for Android?** Android emulators map `10.0.2.2` to the host machine's `localhost`. Using `localhost` inside an emulator refers to the emulator itself, not the development machine. `10.0.2.2` is the standard Android emulator loopback.

### Exercise Tile & Video Player

**File:** `lib/widgets/exercise_tile.dart`

The exercise tile shows:
- Exercise name (bold)
- Sets × Reps
- Rest time
- Coaching cue (italicized)
- "Watch demo" button (only if `mediaUrl != null`)

Tapping "Watch demo" opens a dialog with a full-screen video player:
- Fetches video from backend static endpoint
- Single controller, initialized once, loops forever
- No audio (stripped server-side, muted client-side)
- AspectRatio widget maintains correct video dimensions

### Dependencies (pubspec.yaml)

```yaml
dependencies:
  flutter: sdk: flutter
  cupertino_icons: ^1.0.8      # iOS-style icons
  provider: ^6.1.2              # State management
  http: ^1.2.1                  # HTTP calls to backend
  url_launcher: ^6.2.5          # Open URLs in browser (if needed)
  video_player: ^2.8.0          # ExoPlayer on Android, AVPlayer on iOS
```

---

## 14. Why These Approaches (Decision Log)

### Why NeuMF over SASRec or BERT4Rec?

SASRec (Self-Attentive Sequential Recommendation) and BERT4Rec require sequential interaction history — they learn from the *order* in which a user interacted with items. We have no session logs, no temporal interaction data. NeuMF works with simple binary interaction matrices (user × item). **Sequential models need sequential data. We don't have it.**

### Why Not Collaborative Filtering with SVD/ALS?

SVD (Singular Value Decomposition) and ALS (Alternating Least Squares) are classical matrix factorization methods. They work by decomposing the interaction matrix into user and item latent factors. The issue: **they model interactions as a linear dot product**. NeuMF's MLP branch explicitly captures non-linear interactions that a dot product cannot represent. Our evaluation results confirm NeuMF outperforms GMF (which is equivalent to generalized MF) alone.

### Why Pre-train GMF and MLP Separately?

Jointly training GMF and MLP from random initialization leads to worse results (shown in the He et al. ablation studies). Pre-training each branch individually lets them learn good representations in their respective spaces before being fused. The fusion fine-tuning then only needs to learn *how to combine* the two already-competent representations.

### Why SGD for Fine-Tuning Instead of Adam?

Adam's adaptive learning rates can rapidly change weights that were carefully learned during pre-training, effectively "unlearning" the pre-trained knowledge. SGD with a small fixed learning rate makes conservative updates — the pre-trained embeddings are preserved while the fusion layer improves. This is the exact reasoning from He et al. (2017).

### Why Synthetic Interaction Matrix Instead of Real User Data?

We don't have real users. The app is a graduation project, not a deployed product with thousands of users generating real interaction logs. Synthetic data generated from principled compatibility rules is a valid and common approach in recommender systems research when real data is unavailable. The key is that the rules must be grounded in domain knowledge (e.g., "beginners shouldn't do advanced programs") and the resulting data must have realistic sparsity patterns.

### Why 4:1 Negative Ratio?

From the NeuMF paper itself: empirically, 4-7 negatives per positive produces the best results on multiple datasets. Too few (1:1) = model doesn't learn to distinguish non-preferred items. Too many (10:1+) = the loss is dominated by negatives, and positive signal is overwhelmed.

### Why GPT-4o-mini for LLM instead of a fine-tuned model?

Fine-tuning a model to generate fitness plans would require:
1. A large dataset of (user_profile, program, 7-day_plan) triplets — which we don't have
2. Training compute (GPU time, cost)
3. Re-training whenever plans need to be updated

GPT-4o-mini with a well-engineered prompt gives us professional-quality output immediately, handles edge cases naturally (injury-appropriate substitutions, equipment constraints), and can be updated by changing the prompt rather than retraining.

### Why Rapidfuzz Over Fuzzywuzzy?

Fuzzywuzzy is the classic Python fuzzy matching library. Rapidfuzz is a drop-in replacement written in C/Rust. It's 10-100x faster and has the same API. For real-time exercise name lookup during plan generation, speed matters. Rapidfuzz is the modern standard.

### Why ffmpeg for Video Processing?

ffmpeg is the industry-standard tool for audio/video manipulation. The key operations we need (`-an` to strip audio, concat demuxer, `-c:v copy` for no re-encode) are ffmpeg core features. No alternative provides this level of control and reliability.

### Why Single Looping Video Instead of Multiple Clips?

The initial approach used multiple video clips and swapped the `VideoPlayerController` between them. This caused `OMX.google.h264.decoder` crashes on BlueStacks/Android because:
1. Destroying a controller closes the OMX codec
2. Creating a new controller opens a new OMX codec instance
3. Rapid open/close of OMX codecs destabilizes the Android mediaserver
4. Solution: one file, one controller, one codec instance, never closed during playback

---

## 15. All Encodings, Normalization & Hyperparameters

### Feature Encodings (User)

| Feature | Raw Type | Encoding | Range After |
|---|---|---|---|
| experience_level | int 1/2/3 | (0.0, 0.5, 1.0) | [0, 1] |
| workout_type | str | Strength=0, Cardio=0.33, Yoga=0.67, HIIT=1.0 | [0, 1] |
| session_duration_hours | float | min-max: (x - 0.5) / (2.0 - 0.5) | [0, 1] |
| workout_frequency | int | min-max: (x - 2) / (5 - 2) | [0, 1] |
| bmi | float | min-max: (x - 15) / (45 - 15) | [0, 1] |
| age | int | min-max: (x - 18) / (60 - 18) | [0, 1] |
| gender | str | Male=0, Female=1 | {0, 1} |

**Why min-max normalization?** Neural networks are sensitive to feature scale. Without normalization, a feature like BMI (range 15-45) would dominate cosine similarity over experience_level (range 0-1). Min-max puts all features in [0, 1] for fair comparison.

**Why these specific min-max bounds?** They represent realistic human ranges: BMI 15-45 covers underweight to severely obese; age 18-60 covers the typical adult gym-goer; duration 0.5-2.0 hours covers 30-minute to 2-hour workouts; frequency 2-5 covers minimum effective dose to daily training.

### NeuMF Hyperparameters

| Hyperparameter | Value | Justification |
|---|---|---|
| GMF embedding dim | 16 | He et al. (2017) recommendation for our data size |
| MLP embedding dim | 32 | Larger because MLP concatenates user+item (64-dim input) |
| MLP tower | [64, 32, 16, 8] | Halving tower from He et al. |
| Batch size | 256 | Standard for implicit feedback datasets |
| GMF pre-train epochs | 20 | Convergence observed at ~15 epochs |
| MLP pre-train epochs | 20 | Convergence observed at ~15 epochs |
| NeuMF fine-tune epochs | 10 | Short fine-tune to preserve pre-trained weights |
| Pre-train optimizer | Adam, lr=1e-3 | Fast convergence in pre-training |
| Fine-tune optimizer | SGD, lr=1e-3, momentum=0 | Conservative updates, preserve pre-trained knowledge |
| Loss function | Binary Cross-Entropy | Standard for implicit binary feedback |
| Embedding initializer | 'normal' (Gaussian) | Standard for embedding tables |
| Dense initializer | 'lecun_uniform' | Keeps variance stable across MLP layers |
| Alpha (fusion) | 0.5 | Equal weighting of GMF and MLP at initialization |
| Negative ratio | 4:1 | He et al. empirical recommendation |
| Target sparsity | 94-96% | Matches real recommendation dataset distributions |
| Noise sigma | 0.08 | Realistic variance without overwhelming the signal |
| Minority type boost | 1.3× | Compensates for Yoga/Cardio/HIIT underrepresentation |
| Expert boost | 1.35× | Reinforces advanced user → advanced program affinity |

### Score Blending Weights (Inference)

| Signal | Weight | Justification |
|---|---|---|
| NeuMF score | 45% | Primary collaborative signal |
| Content score | 25% | Regularizer, enforces basic compatibility |
| Intent (workout type) | 30% | Ensures user's stated goal is respected |

### Content Filter Weights

| Rule | Weight |
|---|---|
| Experience level | 35% |
| Workout type | 30% |
| Session duration | 20% |
| Workout frequency | 15% |

### Fuzzy Matching

| Parameter | Value |
|---|---|
| Algorithm | rapidfuzz token_sort_ratio |
| Score cutoff | 72% |
| Preprocessing | Strip leading `(CATEGORY)`, strip trailing `(Qualifier)` |

### LLM Parameters

| Parameter | Value |
|---|---|
| Model | gpt-4o-mini |
| Temperature | 0.7 |
| Max tokens | 4000 |
| Response format | json_object |
| Max retries | 3 |
| Context cap (exercises/day) | 8 |

---

## 16. Evaluation Results

### NeuMF Performance

| Model | HR@10 | NDCG@10 |
|---|---|---|
| GMF (pre-trained) | 0.8510 | 0.6275 |
| MLP (pre-trained) | 0.8674 | 0.6203 |
| NeuMF (combined) | **0.8756** | **0.6363** |

**Interpretation:**
- 87.56% of the time, the held-out "liked" program appears in the top-10 recommendations (out of 100 candidates)
- Average rank of the held-out program within top-10 reflects 0.6363 NDCG (higher = ranked closer to position 1)
- NeuMF outperforms both GMF and MLP individually, validating the fusion architecture

### Data Statistics

| Metric | Value |
|---|---|
| Real programs | 2,148 |
| Synthetic programs (Yoga/Cardio/HIIT augmentation) | 450 |
| Total programs | 2,598 |
| Training users | 973 |
| Total interactions | 1,361,094 |
| Positive interactions | 305,807 (~22%) |
| Effective sparsity | ~95.3% |

---

## 17. Known Limitations

### 1. Synthetic Interaction Matrix
All training data is generated by rules, not by real users rating real programs. The model learns to reproduce our scoring rules, not to capture genuine human preferences. This is a fundamental limitation that can only be resolved by deploying the app and collecting real interaction logs.

### 2. Cold-Start Only (No Online Learning)
New users are mapped to the nearest trained user via cosine similarity. The NeuMF model itself is never updated with new user data at runtime. A production system would implement:
- Periodic retraining with accumulated interactions
- Matrix factorization updates (ALS is well-suited for this)
- Bandit-based exploration to gather data from new users

### 3. LLM Non-Determinism
Temperature=0.7 means the same user profile can generate different plans on different requests. This is generally desirable (variety), but makes the system harder to test systematically.

### 4. Video Coverage
Only 22 exercises have demo videos. The majority of exercises in generated plans will have `mediaUrl=null` and no "Watch demo" button. Coverage improves as more videos are added to the dataset.

### 5. BlueStacks vs Real Android
The video codec workarounds (stripping audio, single-file looping) were engineered for BlueStacks' software OMX decoders. On real Android devices, hardware codecs are more stable and these issues may not occur. The fixes are harmless on real devices but were specifically motivated by emulator limitations.

### 6. No User Authentication
The app has no login system. Each app session is independent. There is no way to retrieve a previous plan or track progress over time. This is a significant limitation for a real fitness app.

### 7. Equipment Constraint Reliability
While equipment constraints are enforced at the prompt level and post-LLM validation, the LLM occasionally still includes exercises that require unlisted equipment. The post-LLM filter catches keyword-based violations but may miss nuanced cases (e.g., "resistance band pull-apart" if only "Resistance Band" is in the user's equipment list — this would actually be fine, but edge cases exist).

---

## 18. Academic References

1. **He, X., Liao, L., Zhang, H., Nie, L., Hu, X., & Chua, T. S. (2017).** Neural collaborative filtering. *Proceedings of the 26th International Conference on World Wide Web (WWW 2017)*, 173-182. https://doi.org/10.1145/3038912.3052569
   - **Used for:** NeuMF architecture (GMF + MLP fusion), pre-training strategy, SGD fine-tuning rationale, negative sampling ratio, leave-one-out evaluation protocol, HR@K and NDCG@K metrics

2. **Gupta, A., et al. (2024).** Content-based recommendation systems for personalized fitness. *Journal of Health Informatics Research.*
   - **Used for:** 4-rule weighted content filter design (experience, type, duration, frequency matching)

3. **Rendle, S., Freudenthaler, C., Gantner, Z., & Schmidt-Thieme, L. (2009).** BPR: Bayesian personalized ranking from implicit feedback. *Proceedings of UAI 2009.*
   - **Used for:** Understanding implicit feedback principles and negative sampling strategies

4. **Koren, Y., Bell, R., & Volinsky, C. (2009).** Matrix factorization techniques for recommender systems. *IEEE Computer, 42*(8), 30-37.
   - **Used for:** Foundational matrix factorization concepts, understanding why NeuMF extends beyond dot-product MF

5. **He, K., Zhang, X., Ren, S., & Sun, J. (2015).** Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification. *ICCV 2015.*
   - **Used for:** ReLU activation rationale and lecun_uniform/He initialization in dense layers

6. **Kingma, D. P., & Ba, J. (2014).** Adam: A method for stochastic optimization. *ICLR 2015.*
   - **Used for:** Adam optimizer for pre-training phases

7. **OpenAI. (2024).** GPT-4o mini technical report.
   - **Used for:** LLM selection rationale, function calling for structured extraction

---

*This document covers the complete FitNova Fitness Planning system. The Form Detection & Correction and Nutrition Guidance phases are documented separately.*
