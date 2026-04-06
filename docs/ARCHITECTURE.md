# System Architecture — FitNova Fitness Planning

## High-Level Overview

FitNova uses a **hybrid 3-layer recommendation pipeline** that combines content-based filtering, neural collaborative filtering, and LLM adaptation to deliver personalized fitness plans from day 1 (cold-start) while maintaining high-quality recommendations powered by collaborative learning.

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        Mobile App (Flutter)                      │
│  [ Onboarding Screen (8 inputs) ] → [ Plan Display Screen ]     │
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTP POST /generate-plan
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│                       FastAPI Backend                            │
│  { age, gender, experience_level, workout_type, ... }          │
│                         ↓                                        │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │         LAYER 1: Content-Based Filtering               │   │
│  │  (Cosine Similarity on Program Features)               │   │
│  │  Input: User profile (8 features)                      │   │
│  │  Output: Top-50 candidate program_ids                 │   │
│  │  Cold-start: YES (works for all users, day 1)         │   │
│  └────────────────────┬────────────────────────────────────┘   │
│                       ↓                                        │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │     LAYER 2: NeuMF (Neural Matrix Factorization)       │   │
│  │  - Maps new user to most-similar trained user          │   │
│  │  - Scores 50 candidates using learned embeddings       │   │
│  │  - Outputs: Top-1 program_id                           │   │
│  │  Cold-start: MITIGATED (nearest-neighbor strategy)     │   │
│  └────────────────────┬────────────────────────────────────┘   │
│                       ↓                                        │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │        LAYER 3: LLM Adaptation (Personalization)       │   │
│  │  - Retrieves full program (exercises, sets, reps)      │   │
│  │  - Sends to Gemini/Claude with user context            │   │
│  │  - Generates personalized weekly plan (JSON)           │   │
│  │  - Handles constraints (time, equipment, volume)       │   │
│  └────────────────────┬────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                        │
        Final Output: Personalized Weekly Plan JSON
    { program_name, week_plan: [...] }
```

## Layer 1: Content-Based Filtering

### Purpose
Solves the **cold-start problem**. On day 1, a new user has no interaction history. Collaborative filtering requires historical data to make recommendations. Content-based filtering immediately produces meaningful candidates by comparing the user's profile to program characteristics.

### Algorithm
**Cosine Similarity on Feature Vectors**

For each new user:
1. Extract 8 features: experience_level, goal, equipment, workout_type, session_duration, workout_frequency, age, gender
2. Normalize features to [0, 1] using dataset min/max
3. Build user feature vector: `[exp_level, goal_1hot, equipment_1hot, worktype_binary, duration_norm, freq_norm, age_norm, gender_binary]`
4. Compute cosine similarity against 2,598 program feature vectors (pre-computed from `program_features.csv`)
5. Rank programs by similarity score
6. Return top-50 program_ids

### Feature Alignment

| User Feature | Program Feature | Transformation |
|---|---|---|
| experience_level (1–3) | level_encoded (0–2) | Shift: user - 1 |
| goal (0–7) | goal_encoded (0–7) | One-hot encode both (8 dims) |
| equipment (0–4) | equipment_encoded (0–4) | One-hot encode both (5 dims) |
| workout_type (0–3) | has_cardio, has_strength, has_yoga, has_hiit | Convert user to 4-dim binary |
| session_duration | time_per_workout_minutes | Normalize: (user_val - min) / (max - min) |
| workout_frequency | program_length_weeks | Normalize both |

### Feature Weights
Weights from Gupta et al. (2024) empirical findings:
- Experience level match: **0.35** (most important)
- Workout type preference: **0.30**
- Session duration compatibility: **0.20**
- Workout frequency compatibility: **0.15**

These weights are applied during cosine similarity computation to prioritize important features.

### Computational Complexity
- Time: O(n) where n = 2,598 programs
- Space: O(n × d) where d = feature dimensionality (~50 dims after one-hot encoding)
- Typical latency: < 100ms

### Output
Top-50 program_ids (filtered from 2,598 total)

---

## Layer 2: Neural Matrix Factorization (NeuMF)

### Purpose
Leverages **collaborative filtering** to re-rank content-filter candidates. NeuMF learns latent patterns from 1.36M user-item interactions, capturing non-obvious preferences and program-program similarities that content-based features miss.

### Architecture
Following **He et al. (2017) "Neural Collaborative Filtering"**:

```
User Input (user_id)
    ↓
┌─────────────────────────────────────────────────┐
│   GMF Pathway (Linear)                          │
│  User Embedding (factors=16)  ⊙  Item Embedding │
│  └─→ Element-wise product (output: 16)         │
└────────────┬────────────────────────────────────┘
             │
             └─→ Concatenate → Output Dense(1, sigmoid)
             │
┌────────────┴────────────────────────────────────┐
│   MLP Pathway (Non-linear)                      │
│  User Embedding (factors=32) ∥ Item Embedding   │
│  └─→ Concatenation (input: 64)                 │
│  └─→ Dense(64) + ReLU                          │
│  └─→ Dense(32) + ReLU                          │
│  └─→ Dense(16) + ReLU                          │
│  └─→ Dense(8) + ReLU (output: 8)               │
└────────────┬────────────────────────────────────┘
             │
        Fusion Layer
    Concat([GMF(16), MLP(8)])  → 24 dims
             │
         Dense(1, sigmoid)
             ↓
    Prediction Score ∈ [0, 1]
```

### Training Protocol
1. **Pre-train GMF** (20 epochs, Adam, BCE loss)
2. **Pre-train MLP** (20 epochs, Adam, BCE loss)
3. **Initialize NeuMF** from pre-trained weights:
   - Copy GMF embeddings/output weights (with 0.5 alpha scaling)
   - Copy MLP embeddings/hidden layer weights (with 0.5 alpha scaling)
4. **Fine-tune NeuMF** (10 epochs, SGD, BCE loss)

### Hyperparameters
- GMF factors: 16
- MLP factors: 32 (per side, concat = 64)
- MLP hidden layers: [64, 32, 16, 8]
- Alpha (weight initialization): 0.5
- Learning rate (pretrain): 0.001 (Adam)
- Learning rate (finetune): 0.001 (SGD)
- Batch size: 256
- Loss: Binary cross-entropy

### Training Data
- **Interactions:** 1,361,094 total
  - Positives: 305,807
  - Negatives: 1,055,287 (3.45:1 ratio)
- **Users:** 973
- **Items (programs):** 2,598
- **Sparsity:** 87.90%

Data was synthetically generated using compatibility rules grounded in Gupta et al. (2024).

### Performance Metrics (Leave-One-Out Evaluation)

| Model | HR@10 | NDCG@10 |
|-------|-------|---------|
| GMF | 0.8510 | 0.6275 |
| MLP | 0.8674 | 0.6203 |
| **NeuMF** | **0.8756** | **0.6363** |

**Interpretation:**
- **HR@10 (Hit Rate):** For 87.56% of test users, the held-out program ranks in the top 10 recommendations
- **NDCG@10 (Normalized DCG):** Accounts for ranking position; top-1 is better than top-10

NeuMF outperforms both components, validating the hybrid architecture.

### Cold-Start Handling
New users have no trained embedding. Strategy:
1. Compute cosine similarity between new user features and all 973 trained users' features
2. Find most similar trained user
3. Use that user_id's embedding for NeuMF inference
4. This nearest-neighbor approach is grounded in NCF cold-start literature

### Computational Complexity
- Time: O(k) where k = 50 (batch predict on candidates)
- Latency: ~100ms on CPU

### Output
Top-1 program_id (the single highest-scored program from Layer 1's candidates)

---

## Layer 3: LLM Adaptation

### Purpose
Transforms a generic program recommendation into a **personalized, context-aware weekly plan** that accounts for the user's specific constraints, goals, and preferences.

### Architecture

```
Program Data (exercises, sets, reps, weeks)
    ↓
┌──────────────────────────────────┐
│  Prompt Engineering              │
│  ┌────────────────────────────┐  │
│  │ System: "You are a PT"     │  │
│  ├────────────────────────────┤  │
│  │ Program exercises (Week 1) │  │
│  │ Program metadata           │  │
│  │ User profile               │  │
│  │ User constraints           │  │
│  │ Output schema (JSON)       │  │
│  └────────────────────────────┘  │
└──────────────┬───────────────────┘
               ↓
        Gemini Flash API
               ↓
     JSON Response Parsing
               ↓
    Retry + Validation (x2)
               ↓
Fallback: Raw program data (on failure)
               ↓
    Personalized Weekly Plan (JSON)
```

### Prompt Design
The prompt includes:
1. **System role:** Personal trainer specializing in program adaptation
2. **Program data:** Full exercises from Week 1, metadata (title, level, total weeks)
3. **User context:** Age, experience level, goal, workout frequency, session duration
4. **Constraints:** Equipment available, time budget, injury history (if provided)
5. **Output schema:** Exact JSON structure expected (day, exercises, sets, reps, rest_seconds, notes)
6. **Instructions:** Adjust volume per experience, enforce time/equipment constraints, ensure 3–5 days/week

### Example Output Schema

```json
{
  "program_name": "Weightlifting Mobility Program",
  "week_plan": [
    {
      "day": 1,
      "day_name": "Monday",
      "focus": "Ankle & Hip Mobility",
      "exercises": [
        {
          "name": "Knee-to-wall ankle dorsiflexion",
          "sets": 3,
          "reps": "12 reps",
          "rest_seconds": 60,
          "notes": "Hold each rep for 2 seconds at the deepest point"
        },
        {
          "name": "90/90 Hip Stretch",
          "sets": 3,
          "reps": "2×30sec each side",
          "rest_seconds": 45,
          "notes": "Keep chest upright, lean forward gently"
        }
      ]
    },
    {
      "day": 2,
      "day_name": "Tuesday",
      "focus": "Rest or Light Mobility",
      "exercises": [...]
    }
  ]
}
```

### Error Handling & Fallback
- Primary: Gemini Flash (free tier 15 RPM)
- JSON validation: Strict schema check
- Retry: Up to 2 attempts with `temperature=0`
- Fallback: Return raw program exercises with default set/rep structure

### Computational Complexity
- Latency: 3–5 seconds (LLM API call + parsing)
- Cost: Free tier sufficient for demo (15 requests/minute)

### Output
Structured JSON weekly plan with personalized exercises, sets, reps, rest times, and notes

---

## Data Flow: End-to-End Example

**Input:** User onboarding form
```
age: 28, gender: Male, experience_level: 2 (Intermediate),
goal: 4 (Muscle & Sculpting), equipment: 2 (Full Gym),
workout_type: 1 (Strength), session_duration: 45 min,
workout_frequency: 4 days/week
```

**Layer 1 Output (Top-50 candidates):**
```
[program_id: 42, program_id: 87, program_id: 15, ...]
(all programs matched to Intermediate/Strength/Full Gym/duration 40-50min)
```

**Layer 2 Output (NeuMF re-ranking):**
```
Most similar trained user: user_id 156
Score for program 42: 0.87
Score for program 87: 0.79
Top selection: program_id 42
```

**Layer 3 Output (Personalized plan):**
```json
{
  "program_name": "Strength Building for Muscle Hypertrophy",
  "week_plan": [
    {
      "day": 1,
      "day_name": "Monday",
      "focus": "Upper Body Push",
      "exercises": [
        {"name": "Barbell Bench Press", "sets": 4, "reps": "6-8", ...},
        {"name": "Incline Dumbbell Press", "sets": 3, "reps": "8-10", ...},
        ...
      ]
    },
    ...
  ]
}
```

**Final Output (Mobile App):**
```
Displays week_plan in expandable day cards with exercises,
sets/reps, rest times, and personalized notes.
```

---

## Technology Stack

**Backend**
- Python 3.10+
- FastAPI (async REST API)
- TensorFlow 2.21 + Keras (NeuMF training & inference)
- NumPy, Pandas, scikit-learn (data processing)
- Gemini Flash API (LLM personalization)

**Frontend**
- Flutter 3.0+
- Dart (UI framework)
- http package (API calls)

**Data & Storage**
- CSV (training data)
- TensorFlow Keras format (models: `.keras`)
- Pickle (metadata, interaction history)

**DevOps**
- Git (version control)
- Pytest (backend testing)
- CI/CD ready (GitHub Actions template)

---

## Performance Characteristics

| Metric | Value | Notes |
|--------|-------|-------|
| Layer 1 latency | < 100ms | Cosine similarity (vectorized) |
| Layer 2 latency | < 200ms | Batch predict (50 candidates) |
| Layer 3 latency | 3–5s | LLM API round-trip |
| **Total latency** | ~5–6s | Per user request |
| Model size | ~3MB | neumf_final.keras |
| Training time (one-shot) | ~15 min | 50 epochs on CPU |
| Memory (inference) | ~200MB | Loaded models + data |

---

## Scalability Considerations

**Current scope:** Single-user inference, no database
- Suitable for mobile app (Flutter calling backend API)
- Backend is stateless (models loaded at startup)
- Can handle 10–50 concurrent requests with 2–4 CPU cores

**Future scaling:**
- Add database (PostgreSQL) for interaction history
- Implement caching (Redis) for frequent recommendations
- Batch retraining weekly/monthly as new interaction data accumulates
- GPU deployment for faster Layer 3 (LLM) calls

---

## References

- **He et al. (2017)** — Architecture, training protocol, evaluation metrics
- **Gupta et al. (2024)** — Feature weights, cold-start strategy, dataset insights
- **TensorFlow/Keras docs** — Model implementation details
