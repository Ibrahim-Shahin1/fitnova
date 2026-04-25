# FitNova — Project Overview

**Type:** AI-Powered Fitness Planning Mobile Application  
**Stack:** Flutter (Android) + FastAPI (Python) + TensorFlow + OpenAI GPT-4o-mini  
**Status:** Fitness Planning ✅ complete. Conversational chat / nutrition ✅ complete. **Form Detection ⚠ in domain-gap recovery** — see HANDOFF.md.

---

## ⚠ Form Analysis — Read This First (2026-04-25)

The form-analysis pipeline is end-to-end functional (camera → MediaPipe → MT-TCN → Flutter UI) and the model passes academic gates on the held-out Fit3D MoCap subject s11. **It does not yet generalise to real phone video** — the synthetic-pretrain-only training distribution leaves a domain gap that surfaces as: classifier collapse on unfamiliar exercises, quality-head saturation near 1.0 regardless of form, and knee/hip joint-error channels flatlining to 0.

**For the next chat session, the canonical handoff documents are:**
- **`HANDOFF.md`** — full state of the form-analysis system, what's been built across Phases A–F, what failed, the two roads forward (MediaPipe-domain pseudo-supervised retrain vs. real-world fine-tune from v4).
- **`FIT3D_AIFIT_USAGE.md`** — academic-honesty ledger of what we used and ignored from the Fit3D dataset and the AIFit paper.
- **`COLAB_RUNBOOK.md`** — exact cell-by-cell procedure to retrain on Colab.

The fitness-planning recommender (NeuMF) and conversational chat layers are **untouched and stable**. Do not modify `backend/services/recommender.py` or its tests when working on form analysis.

---

## Table of Contents

1. [Project Description](#1-project-description)
2. [Architecture Overview](#2-architecture-overview)
3. [AI Pipeline — Fitness Planning](#3-ai-pipeline--fitness-planning)
4. [Feature Breakdown](#4-feature-breakdown)
5. [File Structure](#5-file-structure)
6. [Backend API Reference](#6-backend-api-reference)
7. [Data Pipeline](#7-data-pipeline)
8. [Flutter Frontend](#8-flutter-frontend)
9. [Exercise Video Demo System](#9-exercise-video-demo-system)
10. [Setup & Running](#10-setup--running)
11. [Tech Stack](#11-tech-stack)
12. [Academic Justification](#12-academic-justification)

---

## 1. Project Description

FitNova is a graduation project mobile application that uses a 3-layer AI pipeline to generate personalized weekly fitness plans. The user provides basic profile information through a conversational chat interface, and the system returns a fully personalized 7-day workout plan with exercise-specific coaching cues, sets/reps, rest times, and looping video demonstrations.

**Core user flow:**
1. User registers (age, gender, height, weight, injuries, available equipment)
2. User selects a fitness goal (Strength, Cardio, Hypertrophy, Powerbuilding, etc.)
3. Conversational AI chat gathers: experience level, session duration, training frequency
4. AI pipeline generates a personalized weekly plan
5. User browses the plan day-by-day; taps "Watch demo" on any exercise to see a looping video

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Flutter App (Android)                 │
│  Registration → Goal Selection → Chat → Plan Viewer     │
└─────────────────────────┬───────────────────────────────┘
                          │ HTTP (JSON)
                          ▼
┌─────────────────────────────────────────────────────────┐
│                  FastAPI Backend (Python)                │
│                                                          │
│  POST /chat          POST /generate-plan                 │
│  ┌────────────┐      ┌──────────────────────────────┐   │
│  │ Chat       │      │ 3-Layer AI Recommendation    │   │
│  │ Service    │      │ Pipeline                     │   │
│  │ (GPT-4o-  │      │                              │   │
│  │  mini)    │      │ Layer 1: Content-Based Filter│   │
│  └────────────┘      │ Layer 2: NeuMF Re-Ranker     │   │
│                      │ Layer 3: LLM Adapter         │   │
│                      └──────────────────────────────┘   │
│                                                          │
│  GET /static/exercise_videos/{slug}/demo.mp4             │
└─────────────────────────────────────────────────────────┘
```

---

## 3. AI Pipeline — Fitness Planning

The fitness planning system is a **3-layer hybrid recommender system**, which is the standard architecture in academic and production recommendation systems (Netflix, YouTube, Spotify all use this pattern).

### Layer 1 — Content-Based Filter (`backend/services/content_filter.py`)

Narrows 2,598 training programs to a candidate set using a weighted multi-criteria scoring model based on user profile similarity:

| Criterion | Weight | Logic |
|-----------|--------|-------|
| Experience level | 35% | Exact match preferred; adjacent levels allowed |
| Workout type | 30% | Primary type match + partial secondary type bonus |
| Session duration | 20% | Gaussian similarity on normalized duration |
| Training frequency | 15% | Gaussian similarity on normalized frequency |

All features are normalized using min-max statistics computed from the training dataset (`norm_stats.pkl`). Returns top-N candidates (default: 20).

**Academic basis:** Weighted content-based filtering is the standard candidate generation stage in hybrid recommender pipelines (Burke, 2002; Lops et al., 2011).

---

### Layer 2 — NeuMF Re-Ranker (`backend/services/neumf_ranker.py`)

Re-ranks the candidate set using a pre-trained **Neural Matrix Factorization (NeuMF)** model.

**Architecture** (He et al., 2017 — Neural Collaborative Filtering):
- **GMF branch**: Generalized Matrix Factorization — element-wise product of user/item latent vectors
- **MLP branch**: Multi-Layer Perceptron — concatenated embeddings through dense layers [256 → 128 → 64 → 32]
- **NeuMF**: Concatenates GMF and MLP final layers → single sigmoid output (interaction probability)

**Training:**
- Pre-trains GMF and MLP separately, then combines for fine-tuning
- Trained on 973 synthetic user profiles × 2,598 programs → binary interaction matrix
- Evaluation: Leave-one-out protocol, HR@10 and NDCG@10 metrics
- Trained model: `backend/models/neumf_final.keras`

**Cold-Start Handling:**
New users (not in training set) are mapped to the most similar trained user via cosine similarity on normalized feature vectors. The most similar user's learned embeddings are used for scoring.

**Final Score Blend:**
```
final_score = 0.45 × NeuMF_score + 0.25 × content_filter_score + 0.30 × intent_alignment
```
Where `intent_alignment` measures how well a program matches the user's stated training focus (powerbuilding, hypertrophy, etc.).

---

### Layer 3 — LLM Adapter (`backend/services/llm_adapter.py`)

Transforms the selected program template into a fully personalized 7-day plan using **GPT-4o-mini**.

**What the LLM does:**
- Receives the top-ranked program's Week 1 exercises as a structural template
- Adapts exercise selection to user equipment, experience level, and training focus
- Generates specific coaching cues (2 sentences per exercise)
- Formats output as strict JSON schema

**Hard constraints enforced in the prompt:**
- **Equipment**: Only exercises performable with user's listed equipment. Cables require explicit "Cables" entry.
- **Injuries**: Exercises matching injury keyword blacklist are excluded post-generation
- **Volume**: Sets/reps scaled to experience level (beginner: 2-3 × 10-15, intermediate: 3-4 × 8-12, advanced: 4-5 × 5-8)

**Validation & Fallback:**
- 3 retry attempts on API timeout/connection errors
- JSON schema validation on every response
- Deterministic template fallback if LLM fails entirely
- Post-generation sanitization: reps normalization, coaching cue replacement, injury filtering

**Training Focus Modes:**
| Focus | Behavior |
|-------|----------|
| `powerbuilding` | 1-2 heavy compounds at 3-5 reps, 3-5 hypertrophy accessories at 8-12 |
| `powerlifting` | Big-3 focused, 3-5 reps, 3-5 min rest |
| `hypertrophy` | Classic bodybuilding split, 8-15 reps |
| `general` | Balanced upper/lower/core |

---

### Chat Service (`backend/services/chat_service.py`)

Conversational intake powered by GPT-4o-mini with **function calling** to extract structured user data.

Extracted fields:
- `experience_level` (1=beginner, 2=intermediate, 3=advanced)
- `session_duration_hours` (0.5–4.0)
- `workout_frequency` (1–7 days/week)

Returns `status: "ready"` once all three fields are confirmed. The Flutter app then sends the complete profile to `/generate-plan`.

---

## 4. Feature Breakdown

### ✅ Completed — Fitness Planning

| Feature | Status | Details |
|---------|--------|---------|
| User registration | ✅ | Age, gender, height/weight (auto-BMI), injuries, equipment |
| Goal selection | ✅ | Strength, Cardio, Hypertrophy, Powerbuilding, Powerlifting, General |
| Conversational onboarding | ✅ | GPT-4o-mini chat extracts experience, duration, frequency |
| AI plan generation | ✅ | 3-layer pipeline → 7-day personalized plan |
| Equipment constraint | ✅ | Hard constraint in LLM prompt; cables explicit |
| Injury safety | ✅ | Keyword blacklist for 7 injury types; post-generation filtering |
| Exercise video demos | ✅ | 21 exercises, looping MP4, no audio, no crashes |
| Plan viewer UI | ✅ | Day-by-day tabs, sets/reps/rest chips, coaching cues |

### 🔲 Remaining — Upcoming Features

| Feature | Status |
|---------|--------|
| Form Detection & Correction | 🔲 Not started |
| Nutrition Guidance | 🔲 Not started |

---

## 5. File Structure

```
FitNova Application/
│
├── backend/
│   ├── app.py                          # FastAPI server, all endpoints
│   ├── requirements.txt                # Python dependencies
│   ├── .env                            # OpenAI API key (gitignored)
│   │
│   ├── services/
│   │   ├── content_filter.py           # Layer 1: weighted scoring filter
│   │   ├── neumf_ranker.py             # Layer 2: NeuMF neural re-ranker
│   │   ├── llm_adapter.py              # Layer 3: GPT-4o-mini plan generator
│   │   ├── recommender.py              # Orchestrator: chains Layer 1 + 2
│   │   └── chat_service.py             # Conversational intake (GPT + function calling)
│   │
│   ├── models/
│   │   ├── neumf_final.keras           # Trained NeuMF model
│   │   ├── gmf_pretrained.keras        # Pre-trained GMF branch
│   │   ├── mlp_pretrained.keras        # Pre-trained MLP branch
│   │   └── neumf_metadata.pkl          # num_users, num_items
│   │
│   ├── data/
│   │   ├── program_catalog.pkl         # 2,598 training programs
│   │   ├── norm_stats.pkl              # Feature normalization min/max
│   │   ├── user_features.csv           # 973 synthetic user feature vectors
│   │   ├── program_features.csv        # Program feature vectors
│   │   ├── interactions.csv            # User-program interaction matrix
│   │   ├── injury_exercise_blacklist.json  # 7 injury types, keyword blacklists
│   │   ├── exercise_videos.json        # Exercise name→video slug mapping (gitignored, generated)
│   │   ├── workout_taxonomy.py         # Keyword-based program type classifier
│   │   ├── build_catalog.py            # Builds program_catalog.pkl from raw CSV
│   │   ├── generate_interactions.py    # Generates synthetic interaction matrix
│   │   └── augment_catalog.py          # Augments catalog with Yoga/Cardio/HIIT variants
│   │
│   ├── scripts/
│   │   ├── build_exercise_videos.py            # Auto video selection + concat
│   │   └── build_exercise_videos_manual.py     # Manual video selection + concat
│   │
│   ├── training/
│   │   └── train_neumf.py              # NeuMF training script (He et al. 2017)
│   │
│   └── static/                         # Served at /static (gitignored)
│       └── exercise_videos/
│           ├── bench_press/demo.mp4
│           ├── squat/demo.mp4
│           └── ... (21 exercises)
│
├── lib/
│   ├── main.dart                       # App entry point, routes, theme
│   │
│   ├── config/
│   │   └── api_config.dart             # Base URL (localhost vs 10.0.2.2)
│   │
│   ├── models/
│   │   ├── fitness_plan.dart           # FitnessPlan, DayPlan, Exercise models
│   │   └── chat_models.dart            # ChatMessage, ChatResponse models
│   │
│   ├── providers/
│   │   └── user_provider.dart          # Global state (profile, plan, chat context)
│   │
│   ├── services/
│   │   └── api_service.dart            # HTTP calls: /chat, /generate-plan
│   │
│   ├── screens/
│   │   ├── splash_screen.dart          # 1.5s loading → registration
│   │   ├── registration_screen.dart    # Profile form (age, gender, height, etc.)
│   │   ├── home_screen.dart            # Goal selection grid
│   │   ├── chat_screen.dart            # Conversational intake UI
│   │   └── plan_screen.dart            # 7-day plan viewer with tabs
│   │
│   └── widgets/
│       ├── exercise_tile.dart          # Exercise card + video demo dialog
│       ├── chat_bubble.dart            # Chat message bubble
│       └── goal_card.dart              # Goal selection card
│
├── android/
│   └── app/src/main/AndroidManifest.xml   # usesCleartextTraffic=true for HTTP video
│
├── pubspec.yaml                        # Flutter dependencies
└── PROJECT_OVERVIEW.md                 # This file
```

---

## 6. Backend API Reference

### `GET /health`
Returns `{"status": "healthy"}`. Used to verify the server is running.

---

### `POST /generate-plan`

Generates a personalized 7-day fitness plan.

**Request body:**
```json
{
  "experience_level": 2,
  "workout_type": "Strength",
  "session_duration_hours": 1.5,
  "workout_frequency": 4,
  "age": 22,
  "gender": "Male",
  "bmi": 24.5,
  "injuries": ["knees"],
  "training_focus": "powerbuilding",
  "years_training": 2,
  "equipment": ["Barbell", "Dumbbell"]
}
```

**Response body:**
```json
{
  "program_id": 142,
  "program_title": "Intermediate Powerbuilding Push/Pull/Legs",
  "similar_user_id": 87,
  "personalization_notes": "Knee-safe lower body alternatives selected...",
  "source": "llm_generated",
  "weekly_plan": {
    "day_1": {
      "day_number": 1,
      "focus": "Upper Body Push",
      "is_rest_day": false,
      "exercises": [
        {
          "exercise_name": "Bench Press",
          "sets": 4,
          "reps": "4-6",
          "rest_seconds": 180,
          "coaching_cue": "Keep your shoulder blades retracted...",
          "media_url": "/static/exercise_videos/bench_press/demo.mp4"
        }
      ]
    },
    "day_2": { "is_rest_day": true, "exercises": [] }
  }
}
```

---

### `POST /chat`

Processes one turn of the conversational onboarding chat.

**Request body:**
```json
{
  "conversation": [
    {"role": "user", "content": "I have been lifting for 2 years"}
  ],
  "user_context": {
    "workout_type": "Strength",
    "age": 22,
    "gender": "Male",
    "bmi": 24.5,
    "injuries": [],
    "training_focus": "powerbuilding",
    "equipment": ["Barbell", "Dumbbell"]
  }
}
```

**Response body:**
```json
{
  "status": "continue",
  "message": "Got it! How many days per week can you train?",
  "extracted": {
    "experience_level": 2
  }
}
```
When all 3 fields are extracted: `"status": "ready"`.

---

### `GET /static/exercise_videos/{slug}/demo.mp4`

Serves pre-built looping exercise demo videos. Slugs: `bench_press`, `squat`, `deadlift`, `lat_pulldown`, etc.

---

## 7. Data Pipeline

### Program Catalog (`backend/data/build_catalog.py`)
- Source: `programs_detailed_boostcamp_kaggle.csv` (2,598 real gym programs from Kaggle)
- Extracts: title, goal, equipment, level, workout type, exercises per day/week
- Applies `workout_taxonomy.py` to classify program type (Strength/Cardio/Yoga/HIIT)
- Augmented with synthetic Yoga/Cardio/HIIT variants (`augment_catalog.py`) → 3,048 total
- Output: `program_catalog.pkl`

### Interaction Matrix (`backend/data/generate_interactions.py`)
- Creates 973 synthetic user profiles spanning all experience levels, workout types, durations
- Scores every user-program pair using a weighted compatibility model
- Top-scored programs per user → positive interactions; random negatives sampled
- Output: `interactions.csv`, `user_features.csv`

### NeuMF Training (`backend/training/train_neumf.py`)
- Pre-trains GMF (embedding size 8) and MLP ([256, 128, 64, 32]) separately
- Fine-tunes NeuMF by combining both branches
- Evaluates with leave-one-out: HR@10 and NDCG@10
- Output: `gmf_pretrained.keras`, `mlp_pretrained.keras`, `neumf_final.keras`

### Exercise Video Build (`backend/scripts/build_exercise_videos_manual.py`)
- User hand-picked specific video clips from the BTC dataset for 21 exercises
- ffmpeg concatenates selected clips into a single `demo.mp4` per exercise
- Audio stripped (`-an` flag) to prevent OMX codec crashes on Android emulator
- Output: `backend/static/exercise_videos/{slug}/demo.mp4`
- Index: `backend/data/exercise_videos.json` (318 name→slug aliases + fuzzy matching)

---

## 8. Flutter Frontend

### State Management
Uses `provider` package with a single `UserProvider` (`ChangeNotifier`) that holds:
- Registration data (age, gender, height, weight, BMI, injuries, equipment)
- Selected workout type and training focus
- Chat-extracted fields (experience level, session duration, frequency)
- Generated `FitnessPlan` object

### Navigation Flow
```
SplashScreen (1.5s)
  → RegistrationScreen (profile form)
    → HomeScreen (goal selection)
      → ChatScreen (AI conversational intake)
        → PlanScreen (7-day plan viewer)
```

### Plan Viewer
- Tab bar per day (Mon–Sun)
- Rest days show a recovery message
- Active days list `ExerciseTile` widgets per exercise
- Each tile shows: name, sets × reps, rest time, coaching cue
- "Watch demo" button appears if a video exists for that exercise

### Video Demo
- Single `VideoPlayerController.networkUrl` per dialog open
- `setLooping(true)` + `setVolume(0)` — loops silently
- One codec instance created on open, destroyed on close — no OMX crashes
- Shows loading spinner while initializing

---

## 9. Exercise Video Demo System

### 21 Exercises with Videos

| Exercise | Slug | Clips Used |
|----------|------|-----------|
| Barbell Biceps Curl | `barbell_biceps_curl` | clips 0-3 |
| Barbell Bench Press | `bench_press` | clips 1-4 |
| Chest Fly Machine | `chest_fly_machine` | clips 1-4 |
| Deadlift | `deadlift` | clips 1-4 |
| Hammer Curl | `hammer_curl` | clip 3 |
| Hip Thrust | `hip_thrust` | clip 5 |
| Incline Bench Press | `incline_bench_press` | clip 29 |
| Lat Pulldown | `lat_pulldown` | clips 1-5 |
| Lateral Raise | `lateral_raise` | clips 1-3 |
| Leg Extension | `leg_extension` | clips 1-3 |
| Leg Raises | `leg_raises` | clips 12-13 |
| Plank | `plank` | clips 2-3 |
| Pull Up | `pull_up` | clip 21 |
| Push Up | `push_up` | clips 15-16 |
| Romanian Deadlift | `romanian_deadlift` | clip 11 |
| Russian Twist | `russian_twist` | clip 1 |
| Shoulder Press | `shoulder_press` | clip 16 |
| Squat | `squat` | clips 2-3 |
| T Bar Row | `t_bar_row` | clip 2 |
| Dips | `tricep_dips` | clip 12 |
| Tricep Cable Pushdown | `tricep_pushdown` | clips 2-3 |

### Fuzzy Name Matching

The LLM generates exercise names with variations like `"Bent Over Row (Barbell)"`, `"Lat Pulldown (Close Grip)"`, `"Seated Overhead Press (Dumbbell)"`. The video lookup normalizes these:

1. Strip leading muscle category prefix: `"(BACK) Deadlift"` → `"Deadlift"`
2. Strip trailing qualifier: `"Bent Over Row (Barbell)"` → `"Bent Over Row"`
3. Fuzzy match against 318 aliases using `rapidfuzz.token_sort_ratio` at 72% cutoff

This means any reasonable variation of a covered exercise name will resolve to the correct video.

### User-Confirmed Name Mappings
- `Bent Over Row`, `Barbell Row`, `Dumbbell Row`, `Cable Row` → T-Bar Row video
- `Arnold Press`, `Dumbbell Shoulder Press`, `Seated Overhead Press` → Shoulder Press video
- `Skull Crusher`, `Overhead Tricep Extension`, `French Press`, `Tricep Kickback` → Tricep Pushdown video
- `Glute Bridge` (all variants) → Hip Thrust video
- `Close Grip Bench Press` → Bench Press video

---

## 10. Setup & Running

### Prerequisites
- Python 3.10+
- Flutter SDK 3.x
- Android emulator (BlueStacks or Android Studio AVD)
- ADB (`adb connect localhost:5555` for BlueStacks)
- OpenAI API key

### Backend Setup
```bash
cd "FitNova Application/backend"
pip install -r requirements.txt
cp .env.example .env
# Add your OpenAI API key to .env
python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

### Exercise Videos Setup (one-time)
```bash
# Videos are not in git — regenerate from your BTC dataset:
python -m backend.scripts.build_exercise_videos_manual
# Output: backend/static/exercise_videos/ (21 exercises, ~200MB)
```

### Flutter Setup
```bash
flutter pub get
adb connect localhost:5555   # BlueStacks
flutter run -d localhost:5555
```

### Notes
- Android accesses Windows host at `http://10.0.2.2:8000` (configured in `api_config.dart`)
- `android:usesCleartextTraffic="true"` is set in AndroidManifest.xml to allow HTTP video loading
- Backend takes ~90 seconds to start (TensorFlow model loading)

---

## 11. Tech Stack

### Backend
| Component | Technology |
|-----------|-----------|
| API Server | FastAPI + Uvicorn |
| AI Layer 1 | Custom content-based filter (NumPy) |
| AI Layer 2 | TensorFlow/Keras NeuMF |
| AI Layer 3 | OpenAI GPT-4o-mini |
| Chat AI | OpenAI GPT-4o-mini + function calling |
| Data | Pandas, Scikit-learn |
| Fuzzy Matching | rapidfuzz |
| Video Processing | ffmpeg (via imageio-ffmpeg) |
| Serialization | Pickle, JSON |
| Validation | Pydantic v2 |

### Frontend
| Component | Technology |
|-----------|-----------|
| Framework | Flutter 3.x (Dart) |
| State Management | Provider |
| HTTP | http package |
| Video Playback | video_player (ExoPlayer on Android) |
| Navigation | Named routes |
| UI | Material 3 |

### Data Sources
| Dataset | Purpose |
|---------|---------|
| Boostcamp Kaggle Programs (2,598) | Program catalog |
| Synthetic user profiles (973) | NeuMF training |
| BTC Exercise Video Dataset (652 clips, 22 exercises) | Exercise demos |

---

## 12. Academic Justification

FitNova's AI pipeline follows a **3-layer hybrid recommender architecture**, which is the standard approach in both academic literature and production systems:

### Layer 1 — Content-Based Filtering
Standard candidate generation stage. Used to narrow a large item space before applying expensive neural models. Cited in Burke (2002), Lops et al. (2011), and the Netflix Prize literature.

### Layer 2 — NeuMF Neural Collaborative Filtering
Implements He et al. (2017) "Neural Collaborative Filtering" (WWW 2017, ACM). The paper proposes combining GMF (matrix factorization) with MLP (deep learning) through a unified NeuMF framework. Our implementation follows the original architecture: separate pre-training + joint fine-tuning. Cold-start is handled via cosine similarity on user features — a standard cold-start mitigation technique (Lika et al., 2014).

### Layer 3 — LLM-Based Plan Generation
Large language models as personalization engines is an active research area. Our use of GPT-4o-mini with structured JSON output and hard constraint enforcement (injuries, equipment) is consistent with the "LLM as recommender" paradigm (Dai et al., 2023; Bao et al., 2023).

### References
- He, X. et al. (2017). Neural Collaborative Filtering. WWW 2017.
- Burke, R. (2002). Hybrid Recommender Systems: Survey and Experiments. User Modeling and User-Adapted Interaction.
- Lops, P. et al. (2011). Content-based Recommender Systems. Recommender Systems Handbook.
- Lika, B. et al. (2014). Facing the cold start problem in recommender systems. Expert Systems with Applications.
- Dai, S. et al. (2023). AugLLM: Augmenting Large Language Models for Recommendation. arXiv.
