# Development Roadmap — FitNova Fitness Planning

This document outlines the 7-phase implementation plan for the FitNova fitness planning feature. Each phase is sequential, with clearly defined deliverables and dependencies.

## Current Status: Phase 1 (Starting)

**Completed:**
- ✅ NeuMF model trained and saved
- ✅ Training pipeline implemented
- ✅ Project initialized with Git
- ✅ Documentation structure in place

**Next:** Program Catalog Builder

---

## Phase 1: Data Pipeline & Program Catalog

**Goal:** Build the artifact that bridges NeuMF's numeric program_ids to actual exercise schedules.

### What It Does
- Loads `programs_detailed_boostcamp_kaggle.csv` (600K+ rows, 2,598 programs)
- Reconstructs the program_id assignment (replaying the exact `groupby('title', sort=False)` logic)
- For each program_id, aggregates all exercises into a structured list
- Saves as `backend/data/program_catalog.pkl`

### Why This Matters
NeuMF outputs a program_id (e.g., 42). Without this catalog, we cannot retrieve what exercises are in that program. Layer 3 (LLM) needs the actual exercise data to personalize the plan.

### Deliverables
- `backend/data/build_catalog.py` — catalog builder script
- `backend/data/program_catalog.pkl` — compiled catalog (dict mapping program_id → program data)

### Verification
- `len(catalog) == 3048` (2,598 real + 450 synthetic augmentation)
- `catalog[0]['title'] == "Weightlifting Mobility program"`
- Spot-check 5 random programs against source CSV

### Dependencies
None — this is the foundation.

---

## Phase 2: Layer 1 — Content-Based Filtering

**Goal:** Match new user profiles to candidate programs using cosine similarity.

### What It Does
- Takes 8-feature user profile (age, gender, experience level, goal, equipment, workout type, session duration, workout frequency)
- Computes cosine similarity against 2,598 program feature vectors
- Returns top-50 most compatible program_ids

### Why This Matters
Solves the **cold-start problem**. A brand-new user has no interaction history, so collaborative filtering alone can't help. Content-based filtering works immediately for day-one users.

### Deliverables
- `backend/services/content_filter.py` — ContentBasedFilter class
- Feature alignment design with normalized dimensions
- Top-N candidate selection

### Feature Alignment
| User Field | Program Field | Encoding |
|---|---|---|
| experience_level (1–3) | level_encoded (0–2) | Shift by -1 |
| goal (0–7) | goal_encoded (0–7) | One-hot (8 dims) |
| equipment (0–4) | equipment_encoded (0–4) | One-hot (5 dims) |
| workout_type (single 0–3) | has_cardio/strength/yoga/hiit | Convert to 4-dim binary |
| session_duration (min) | time_per_workout_minutes | Normalize with data min/max |
| workout_frequency (days/week) | program_length_weeks | Normalize with data min/max |
| age, gender | (no program equivalent) | Exclude from cosine, pass to Layer 3 |

### Verification
- Test with beginner/strength profile → top candidates are beginner strength programs
- Measure overlap between top-50 and actual positives for 10 real users → expect 15–30%
- Latency < 100ms for 2,598 programs

### Dependencies
- Phase 1 (needs program feature matrix)

---

## Phase 3: Layer 1+2 Integration — Recommendation Pipeline

**Goal:** Wire content filter output into NeuMF for re-ranking.

### What It Does
- Content filter produces top-50 candidate program_ids
- Maps new user to most-similar existing user (nearest-neighbor cold-start)
- NeuMF scores all 50 candidates for that user
- Returns top-1 program_id

### Why This Matters
Combines the strengths of both approaches:
- Content-based handles cold-start (day 1 users)
- NeuMF leverages collaborative signal from 973 trained users
- Hybrid approach outperforms either alone

### Cold-Start Strategy
For a new user with no embedding:
1. Compute cosine similarity between new user's profile and all 973 rows in `user_features.csv`
2. Find the most similar existing user
3. Use that user_id for NeuMF inference
4. This is academically defensible (nearest-neighbor cold-start in NCF literature)

### Deliverables
- `backend/services/neumf_ranker.py` — NeuMFRanker class
- `backend/services/recommender.py` — orchestrator (profile → filter → rank → top-1 program_id)

### Verification
- End-to-end test: raw user profile → program_id → expect order different from content-filter-only
- Latency < 2 seconds (content filter + NeuMF)
- Cold-start correctness: new user mapped to sensible existing user

### Dependencies
- Phase 1 (catalog), Phase 2 (content filter)

---

## Phase 4: Layer 3 — LLM Adaptation

**Goal:** Personalize the top program into a weekly plan using an LLM.

### What It Does
- Takes program_id from Phase 3
- Retrieves full program data from Phase 1 catalog
- Builds prompt: program exercises + user context (age, experience, goals, constraints)
- Sends to Gemini Flash or Claude API
- Parses response as structured JSON

### Why This Matters
Raw programs are generic. LLM adaptation makes them personal:
- Adjust volume (sets/reps) based on experience level
- Modify duration to fit user's available time
- Suggest alternatives for unavailable equipment
- Add motivational notes based on goal

### Deliverables
- `backend/services/llm_adapter.py` — LLMAdapter class
- Prompt engineering for fitness context
- JSON schema definition and parsing
- Retry logic + fallback (return raw plan on LLM failure)

### Output Schema
```json
{
  "program_name": "Weightlifting Mobility Program",
  "week_plan": [
    {
      "day": 1,
      "day_name": "Monday",
      "focus": "Mobility",
      "exercises": [
        {
          "name": "Knee-to-wall ankle dorsiflexion",
          "sets": 3,
          "reps": "12 reps",
          "rest_seconds": 60,
          "notes": "Hold each rep for 2 seconds"
        }
      ]
    }
  ]
}
```

### LLM Provider
- Primary: Gemini Flash (free tier: 15 RPM, sufficient for demo)
- Fallback: OpenAI (if needed)
- Abstracted in code for easy switching

### Verification
- 3 profiles (beginner/intermediate/expert) → different volumes
- Valid JSON output 100% of time (with fallback)
- Latency < 5 seconds per request
- No hallucinations (exercise names match program data)

### Dependencies
- Phase 1 (catalog), Phase 3 (recommender)

---

## Phase 5: FastAPI Server

**Goal:** Expose `POST /generate-plan` REST API.

### What It Does
- Loads all models at startup
- Accepts user profile dict
- Orchestrates Phases 1–4 pipeline
- Returns personalized weekly plan JSON

### Deliverables
- `backend/main.py` — FastAPI app with lifespan manager
- `backend/routers/plan.py` — endpoint implementation
- `backend/schemas.py` — Pydantic request/response models
- `backend/config.py` — settings from .env
- `backend/utils/encoding.py` — encoding helpers (string → int)

### API Specification

**Endpoint:** `POST /generate-plan`

**Request Body:**
```json
{
  "age": 28,
  "gender": "Male",
  "experience_level": 2,
  "workout_type": "Strength",
  "session_duration": 45,
  "workout_frequency": 4,
  "goal": "Muscle & Sculpting",
  "equipment": "Full Gym"
}
```

**Response (200):**
```json
{
  "program_name": "...",
  "week_plan": [...]
}
```

**Error Responses:**
- 422: Validation error (missing/invalid field)
- 500: LLM failure (with fallback plan)
- 503: Models not loaded

### Threading & Async
- TensorFlow is not thread-safe → use asyncio.Lock around model.predict()
- All I/O operations async (database, LLM calls)

### Verification
- curl test with valid/invalid inputs
- 10 concurrent requests → no crashes
- Latency: 2–5 seconds per request

### Dependencies
- All previous phases (1–4)

---

## Phase 6: Flutter Frontend

**Goal:** Build mobile UI for onboarding and plan display.

### Screens

**1. Profile Screen**
- Age: TextFormField
- Gender: DropdownButton (Male/Female)
- Experience Level: SegmentedButton (Beginner / Intermediate / Expert)
- Goal: Dropdown (8 options)
- Workout Type: Dropdown (Cardio / Strength / Yoga / HIIT)
- Equipment: Dropdown (At Home / Dumbbell Only / Full Gym / Garage Gym / Unknown)
- Session Duration: Slider (15–120 min)
- Workout Frequency: Slider (1–7 days/week)
- Button: "Generate My Plan"

**2. Plan Screen**
- Program name header
- Expandable cards per day
- Exercise rows: name | sets × reps | rest time | notes
- Loading spinner during API call
- Error state with retry button

### Deliverables
- `lib/screens/profile_screen.dart`
- `lib/screens/plan_screen.dart`
- `lib/services/api_service.dart`
- `lib/models/plan_model.dart`
- Updated `pubspec.yaml` (add `http: ^1.2.0`)

### State Management
StatefulWidget is sufficient for this scope (no BLoC/Riverpod needed).

### API Base URL
- Android emulator: `http://10.0.2.2:8000`
- Web: `http://localhost:8000`
- Physical device: `http://<machine-ip>:8000`

### Verification
- Full flow: fill form → tap generate → see personalized plan
- Error handling: network failure → show message + retry
- Loading state: spinner appears immediately

### Dependencies
- Phase 5 (FastAPI server running)

---

## Phase 7: Testing & Academic Documentation

**Goal:** Ensure robustness and document the project for academic review.

### Deliverables

**Unit Tests**
- `backend/tests/test_content_filter.py` — feature alignment, similarity matching
- `backend/tests/test_neumf_ranker.py` — model loading, prediction
- `backend/tests/test_llm_adapter.py` — prompt generation, JSON parsing
- `backend/tests/test_api.py` — endpoint validation, error handling

**Integration Tests**
- Full pipeline test: raw profile → final JSON plan
- Concurrent request test: 10 users in parallel

**Documentation**
- Update README with final results
- Add API documentation (Swagger auto-generated at `/docs`)
- Performance comparison table (GMF vs MLP vs NeuMF)
- Methodology write-up with citations

### Academic Rigor
- Cite all references (He et al. 2017, Gupta et al. 2024)
- Document feature weights and their justification
- Explain cold-start strategy with literature basis
- Include model performance metrics

### Verification
- `pytest backend/tests/` → all pass
- `/docs` endpoint shows interactive API spec
- README is complete and accurate

### Dependencies
- All previous phases (1–6)

---

## Quick Commands

```bash
# Setup backend
cd backend
pip install -r requirements.txt

# Re-train model (optional)
python training/train_neumf.py

# Run FastAPI (Phase 5+)
python main.py

# Run tests (Phase 7+)
pytest tests/

# Run Flutter (Phase 6+)
flutter run
```

---

## Timeline Estimate

Assuming 2–3 hours per phase for implementation + testing:

- Phase 1: 2 hours
- Phase 2: 3 hours
- Phase 3: 2 hours
- Phase 4: 3 hours
- Phase 5: 2 hours
- Phase 6: 4 hours (UI design takes time)
- Phase 7: 3 hours

**Total:** ~19 hours of active development

---

## Key Files Reference

- Training: `backend/training/train_neumf.py`
- Trained models: `backend/models/*.keras`
- Program data: `backend/data/datasets/programs_detailed_boostcamp_kaggle.csv`
  - **Note:** This 282MB file is gitignored. Download from Kaggle (Boostcamp dataset)
    and place at the path above, or set `FITNOVA_PROGRAM_CSV` env var to your location.
    The catalog artifact (`program_catalog.pkl`) IS tracked in Git, so rebuilding is
    only needed if you change the classification logic.
- User data: `backend/data/user_features.csv`
- Program features: `backend/data/program_features.csv`
- Gym members raw data: `backend/data/datasets/gym_members_exercise_tracking.csv` (tracked, 64KB)
