<!-- refreshed: 2026-05-18 -->
# Architecture

**Analysis Date:** 2026-05-18

## System Overview

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Flutter Client  (lib/)                               │
│                                                                             │
│  Screens: Splash → Register → ModeSelect → Home → Chat → Plan              │
│           ExerciseSelect → Guidelines → FormCheck / VideoUpload             │
│           FormResults → FormReplay                                          │
│                                                                             │
│  Providers (Provider package)                                               │
│   UserProvider              FormSessionProvider                             │
│   `lib/providers/user_provider.dart`  `lib/providers/form_session_provider.dart`
│                                                                             │
│  Services (HTTP + WebSocket)                                                │
│   ApiService (HTTP REST)          FormSessionService (WS)                   │
│   `lib/services/api_service.dart` `lib/services/form_session_service.dart` │
└──────────────────────────┬──────────────────────────────────────────────────┘
                           │  HTTP (port 8000) + WebSocket (ws://)
                           ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    FastAPI Server  (backend/app.py)                         │
│                                                                             │
│  REST endpoints:                                                            │
│   GET  /health                 GET  /api/exercises                          │
│   POST /chat                   POST /generate-plan                          │
│   POST /analyze-form-video                                                  │
│                                                                             │
│  WebSocket:                                                                 │
│   WS /ws/form-session                                                       │
│                                                                             │
│  Static files: /static/exercise_videos/**                                  │
└──────┬──────────────┬──────────────────────────┬────────────────────────────┘
       │              │                          │
       ▼              ▼                          ▼
┌────────────┐ ┌────────────────────┐  ┌──────────────────────────────────────┐
│ Recommendation  │  Chat Pipeline   │  │       Form Analysis Pipeline         │
│  Pipeline  │  │ `services/        │  │                                      │
│            │  │  chat_service.py` │  │  FormAnalyzer  `services/            │
│ Layer 1:   │  │                   │  │  form_analyzer.py`                   │
│ ContentFilter  │  GPT-4o-mini      │  │  - MediaPipe Pose (VIDEO mode)       │
│ `services/ │  │  function-calling │  │  - 33→15 joint mapping               │
│ content_   │  │  extraction       │  │  - 22 angular features               │
│ filter.py` │  │                   │  │  - ST-GCN v6 / v5.2 / v4 dispatch   │
│            │  └────────────────────┘  │                                      │
│ Layer 2:   │                          │  FormSession  `services/             │
│ NeuMFRanker│                          │  form_session.py`                    │
│ `services/ │                          │  - 64-frame sliding window           │
│ neumf_     │                          │  - EMA smoothing                     │
│ ranker.py` │                          │  - rep counting                      │
│            │                          │  - GPT-4o-mini coaching feedback     │
│ Layer 3:   │                          └──────────────────────────────────────┘
│ LLMAdapter │
│ `services/ │
│ llm_       │
│ adapter.py`│
└────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Data Artefacts (backend/data/)                      │
│                                                                             │
│  program_catalog.pkl   norm_stats.pkl   user_features.csv                  │
│  interactions.csv      exercise_videos.json   injury_exercise_blacklist.json│
└─────────────────────────────────────────────────────────────────────────────┘
```

## Component Responsibilities

| Component | Responsibility | File |
|-----------|----------------|------|
| `FitNovaApp` | Root widget, route table, theme, provider wiring | `lib/main.dart` |
| `ApiConfig` | Base URL / WS URL resolution per platform | `lib/config/api_config.dart` |
| `ApiService` | All REST calls (chat, generate-plan, exercises, video upload) | `lib/services/api_service.dart` |
| `FormSessionService` | WebSocket lifecycle + frame streaming for live camera mode | `lib/services/form_session_service.dart` |
| `UserProvider` | User profile, chat-extracted fields, plan storage | `lib/providers/user_provider.dart` |
| `FormSessionProvider` | Live form session state machine + per-frame metrics | `lib/providers/form_session_provider.dart` |
| `FastAPI app` | Server entry point, route definitions, lifespan loader | `backend/app.py` |
| `Recommender` | Chains Layer 1 + Layer 2 into a single `recommend()` call | `backend/services/recommender.py` |
| `ContentBasedFilter` | Scores all 2,598 programs against user profile (Gupta et al. weights) | `backend/services/content_filter.py` |
| `NeuMFRanker` | NeuMF re-ranking + cold-start nearest-neighbor user mapping | `backend/services/neumf_ranker.py` |
| `LLMAdapter` | Converts recommended program into a 7-day weekly plan via GPT-4o-mini | `backend/services/llm_adapter.py` |
| `ChatService` | Stateless GPT-4o-mini function-calling intake to extract 3 fitness params | `backend/services/chat_service.py` |
| `FormAnalyzer` | Per-frame: decode JPEG → MediaPipe → 15 canonical joints → 22 angles | `backend/services/form_analyzer.py` |
| `FormSession` | Per-connection: sliding window, inference dispatch, rep counting, session summary | `backend/services/form_session.py` |
| `mediapipe_config` | Pinned MediaPipe options (single truth for training + inference) | `backend/services/mediapipe_config.py` |
| `exercises.py` | SSOT for exercise metadata (loaded from `exercises.json`, lru_cached) | `backend/config/exercises.py` |
| `ExerciseMismatchDetector` | Detects mismatch between user-selected and classifier-predicted exercise | `backend/services/exercise_sanity.py` |

## Pattern Overview

**Overall:** Client-Server with layered ML pipeline

**Key Characteristics:**
- Flutter client is stateless regarding ML — all AI logic runs server-side
- Two transport modes: REST (plan generation, video upload) and WebSocket (live camera streaming)
- Recommendation pipeline is strictly layered (Layer 1 → Layer 2 → Layer 3); each layer takes previous output as input
- Form analysis is model-version-aware: the same `FormAnalyzer` / `FormSession` classes dispatch to v6, v5.2, or v4 based on which weights exist on disk
- Providers follow the Provider package pattern (ChangeNotifier); screens read via `context.watch` / `context.read`

## Layers

**Flutter Presentation Layer:**
- Purpose: Screen rendering, navigation, camera capture, user input
- Location: `lib/screens/`
- Contains: 12 screen widgets, all `StatefulWidget` or `StatelessWidget`
- Depends on: Providers for state, ApiService / FormSessionService for data
- Used by: End user

**Flutter State Layer (Providers):**
- Purpose: Shared mutable state across the widget tree; keeps screens decoupled from each other
- Location: `lib/providers/`
- Contains: `UserProvider` (user profile + plan), `FormSessionProvider` (form session lifecycle + live metrics), `ThemeController` (`lib/theme/theme_controller.dart`)
- Depends on: Models, Flutter foundation
- Used by: Screens via `Provider.of` / `context.watch`

**Flutter Service Layer:**
- Purpose: Network calls; wraps HTTP and WebSocket transport details
- Location: `lib/services/`
- Contains: `ApiService` (static methods, http package), `FormSessionService` (WebSocket, web_socket_channel)
- Depends on: `ApiConfig`, Dart `http`, `web_socket_channel`
- Used by: Screens and providers

**FastAPI Route Layer:**
- Purpose: HTTP/WebSocket routing, request validation, error handling, lifespan startup
- Location: `backend/app.py`
- Contains: Pydantic models for request/response, all endpoint definitions
- Depends on: All backend services
- Used by: Flutter client

**Backend Service Layer:**
- Purpose: Business logic: recommendation, LLM calls, form analysis
- Location: `backend/services/`
- Contains: `Recommender`, `ContentBasedFilter`, `NeuMFRanker`, `LLMAdapter`, `ChatService`, `FormAnalyzer`, `FormSession`, `ExerciseMismatchDetector`, `FormGeometry` (retained but disconnected from runtime)
- Depends on: ML models on disk, OpenAI API, data artefacts in `backend/data/`
- Used by: FastAPI route handlers via `app.state.*`

**ML Training Layer:**
- Purpose: Offline data pipeline and model training (not loaded at server runtime)
- Location: `backend/training/`
- Contains: preprocessing, model definitions, evaluation scripts, diagnostics
- Depends on: `backend/data/`, TensorFlow, MediaPipe, NumPy
- Used by: Colab notebooks, offline scripts; NOT imported at serve time

**Config / Data Layer:**
- Purpose: Static configuration and serialised data artefacts
- Location: `backend/config/`, `backend/data/`
- Contains: `exercises.json` (SSOT for 27 exercises), `exercises.py` (loader), `program_catalog.pkl`, `norm_stats.pkl`, `user_features.csv`, `injury_exercise_blacklist.json`, `exercise_videos.json`
- Depends on: Nothing at runtime (loaded once, lru_cached where applicable)
- Used by: Service layer

## Data Flow

### Fitness Plan Generation (Primary Path)

1. User fills registration + selects goal → `UserProvider.setProfile()` + `setWorkoutType()` (`lib/providers/user_provider.dart`)
2. `ChatScreen` calls `ApiService.sendChat()` → `POST /chat` (`lib/services/api_service.dart:9`)
3. `ChatService.process_message()` calls GPT-4o-mini with function-calling; returns `{status, message, extracted}` (`backend/services/chat_service.py:113`)
4. Client accumulates extracted fields into `UserProvider.updateExtracted()` until `status == "ready"`
5. `ChatScreen` calls `ApiService.generatePlan()` → `POST /generate-plan` (`lib/services/api_service.dart:32`)
6. `Recommender.recommend()`: ContentBasedFilter scores 2,598 programs → top-50; NeuMFRanker cold-starts user to nearest trained user → re-ranks → returns `program_id` (`backend/services/recommender.py:20`)
7. `LLMAdapter.generate_plan()`: builds GPT-4o-mini prompt from program week-1 exercises + user profile → returns validated 7-day plan JSON with media URLs (`backend/services/llm_adapter.py:140`)
8. Response deserialized to `FitnessPlan` → stored in `UserProvider.currentPlan`; user navigated to `PlanScreen`

### Live Form Analysis (WebSocket Path)

1. User selects exercise → `ExerciseSelectionScreen` → `/form-check` with `exerciseHint`
2. `FormCheckScreen` initialises camera (front-facing, medium resolution, 100ms polling timer) (`lib/screens/form_check_screen.dart`)
3. `FormSessionService.connect()` opens `ws://host/ws/form-session`, sends `start_session` with `selected_exercise` (`lib/services/form_session_service.dart:18`)
4. Each 100ms timer tick: `takePicture()` → JPEG bytes → `FormSessionService.sendFrame()` base64-encoded (`lib/screens/form_check_screen.dart`)
5. `FormSession.add_frame()` calls `FormAnalyzer.process_frame()`: MediaPipe → 15 canonical joints → 22 angular features (`backend/services/form_session.py:233`)
6. Every 10th frame with a full 64-frame window: `FormAnalyzer.predict_window()` dispatches to v6/v5.2/v4 ST-GCN (`backend/services/form_analyzer.py:481`)
7. EMA-smoothed quality + joint errors streamed back as `FormFrameResult` JSON → `FormSessionProvider.updateFrame()` → `SkeletonPainter` overlay + metrics HUD
8. On `end_session`: `FormSession.end_session()` calls GPT-4o-mini for coaching feedback → `FormSessionSummary` → navigation to `/form-results`

### Video Upload Path (REST Alternative)

1. `VideoUploadScreen` picks video from gallery → `ApiService.uploadFormVideo()` → `POST /analyze-form-video` (`lib/services/api_service.dart:60`)
2. Server samples video at ~10fps with OpenCV, runs same `FormSession` pipeline on each sampled frame (`backend/app.py:397`)
3. Returns `FormSessionSummary` + per-frame `timeline` array
4. If `timeline` is non-empty, navigates to `/form-replay` (video playback with skeleton overlay); otherwise falls back to `/form-results`

### ML Form Analysis Pipeline (Internal)

1. JPEG bytes → `cv2.imdecode` → RGB conversion
2. `MediaPipe PoseLandmarker.detect_for_video()` (VIDEO mode, monotonic timestamps) → 33 world + image landmarks
3. `extract_canonical_from_mediapipe()` → 15 canonical joints (world + image coords separately) (`backend/training/preprocessing/normalize.py`)
4. `normalize_skeleton()` → skeleton-normalised world coords (pelvis at origin, unit scale)
5. `compute_frame_angles()` → 22 angular features (16 articulation angles + 6 axis-relative) (`backend/training/preprocessing/angular_features.py`)
6. Features buffered in `FormSession` sliding deques (64-frame window)
7. `FormAnalyzer.predict_window()` → v6 ST-GCN: inputs are `(B,64,15,4)` pose + `(B,64,22)` angles + `(B,)` exercise_id → outputs quality, action (25 classes), rep_count, boundary (64,), joint_err (64,10)

**State Management:**
- Flutter: Provider package (`ChangeNotifier`); `UserProvider` and `FormSessionProvider` held at root via `MultiProvider` in `main.dart`
- Backend: Services instantiated once in FastAPI `lifespan()` and stored on `app.state.*`; `FormSession` is per-connection stateful (each WebSocket gets its own instance)

## Key Abstractions

**FormAnalyzer (stateful but logically stateless per-frame):**
- Purpose: Pose estimation + feature extraction; shared across all sessions
- Examples: `backend/services/form_analyzer.py`
- Pattern: Loaded once at startup, `process_frame()` is the public API; `_model_version` field drives dispatch; `reset_video_state()` must be called at session start to reset MediaPipe's internal timestamp counter

**FormSession (stateful, one per WebSocket connection):**
- Purpose: Accumulate frames, run windowed inference, manage rep tracking, produce end-of-session summary
- Examples: `backend/services/form_session.py`
- Pattern: Instantiated fresh per WebSocket accept; holds deques for the 64-frame sliding window; EMA state seeded on first inference; `end_session()` returns the full summary dict

**UserProvider (Flutter global state):**
- Purpose: Single store for user identity, chat-extracted fitness params, and the generated plan
- Examples: `lib/providers/user_provider.dart`
- Pattern: `ChangeNotifier`; `generatePlanPayload` and `chatUserContext` are computed getters that assemble the API payloads

**Model Version Dispatch (FormAnalyzer):**
- Purpose: Backward-compatible loading — server auto-detects v6 → v5.2 → v4 weights based on file presence
- Examples: `backend/services/form_analyzer.py:152`, `backend/app.py:184`
- Pattern: File-presence detection at startup sets `_model_version`; `predict_window()` dispatches to `_predict_window_v6/v5_2/v4` accordingly. v5.2's 5-group joint_err is remapped to the 10-group schema used by v4 and Flutter for UI compatibility.

## Entry Points

**Flutter App:**
- Location: `lib/main.dart:22`
- Triggers: Flutter engine start
- Responsibilities: Init `ThemeController`, wire `MultiProvider` with `UserProvider` + `FormSessionProvider` + `ThemeController`, declare route table, launch on `/splash`

**FastAPI Server:**
- Location: `backend/app.py`
- Triggers: `uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000`
- Responsibilities: CORS middleware, static file mount, `lifespan()` startup loading of `Recommender`, `LLMAdapter`, `ChatService`, `FormAnalyzer` into `app.state`

**ML Training Entry Points:**
- v6 training: `backend/training/train_form_model_v6.py`
- v6.1 (distillation/VLM): `backend/training/train_form_model_v6_1.py`
- NeuMF training: `backend/training/train_neumf.py`
- QEVD preprocessing: `backend/training/preprocessing/qevd_extractor.py`

## Architectural Constraints

- **Threading:** FastAPI runs on a single-threaded async event loop. `FormAnalyzer.predict_window()` calls `model.predict()` synchronously (blocking the event loop during inference). `FormSession` instances must NOT be shared across WebSocket connections — `process_frame()` mutates `_ts_ms` and `_prev_result` without a lock.
- **Global state:** `app.state.form_analyzer` is a single `FormAnalyzer` instance shared across all sessions. `FormSession` calls `analyzer.reset_video_state()` at construction — this is safe because `reset_video_state()` recreates the internal MediaPipe landmarker; concurrent sessions on the same analyzer would corrupt each other's timestamp state.
- **MediaPipe timestamp requirement:** `detect_for_video()` requires monotonically increasing timestamps per landmarker instance. `reset_video_state()` destroys and recreates the landmarker to reset the counter. Passing a timestamp <= the last one raises an exception from MediaPipe.
- **Model version discovery:** Done by file-presence check at server startup (`backend/app.py:184`). Override via `FITNOVA_MODEL_DIR` environment variable.
- **Training/inference parity:** `backend/services/mediapipe_config.py` is the single source of truth for MediaPipe settings. Both `FormAnalyzer` and `qevd_extractor.py` import from it. Changing values there requires re-extraction and retraining.

## Anti-Patterns

### Geometric Rule Layer (Removed at Runtime)

**What happens:** `form_geometry.py` and `form_session.py` previously included rule-based angle checks. As of 2026-05-11 the geometric layer is imported but not called at runtime (`FormSession.__init__` comment: "Geometric form validator instantiation removed (Day 0.5)").

**Why it's wrong:** Rules misfired in both directions on real phone video. They competed with the ML signal and caused unreliable rep counting.

**Do this instead:** Use purely the ML signal from `FormAnalyzer.predict_window()`. If rule-based augmentation is needed in future, implement it as a separate trained module, not inline heuristics in `FormSession`.

### Sharing FormAnalyzer Across Concurrent Sessions

**What happens:** `app.state.form_analyzer` is a single object. `reset_video_state()` (called per-session construction) recreates the MediaPipe landmarker in-place.

**Why it's wrong:** If two WebSocket sessions are live simultaneously, `reset_video_state()` from session B will break session A's landmarker.

**Do this instead:** For multi-user production, either create one `FormAnalyzer` per WebSocket connection, or add a session-scoped lock. Current design is safe for single-user (one active session at a time).

## Error Handling

**Strategy:** Fail-fast at the HTTP layer; graceful degradation in ML components.

**Patterns:**
- FastAPI endpoint handlers catch service exceptions and raise `HTTPException(500)` with detail string
- `LLMAdapter.generate_plan()` catches all exceptions and falls back to `_build_template_plan()` (deterministic, no LLM) — `source` field in response indicates `"llm"` vs `"template_fallback"`
- `FormSession._generate_llm_feedback()` falls back to `_fallback_feedback()` (deterministic text) when `openai_client` is None or raises
- `FormAnalyzer.predict_window()` returns a neutral all-zeros dict when `model_ready` is False, allowing the UI to operate without a trained model
- MediaPipe unavailability: `FormAnalyzer.process_frame()` returns `status: "mediapipe_unavailable"` rather than raising

## Cross-Cutting Concerns

**Logging:** Python standard `logging`; loggers named `"fitnova"` (app), `"fitnova.chat"`, `"fitnova.llm"`. Debug detail gated on `FITNOVA_DEBUG=1` env var in `FormAnalyzer`.

**Validation:** Pydantic v2 models at the API boundary (`UserProfileRequest`, `ChatRequest`, etc. in `backend/app.py`). No validation layer in the Flutter client beyond type-safe `fromJson` deserialization.

**Authentication:** None. The API accepts requests from any origin (CORS `allow_origins=["*"]`). There is no user authentication system.

---

*Architecture analysis: 2026-05-18*
