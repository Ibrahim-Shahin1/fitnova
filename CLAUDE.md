<!-- GSD:project-start source:PROJECT.md -->

## Project

**FitNova**

FitNova is an AI-powered fitness mobile app (Flutter client + FastAPI/Python backend) with two features: a **workout plan generator** (conversational intake + a layered recommender that produces a personalized 7-day plan) and **exercise form correction** (camera- or video-based detection of form errors with feedback). This milestone rebuilds the form-correction feature from scratch on the Fitness-AQA dataset; the plan generator already works and stays as-is.

**Core Value:** A user can record or upload themselves doing a lift and get trustworthy, plain-language feedback on specific form errors — feedback grounded in a published dataset and method, not guesswork.

### Constraints

- **Tech stack**: Form model in PyTorch (R(2+1)D-18 / ResNet-18 — official code and paper architectures are torch-native) — Why: faithful reproduction, lower risk. The existing recommender stays TensorFlow; FastAPI loads both.
- **Method fidelity**: use the dataset's official train/val/test splits verbatim and report F1-score per error — Why: identical metrics to published work for a defensible comparison.
- **Compute**: heavy training on Colab; every notebook checkpoints to Google Drive and resumes after a disconnect — Why: Colab runtimes are not durable.
- **Working mode**: interactive — one runnable unit at a time, user runs and pastes outputs back, no blind runs — Why: the prior failure came from building on un-understood data.
- **Data**: Fitness-AQA is non-commercial / research-use only.

<!-- GSD:project-end -->

<!-- GSD:stack-start source:codebase/STACK.md -->

## Technology Stack

## Languages

- Dart 3.7+ — Flutter client (`lib/`)
- Python 3.x (no `.python-version` pinned; inferred from syntax) — FastAPI backend + ML pipeline (`backend/`)
- None

## Runtime

- Flutter SDK (Dart ^3.7.0 per `pubspec.yaml`)
- Android target: `android:usesCleartextTraffic="true"` set in `android/app/src/main/AndroidManifest.xml` (cleartext HTTP to local backend)
- Package manager: `pub` (Flutter's built-in)
- Lockfile: `pubspec.lock` present and committed
- Python runtime (no `.python-version` or `Pipfile`; dependencies managed via `pip`)
- Package manager: `pip` with `backend/requirements.txt`
- No lockfile — only range-pinned `requirements.txt`

## Frameworks

- Flutter (Material Design + Cupertino) — cross-platform UI framework; `uses-material-design: true` in `pubspec.yaml`
- Provider `^6.1.2` (resolved `6.1.x`) — state management; used in `lib/providers/user_provider.dart` and `lib/providers/form_session_provider.dart`
- FastAPI `>=0.110.0` — REST + WebSocket server; entry point `backend/app.py`
- Uvicorn `>=0.29.0` (standard extras) — ASGI server; run command: `uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000`
- Pydantic `>=2.0.0` — request/response schema validation; all API models defined inline in `backend/app.py`
- TensorFlow `>=2.13.0` — model training and inference for NeuMF recommendation (`backend/services/neumf_ranker.py`) and form analysis (`backend/services/form_analyzer.py`)
- MediaPipe `>=0.10.9` (pinned operationally to `0.10.33` in `backend/services/mediapipe_config.py`) — pose landmark extraction; Tasks API, VIDEO running mode
- scikit-learn `>=1.3.0` — cosine similarity for cold-start user mapping (`backend/services/neumf_ranker.py`)
- pytest `>=8.0.0` — Python unit/integration tests (`backend/services/test_*.py`, `backend/config/test_*.py`)
- flutter_test (Flutter SDK) — Dart test framework (dev dependency, no test files authored yet)
- flutter_lints `^5.0.0` / `flutter analyze` — static lint analysis; config in `analysis_options.yaml`
- python-dotenv `>=1.0.0` — loads `backend/.env` at runtime (`backend/services/llm_adapter.py`, `backend/services/chat_service.py`)

## Key Dependencies

- `camera: 0.11.2+1` (resolved) — live camera feed for real-time form analysis (`lib/screens/form_check_screen.dart`)
- `web_socket_channel: ^3.0.0` — WebSocket client for streaming frames to backend (`lib/services/form_session_service.dart`)
- `http: 1.6.0` (resolved) — REST calls to backend (`lib/services/api_service.dart`)
- `provider: ^6.1.2` — app-wide state for user profile and form session
- `video_player: ^2.8.0` — playback of form replay and exercise demo videos (`lib/screens/form_replay_screen.dart`)
- `image_picker: 1.2.1` (resolved) — select exercise video files from device for upload (`lib/screens/video_upload_screen.dart`)
- `flutter_svg: 2.2.2` (resolved) — SVG asset rendering
- `cupertino_icons: 1.0.8` (resolved) — iOS-style icon set
- `shared_preferences: ^2.2.3` — local key-value persistence (theme mode in `lib/theme/theme_controller.dart`)
- `url_launcher: ^6.2.5` — open external URLs
- `openai: >=1.30.0` — GPT-4o-mini calls for plan generation (`backend/services/llm_adapter.py`) and session-end coaching feedback (`backend/services/form_session.py`); also function-calling in `backend/services/chat_service.py`
- `mediapipe: >=0.10.9` — pose landmark extraction (33 landmarks, 22 angular features); pinned `0.10.33` + protobuf `4.25.3` in `backend/services/mediapipe_config.py`
- `tensorflow: >=2.13.0` — NeuMF recommendation model (`backend/models/neumf_final.keras`) and form-analysis MT-TCN/ST-GCN models
- `numpy: >=1.24.0` — angular feature computation, normalisation
- `pandas: >=2.0.0` — user feature CSV loading (`backend/data/user_features.csv`, `backend/data/program_features.csv`)
- `rapidfuzz: >=3.0.0` — fuzzy exercise name matching for video URL lookup (`backend/services/llm_adapter.py`)
- `opencv-python-headless: >=4.8.0` — video frame decoding in REST upload endpoint (`backend/app.py`: `analyze_form_video`)
- `scipy: >=1.11.0` — signal processing for form geometry
- `websockets: >=12.0` — WebSocket transport (used by uvicorn standard extras)
- `httpx: >=0.27.0` — async HTTP client (available for outgoing calls)
- `imageio-ffmpeg: >=0.4.9` — exercise video processing scripts (`backend/scripts/build_exercise_videos.py`)

## Custom ML Models

- `backend/models/neumf_final.keras` — Neural Matrix Factorisation; pre-trained GMF + MLP components at `backend/models/gmf_pretrained.keras` and `backend/models/mlp_pretrained.keras`
- `backend/models/neumf_metadata.pkl` — user/item count metadata
- `backend/data/program_catalog.pkl` — 2,598 programs with Week 1 exercises
- `backend/data/norm_stats.pkl` — feature normalisation statistics
- Active weights selected at startup with auto-detection priority: v6 (`backend/models/form_model_v6/v6_supervised.weights.h5`) → v5.2 (`backend/models/form_model_v5_2/v5_2_supervised.weights.h5`) → v4 legacy (`backend/models/form_model/mt_tcn_weights.weights.h5`)
- Override via `FITNOVA_MODEL_DIR` environment variable
- Model architecture definitions: `backend/training/models/mt_tcn.py` (v4 MT-TCN), `backend/training/models/st_gcn.py` (v5.x ST-GCN), `backend/training/models/st_gcn_v6.py` / `st_gcn_v6_1.py` (v6/v6.1)
- MediaPipe model asset: `backend/models/pose_landmarker_full.task` (~29 MB float16, auto-downloaded from Google on first use)

## Configuration

- Backend secrets live in `backend/.env` (not committed; `backend/.env.example` committed as template)
- Only one required key: `OPENAI_API_KEY`
- Optional override: `FITNOVA_MODEL_DIR` — selects which model version directory to load
- Configured in `lib/config/api_config.dart`
- Android emulator: `http://10.0.2.2:8000` / `ws://10.0.2.2:8000`
- Web/desktop/iOS: `http://localhost:8000` / `ws://localhost:8000`
- No production URL configured — local development only
- `analysis_options.yaml` — extends `package:flutter_lints/flutter.yaml`; no custom rules added
- `pubspec.yaml` — `publish_to: 'none'` (private, not pub.dev publishable)
- Font: custom Sora variable font (`assets/fonts/Sora-Variable.ttf`), weights 400–800
- Assets: `assets/logo/` (app logo)

## Platform Requirements

- Flutter SDK with Dart ^3.7.0
- Python (no enforced version; `>=3.10` needed for `str | None` union syntax used throughout backend)
- `pip install -r backend/requirements.txt`
- `OPENAI_API_KEY` in `backend/.env`
- MediaPipe model auto-downloads on first backend startup if absent
- Backend runs on port 8000; Flutter connects to 10.0.2.2:8000 from Android emulator
- No deployment configuration found — no Dockerfile, no `render.yaml`, no `fly.toml`, no `Procfile`
- Intended as a local graduation-project demo server

<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->

## Conventions

## Two Codebases, Two Convention Sets

## Naming Patterns

### Dart / Flutter (`lib/`)

- Screens: `snake_case_screen.dart` — e.g., `form_check_screen.dart`, `chat_screen.dart`
- Providers: `snake_case_provider.dart` — e.g., `user_provider.dart`, `form_session_provider.dart`
- Services: `snake_case_service.dart` — e.g., `api_service.dart`, `form_session_service.dart`
- Models: `snake_case_models.dart` or `noun.dart` — e.g., `fitness_plan.dart`, `form_models.dart`
- Widgets: `snake_case.dart` — e.g., `skeleton_painter.dart`, `app_button.dart`
- Theme files: `app_<token>.dart` — e.g., `app_colors.dart`, `app_spacing.dart`, `app_typography.dart`
- Config: `api_config.dart`
- `UpperCamelCase` for all types: `UserProvider`, `FormSessionProvider`, `SkeletonPainter`, `AppButton`
- Widget state classes: `_<WidgetName>State` — e.g., `_ChatScreenState`, `_FormCheckScreenState`
- Private classes prefixed with `_`
- `lowerCamelCase` for all methods and functions: `setProfile()`, `updateExtracted()`, `buildJointErrorMap()`
- Private methods prefixed with `_`: `_fetchGreeting()`, `_sendMessage()`, `_initCamera()`
- Event handlers: `_on<Event>()` or `_<verb><Noun>()` pattern
- `lowerCamelCase`: `experienceLevel`, `sessionDurationHours`, `workoutFrequency`
- Private provider fields: underscore-prefixed (`_state`, `_latestFrame`, `_liveQuality`)
- Constants: `lowerCamelCase` for `static const`, `SCREAMING_SNAKE` for file-level constants — e.g., `_frameIntervalMs = 100`, `_connections`
- `UpperCamelCase` for type, `lowerCamelCase` for values: `FormSessionState.idle`, `AppButtonVariant.primary`

### Python / Backend (`backend/`)

- Services: `snake_case.py` — e.g., `form_analyzer.py`, `llm_adapter.py`, `chat_service.py`
- Tests live both in `backend/tests/` and co-located next to source modules (e.g., `backend/services/test_form_geometry.py`, `backend/config/test_defect_to_region.py`)
- Training preprocessing: `snake_case.py` in `backend/training/preprocessing/`
- `UpperCamelCase`: `FormAnalyzer`, `LLMAdapter`, `Recommender`, `ContentBasedFilter`
- Pydantic models: `UpperCamelCase` suffixed by role — `UserProfileRequest`, `GeneratePlanResponse`, `ChatRequest`, `ChatResponse`
- Public: `snake_case` — `generate_plan()`, `recommend()`, `process_message()`
- Private helpers: `_snake_case` — `_call_llm()`, `_build_user_prompt()`, `_validate_response()`, `_sanitize_reps()`
- Module-level helpers in test files: `_snake_case` — `_sample_plan()`, `_make_response()`
- Module-level constants: `SCREAMING_SNAKE` — `WINDOW_SIZE`, `INFERENCE_INTERVAL`, `MAX_FILL_FRAMES`
- Private module-level paths: `_snake_case` — `_BASE_DIR`, `_DATA_DIR`, `_CATALOG_PATH`
- Local variables: `snake_case`

## Code Style

### Dart / Flutter

- Tool: `dart format` (enforced by Flutter toolchain)
- No additional `.prettierrc` or `biome.json` — relies on `dart format` defaults
- Single quotes for strings (Flutter convention)
- Tool: `flutter_lints ^5.0.0` (declared in `pubspec.yaml`)
- Config: `analysis_options.yaml` at repo root — uses `package:flutter_lints/flutter.yaml`
- No custom rule overrides are currently active; the commented-out lines show `prefer_single_quotes` and `avoid_print` are under consideration but not enabled
- Suppress per-file with `// ignore_for_file: rule_name` or per-line with `// ignore: rule_name`
- `super.key` parameter passed to widget constructors (modern Flutter idiom): `const AppButton({super.key, ...})`
- `const` constructors used throughout for stateless widgets
- Dart 3 pattern matching used: `switch` expressions with destructuring `(bg, fg, border) = switch (variant) {...}`
- `context.read<T>()` preferred over `Provider.of<T>(context, listen: false)` in event handlers

### Python / Backend

- No `.flake8`, `pyproject.toml`, or `setup.cfg` found — no enforced formatter
- Line lengths are not rigidly controlled; long lines appear in prompt strings and constants
- Backslash line continuation used sparingly for long string literals
- No `mypy.ini` or `pyright` config found
- Type annotations are used consistently in all service and config modules
- `from __future__ import annotations` present in virtually all backend Python files — enables PEP 563 postponed evaluation for forward references and Python 3.10-style union syntax (`X | Y`) on older runtimes
- Module docstrings at file top: triple-quoted, multi-line, describe purpose and pipeline role (see `backend/app.py`, `backend/services/llm_adapter.py`)
- Section separators using `# ─────────...─────────` (box-drawing dashes) are used heavily to group endpoint definitions, fixtures, and test suites within a file
- Two-space comment style for inline alignment: `self._ts_ms: int = 0   # VIDEO mode state`

## Import Organization

### Dart

- Relative paths only: `'../models/fitness_plan.dart'`, `'../config/api_config.dart'`
- No barrel (`index.dart`) files found — each import is explicit

### Python

- Absolute package imports used throughout: `from backend.services.recommender import Recommender`
- No `__init__.py` re-exports for convenience — always import from the exact module

## Error Handling

### Dart

- All async API calls wrapped in `try/catch` inside screen methods
- Errors surfaced to users via `ScaffoldMessenger.of(context).showSnackBar(...)` or `_showError()` helper
- On API failure: throws `Exception('description: $statusCode $body')` from `ApiService`
- Screen-level error state stored in local `String?` fields (e.g., `_cameraError`)
- No global error boundary — errors bubble to the screen that made the request

### Python

- FastAPI endpoints catch all service exceptions and re-raise as `HTTPException` with a descriptive `detail` field:
- Service classes raise `ValueError` for schema/validation failures, `RuntimeError` for unexpected states
- LLM adapter uses a retry loop with specific exception types: `APITimeoutError`, `APIConnectionError` caught separately; `json.JSONDecodeError` caught and re-raised as `ValueError`
- Graceful degradation is a recurring pattern: services fall back to a template/default rather than failing hard (e.g., `LLMAdapter._build_template_plan()`, `FormAnalyzer` no-pose forward-fill)
- `logger.exception()` used (not `logger.error()`) when inside an `except` block so the traceback is always captured

## Logging

### Dart

- No structured logging framework; screens use `debugPrint()` or skip logging entirely for non-critical paths
- No `print()` calls found in `lib/` — consistent with `avoid_print` lint guidance

### Python

- `logging.getLogger(name)` at module top — two naming styles used:
- `logger.info()` for lifecycle events, `logger.warning()` for degraded states, `logger.exception()` inside `except` blocks

## Comments

### Dart

- Doc comments on public-facing widgets and helper functions using `///`
- Inline `//` comments explain non-obvious decisions (camera capture strategy, timer interval rationale)
- No JSDoc-style parameter documentation

### Python

- Module docstrings describe the module's role in the pipeline (e.g., `backend/app.py` lists all 4 layers)
- Inline comments explain ML constants, threshold rationale, and bug fixes with dates (e.g., `# B6 fix (2026-04-18): switched from VisionRunningMode.IMAGE to VIDEO`)
- Fix comments always include the bug label + date: `# B7 fix (2026-04-18): forward-fill on no-pose frames.`
- Section separator lines (`# ─────────`) used inside long files to delineate logical groups

## Function Design

### Dart

- Screen state methods are private (`_`) and purpose-named: `_initCamera()`, `_fetchGreeting()`, `_sendMessage()`
- Provider mutation methods are public and imperative: `setProfile()`, `setInjuries()`, `setPlan()`
- `dispose()` always overridden in StatefulWidgets that create controllers — consistently calls `super.dispose()` last
- Computed properties used for derived data: `bool get isReadyForPlan`, `Map get generatePlanPayload`

### Python

- Service methods are instance methods with `self`; pure helpers that don't need instance state are `@staticmethod`
- `@lru_cache(maxsize=1)` used for expensive one-time loads (e.g., `backend/config/exercises.py` loads JSON once per process)
- Functions have docstrings on public methods listing `Args:` and `Returns:` with types
- Private helpers prefixed `_` and kept near their callers within the class

## Module Design

### Dart

- No barrel (index) files; every import is direct
- Providers: state + mutation only; no business logic
- Services (`ApiService`, `FormSessionService`): all methods are `static` — no instance state
- Models: plain data classes with `fromJson()` factory constructors; no serialization libraries

### Python

- Each service is a class (even single-responsibility ones like `Recommender`) for testability and lifecycle management
- Pydantic `BaseModel` with `Field(...)` validators used for all HTTP request/response models in `backend/app.py`
- `app.state.*` is the injection point: all heavy services (recommender, llm_adapter, form_analyzer) are loaded once in the `lifespan` context manager and attached to `app.state`
- No dependency injection framework — services are instantiated directly

<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->

## Architecture

## System Overview

```text

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

- Flutter client is stateless regarding ML — all AI logic runs server-side
- Two transport modes: REST (plan generation, video upload) and WebSocket (live camera streaming)
- Recommendation pipeline is strictly layered (Layer 1 → Layer 2 → Layer 3); each layer takes previous output as input
- Form analysis is model-version-aware: the same `FormAnalyzer` / `FormSession` classes dispatch to v6, v5.2, or v4 based on which weights exist on disk
- Providers follow the Provider package pattern (ChangeNotifier); screens read via `context.watch` / `context.read`

## Layers

- Purpose: Screen rendering, navigation, camera capture, user input
- Location: `lib/screens/`
- Contains: 12 screen widgets, all `StatefulWidget` or `StatelessWidget`
- Depends on: Providers for state, ApiService / FormSessionService for data
- Used by: End user
- Purpose: Shared mutable state across the widget tree; keeps screens decoupled from each other
- Location: `lib/providers/`
- Contains: `UserProvider` (user profile + plan), `FormSessionProvider` (form session lifecycle + live metrics), `ThemeController` (`lib/theme/theme_controller.dart`)
- Depends on: Models, Flutter foundation
- Used by: Screens via `Provider.of` / `context.watch`
- Purpose: Network calls; wraps HTTP and WebSocket transport details
- Location: `lib/services/`
- Contains: `ApiService` (static methods, http package), `FormSessionService` (WebSocket, web_socket_channel)
- Depends on: `ApiConfig`, Dart `http`, `web_socket_channel`
- Used by: Screens and providers
- Purpose: HTTP/WebSocket routing, request validation, error handling, lifespan startup
- Location: `backend/app.py`
- Contains: Pydantic models for request/response, all endpoint definitions
- Depends on: All backend services
- Used by: Flutter client
- Purpose: Business logic: recommendation, LLM calls, form analysis
- Location: `backend/services/`
- Contains: `Recommender`, `ContentBasedFilter`, `NeuMFRanker`, `LLMAdapter`, `ChatService`, `FormAnalyzer`, `FormSession`, `ExerciseMismatchDetector`, `FormGeometry` (retained but disconnected from runtime)
- Depends on: ML models on disk, OpenAI API, data artefacts in `backend/data/`
- Used by: FastAPI route handlers via `app.state.*`
- Purpose: Offline data pipeline and model training (not loaded at server runtime)
- Location: `backend/training/`
- Contains: preprocessing, model definitions, evaluation scripts, diagnostics
- Depends on: `backend/data/`, TensorFlow, MediaPipe, NumPy
- Used by: Colab notebooks, offline scripts; NOT imported at serve time
- Purpose: Static configuration and serialised data artefacts
- Location: `backend/config/`, `backend/data/`
- Contains: `exercises.json` (SSOT for 27 exercises), `exercises.py` (loader), `program_catalog.pkl`, `norm_stats.pkl`, `user_features.csv`, `injury_exercise_blacklist.json`, `exercise_videos.json`
- Depends on: Nothing at runtime (loaded once, lru_cached where applicable)
- Used by: Service layer

## Data Flow

### Fitness Plan Generation (Primary Path)

### Live Form Analysis (WebSocket Path)

### Video Upload Path (REST Alternative)

### ML Form Analysis Pipeline (Internal)

- Flutter: Provider package (`ChangeNotifier`); `UserProvider` and `FormSessionProvider` held at root via `MultiProvider` in `main.dart`
- Backend: Services instantiated once in FastAPI `lifespan()` and stored on `app.state.*`; `FormSession` is per-connection stateful (each WebSocket gets its own instance)

## Key Abstractions

- Purpose: Pose estimation + feature extraction; shared across all sessions
- Examples: `backend/services/form_analyzer.py`
- Pattern: Loaded once at startup, `process_frame()` is the public API; `_model_version` field drives dispatch; `reset_video_state()` must be called at session start to reset MediaPipe's internal timestamp counter
- Purpose: Accumulate frames, run windowed inference, manage rep tracking, produce end-of-session summary
- Examples: `backend/services/form_session.py`
- Pattern: Instantiated fresh per WebSocket accept; holds deques for the 64-frame sliding window; EMA state seeded on first inference; `end_session()` returns the full summary dict
- Purpose: Single store for user identity, chat-extracted fitness params, and the generated plan
- Examples: `lib/providers/user_provider.dart`
- Pattern: `ChangeNotifier`; `generatePlanPayload` and `chatUserContext` are computed getters that assemble the API payloads
- Purpose: Backward-compatible loading — server auto-detects v6 → v5.2 → v4 weights based on file presence
- Examples: `backend/services/form_analyzer.py:152`, `backend/app.py:184`
- Pattern: File-presence detection at startup sets `_model_version`; `predict_window()` dispatches to `_predict_window_v6/v5_2/v4` accordingly. v5.2's 5-group joint_err is remapped to the 10-group schema used by v4 and Flutter for UI compatibility.

## Entry Points

- Location: `lib/main.dart:22`
- Triggers: Flutter engine start
- Responsibilities: Init `ThemeController`, wire `MultiProvider` with `UserProvider` + `FormSessionProvider` + `ThemeController`, declare route table, launch on `/splash`
- Location: `backend/app.py`
- Triggers: `uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000`
- Responsibilities: CORS middleware, static file mount, `lifespan()` startup loading of `Recommender`, `LLMAdapter`, `ChatService`, `FormAnalyzer` into `app.state`
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

### Sharing FormAnalyzer Across Concurrent Sessions

## Error Handling

- FastAPI endpoint handlers catch service exceptions and raise `HTTPException(500)` with detail string
- `LLMAdapter.generate_plan()` catches all exceptions and falls back to `_build_template_plan()` (deterministic, no LLM) — `source` field in response indicates `"llm"` vs `"template_fallback"`
- `FormSession._generate_llm_feedback()` falls back to `_fallback_feedback()` (deterministic text) when `openai_client` is None or raises
- `FormAnalyzer.predict_window()` returns a neutral all-zeros dict when `model_ready` is False, allowing the UI to operate without a trained model
- MediaPipe unavailability: `FormAnalyzer.process_frame()` returns `status: "mediapipe_unavailable"` rather than raising

## Cross-Cutting Concerns

<!-- GSD:architecture-end -->

<!-- GSD:skills-start source:skills/ -->

## Project Skills

No project skills found. Add skills to any of: `.claude/skills/`, `.agents/skills/`, `.cursor/skills/`, `.github/skills/`, or `.codex/skills/` with a `SKILL.md` index file.
<!-- GSD:skills-end -->

<!-- GSD:workflow-start source:GSD defaults -->

## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:

- `/gsd:quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd:debug` for investigation and bug fixing
- `/gsd:execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->

<!-- GSD:profile-start -->

## Developer Profile

> Profile not yet configured. Run `/gsd:profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->
