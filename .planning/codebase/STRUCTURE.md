# Codebase Structure

**Analysis Date:** 2026-05-18

## Directory Layout

```
FitNova Application/
├── lib/                         # Flutter client source (Dart)
│   ├── config/                  # Client-side configuration
│   ├── models/                  # Dart data models (mirrors backend JSON shapes)
│   ├── providers/               # ChangeNotifier state providers (Provider package)
│   ├── screens/                 # 12 full-page screen widgets
│   ├── services/                # HTTP REST + WebSocket service classes
│   ├── theme/                   # AppTheme, AppColors, AppSpacing, AppTypography
│   ├── widgets/                 # Shared widgets (domain + UI primitives)
│   │   └── ui/                  # Generic UI primitives (button, card, loader, etc.)
│   └── main.dart                # App entry point, route table, MultiProvider
│
├── backend/                     # Python FastAPI server + ML components
│   ├── app.py                   # Server entry point, all routes, lifespan startup
│   ├── config/                  # Exercise metadata SSOT + exercise rules
│   │   ├── exercises.json       # 27 exercises with metadata (idx, display_name, joints, etc.)
│   │   ├── exercises.py         # Python loader (lru_cached)
│   │   ├── defect_to_region.py  # Maps model defect codes to body regions
│   │   └── exercise_rules/      # Per-exercise geometric rules (squat.py; disconnected at runtime)
│   ├── data/                    # Serialised artefacts + synthetic dataset
│   │   ├── program_catalog.pkl  # 2,598 programs (dict, keyed by program_id)
│   │   ├── norm_stats.pkl       # Feature normalisation statistics
│   │   ├── user_features.csv    # 72,000 synthetic user profiles for NeuMF cold-start
│   │   ├── interactions.csv     # Synthetic user-program interaction matrix
│   │   ├── exercise_videos.json # Slug → demo filename mapping for LLM plan media URLs
│   │   ├── injury_exercise_blacklist.json  # Per-injury keyword exclusion lists
│   │   ├── exercise_media.json  # Extended exercise media metadata
│   │   └── qevd_extracted/      # Extracted QEVD .npy feature files (training only)
│   ├── models/                  # Trained model weights (served at runtime)
│   │   ├── neumf_final.keras    # NeuMF recommendation model
│   │   ├── neumf_metadata.pkl   # NeuMF user/item counts
│   │   ├── form_model/          # v4 legacy MT-TCN weights (fallback)
│   │   ├── form_model_v5_2/     # v5.2 ST-GCN Fit3D weights
│   │   └── form_model_v6/       # v6 ST-GCN QEVD weights (current target)
│   ├── services/                # Runtime business-logic services (loaded at startup)
│   │   ├── chat_service.py      # GPT-4o-mini conversational intake
│   │   ├── content_filter.py    # Layer 1: content-based program filter
│   │   ├── exercise_sanity.py   # Exercise mismatch detection
│   │   ├── form_analyzer.py     # Per-frame pose + feature extraction + ML dispatch
│   │   ├── form_geometry.py     # Geometric rules (retained, NOT wired at runtime)
│   │   ├── form_session.py      # Per-connection stateful session manager
│   │   ├── llm_adapter.py       # Layer 3: GPT-4o-mini plan generation
│   │   ├── mediapipe_config.py  # Pinned MediaPipe settings (training/inference parity)
│   │   ├── neumf_ranker.py      # Layer 2: NeuMF re-ranker
│   │   └── recommender.py       # Chains Layer 1 + 2
│   ├── scripts/                 # One-off data build scripts
│   │   ├── build_exercise_media.py
│   │   └── build_exercise_videos.py
│   ├── static/                  # Served by FastAPI /static mount
│   │   └── exercise_videos/     # 23 exercise demo video directories
│   ├── tests/                   # pytest test suite
│   └── training/                # Offline ML training pipeline (not loaded at runtime)
│       ├── models/              # TF model definitions (mt_tcn, st_gcn, st_gcn_v6, st_gcn_v6_1)
│       ├── preprocessing/       # Feature extraction, dataset builders, augmentation
│       ├── evaluation/          # Reality checks, offline eval, visualise pipeline
│       └── diagnostics/         # Phase-gate verification scripts
│
├── assets/                      # Flutter app assets
│   ├── fonts/                   # Sora variable font
│   └── logo/                    # App logo images
│
├── test/                        # Flutter widget tests (minimal)
├── pubspec.yaml                 # Flutter dependency manifest
├── analysis_options.yaml        # Dart linting config
│
├── android/ ios/ macos/ linux/ windows/ web/   # Generated Flutter platform scaffolding
├── build/  .dart_tool/          # Generated build artefacts — do not edit
│
├── colab_notebooks/             # Colab notebooks for Colab training runs
├── docs/                        # Project documentation
└── _build_*.py / bundle_for_colab.ps1  # Bundle scripts to zip source for Colab
```

## Directory Purposes

**`lib/config/`:**
- Purpose: Client-side constants and configuration helpers
- Key files: `api_config.dart` — resolves `baseUrl` and `wsUrl` per platform (Android uses `10.0.2.2:8000` for emulator host, others use `localhost:8000`)

**`lib/models/`:**
- Purpose: Dart representations of all JSON shapes exchanged with the backend
- Key files: `chat_models.dart` (ChatMessage, ChatResponse), `fitness_plan.dart` (FitnessPlan, DayPlan, ExerciseDetail), `form_models.dart` (FormFrameResult, FormSessionSummary, RepResult, FormFrameTimeline), `exercise_meta.dart` (ExerciseMeta for the exercise picker)

**`lib/providers/`:**
- Purpose: Application state shared across the widget tree; all extend `ChangeNotifier`
- Key files: `user_provider.dart`, `form_session_provider.dart`
- Note: `ThemeController` lives in `lib/theme/theme_controller.dart` — not in this directory

**`lib/screens/`:**
- Purpose: One file per navigation destination
- Key files (navigation order): `splash_screen.dart` → `registration_screen.dart` → `mode_select_screen.dart` → `home_screen.dart` → `chat_screen.dart` → `plan_screen.dart`; `exercise_selection_screen.dart` → `guidelines_screen.dart` → `form_check_screen.dart` (live camera) or `video_upload_screen.dart` (recorded video) → `form_results_screen.dart` or `form_replay_screen.dart`

**`lib/services/`:**
- Purpose: Network abstraction; screens call these instead of importing `http` directly
- Key files: `api_service.dart` (static methods), `form_session_service.dart` (WebSocket lifecycle)

**`lib/widgets/`:**
- Purpose: Reusable widgets split by type
- Domain widgets: `chat_bubble.dart`, `exercise_tile.dart`, `goal_card.dart`, `mismatch_banner.dart`, `skeleton_painter.dart`
- Primitive UI (`lib/widgets/ui/`): `app_button.dart`, `app_card.dart`, `app_empty_state.dart`, `app_error_state.dart`, `app_loader.dart`, `app_text_field.dart`

**`lib/theme/`:**
- Purpose: Design system tokens; use these everywhere instead of hardcoded values
- Key files: `app_colors.dart` (colour tokens), `app_spacing.dart` (spacing constants), `app_typography.dart` (text styles), `app_theme.dart` (light + dark `ThemeData`), `theme_controller.dart` (persists light/dark mode to SharedPreferences)

**`backend/services/`:**
- Purpose: All code that runs at server startup and handles requests — this is the primary development target for backend features
- Note: `form_geometry.py` is present but NOT wired into `FormSession` (pure ML path only as of 2026-05-11)

**`backend/training/`:**
- Purpose: Offline pipeline run in Google Colab; completely separate from the server runtime
- Models trained here produce `.weights.h5` / `.keras` files that are placed in `backend/models/` for server use
- Not imported by `backend/app.py` or any `backend/services/` module (except `FormAnalyzer._load_model()` which imports model-definition classes at load time only)

**`backend/data/`:**
- Purpose: Static data artefacts consumed at runtime. Generated offline and committed.
- `program_catalog.pkl`: dict of 2,598 programs; key is `program_id` (int)
- `user_features.csv`: 72,000 rows of synthetic users used by NeuMFRanker cold-start lookup
- `exercise_videos.json`: fuzzy-match lookup from exercise name to demo video slug; consumed by `LLMAdapter`

**`backend/config/exercises.json`:**
- Purpose: Single source of truth for all 27 supported exercises. Keys are canonical exercise IDs (e.g. `"squat"`, `"deadlift"`). Each entry has `idx`, `display_name`, `muscle_groups`, `key_errors_detected`, `relevant_joint_indices`, and camera setup fields.
- This file is loaded by `backend/config/exercises.py` and also served to Flutter via `GET /api/exercises`.

## Key File Locations

**Entry Points:**
- `lib/main.dart`: Flutter app entry (`void main()`)
- `backend/app.py`: FastAPI server entry (`app = FastAPI(...)` + all routes)

**Configuration:**
- `lib/config/api_config.dart`: API base URL + WebSocket URL per platform
- `backend/config/exercises.json`: Exercise metadata SSOT (27 exercises)
- `backend/services/mediapipe_config.py`: Pinned MediaPipe settings (training/inference parity)
- `pubspec.yaml`: Flutter package dependencies + font/asset declarations

**Core ML / Business Logic:**
- `backend/services/form_analyzer.py`: JPEG → pose → features → model inference dispatch
- `backend/services/form_session.py`: Sliding-window session state + rep counting + LLM feedback
- `backend/services/recommender.py`: Layer 1+2 chain
- `backend/services/llm_adapter.py`: Layer 3 plan generation + template fallback

**Model Definitions:**
- `backend/training/models/st_gcn_v6.py`: v6 ST-GCN (QEVD, 10-channel joint_err, 25 classes)
- `backend/training/models/st_gcn.py`: v5.2 ST-GCN (Fit3D, 5-channel joint_err, 15 classes)
- `backend/training/models/mt_tcn.py`: v4 legacy MT-TCN

**Data Preprocessing:**
- `backend/training/preprocessing/angular_features.py`: 22-feature angle computation from 15 canonical joints
- `backend/training/preprocessing/normalize.py`: `extract_canonical_from_mediapipe()`, `normalize_skeleton()`, `AngleNormalizer`
- `backend/training/preprocessing/joint_mapping.py`: MediaPipe 33-joint to canonical 15-joint mapping

**Testing:**
- `backend/tests/`: pytest suite (app, chat, recommender, form analyzer, content filter, etc.)
- `backend/services/test_form_analyzer_v6.py`, `test_form_session_v6_routing.py`: Service-level tests co-located with implementation
- `test/`: Flutter widget tests (minimal)

## Naming Conventions

**Files (Flutter/Dart):**
- `snake_case.dart` for all files: `form_session_service.dart`, `app_button.dart`
- Screen files: `*_screen.dart`
- Provider files: `*_provider.dart`
- Service files: `*_service.dart`
- Model files: use domain noun, e.g. `fitness_plan.dart`, `form_models.dart`
- Widget files: descriptive noun, e.g. `skeleton_painter.dart`, `mismatch_banner.dart`

**Classes (Flutter/Dart):**
- `PascalCase`: `FormSessionService`, `UserProvider`, `AppButton`
- Screen classes: `*Screen` + `_*State` for stateful
- Provider classes: `*Provider`
- Service classes: `*Service`

**Files (Python/backend):**
- `snake_case.py` for all modules
- Test files: `test_*.py` (some co-located in services/, some in tests/)
- Version suffixes on model/training files: `_v6.py`, `_v6_1.py`, `_v5_2.py`

**Classes (Python):**
- `PascalCase`: `ContentBasedFilter`, `NeuMFRanker`, `FormSession`
- Training models: `build_*_model()` factory functions (not classes)

## Where to Add New Code

**New Screen:**
- Implementation: `lib/screens/<name>_screen.dart`
- Register route in `lib/main.dart` routes map or `onGenerateRoute` (if it takes arguments)
- Access shared state via `context.watch<UserProvider>()` / `context.read<FormSessionProvider>()`

**New REST Endpoint:**
- Add Pydantic request/response models in `backend/app.py` (top of file with other models)
- Add `@app.post(...)` or `@app.get(...)` handler in `backend/app.py`
- If it needs a new service, add the service in `backend/services/<name>.py` and instantiate it in the `lifespan()` function, storing on `app.state`
- Add corresponding static method to `lib/services/api_service.dart`

**New Exercise:**
- Add entry to `backend/config/exercises.json` with a new canonical ID and all required fields (`idx`, `display_name`, `relevant_joint_indices`, etc.)
- If the exercise needs a demo video: add video files under `backend/static/exercise_videos/<slug>/`, update `backend/data/exercise_videos.json`

**New Widget (domain):**
- `lib/widgets/<name>.dart`
- Use theme tokens from `lib/theme/` (AppColors, AppSpacing, AppTypography)

**New Primitive UI Widget:**
- `lib/widgets/ui/<name>.dart`

**New Provider:**
- `lib/providers/<name>_provider.dart`
- Register in `MultiProvider` list in `lib/main.dart`

**New Backend Data Artefact:**
- Generate offline and place in `backend/data/`
- Load in the relevant service's `__init__` or use `lru_cache` pattern from `backend/config/exercises.py`

**New ML Model Version:**
- Model definition: `backend/training/models/<name>.py`
- Training script: `backend/training/train_form_model_<version>.py`
- Weights go to: `backend/models/form_model_<version>/`
- Add version detection logic to `FormAnalyzer._load_model()` in `backend/services/form_analyzer.py`
- Add `_predict_window_<version>()` method to `FormAnalyzer`
- Add `_load_<version>_model()` method to `FormAnalyzer`

## Special Directories

**`backend/models/`:**
- Purpose: Trained model weights served at runtime; also contains versioned backups
- Generated: Yes (by Colab training scripts)
- Committed: Weights are large binary files; `.keras` and `.weights.h5` are typically NOT committed to git (referenced by Colab)

**`backend/data/`:**
- Purpose: Serialised dataset artefacts and static config used at runtime
- Generated: Yes (by offline scripts in `backend/data/` and `backend/training/preprocessing/`)
- Committed: Yes — `program_catalog.pkl`, `norm_stats.pkl`, `user_features.csv`, `exercise_videos.json`, `injury_exercise_blacklist.json` etc. are committed

**`backend/static/exercise_videos/`:**
- Purpose: Served as static files under `/static/exercise_videos/<slug>/<file>`
- Generated: Yes (by `backend/scripts/build_exercise_videos.py`)
- Committed: Video files are large; likely excluded from git

**`backend/_archive_v5_2/`:**
- Purpose: Archived v5.2 training preprocessing scripts; not used in any active pipeline
- Generated: No
- Committed: Yes (historical reference)

**`colab_notebooks/`:**
- Purpose: Jupyter notebooks for Colab-based training and evaluation runs
- Generated: Partially (notebooks may be generated by `_build_*.py` scripts)
- Committed: Yes

**`build/` and `.dart_tool/`:**
- Purpose: Flutter build output; never edit
- Generated: Yes
- Committed: No (`.gitignore`)

---

*Structure analysis: 2026-05-18*
