# Technology Stack

**Analysis Date:** 2026-05-18

## Languages

**Primary:**
- Dart 3.7+ — Flutter client (`lib/`)
- Python 3.x (no `.python-version` pinned; inferred from syntax) — FastAPI backend + ML pipeline (`backend/`)

**Secondary:**
- None

## Runtime

**Flutter Client:**
- Flutter SDK (Dart ^3.7.0 per `pubspec.yaml`)
- Android target: `android:usesCleartextTraffic="true"` set in `android/app/src/main/AndroidManifest.xml` (cleartext HTTP to local backend)
- Package manager: `pub` (Flutter's built-in)
- Lockfile: `pubspec.lock` present and committed

**Python Backend:**
- Python runtime (no `.python-version` or `Pipfile`; dependencies managed via `pip`)
- Package manager: `pip` with `backend/requirements.txt`
- No lockfile — only range-pinned `requirements.txt`

## Frameworks

**Flutter Client:**
- Flutter (Material Design + Cupertino) — cross-platform UI framework; `uses-material-design: true` in `pubspec.yaml`
- Provider `^6.1.2` (resolved `6.1.x`) — state management; used in `lib/providers/user_provider.dart` and `lib/providers/form_session_provider.dart`

**Python Backend — API:**
- FastAPI `>=0.110.0` — REST + WebSocket server; entry point `backend/app.py`
- Uvicorn `>=0.29.0` (standard extras) — ASGI server; run command: `uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000`
- Pydantic `>=2.0.0` — request/response schema validation; all API models defined inline in `backend/app.py`

**Python Backend — ML:**
- TensorFlow `>=2.13.0` — model training and inference for NeuMF recommendation (`backend/services/neumf_ranker.py`) and form analysis (`backend/services/form_analyzer.py`)
- MediaPipe `>=0.10.9` (pinned operationally to `0.10.33` in `backend/services/mediapipe_config.py`) — pose landmark extraction; Tasks API, VIDEO running mode
- scikit-learn `>=1.3.0` — cosine similarity for cold-start user mapping (`backend/services/neumf_ranker.py`)

**Testing:**
- pytest `>=8.0.0` — Python unit/integration tests (`backend/services/test_*.py`, `backend/config/test_*.py`)
- flutter_test (Flutter SDK) — Dart test framework (dev dependency, no test files authored yet)
- flutter_lints `^5.0.0` / `flutter analyze` — static lint analysis; config in `analysis_options.yaml`

**Build/Dev:**
- python-dotenv `>=1.0.0` — loads `backend/.env` at runtime (`backend/services/llm_adapter.py`, `backend/services/chat_service.py`)

## Key Dependencies

**Flutter Client — Critical:**
- `camera: 0.11.2+1` (resolved) — live camera feed for real-time form analysis (`lib/screens/form_check_screen.dart`)
- `web_socket_channel: ^3.0.0` — WebSocket client for streaming frames to backend (`lib/services/form_session_service.dart`)
- `http: 1.6.0` (resolved) — REST calls to backend (`lib/services/api_service.dart`)
- `provider: ^6.1.2` — app-wide state for user profile and form session
- `video_player: ^2.8.0` — playback of form replay and exercise demo videos (`lib/screens/form_replay_screen.dart`)
- `image_picker: 1.2.1` (resolved) — select exercise video files from device for upload (`lib/screens/video_upload_screen.dart`)

**Flutter Client — UI:**
- `flutter_svg: 2.2.2` (resolved) — SVG asset rendering
- `cupertino_icons: 1.0.8` (resolved) — iOS-style icon set
- `shared_preferences: ^2.2.3` — local key-value persistence (theme mode in `lib/theme/theme_controller.dart`)
- `url_launcher: ^6.2.5` — open external URLs

**Python Backend — Critical:**
- `openai: >=1.30.0` — GPT-4o-mini calls for plan generation (`backend/services/llm_adapter.py`) and session-end coaching feedback (`backend/services/form_session.py`); also function-calling in `backend/services/chat_service.py`
- `mediapipe: >=0.10.9` — pose landmark extraction (33 landmarks, 22 angular features); pinned `0.10.33` + protobuf `4.25.3` in `backend/services/mediapipe_config.py`
- `tensorflow: >=2.13.0` — NeuMF recommendation model (`backend/models/neumf_final.keras`) and form-analysis MT-TCN/ST-GCN models
- `numpy: >=1.24.0` — angular feature computation, normalisation
- `pandas: >=2.0.0` — user feature CSV loading (`backend/data/user_features.csv`, `backend/data/program_features.csv`)
- `rapidfuzz: >=3.0.0` — fuzzy exercise name matching for video URL lookup (`backend/services/llm_adapter.py`)

**Python Backend — Infrastructure:**
- `opencv-python-headless: >=4.8.0` — video frame decoding in REST upload endpoint (`backend/app.py`: `analyze_form_video`)
- `scipy: >=1.11.0` — signal processing for form geometry
- `websockets: >=12.0` — WebSocket transport (used by uvicorn standard extras)
- `httpx: >=0.27.0` — async HTTP client (available for outgoing calls)
- `imageio-ffmpeg: >=0.4.9` — exercise video processing scripts (`backend/scripts/build_exercise_videos.py`)

## Custom ML Models

**NeuMF Recommendation (Layer 2):**
- `backend/models/neumf_final.keras` — Neural Matrix Factorisation; pre-trained GMF + MLP components at `backend/models/gmf_pretrained.keras` and `backend/models/mlp_pretrained.keras`
- `backend/models/neumf_metadata.pkl` — user/item count metadata
- `backend/data/program_catalog.pkl` — 2,598 programs with Week 1 exercises
- `backend/data/norm_stats.pkl` — feature normalisation statistics

**Form Analysis (Layer 4):**
- Active weights selected at startup with auto-detection priority: v6 (`backend/models/form_model_v6/v6_supervised.weights.h5`) → v5.2 (`backend/models/form_model_v5_2/v5_2_supervised.weights.h5`) → v4 legacy (`backend/models/form_model/mt_tcn_weights.weights.h5`)
- Override via `FITNOVA_MODEL_DIR` environment variable
- Model architecture definitions: `backend/training/models/mt_tcn.py` (v4 MT-TCN), `backend/training/models/st_gcn.py` (v5.x ST-GCN), `backend/training/models/st_gcn_v6.py` / `st_gcn_v6_1.py` (v6/v6.1)
- MediaPipe model asset: `backend/models/pose_landmarker_full.task` (~29 MB float16, auto-downloaded from Google on first use)

## Configuration

**Environment:**
- Backend secrets live in `backend/.env` (not committed; `backend/.env.example` committed as template)
- Only one required key: `OPENAI_API_KEY`
- Optional override: `FITNOVA_MODEL_DIR` — selects which model version directory to load

**Flutter API Endpoint:**
- Configured in `lib/config/api_config.dart`
- Android emulator: `http://10.0.2.2:8000` / `ws://10.0.2.2:8000`
- Web/desktop/iOS: `http://localhost:8000` / `ws://localhost:8000`
- No production URL configured — local development only

**Build:**
- `analysis_options.yaml` — extends `package:flutter_lints/flutter.yaml`; no custom rules added
- `pubspec.yaml` — `publish_to: 'none'` (private, not pub.dev publishable)
- Font: custom Sora variable font (`assets/fonts/Sora-Variable.ttf`), weights 400–800
- Assets: `assets/logo/` (app logo)

## Platform Requirements

**Development:**
- Flutter SDK with Dart ^3.7.0
- Python (no enforced version; `>=3.10` needed for `str | None` union syntax used throughout backend)
- `pip install -r backend/requirements.txt`
- `OPENAI_API_KEY` in `backend/.env`
- MediaPipe model auto-downloads on first backend startup if absent
- Backend runs on port 8000; Flutter connects to 10.0.2.2:8000 from Android emulator

**Production:**
- No deployment configuration found — no Dockerfile, no `render.yaml`, no `fly.toml`, no `Procfile`
- Intended as a local graduation-project demo server

---

*Stack analysis: 2026-05-18*
