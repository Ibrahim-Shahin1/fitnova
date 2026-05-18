# External Integrations

**Analysis Date:** 2026-05-18

## APIs & External Services

**OpenAI:**
- Service: OpenAI Chat Completions API
- Used for three distinct purposes:
  1. **Plan generation (Layer 3)** — GPT-4o-mini with `response_format: json_object`; called in `backend/services/llm_adapter.py` via `self.client.chat.completions.create(model="gpt-4o-mini", ...)`
  2. **Conversational intake** — GPT-4o-mini with function calling (`extract_fitness_params` tool); called in `backend/services/chat_service.py`
  3. **Session-end coaching feedback** — GPT-4o-mini generates post-session LLM feedback summary; called in `backend/services/form_session.py`
- SDK: `openai>=1.30.0` (Python); client instantiated as `OpenAI(api_key=...)`
- Auth: `OPENAI_API_KEY` environment variable, loaded from `backend/.env` via `python-dotenv`
- Retry logic: 3 attempts with timeout=60s for plan generation; no explicit retry in chat/form_session

**Google MediaPipe (model asset download):**
- Service: Google's MediaPipe Model Zoo (CDN)
- Used for: one-time download of `pose_landmarker_full.task` (~29 MB float16) on first backend startup
- URL: `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task`
- Config: `backend/services/mediapipe_config.py` (`POSE_LANDMARKER_MODEL_URL`, `ensure_model_downloaded()`)
- After download, asset is cached at `backend/models/pose_landmarker_full.task`; no further network calls

## Data Storage

**Databases:**
- None. No database (SQL or NoSQL) is used anywhere in the stack.
- All persistent application data is stored as serialised files on the local filesystem:
  - `backend/data/program_catalog.pkl` — 2,598 workout programs
  - `backend/data/norm_stats.pkl` — content-filter normalisation stats
  - `backend/data/user_features.csv` — pre-computed user feature vectors for cold-start
  - `backend/data/program_features.csv` — pre-computed program feature vectors
  - `backend/data/interactions.csv` — synthetic user-program interaction matrix
  - `backend/models/neumf_metadata.pkl` — NeuMF user/item count metadata
  - `backend/models/user_pos_items.pkl` — positive interaction items per user
  - `backend/data/exercise_videos.json` — exercise video slug/alias lookup
  - `backend/data/exercise_media.json` — exercise metadata including image paths
  - `backend/data/injury_exercise_blacklist.json` — exercise keywords to avoid per injury

**Flutter Client Persistence:**
- `shared_preferences: ^2.2.3` — device key-value store; used only for theme mode (`lib/theme/theme_controller.dart`)
- No local database (sqflite, Hive, etc.) used on the Flutter side

**File Storage:**
- Local filesystem only
- Exercise demo videos served from `backend/static/exercise_videos/<slug>/` via FastAPI's `StaticFiles` mount at `/static`
- 22 exercise directories present under `backend/static/exercise_videos/` (bench_press, squat, deadlift, etc.)
- Static mount configured in `backend/app.py`: `app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")`

**Caching:**
- No runtime caching layer (no Redis, Memcached, or in-process LRU cache)
- ML model artifacts are loaded once into `app.state` at server startup via `lifespan` context manager in `backend/app.py` and held in memory for the process lifetime

## Authentication & Identity

**Auth Provider:**
- None. No authentication layer exists anywhere in the stack.
- The backend has no login, JWT, session cookies, or API keys for client calls
- `CORSMiddleware` is configured with `allow_origins=["*"]` — open to all origins (`backend/app.py`)
- User profile data (age, BMI, experience level, etc.) is submitted per-request in the Flutter client via `lib/providers/user_provider.dart` and `lib/screens/registration_screen.dart`; nothing is persisted server-side per user

## Monitoring & Observability

**Error Tracking:**
- None. No Sentry, Rollbar, or similar service is integrated.

**Logs:**
- Python `logging` module only; logger named `"fitnova"` (`backend/app.py`) with sub-loggers `"fitnova.llm"` (`backend/services/llm_adapter.py`) and `"fitnova.chat"` (`backend/services/chat_service.py`)
- Log output goes to stdout; no log aggregation or file rotation configured
- Flutter: no structured logging; standard `print`/`debugPrint` pattern

## CI/CD & Deployment

**Hosting:**
- Not applicable. No deployment configuration found (no Dockerfile, `render.yaml`, `fly.toml`, `heroku.yml`, or `Procfile`).
- Backend is run locally: `uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000`

**CI Pipeline:**
- None. No GitHub Actions, CircleCI, or equivalent pipeline configuration files found.

## Internal Service Communication

**Flutter → Backend (REST):**
- Client: `http: 1.6.0` (Dart) in `lib/services/api_service.dart`
- Endpoints consumed:
  - `POST /chat` — conversational intake
  - `POST /generate-plan` — full recommendation + plan generation pipeline
  - `GET /api/exercises` — exercise metadata catalogue
  - `POST /analyze-form-video` — multipart video upload for REST form analysis
  - `GET /health` — liveness check with model version info

**Flutter → Backend (WebSocket):**
- Client: `web_socket_channel: ^3.0.0` (Dart) in `lib/services/form_session_service.dart`
- Endpoint: `WS /ws/form-session`
- Protocol: JSON messages; client sends `start_session` → repeated `frame` (base64 JPEG) → `end_session`; server returns per-frame `FormFrameResult` JSON and final `FormSessionSummary` JSON

## Environment Configuration

**Required env vars (backend):**
- `OPENAI_API_KEY` — must be set in `backend/.env` or OS environment; backend raises `ValueError` on startup if absent

**Optional env vars (backend):**
- `FITNOVA_MODEL_DIR` — absolute path to override model directory selection; if unset, auto-detection picks v6 → v5.2 → v4 based on file presence

**Secrets location:**
- `backend/.env` — not committed to git
- `backend/.env.example` — committed template showing only `OPENAI_API_KEY=sk-your-key-here`

## Webhooks & Callbacks

**Incoming:**
- None

**Outgoing:**
- None (all external calls are synchronous request/response: OpenAI Chat Completions, MediaPipe one-time download)

---

*Integration audit: 2026-05-18*
