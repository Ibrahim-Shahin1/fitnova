# FitNova

An AI-powered fitness app with a Flutter client and a FastAPI backend. It offers three
features: a personalized workout-plan generator, a nutrition and meal planner, and an
exercise form-correction benchmark for the squat, the overhead press, and squat depth.

## Requirements

- Python 3.10+ (backend and ML)
- Flutter SDK 3.7+ with Dart 3.7+ (client)
- A Supabase project (authentication + Postgres) and an OpenAI API key

## Backend (FastAPI)

Run these in the VS Code terminal from the project root:

```bash
pip install -r backend/requirements.txt
# create backend/.env from the template and fill in the values
uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000
```

Copy `backend/.env.example` to `backend/.env` and set `OPENAI_API_KEY`, `SUPABASE_URL`,
`SUPABASE_ANON_KEY`, `SUPABASE_JWT_SECRET`, and `SUPABASE_DB_URL`. The server then runs at
`http://localhost:8000` (interactive docs at `/docs`). On first start it builds a small
isolated environment for the plan and nutrition crews from `backend/requirements-crew.txt`.

## Frontend (Flutter)

In a second VS Code terminal from the project root:

```bash
flutter pub get
flutter run
```

Choose a device when prompted. The client reads the backend URL per platform from
`lib/config/api_config.dart`.

## Layout

- `lib/` — Flutter client (screens, providers, services)
- `backend/` — FastAPI server, ML services, model weights, and training code
- `docs/` — project report, evaluation write-ups, figures, and notebooks
- `Fitness-AQA/` — the form-correction dataset (non-commercial, research use only)
