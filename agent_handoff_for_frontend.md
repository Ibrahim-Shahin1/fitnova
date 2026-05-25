# FitNova — Handoff: merge the frontend / auth / DB / AI-coach stream into the main repo

This stream (frontend + Supabase auth + per-user database + AI coach + plan
generation) was built in a separate git worktree to avoid disturbing the
form-correction / model-training work. It is complete and committed. This doc
explains how to consolidate it into the main repo and run it **without breaking
form correction**.

## Repo state (same `.git`, two worktrees)

- Main worktree: `C:\Users\tsh_x\Desktop\FitNova Application` → branch `fresh-start`
  (form-correction / training; 64 commits past the shared base `a0841b4`).
- Frontend worktree: `C:\Users\tsh_x\Desktop\FitNova-frontend` → branch
  **`feature/frontend-user-profile`** (35 commits past `a0841b4`, HEAD `3f5a900`, clean).
- **Verified: the two branches changed NO files in common → the merge is conflict-free.**
  The form/training serving files (`app.py`, `requirements.txt`, `llm_adapter.py`,
  `recommender.py`, …) were *not* touched on `fresh-start`; all plan/LLM/coach
  changes are exclusively on the feature branch.

## STEP 1 — Merge (from `C:\Users\tsh_x\Desktop\FitNova Application`)

```
git merge feature/frontend-user-profile
```

Expect a clean merge. Afterward, sanity-check that `backend/app.py` and
`backend/requirements.txt` contain BOTH the form endpoints/deps and the new
routers (`plan`, `coach`, `logs`) + `psycopg[binary,pool]`.

## STEP 2 — Provision local (gitignored) artifacts into the main dir

These are intentionally NOT in git:

1. **`backend/.env`** — copy from `...\FitNova-frontend\backend\.env`. Holds
   `OPENAI_API_KEY` + `SUPABASE_URL`, `SUPABASE_ANON_KEY`, `SUPABASE_JWT_SECRET`,
   `SUPABASE_DB_URL`. Keep it untracked.
2. **`backend/.crewenv`** — the isolated CrewAI venv (the multi-agent plan
   generator runs here as a subprocess). Fastest: set env var
   `FITNOVA_CREW_PYTHON=C:\Users\tsh_x\Desktop\FitNova-frontend\backend\.crewenv\Scripts\python.exe`
   to reuse the existing one. Portable alternative:
   `python -m venv backend\.crewenv && backend\.crewenv\Scripts\python -m pip install -r backend\requirements-crew.txt`.
   **⚠️ Do NOT `pip install crewai` into the main backend env** — CrewAI forces
   `protobuf<6`, which would downgrade protobuf 7 and break TensorFlow 2.21 /
   mediapipe (the form stack). That is the entire reason it is isolated in a
   subprocess.
3. **`backend/data/exercise_videos.json` + `backend/static/exercise_videos/`** —
   exercise demo clips. Already present in the main dir (the worktree copied them
   FROM here). Leave them.
4. Main backend env: `pip install -r backend\requirements.txt` (adds
   `psycopg[binary,pool]`; auth uses `httpx`, already present — no PyJWT).
5. Flutter: `flutter pub get` (adds `supabase_flutter`, `fl_chart`, `intl`).
   Supabase URL/anon key are hardcoded in `lib/config/supabase_config.dart`; deep
   link `io.fitnova.app://auth-callback` is in AndroidManifest/Info.plist.

## STEP 3 — Database (shared Supabase, already live & current)

The schema (`supabase/migrations/0001_init.sql`: `profiles, plans, plan_days,
plan_exercises, workout_logs, conversations, conversation_messages`, all
RLS-enabled, `handle_new_user` trigger) was applied via `supabase db push`. One
live change was also made directly: `profiles.training_focus` CHECK now allows
`bodybuilding|powerbuilding|powerlifting|cardio|general` (old `hypertrophy` rows
migrated to `bodybuilding`); the migration file reflects this. **No DB action
needed** — same shared project.

## What was built

- **Auth:** Supabase email auth (sign-in/up, email-verify deep link, password
  reset). `AuthProvider` + `AuthGate` route splash → auth → onboarding → 6-tab
  shell. Backend `deps/auth.py` `require_user` verifies tokens **remotely** via
  `GET {SUPABASE_URL}/auth/v1/user` (algorithm-agnostic — the project uses
  asymmetric signing keys, so local HS256/PyJWT was abandoned).
- **DB layer:** `backend/db/supabase_client.py` (psycopg pool against the Supabase
  **session pooler**) + `backend/db/repositories/` (`profile_repo`, `plan_repo`,
  `log_repo`, `conversation_repo`).
- **Nav shell (6 tabs):** Home (dashboard), Planning, Progress, Form (wraps the
  existing form flow **unchanged**), Nutrition (placeholder), Profile.
- **Onboarding/Profile:** collects name/age/gender/height/weight + **experience
  level, training goal, workouts/week, equipment, injuries**; editable later in
  Profile.
- **Plan generation = CrewAI 4-agent pipeline** (`backend/services/crew_runner.py`,
  runs in `.crewenv` subprocess): **Profiler → Generator → Critic → Optimizer**
  with a critic↔optimizer loop. Grounded **100% in the recommended program's
  catalog exercises** (no template fallback, no invented exercises). A
  deterministic engine (`backend/services/plan_validators.py`: movement-group
  classifier + `repair_plan`) guarantees coherent splits (no chest on a pull day),
  injury/equipment safety, and dataset-only provenance. `backend/services/plan_crew.py`
  is the in-process bridge (no crewai import) that runs the subprocess, hard-gates,
  normalizes, and persists. `llm_adapter.generate_plan()` now delegates to it.
- **Goal-aware program pick:** `FOCUS_TO_GOALS` + `select_program_id` choose a
  recommender candidate whose dataset `goal` matches the user's goal (the content
  filter ignored goal before).
- **Live generation screen:** SSE `POST /api/plan/generate/stream` streams
  per-agent progress (+ the Critic's findings/score per round) to a full-screen
  Flutter view with a "Proceed" gate before revealing the plan.
- **Persistent AI coach** (`coach_service.py`, gpt-4o-mini tool loop): greets
  first as "FitNova AI", confirms the saved profile (never re-asks), and on
  readiness calls `prepare_plan` → surfaces the "Generate Plan" button. Tools in
  `coach_tools.py` (`get_user_profile`, `get_active_plan`, `prepare_plan`).
- **Per-set logging + Progress tab:** `workout_logs` CRUD (`/api/logs`), plan
  day/exercise completion (`PATCH /api/plan/...`), Progress tab with estimated-1RM
  charts, PRs, weekly volume, week-vs-week (fl_chart).
- **"View exercise" demo videos** restored on the plan view.

## New backend endpoints

All `Depends(require_user)`: `/api/me`, `/api/plan/generate`,
`/api/plan/generate/stream` (SSE), `/api/plan/active`,
`/api/plan/exercises/{id}/complete`, `/api/plan/days/{id}/complete`,
`/api/chat/start`, `/api/chat/conversation`, `/api/chat/messages`, `/api/logs`
(POST/GET/DELETE). Existing endpoints (`/chat`, `/generate-plan`,
`/ws/form-session`, `/analyze-form-video`, `/health`, `/api/exercises`) are
untouched.

## Not touched (safe)

All form-correction (`form_*.py`, `mediapipe_*.py`, `backend/training/**`, the
form screens, `/ws/form-session`), `recommender.py` / `content_filter.py` /
`neumf_ranker.py` (consumed read-only), the program catalog/`.pkl` data,
`chat_service.py` (intake), and the existing dynamic routes in `lib/main.dart`.

## Run + smoke test

1. Backend (from the main dir): `uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000`.
2. Flutter on the device/emulator: `flutter run`.
3. Sign up → fill profile (goal/level/frequency/equipment/injuries) → open
   **Coach**: it greets as FitNova AI and recaps the profile → say "yes" →
   **Generate Plan** button → watch the 4 agents → **Proceed** → plan appears in
   the Planning tab. Log a set → check the Progress tab. Confirm the **Form** tab
   still runs the camera/analysis flow exactly as before.

The full per-commit history is on `feature/frontend-user-profile` (35 commits;
messages are descriptive).
