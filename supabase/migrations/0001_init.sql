-- FitNova — initial per-user schema (auth, profiles, plans, workout logs, coach conversations)
--
-- Security model:
--   * Supabase Auth (auth.users) is the identity source. auth.uid() returns the
--     authenticated user's UUID from the verified JWT.
--   * Every table below has Row-Level Security ENABLED. The Flutter client connects
--     with the public anon key + the user's JWT, so it can only ever see/modify rows
--     where the row's owner == auth.uid().
--   * The FastAPI backend connects with the service-role key, which BYPASSES RLS.
--     The backend always verifies the user's JWT first (backend/deps/auth.py) and
--     asserts the row's user_id == the JWT 'sub' before writing. RLS is the safety net;
--     the backend's own checks are the first line.
--
-- Idempotent: safe to re-run (uses IF NOT EXISTS / CREATE OR REPLACE / DROP POLICY IF EXISTS).

-- ─────────────────────────────────────────────────────────────────────────────
-- Helper functions
-- ─────────────────────────────────────────────────────────────────────────────

-- Bump updated_at on every UPDATE.
create or replace function public.set_updated_at()
returns trigger
language plpgsql
as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

-- Create a blank profile row when a new auth user signs up.
-- SECURITY DEFINER so it can insert into public.profiles regardless of RLS.
-- search_path pinned to '' to prevent search-path hijacking (Supabase linter rule).
create or replace function public.handle_new_user()
returns trigger
language plpgsql
security definer
set search_path = ''
as $$
begin
  insert into public.profiles (id, display_name)
  values (new.id, nullif(new.raw_user_meta_data->>'display_name', ''));
  return new;
end;
$$;

-- ─────────────────────────────────────────────────────────────────────────────
-- profiles — one row per auth user. Detail fields nullable; filled during onboarding.
-- ─────────────────────────────────────────────────────────────────────────────

create table if not exists public.profiles (
  id                      uuid primary key references auth.users(id) on delete cascade,
  display_name            text,
  age                     int          check (age between 10 and 100),
  gender                  text         check (gender in ('Male', 'Female')),
  height_cm               numeric(5,2) check (height_cm between 80 and 250),
  weight_kg               numeric(5,2) check (weight_kg between 20 and 400),
  injuries                text[]       not null default '{}',
  training_focus          text         check (training_focus in ('bodybuilding', 'powerbuilding', 'powerlifting', 'cardio', 'general')),
  years_training          int          check (years_training between 0 and 80),
  equipment               text[]       not null default '{}',
  experience_level        int          check (experience_level in (1, 2, 3)),
  session_duration_hours  numeric(3,1) check (session_duration_hours > 0 and session_duration_hours <= 4.0),
  workout_frequency       int          check (workout_frequency between 1 and 7),
  unit_preference         text         not null default 'metric' check (unit_preference in ('metric', 'imperial')),
  onboarding_completed    boolean      not null default false,
  created_at              timestamptz  not null default now(),
  updated_at              timestamptz  not null default now()
);

drop trigger if exists profiles_set_updated_at on public.profiles;
create trigger profiles_set_updated_at
  before update on public.profiles
  for each row execute function public.set_updated_at();

alter table public.profiles enable row level security;

drop policy if exists profiles_select_self on public.profiles;
create policy profiles_select_self on public.profiles
  for select using (auth.uid() = id);

drop policy if exists profiles_update_self on public.profiles;
create policy profiles_update_self on public.profiles
  for update using (auth.uid() = id) with check (auth.uid() = id);
-- INSERT handled by handle_new_user() trigger (SECURITY DEFINER); DELETE cascades from auth.users.

grant select, update on public.profiles to authenticated;

-- ─────────────────────────────────────────────────────────────────────────────
-- plans — a generated 7-day plan. One active plan per user (partial unique index).
-- ─────────────────────────────────────────────────────────────────────────────

create table if not exists public.plans (
  id                    uuid primary key default gen_random_uuid(),
  user_id               uuid not null references auth.users(id) on delete cascade,
  program_id            int  not null,
  similar_user_id       int,                       -- recommender kNN catalog id; NOT a FK to auth.users (audit trace only)
  program_title         text not null,
  personalization_notes text,
  source                text not null check (source in ('llm', 'template_fallback')),
  is_active             boolean not null default true,
  created_at            timestamptz not null default now()
);

create index if not exists plans_user_idx on public.plans (user_id);
create unique index if not exists plans_one_active_per_user on public.plans (user_id) where is_active;

alter table public.plans enable row level security;

drop policy if exists plans_all_self on public.plans;
create policy plans_all_self on public.plans
  for all using (auth.uid() = user_id) with check (auth.uid() = user_id);

grant select, insert, update, delete on public.plans to authenticated;

-- ─────────────────────────────────────────────────────────────────────────────
-- plan_days — 7 days per plan. Ownership tested via join to plans.
-- ─────────────────────────────────────────────────────────────────────────────

create table if not exists public.plan_days (
  id           uuid primary key default gen_random_uuid(),
  plan_id      uuid not null references public.plans(id) on delete cascade,
  day_number   int  not null check (day_number between 1 and 7),
  focus        text not null,
  is_rest_day  boolean not null,
  is_completed boolean not null default false,
  completed_at timestamptz,
  unique (plan_id, day_number)
);

create index if not exists plan_days_plan_idx on public.plan_days (plan_id);

alter table public.plan_days enable row level security;

drop policy if exists plan_days_all_self on public.plan_days;
create policy plan_days_all_self on public.plan_days
  for all
  using (exists (
    select 1 from public.plans p
    where p.id = plan_days.plan_id and p.user_id = auth.uid()
  ))
  with check (exists (
    select 1 from public.plans p
    where p.id = plan_days.plan_id and p.user_id = auth.uid()
  ));

grant select, insert, update, delete on public.plan_days to authenticated;

-- ─────────────────────────────────────────────────────────────────────────────
-- plan_exercises — exercises within a day. Ownership tested via join plan_days→plans.
-- ─────────────────────────────────────────────────────────────────────────────

create table if not exists public.plan_exercises (
  id           uuid primary key default gen_random_uuid(),
  plan_day_id  uuid not null references public.plan_days(id) on delete cascade,
  position     int  not null,
  exercise_name text not null,
  sets         int  not null,
  reps         text not null,            -- string: "8-10", "30 sec", "AMRAP"
  rest_seconds int  not null,
  coaching_cue text not null,
  media_url    text,
  is_completed boolean not null default false,
  completed_at timestamptz,
  unique (plan_day_id, position)
);

create index if not exists plan_exercises_day_idx on public.plan_exercises (plan_day_id);

alter table public.plan_exercises enable row level security;

drop policy if exists plan_exercises_all_self on public.plan_exercises;
create policy plan_exercises_all_self on public.plan_exercises
  for all
  using (exists (
    select 1 from public.plan_days d
    join public.plans p on p.id = d.plan_id
    where d.id = plan_exercises.plan_day_id and p.user_id = auth.uid()
  ))
  with check (exists (
    select 1 from public.plan_days d
    join public.plans p on p.id = d.plan_id
    where d.id = plan_exercises.plan_day_id and p.user_id = auth.uid()
  ));

grant select, insert, update, delete on public.plan_exercises to authenticated;

-- ─────────────────────────────────────────────────────────────────────────────
-- workout_logs — one logged set. user_id denormalized for fast RLS + index.
-- exercise_name is a snapshot so logs survive plan deletion (FK set null).
-- ─────────────────────────────────────────────────────────────────────────────

create table if not exists public.workout_logs (
  id               uuid primary key default gen_random_uuid(),
  user_id          uuid not null references auth.users(id) on delete cascade,
  plan_exercise_id uuid references public.plan_exercises(id) on delete set null,
  exercise_name    text not null,
  set_number       int  not null check (set_number >= 1),
  reps_completed   int  check (reps_completed >= 0),
  weight_kg        numeric(6,2) check (weight_kg >= 0),
  duration_seconds int  check (duration_seconds >= 0),
  rpe              numeric(3,1) check (rpe between 1 and 10),
  notes            text,
  performed_at     timestamptz not null default now()
);

create index if not exists workout_logs_user_time_idx on public.workout_logs (user_id, performed_at desc);
create index if not exists workout_logs_user_exercise_idx on public.workout_logs (user_id, exercise_name, performed_at desc);
create index if not exists workout_logs_plan_exercise_idx on public.workout_logs (plan_exercise_id);

alter table public.workout_logs enable row level security;

drop policy if exists workout_logs_all_self on public.workout_logs;
create policy workout_logs_all_self on public.workout_logs
  for all using (auth.uid() = user_id) with check (auth.uid() = user_id);

grant select, insert, update, delete on public.workout_logs to authenticated;

-- ─────────────────────────────────────────────────────────────────────────────
-- conversations — coach chat threads. v1 = one row per user ('Fitness Coach').
-- ─────────────────────────────────────────────────────────────────────────────

create table if not exists public.conversations (
  id              uuid primary key default gen_random_uuid(),
  user_id         uuid not null references auth.users(id) on delete cascade,
  title           text not null default 'Fitness Coach',
  created_at      timestamptz not null default now(),
  last_message_at timestamptz not null default now()
);

create index if not exists conversations_user_recent_idx on public.conversations (user_id, last_message_at desc);

alter table public.conversations enable row level security;

drop policy if exists conversations_all_self on public.conversations;
create policy conversations_all_self on public.conversations
  for all using (auth.uid() = user_id) with check (auth.uid() = user_id);

grant select, insert, update, delete on public.conversations to authenticated;

-- ─────────────────────────────────────────────────────────────────────────────
-- conversation_messages — full multi-turn transcript incl. tool calls/results.
-- user_id denormalized for fast RLS.
-- ─────────────────────────────────────────────────────────────────────────────

create table if not exists public.conversation_messages (
  id              uuid primary key default gen_random_uuid(),
  conversation_id uuid not null references public.conversations(id) on delete cascade,
  user_id         uuid not null references auth.users(id) on delete cascade,
  role            text not null check (role in ('user', 'assistant', 'tool', 'system')),
  content         text,
  tool_call_id    text,
  tool_name       text,
  tool_args       jsonb,
  tool_result     jsonb,
  created_at      timestamptz not null default now()
);

create index if not exists conversation_messages_thread_idx on public.conversation_messages (conversation_id, created_at);
create index if not exists conversation_messages_user_idx on public.conversation_messages (user_id);

alter table public.conversation_messages enable row level security;

drop policy if exists conversation_messages_all_self on public.conversation_messages;
create policy conversation_messages_all_self on public.conversation_messages
  for all using (auth.uid() = user_id) with check (auth.uid() = user_id);

grant select, insert, update, delete on public.conversation_messages to authenticated;

-- ─────────────────────────────────────────────────────────────────────────────
-- Trigger: blank profile on signup. Created last so public.profiles exists.
-- ─────────────────────────────────────────────────────────────────────────────

drop trigger if exists on_auth_user_created on auth.users;
create trigger on_auth_user_created
  after insert on auth.users
  for each row execute function public.handle_new_user();
