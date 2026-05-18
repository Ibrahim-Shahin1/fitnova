# Coding Conventions

**Analysis Date:** 2026-05-18

---

## Two Codebases, Two Convention Sets

FitNova has a Flutter/Dart client in `lib/` and a Python backend in `backend/`. They use
different languages but share several structural patterns (layered modules, thin service
objects, provider/state classes, and explicit error propagation). Differences are called
out under each section.

---

## Naming Patterns

### Dart / Flutter (`lib/`)

**Files:**
- Screens: `snake_case_screen.dart` — e.g., `form_check_screen.dart`, `chat_screen.dart`
- Providers: `snake_case_provider.dart` — e.g., `user_provider.dart`, `form_session_provider.dart`
- Services: `snake_case_service.dart` — e.g., `api_service.dart`, `form_session_service.dart`
- Models: `snake_case_models.dart` or `noun.dart` — e.g., `fitness_plan.dart`, `form_models.dart`
- Widgets: `snake_case.dart` — e.g., `skeleton_painter.dart`, `app_button.dart`
- Theme files: `app_<token>.dart` — e.g., `app_colors.dart`, `app_spacing.dart`, `app_typography.dart`
- Config: `api_config.dart`

**Classes:**
- `UpperCamelCase` for all types: `UserProvider`, `FormSessionProvider`, `SkeletonPainter`, `AppButton`
- Widget state classes: `_<WidgetName>State` — e.g., `_ChatScreenState`, `_FormCheckScreenState`
- Private classes prefixed with `_`

**Functions / Methods:**
- `lowerCamelCase` for all methods and functions: `setProfile()`, `updateExtracted()`, `buildJointErrorMap()`
- Private methods prefixed with `_`: `_fetchGreeting()`, `_sendMessage()`, `_initCamera()`
- Event handlers: `_on<Event>()` or `_<verb><Noun>()` pattern

**Variables:**
- `lowerCamelCase`: `experienceLevel`, `sessionDurationHours`, `workoutFrequency`
- Private provider fields: underscore-prefixed (`_state`, `_latestFrame`, `_liveQuality`)
- Constants: `lowerCamelCase` for `static const`, `SCREAMING_SNAKE` for file-level constants — e.g., `_frameIntervalMs = 100`, `_connections`

**Enums:**
- `UpperCamelCase` for type, `lowerCamelCase` for values: `FormSessionState.idle`, `AppButtonVariant.primary`

### Python / Backend (`backend/`)

**Files:**
- Services: `snake_case.py` — e.g., `form_analyzer.py`, `llm_adapter.py`, `chat_service.py`
- Tests live both in `backend/tests/` and co-located next to source modules (e.g., `backend/services/test_form_geometry.py`, `backend/config/test_defect_to_region.py`)
- Training preprocessing: `snake_case.py` in `backend/training/preprocessing/`

**Classes:**
- `UpperCamelCase`: `FormAnalyzer`, `LLMAdapter`, `Recommender`, `ContentBasedFilter`
- Pydantic models: `UpperCamelCase` suffixed by role — `UserProfileRequest`, `GeneratePlanResponse`, `ChatRequest`, `ChatResponse`

**Functions / Methods:**
- Public: `snake_case` — `generate_plan()`, `recommend()`, `process_message()`
- Private helpers: `_snake_case` — `_call_llm()`, `_build_user_prompt()`, `_validate_response()`, `_sanitize_reps()`
- Module-level helpers in test files: `_snake_case` — `_sample_plan()`, `_make_response()`

**Variables / Constants:**
- Module-level constants: `SCREAMING_SNAKE` — `WINDOW_SIZE`, `INFERENCE_INTERVAL`, `MAX_FILL_FRAMES`
- Private module-level paths: `_snake_case` — `_BASE_DIR`, `_DATA_DIR`, `_CATALOG_PATH`
- Local variables: `snake_case`

---

## Code Style

### Dart / Flutter

**Formatting:**
- Tool: `dart format` (enforced by Flutter toolchain)
- No additional `.prettierrc` or `biome.json` — relies on `dart format` defaults
- Single quotes for strings (Flutter convention)

**Linting:**
- Tool: `flutter_lints ^5.0.0` (declared in `pubspec.yaml`)
- Config: `analysis_options.yaml` at repo root — uses `package:flutter_lints/flutter.yaml`
- No custom rule overrides are currently active; the commented-out lines show `prefer_single_quotes` and `avoid_print` are under consideration but not enabled
- Suppress per-file with `// ignore_for_file: rule_name` or per-line with `// ignore: rule_name`

**Key style observations:**
- `super.key` parameter passed to widget constructors (modern Flutter idiom): `const AppButton({super.key, ...})`
- `const` constructors used throughout for stateless widgets
- Dart 3 pattern matching used: `switch` expressions with destructuring `(bg, fg, border) = switch (variant) {...}`
- `context.read<T>()` preferred over `Provider.of<T>(context, listen: false)` in event handlers

### Python / Backend

**Formatting:**
- No `.flake8`, `pyproject.toml`, or `setup.cfg` found — no enforced formatter
- Line lengths are not rigidly controlled; long lines appear in prompt strings and constants
- Backslash line continuation used sparingly for long string literals

**Linting / Type Checking:**
- No `mypy.ini` or `pyright` config found
- Type annotations are used consistently in all service and config modules
- `from __future__ import annotations` present in virtually all backend Python files — enables PEP 563 postponed evaluation for forward references and Python 3.10-style union syntax (`X | Y`) on older runtimes

**Key style observations:**
- Module docstrings at file top: triple-quoted, multi-line, describe purpose and pipeline role (see `backend/app.py`, `backend/services/llm_adapter.py`)
- Section separators using `# ─────────...─────────` (box-drawing dashes) are used heavily to group endpoint definitions, fixtures, and test suites within a file
- Two-space comment style for inline alignment: `self._ts_ms: int = 0   # VIDEO mode state`

---

## Import Organization

### Dart

**Order (observed pattern):**
1. `dart:*` SDK imports (`dart:convert`, `dart:io`, `dart:async`)
2. `package:flutter/*` framework
3. `package:*` third-party packages
4. Relative project imports (`'../models/...'`, `'../providers/...'`)

**Path style:**
- Relative paths only: `'../models/fitness_plan.dart'`, `'../config/api_config.dart'`
- No barrel (`index.dart`) files found — each import is explicit

### Python

**Order (observed pattern):**
1. `from __future__ import annotations` (always first when present)
2. Standard library (`json`, `logging`, `os`, `typing`)
3. Third-party packages (`fastapi`, `pydantic`, `numpy`, `openai`)
4. Internal project imports (`from backend.services.*`, `from backend.config.*`)

**Style:**
- Absolute package imports used throughout: `from backend.services.recommender import Recommender`
- No `__init__.py` re-exports for convenience — always import from the exact module

---

## Error Handling

### Dart

- All async API calls wrapped in `try/catch` inside screen methods
- Errors surfaced to users via `ScaffoldMessenger.of(context).showSnackBar(...)` or `_showError()` helper
- On API failure: throws `Exception('description: $statusCode $body')` from `ApiService`
- Screen-level error state stored in local `String?` fields (e.g., `_cameraError`)
- No global error boundary — errors bubble to the screen that made the request

Example from `lib/screens/chat_screen.dart`:
```dart
try {
  final resp = await ApiService.sendChat(...);
  // handle success
} catch (e) {
  _showError('Could not connect to server. Is the backend running?');
} finally {
  setState(() => _isLoading = false);
}
```

### Python

- FastAPI endpoints catch all service exceptions and re-raise as `HTTPException` with a descriptive `detail` field:
  ```python
  except Exception as exc:
      logger.exception("Recommendation engine failed")
      raise HTTPException(status_code=500, detail=f"Recommendation failed: {exc}")
  ```
- Service classes raise `ValueError` for schema/validation failures, `RuntimeError` for unexpected states
- LLM adapter uses a retry loop with specific exception types: `APITimeoutError`, `APIConnectionError` caught separately; `json.JSONDecodeError` caught and re-raised as `ValueError`
- Graceful degradation is a recurring pattern: services fall back to a template/default rather than failing hard (e.g., `LLMAdapter._build_template_plan()`, `FormAnalyzer` no-pose forward-fill)
- `logger.exception()` used (not `logger.error()`) when inside an `except` block so the traceback is always captured

---

## Logging

### Dart
- No structured logging framework; screens use `debugPrint()` or skip logging entirely for non-critical paths
- No `print()` calls found in `lib/` — consistent with `avoid_print` lint guidance

### Python
- `logging.getLogger(name)` at module top — two naming styles used:
  - Hierarchical: `"fitnova"` (app), `"fitnova.chat"`, `"fitnova.llm"` for core services
  - `__name__` for secondary modules: `form_analyzer.py`, `form_geometry.py`, `form_session.py`
- `logger.info()` for lifecycle events, `logger.warning()` for degraded states, `logger.exception()` inside `except` blocks

---

## Comments

### Dart
- Doc comments on public-facing widgets and helper functions using `///`
- Inline `//` comments explain non-obvious decisions (camera capture strategy, timer interval rationale)
- No JSDoc-style parameter documentation

Example from `lib/widgets/skeleton_painter.dart`:
```dart
/// Draws the 15-joint skeleton on top of the camera preview.
class SkeletonPainter extends CustomPainter {
  /// 15 joints × [x, y, z] in image-normalised coordinates (0-1).
  final List<List<double>>? landmarks;
```

### Python
- Module docstrings describe the module's role in the pipeline (e.g., `backend/app.py` lists all 4 layers)
- Inline comments explain ML constants, threshold rationale, and bug fixes with dates (e.g., `# B6 fix (2026-04-18): switched from VisionRunningMode.IMAGE to VIDEO`)
- Fix comments always include the bug label + date: `# B7 fix (2026-04-18): forward-fill on no-pose frames.`
- Section separator lines (`# ─────────`) used inside long files to delineate logical groups

---

## Function Design

### Dart
- Screen state methods are private (`_`) and purpose-named: `_initCamera()`, `_fetchGreeting()`, `_sendMessage()`
- Provider mutation methods are public and imperative: `setProfile()`, `setInjuries()`, `setPlan()`
- `dispose()` always overridden in StatefulWidgets that create controllers — consistently calls `super.dispose()` last
- Computed properties used for derived data: `bool get isReadyForPlan`, `Map get generatePlanPayload`

### Python
- Service methods are instance methods with `self`; pure helpers that don't need instance state are `@staticmethod`
  - Static: `LLMAdapter._sanitize_reps()`, `LLMAdapter._default_coaching_cue()`, `FormSession._resolve_exercise_idx()`
- `@lru_cache(maxsize=1)` used for expensive one-time loads (e.g., `backend/config/exercises.py` loads JSON once per process)
- Functions have docstrings on public methods listing `Args:` and `Returns:` with types
- Private helpers prefixed `_` and kept near their callers within the class

---

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

---

*Convention analysis: 2026-05-18*
