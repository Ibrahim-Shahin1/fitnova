# Testing Patterns

**Analysis Date:** 2026-05-18

---

## Two Test Suites

FitNova maintains two independent test suites that are completely separate:

| Side | Framework | Location(s) |
|------|-----------|-------------|
| Flutter / Dart | `flutter_test` | `test/` |
| Python / Backend | `pytest` | `backend/tests/`, `backend/services/test_*.py`, `backend/config/test_*.py`, `backend/training/preprocessing/test_*.py` |

---

## Python Backend Tests

### Test Framework

**Runner:**
- `pytest >= 8.0.0` (declared in `backend/requirements.txt`)
- Evidence of use: `.pytest_cache/` directory exists at repo root
- No `pytest.ini`, `pyproject.toml`, or `setup.cfg` found — pytest runs with default discovery settings

**Run Commands:**
```bash
# Run all backend tests from repo root
pytest backend/

# Run a specific test file
pytest backend/tests/test_app.py -v

# Run co-located service tests
pytest backend/services/test_form_geometry.py -v

# Run with output
pytest backend/tests/ -v --tb=short

# Run the noise-budget script (not a pytest file — run directly)
python -m backend.tests.check_noise_budget
```

**Assertion style:**
- Plain `assert` statements throughout — no assertion library
- Failure messages use f-strings: `assert counts["Yoga"] == N_PER_TYPE, f"Expected {N_PER_TYPE} Yoga, got {counts['Yoga']}"`

### Test File Organization

**Locations — three distinct patterns:**

1. **`backend/tests/`** — endpoint and service integration tests
   - `test_app.py` — FastAPI endpoint tests
   - `test_chat.py` — `/chat` endpoint tests
   - `test_recommender.py` — Recommender pipeline
   - `test_content_filter.py` — ContentBasedFilter
   - `test_llm_adapter.py` — LLMAdapter (largest test file, 400+ lines)
   - `test_augmentation.py` — catalog augmentation
   - `test_angular_features_rotation_invariance.py` — ML invariance GA gate
   - `test_workout_taxonomy.py` — taxonomy classification
   - `test_exercise_sanity.py` — ExerciseMismatchDetector
   - `test_exercises_consistency.py` — exercises SSOT vs trained labels
   - `check_noise_budget.py` — noise budget audit script (not pytest-discoverable; run with `python -m`)
   - `validate_user_squats.py` — manual validation script

2. **Co-located in `backend/services/`** — module-adjacent tests
   - `test_form_geometry.py` — GeometricFormValidator, RepStateMachine (90+ test functions)
   - `test_form_session_v6_routing.py` — FormSession exercise-idx routing
   - `test_form_analyzer_v6.py` — FormAnalyzer v6 smoke tests

3. **Co-located in `backend/config/`** and `backend/training/preprocessing/`**
   - `backend/config/test_defect_to_region.py` — defect-to-region mapping
   - `backend/training/preprocessing/test_qevd_label_builder.py`
   - `backend/training/preprocessing/test_qevd_label_builder_v2.py`
   - `backend/training/preprocessing/test_qevd_class_space.py`
   - `backend/training/preprocessing/test_qevd_dataset.py`
   - `backend/training/preprocessing/test_qevd_dataset_v6_1.py`
   - `backend/training/preprocessing/test_qevd_extractor.py`

**Naming:**
- All test files: `test_<subject>.py`
- All test functions: `def test_<what_it_checks>()`
- Helper factories: `def _<name>()` — underscore-prefixed, not discovered by pytest

**Total test function count:** ~389 across the backend

### Test Structure

**Module-level docstring (present on most test files):**
```python
"""
Tests for backend/app.py — FastAPI endpoints.
All heavy services (Recommender, LLMAdapter) are mocked.
"""
```

**Section separators inside large test files:**
```python
# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────
```

**Standard test function signature:**
```python
def test_<behavior>(client, mock_services):
    # arrange
    # act
    result = some_function(...)
    # assert
    assert result["field"] == expected_value
```

**Parametrize pattern (used widely):**
```python
@pytest.mark.parametrize("variation,expected", [
    ("knees over toes",   {REGION_KNEES}),
    ("back not straight", {REGION_TRUNK}),
    ("elbows flared",     {REGION_ELBOWS}),
])
def test_named_body_part_maps_to_region(variation, expected):
    assert regions_for_defect(variation) == expected
```

**Skip pattern for live API/dataset-dependent tests:**
```python
@pytest.mark.skipif(not _has_api_key(), reason="OPENAI_API_KEY not configured")
def test_live_llm_call():
    ...

@pytest.mark.skipif(
    not os.path.exists(QEVD_DIR),
    reason="QEVD dataset not available on this machine"
)
def test_qevd_extractor_real_data():
    ...
```

### Mocking

**Primary tool:** `unittest.mock.MagicMock` — no third-party mock library

**Pattern 1 — `autouse=True` fixture injecting into `app.state`:**
```python
@pytest.fixture(autouse=True)
def mock_services():
    mock_rec = MagicMock()
    mock_rec.recommend.return_value = {
        "program_id": 42,
        "content_candidates": list(range(50)),
        ...
    }
    mock_llm = MagicMock()
    mock_llm.generate_plan.return_value = { ... }
    app.state.recommender = mock_rec
    app.state.llm_adapter = mock_llm
    yield mock_rec, mock_llm
```
Used in: `backend/tests/test_app.py`, `backend/tests/test_chat.py`

**Pattern 2 — Hand-rolled fake class:**
```python
class FakeClient:
    def __init__(self, payload=None, error=None):
        self.payload = payload
        self.error = error
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        if self.error:
            raise self.error
        return _make_response(self.payload)
```
Used in: `backend/tests/test_llm_adapter.py` — injects `FakeClient` into `LLMAdapter(client=...)`

**Pattern 3 — `monkeypatch` (pytest built-in):**
```python
def _make_adapter(monkeypatch, client=None):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    return LLMAdapter(client=client or FakeClient(payload=_valid_plan_payload()))
```
Used in: `backend/tests/test_llm_adapter.py` — also patches env vars

**Pattern 4 — `patch()` for module-level patching:**
```python
from unittest.mock import patch
# Used selectively in test_chat.py for OpenAI response construction
```

**Pattern 5 — `tempfile.TemporaryDirectory()` for file-system isolation:**
```python
@pytest.fixture
def tmp_model_dir():
    with tempfile.TemporaryDirectory() as d:
        labels = {"squat": 0, "deadlift": 1}
        with open(os.path.join(d, "exercise_labels.json"), "w") as f:
            json.dump(labels, f)
        # Reset class-level singleton state between tests
        ExerciseMismatchDetector._class_index = None
        ExerciseMismatchDetector._neighbours  = None
        ExerciseMismatchDetector._mode        = "uninitialised"
        yield d
```
Used in: `backend/tests/test_exercise_sanity.py`

**What to mock:**
- All external I/O at the unit test boundary: OpenAI API calls, heavy ML model loading
- `app.state` services (inject mocks before each test via `autouse=True` fixture)
- Environment variables via `monkeypatch.setenv()`
- File system via `tempfile.TemporaryDirectory()` when testing file-dependent classes

**What NOT to mock:**
- Pure computation (angular feature math, taxonomy inference, reps sanitization) — tested against real inputs
- Pydantic validation — tested via `TestClient` with real FastAPI request routing
- In-process data transformations

### Fixtures and Factories

**Shared request bodies defined at module top:**
```python
VALID_BODY = {
    "experience_level": 1,
    "workout_type": "Strength",
    "session_duration_hours": 1.0,
    "workout_frequency": 3,
}
```

**Helper factories return realistic test data:**
```python
def _sample_plan(workout_days=(1, 3, 5)):
    plan = {}
    for d in range(1, 8):
        key = f"day_{d}"
        if d in workout_days:
            plan[key] = { "day_number": d, "is_rest_day": False, "exercises": [...] }
        else:
            plan[key] = { "day_number": d, "is_rest_day": True, "exercises": [] }
    return plan
```

**Synthetic skeleton helpers in geometry tests:**
```python
def make_standing_lms(hip_y=0.50, trunk_lean_deg=0.0, knee_x_offset=0.0,
                       hip_minus_knee=-0.20) -> np.ndarray:
    """Return a (15, 3) array representing a person standing."""
    ...
```
Used in: `backend/services/test_form_geometry.py` — 50+ tests all build landmark arrays via this factory

**FastAPI test client fixture:**
```python
@pytest.fixture
def client():
    return TestClient(app, raise_server_exceptions=False)
```
`raise_server_exceptions=False` lets HTTP 500 responses be asserted rather than re-raising

### Test Types

**Unit tests:**
- Pure function coverage: `_sanitize_reps()`, `_default_coaching_cue()`, `_sanitize_title()`, `regions_for_defect()`, `infer_program_type_profile()`
- Class method testing with synthetic data: `RepStateMachine`, `GeometricFormValidator`, `ExerciseMismatchDetector`
- Invariance / property-based tests: `test_all_22_features_invariant_under_y_rotation` tests 20 subjects × 3 frames × 4 rotation angles

**Integration tests:**
- FastAPI endpoint tests via `TestClient` (real routing, Pydantic validation, HTTP status codes)
- Service pipeline tests: `test_recommender_returns_sorted_rankings_and_valid_ids()` instantiates real `Recommender` and checks the full Layer 1+2 output shape
- Data consistency tests: `test_exercises_consistency.py` cross-checks `exercises.json` SSOT against the trained label file

**Live/external tests (skipped in CI if missing):**
- `test_live_llm_call()` — requires `OPENAI_API_KEY` in environment
- QEVD dataset tests — require local dataset directory to exist

**Utility scripts (not pytest-discovered):**
- `backend/tests/check_noise_budget.py` — runs as `python -m backend.tests.check_noise_budget`
- `backend/tests/validate_user_squats.py` — manual validation, not a test suite

### Coverage

**Requirements:** None enforced (no `.coveragerc`, no `--cov` in any config file)

**View coverage manually:**
```bash
pip install pytest-cov
pytest backend/ --cov=backend --cov-report=term-missing
```

**Notably well-covered:**
- `LLMAdapter` — 30+ test functions covering every fallback branch
- `GeometricFormValidator` / `RepStateMachine` — 50+ tests including edge cases (half-reps, shallow bobbing, double-count prevention)
- `ExerciseMismatchDetector` — 12 tests covering smart/degraded/disabled modes and v5.2/v6 label file variants

**Potentially under-covered:**
- `FormSession.add_frame()` and `end_session()` — complex sliding-window logic; no test file found in `backend/tests/` for the full session lifecycle (only routing tests in `backend/services/test_form_session_v6_routing.py`)
- `backend/services/form_analyzer.py` — only v6-specific smoke tests in `backend/services/test_form_analyzer_v6.py`

---

## Flutter / Dart Tests

### Test Framework

**Runner:**
- `flutter_test` (Flutter SDK built-in)
- `flutter_lints ^5.0.0` for static analysis

**Run Commands:**
```bash
flutter test                          # Run all tests
flutter test test/widget_test.dart    # Run specific file
flutter analyze                       # Static analysis (dart analyze)
```

### Test File Organization

**Location:**
- `test/` directory at repo root — single file: `test/widget_test.dart`
- No co-located test files inside `lib/`

**Current coverage:** One widget test covering app launch and splash screen rendering.

### Test Structure

**Widget test pattern:**
```dart
void main() {
  testWidgets('App launches and shows splash screen', (WidgetTester tester) async {
    // Arrange: mock SharedPreferences, init ThemeController
    SharedPreferences.setMockInitialValues(<String, Object>{});
    final themeController = ThemeController();
    await themeController.init();

    // Act: pump the full widget tree with MultiProvider
    await tester.pumpWidget(
      MultiProvider(
        providers: [
          ChangeNotifierProvider.value(value: themeController),
          ChangeNotifierProvider(create: (_) => UserProvider()),
          ChangeNotifierProvider(create: (_) => FormSessionProvider()),
        ],
        child: const FitNovaApp(),
      ),
    );

    // Assert: verify UI elements
    expect(find.image(const AssetImage('assets/logo/wordmark.png')), findsOneWidget);
    expect(find.text('AI-Powered Fitness Planning'), findsOneWidget);

    // Drain timers to avoid "pending Timer" warnings
    await tester.pumpAndSettle(const Duration(seconds: 2));
  });
}
```

**Key patterns:**
- `SharedPreferences.setMockInitialValues({})` to stub persistent storage
- Full `MultiProvider` tree constructed in test — same providers as `main.dart`
- `pumpAndSettle(Duration)` used to flush scheduled timers (e.g., 1500 ms splash navigation delay)
- `find.image()` and `find.text()` finders used for assertions

### Mocking (Dart)

**Used:**
- `shared_preferences` test mock via `SharedPreferences.setMockInitialValues()`
- No `mockito` or `mocktail` packages in `pubspec.yaml`

**Not mocked:**
- `ApiService` — all HTTP calls would fail in a widget test environment; no test currently exercises the network layer

### Coverage

**Requirements:** None enforced

**Current state:** The single widget test only covers the splash screen. All screens making real API calls (`ChatScreen`, `FormCheckScreen`, `PlanScreen`) have no widget tests. This is the primary Flutter test coverage gap.

---

## CI / CD

**No CI configuration found** (no `.github/`, `Makefile`, `Jenkinsfile`, or similar). Tests are run manually from the developer's machine.

**Recommended manual test run (full suite):**
```bash
# Python
pytest backend/ -v

# Flutter
flutter test
flutter analyze
```

---

*Testing analysis: 2026-05-18*
