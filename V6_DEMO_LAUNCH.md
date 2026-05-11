# v6 Demo Launch — BlueStacks + Flutter

How to run the FitNova app with the **v6 (QEVD-trained) model** and judge the
form-correction signal end-to-end. The model is the one trained 2026-05-09 in
the D6 Colab notebook; it failed the D9 reality-check (quality_gap = +0.004
vs the +0.15 needed) but is structurally sound. Run this demo to feel how
broken the user-facing UX is, before committing to the Tier 2 GPT-4o-mini
relabel + retrain recovery.

---

## Pre-flight checklist (do these once)

1. **v6 weights local.** Already verified at:
   `backend/models/form_model_v6/v6_supervised.weights.h5`

2. **Backend dependencies.** From repo root:
   ```powershell
   pip install -r backend/requirements.txt
   ```

3. **Flutter SDK + plugins.** From repo root:
   ```powershell
   flutter pub get
   ```

4. **BlueStacks installed.** Should already be set up per your earlier notes.

---

## Two terminals — start in this order

### Terminal 1 — backend (always first)

```powershell
cd "C:\Users\tsh_x\Desktop\FitNova Application"
$env:FITNOVA_DEBUG = "1"        # turn on per-frame v6 prediction logging
python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

**What you should see in the boot log:**
```
INFO ... v6 weights detected -> using ...\backend\models\form_model_v6
INFO ... v6 ST-GCN weights loaded (4.5 MB, 1,083,385 params).
INFO ... v6 QEVD exercise map loaded: 25 classes (includes '__other__'=24).
INFO ... FitNova backend ready.
```

If you see `v5.2 weights detected` or `falling back to v4`, the v6 weights
aren't being picked up — check that
`backend/models/form_model_v6/v6_supervised.weights.h5` exists.

`--host 0.0.0.0` binds to all interfaces so BlueStacks can reach it. The
backend logs will print `v6 input: ...` and `v6 output: ...` lines for every
window-inference (~1 per second). That's the per-frame diagnostic.

### Terminal 2 — Flutter on BlueStacks

```powershell
cd "C:\Users\tsh_x\Desktop\FitNova Application"
flutter devices    # confirm BlueStacks is listed (e.g. "Bluestacks emulator-5554")
flutter run        # pick the BlueStacks device when prompted
```

The app will hot-deploy to the BlueStacks instance. The default `api_config.dart`
maps Android → `http://10.0.2.2:8000` and `ws://10.0.2.2:8000`, which works for
standard Android emulators AND most BlueStacks configurations.

**If the app can't reach the backend** (red error banner / WebSocket refused):

```powershell
ipconfig
# Look for IPv4 Address on your active adapter (Wi-Fi or Ethernet)
# Example: 192.168.1.42
```

Then edit `lib/config/api_config.dart`:
```dart
if (Platform.isAndroid) return 'http://192.168.1.42:8000';   // your IP
// and same for the ws line
```

Hot-restart the Flutter app (capital R in the `flutter run` terminal).

---

## What to do in the app

1. From the home screen, navigate to the workout / form-tracking screen.
2. Pick **Squat** as the exercise.
3. Stand in front of your webcam (BlueStacks should pipe webcam to the app's
   camera permission). Do a few squats — both good form and intentionally bad
   form (e.g. knees caving in, rounded back).
4. Watch the in-app feedback:
   - Live skeleton overlay (driven by MediaPipe extraction in the backend)
   - Quality bar / score
   - Joint-error heat highlights (knee, hip, shoulder, etc)
   - Rep counter
   - End-of-session coaching summary (if OPENAI_API_KEY is set in the env)

---

## Honest expectations from the D9 numbers

| What you'll see | Why (tied to D9 results) |
|------------------|-------------------------|
| Squat detection works (top-1 = "squats") | Action head learned this fine |
| Quality score ~0.65-0.75 regardless of form | quality_gap = +0.004 → no good/bad discrimination |
| Knee/hip warnings firing constantly | knee_hip_max ~0.99 across all clips → over-active |
| Rep counter doesn't tick | boundary head is dead (predicts ~0 always) |
| Hip warnings sometimes spike on clearly bad reps | joint_err head DID learn something; the gap is just too small |

**This is the failure scenario, not a bug in the integration.** The model is
structurally working — every head fires, the WebSocket payload is correctly
shaped, the Flutter UI receives all 10 joint-group channels. The model itself
needs better labels (Tier 2 GPT-4o-mini relabel) before another retrain.

---

## After your judgment call

Tell me what you actually saw versus what you expected. Specifically:
- Does the live skeleton overlay match your body? (validates MediaPipe path)
- Does pre-recorded form-good vs form-bad behave differently AT ALL? (validates
  whether ANY signal exists, even if too small for the gates)
- Is the UX broken in ways you didn't expect from the metrics? (we may have
  missed a failure mode)

Then we either:
- Proceed to Tier 2 (GPT-4o-mini relabel + retrain) — my current recommendation
- Abandon the QEVD approach and pivot — only if you saw something dramatically
  worse than the metrics predicted

---

## Cleanup (when done)

In the backend terminal, Ctrl+C. Flutter terminal, q. Don't change `FITNOVA_DEBUG`
back unless you're committing — the debug logs are noise but harmless.
