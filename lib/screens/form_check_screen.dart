import 'dart:async';
import 'dart:io';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:provider/provider.dart';

import '../models/form_models.dart';
import '../providers/form_session_provider.dart';
import '../services/form_session_service.dart';
import '../widgets/mismatch_banner.dart';
import '../widgets/skeleton_painter.dart';

class FormCheckScreen extends StatefulWidget {
  final String? exerciseHint;
  const FormCheckScreen({super.key, this.exerciseHint});

  @override
  State<FormCheckScreen> createState() => _FormCheckScreenState();
}

class _FormCheckScreenState extends State<FormCheckScreen> {
  CameraController? _cameraController;
  final _formService = FormSessionService();
  StreamSubscription<dynamic>? _wsSub;

  bool _cameraReady  = false;
  bool _sessionActive = false;
  String? _cameraError;

  // Periodic JPEG capture timer.
  // We do NOT use `startImageStream`: on Android it delivers YUV regardless of
  // the requested ImageFormatGroup, and the backend's cv2.imdecode silently
  // fails on YUV bytes (decode_error → empty UI). Polling `takePicture()`
  // returns real JPEG-encoded files that the backend can actually decode.
  //
  // ResolutionPreset.medium (480×640) — bumped up from low on 2026-05-11
  // because the .low preview looked pixelated when stretched to fullscreen
  // (BoxFit.cover). takePicture() latency is ~150 ms at medium vs ~70 ms at
  // low — still feasible, and the preview looks dramatically better.
  Timer? _frameTimer;
  bool _capturingFrame = false;
  // Interval dropped from 150ms to 100ms (target 10 fps) to make the
  // skeleton+metrics overlay feel less laggy. takePicture() is the
  // bottleneck — the timer fires but actual captures gate on
  // _capturingFrame and isTakingPicture so we never queue.
  static const _frameIntervalMs = 100;
  int _framesSent = 0;
  int _framesFailed = 0;
  bool _isFrontCamera = false;

  @override
  void initState() {
    super.initState();
    _initCamera();
  }

  @override
  void dispose() {
    _frameTimer?.cancel();
    _wsSub?.cancel();
    _formService.disconnect();
    _cameraController?.dispose();
    super.dispose();
  }

  // ── Camera init ────────────────────────────────────────────────────────────

  Future<void> _initCamera() async {
    final cameras = await availableCameras();
    if (cameras.isEmpty) {
      setState(() => _cameraError = 'No cameras available.');
      return;
    }

    // Prefer front camera for self-recording; fall back to first available
    final camera = cameras.firstWhere(
      (c) => c.lensDirection == CameraLensDirection.front,
      orElse: () => cameras.first,
    );

    final controller = CameraController(
      camera,
      ResolutionPreset.medium,    // 480×640 — better preview for fullscreen UI
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.jpeg,
    );

    try {
      await controller.initialize();
      if (mounted) {
        setState(() {
          _cameraController = controller;
          _cameraReady = true;
          _isFrontCamera =
              camera.lensDirection == CameraLensDirection.front;
        });
      }
    } catch (e) {
      if (mounted) setState(() => _cameraError = 'Camera error: $e');
    }
  }

  // ── Session control ────────────────────────────────────────────────────────

  void _startSession() {
    final provider = context.read<FormSessionProvider>();
    provider.startSession();

    final stream = _formService.connect(widget.exerciseHint);
    _wsSub = stream.listen(
      _onServerMessage,
      onError: (e) => provider.setError(e.toString()),
    );

    // Periodic JPEG capture via takePicture(). This is the reliable cross-
    // platform path — startImageStream delivers YUV which the backend can't
    // decode without metadata. 4 fps is plenty for our analysis cadence.
    _framesSent = 0;
    _framesFailed = 0;
    _frameTimer = Timer.periodic(
      const Duration(milliseconds: _frameIntervalMs),
      (_) => _captureAndSendFrame(),
    );
    setState(() => _sessionActive = true);
  }

  void _stopSession() {
    _frameTimer?.cancel();
    _frameTimer = null;
    context.read<FormSessionProvider>().setEnding();
    _formService.endSession();
    setState(() => _sessionActive = false);
    debugPrint('FormCheck: session stopped — sent=$_framesSent failed=$_framesFailed');
  }

  Future<void> _captureAndSendFrame() async {
    if (!_sessionActive) return;
    if (_cameraController == null || !_cameraController!.value.isInitialized) return;
    if (_capturingFrame) return; // skip if previous still pending
    if (_cameraController!.value.isTakingPicture) return;

    _capturingFrame = true;
    try {
      final xfile = await _cameraController!.takePicture();
      final bytes = await File(xfile.path).readAsBytes();
      _formService.sendFrame(bytes, DateTime.now().millisecondsSinceEpoch);
      _framesSent += 1;
      // Best-effort cleanup of the temp JPEG file
      try {
        await File(xfile.path).delete();
      } catch (_) {}
    } catch (e) {
      _framesFailed += 1;
      debugPrint('FormCheck: frame capture failed (#$_framesFailed): $e');
    } finally {
      _capturingFrame = false;
    }
  }

  void _onServerMessage(dynamic msg) {
    if (msg is! Map<String, dynamic>) return;
    final type = msg['type'] as String?;
    final provider = context.read<FormSessionProvider>();

    if (type == 'frame_result') {
      provider.updateFrame(FormFrameResult.fromJson(msg));
      // Parse mismatch warning if present
      if (msg.containsKey('mismatch_warning')) {
        final warning = msg['mismatch_warning'] as Map<String, dynamic>;
        provider.setMismatchWarning(
          selected: warning['selected'] as String? ?? '',
          predicted: warning['predicted'] as String? ?? '',
          confidence: (warning['confidence'] as num?)?.toDouble() ?? 0.0,
        );
      }
    } else if (type == 'session_summary') {
      provider.setSummary(FormSessionSummary.fromJson(msg));
      Navigator.of(context).pushReplacementNamed('/form-results');
    } else if (type == 'error') {
      provider.setError(msg['message'] as String? ?? 'Unknown error');
    }
  }

  // ── Build ──────────────────────────────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    final exerciseTitle = widget.exerciseHint != null
        ? widget.exerciseHint!
            .replaceAll('_', ' ')
            .split(' ')
            .map((w) => w.isEmpty ? w : '${w[0].toUpperCase()}${w.substring(1)}')
            .join(' ')
        : 'Form Check';

    return Scaffold(
      backgroundColor: Colors.black,
      // Full-screen camera: AppBar floats over the preview with transparent
      // background. extendBodyBehindAppBar lets the camera fill behind it.
      extendBodyBehindAppBar: true,
      appBar: AppBar(
        backgroundColor: Colors.black.withValues(alpha: 0.35),
        elevation: 0,
        foregroundColor: Colors.white,
        systemOverlayStyle: SystemUiOverlayStyle.light,
        title: Text(exerciseTitle, style: const TextStyle(fontSize: 16)),
      ),
      body: Stack(
        fit: StackFit.expand,
        children: [
          // ── Layer 1: full-screen camera preview (cover-fit) ────────────────
          _buildFullScreenCamera(),
          // ── Layer 2: skeleton overlay (matches camera coords) ──────────────
          _buildSkeletonOverlay(),
          // ── Layer 3: floating UI overlays ──────────────────────────────────
          // Top-center: BIG REP COUNTER (the headline metric)
          Positioned(
            top: kToolbarHeight + MediaQuery.of(context).padding.top + 12,
            left: 0,
            right: 0,
            child: Center(
              child: Consumer<FormSessionProvider>(
                builder: (ctx, provider, _) =>
                    _RepCounterPill(count: provider.liveRepCount),
              ),
            ),
          ),
          // Top-left: exercise label (small chip, doesn't compete with rep counter)
          Positioned(
            top: kToolbarHeight + MediaQuery.of(context).padding.top + 8,
            left: 12,
            child: Consumer<FormSessionProvider>(
              builder: (ctx, provider, _) => _LabelChip(
                text: provider.liveExercise.replaceAll('_', ' '),
              ),
            ),
          ),
          // Top-right: small phase indicator (STANDING / DESCENDING / ASCENDING)
          // Only shown when geometry has calibrated.
          Positioned(
            top: kToolbarHeight + MediaQuery.of(context).padding.top + 8,
            right: 12,
            child: Consumer<FormSessionProvider>(
              builder: (ctx, provider, _) {
                if (!provider.liveGeometricReady || provider.livePhase == null) {
                  return const SizedBox.shrink();
                }
                return _PhasePill(phase: provider.livePhase!);
              },
            ),
          ),
          // Top (below counter): mismatch warning banner
          Positioned(
            top: kToolbarHeight + MediaQuery.of(context).padding.top + 80,
            left: 12,
            right: 12,
            child: Consumer<FormSessionProvider>(
              builder: (ctx, provider, _) {
                if (!provider.hasMismatchWarning) {
                  return const SizedBox.shrink();
                }
                return MismatchBanner(
                  selected: provider.mismatchSelected,
                  predicted: provider.mismatchPredicted,
                  confidence: provider.mismatchConfidence,
                  onDismiss: () => provider.dismissMismatch(),
                );
              },
            ),
          ),
          // Bottom: floating controls + metrics
          Positioned(
            left: 0,
            right: 0,
            bottom: 0,
            child: SafeArea(
              top: false,
              child: _buildFloatingControls(),
            ),
          ),
        ],
      ),
    );
  }

  // ── Camera preview filling the entire screen (BoxFit.cover) ──────────────
  Widget _buildFullScreenCamera() {
    if (_cameraError != null) {
      return Center(
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Text(_cameraError!, style: const TextStyle(color: Colors.red)),
        ),
      );
    }

    if (!_cameraReady || _cameraController == null) {
      return const Center(child: CircularProgressIndicator(color: Colors.white));
    }

    final preview = _cameraController!;
    final size = preview.value.previewSize;
    if (size == null) {
      return const Center(child: CircularProgressIndicator(color: Colors.white));
    }

    // The camera plugin reports previewSize in landscape (sensor) orientation.
    // For portrait UI we swap width/height. FittedBox(fit: cover) then scales
    // the natural-aspect preview to FILL the screen (cropping sides if the
    // camera is more square than the screen). This is the standard pattern
    // for full-screen camera preview in Flutter.
    return ClipRect(
      child: SizedBox.expand(
        child: FittedBox(
          fit: BoxFit.cover,
          child: SizedBox(
            width: size.height,
            height: size.width,
            child: CameraPreview(preview),
          ),
        ),
      ),
    );
  }

  // ── Skeleton overlay aligned to the same FittedBox crop ──────────────────
  Widget _buildSkeletonOverlay() {
    if (!_cameraReady || _cameraController == null) {
      return const SizedBox.shrink();
    }
    final preview = _cameraController!;
    final size = preview.value.previewSize;
    if (size == null) return const SizedBox.shrink();

    return Consumer<FormSessionProvider>(
      builder: (context, provider, _) {
        final jointErrorMap = buildJointErrorMap(provider.liveJointErrors);
        return ClipRect(
          child: SizedBox.expand(
            child: FittedBox(
              fit: BoxFit.cover,
              child: SizedBox(
                width: size.height,
                height: size.width,
                child: provider.liveLandmarks == null
                    ? const SizedBox.shrink()
                    : CustomPaint(
                        painter: SkeletonPainter(
                          landmarks: provider.liveLandmarks,
                          jointErrors: jointErrorMap,
                          mirror: _isFrontCamera,
                        ),
                      ),
              ),
            ),
          ),
        );
      },
    );
  }

  // ── Floating bottom controls: active flags + quality LABEL + Start/Stop ──
  // No percentages. Active flags are shown as severity-coloured chips
  // (e.g. "Small Knee Caving"). The quality scalar collapses to a single
  // categorical label (e.g. "Good Form"). All visual signals come from
  // the backend's geometric rule layer.
  Widget _buildFloatingControls() {
    return Consumer<FormSessionProvider>(
      builder: (context, provider, _) {
        final qualityLabel = provider.liveQualityLabel
            ?? _qualityFallbackLabel(provider.liveQuality);
        final qualityColor = _qualityColor(provider.liveQuality);
        final flags = provider.liveActiveFlags;

        return Container(
          decoration: BoxDecoration(
            gradient: LinearGradient(
              begin: Alignment.topCenter,
              end: Alignment.bottomCenter,
              colors: [
                Colors.transparent,
                Colors.black.withValues(alpha: 0.85),
              ],
            ),
          ),
          padding: const EdgeInsets.fromLTRB(20, 28, 20, 22),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              // Active red-flag chips (categorical labels). Empty when form
              // is clean. Scrollable horizontally if many flags fire at once.
              if (flags.isNotEmpty)
                SizedBox(
                  height: 32,
                  child: ListView.separated(
                    scrollDirection: Axis.horizontal,
                    itemCount: flags.length,
                    separatorBuilder: (_, __) => const SizedBox(width: 6),
                    itemBuilder: (_, i) => _FlagChip(flag: flags[i]),
                  ),
                ),
              if (flags.isNotEmpty) const SizedBox(height: 10),
              // Quality label — single line, single colour, no number
              Center(
                child: Container(
                  padding: const EdgeInsets.symmetric(
                      horizontal: 14, vertical: 6),
                  decoration: BoxDecoration(
                    color: qualityColor.withValues(alpha: 0.2),
                    borderRadius: BorderRadius.circular(20),
                    border: Border.all(color: qualityColor, width: 1.5),
                  ),
                  child: Text(
                    qualityLabel,
                    style: TextStyle(
                      color: qualityColor,
                      fontWeight: FontWeight.bold,
                      fontSize: 15,
                    ),
                  ),
                ),
              ),
              const SizedBox(height: 16),
              // Big start/stop button (centered)
              Center(
                child: _StartStopButton(
                  isActive: _sessionActive,
                  onStart: _startSession,
                  onStop: _stopSession,
                ),
              ),
            ],
          ),
        );
      },
    );
  }

  // Fallback when backend doesn't send a quality_label (e.g. v6 only, no
  // geometric layer for the selected exercise). Same buckets as backend.
  static String _qualityFallbackLabel(double q) {
    if (q >= 0.85) return 'Excellent Form';
    if (q >= 0.70) return 'Good Form';
    if (q >= 0.50) return 'Fair Form';
    if (q >= 0.30) return 'Needs Work';
    return 'Bad Form';
  }

  static Color _qualityColor(double q) {
    if (q >= 0.70) return Colors.greenAccent.shade400;
    if (q >= 0.40) return Colors.amberAccent.shade400;
    return Colors.redAccent.shade400;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper widgets
// ─────────────────────────────────────────────────────────────────────────────

class _LabelChip extends StatelessWidget {
  final String text;
  const _LabelChip({required this.text});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
      decoration: BoxDecoration(
        color: Colors.black.withValues(alpha: 0.55),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Text(
        text,
        style: const TextStyle(color: Colors.white, fontSize: 13),
      ),
    );
  }
}

class _StartStopButton extends StatelessWidget {
  final bool isActive;
  final VoidCallback onStart;
  final VoidCallback onStop;
  const _StartStopButton({
    required this.isActive,
    required this.onStart,
    required this.onStop,
  });

  @override
  Widget build(BuildContext context) {
    final bgColor = isActive ? Colors.redAccent.shade400 : Colors.greenAccent.shade400;
    return Material(
      color: Colors.transparent,
      child: InkWell(
        borderRadius: BorderRadius.circular(40),
        onTap: isActive ? onStop : onStart,
        child: Container(
          width: 76,
          height: 76,
          decoration: BoxDecoration(
            shape: BoxShape.circle,
            color: bgColor,
            boxShadow: [
              BoxShadow(
                color: bgColor.withValues(alpha: 0.4),
                blurRadius: 12,
                spreadRadius: 2,
              ),
            ],
            border: Border.all(color: Colors.white, width: 3),
          ),
          child: Icon(
            isActive ? Icons.stop : Icons.play_arrow,
            color: Colors.white,
            size: 38,
          ),
        ),
      ),
    );
  }
}


/// Big top-of-screen rep counter pill. The headline number — visible from
/// across the room. Uses a monospace digit slot so the number doesn't
/// wobble when transitioning from 9 → 10.
class _RepCounterPill extends StatelessWidget {
  final int count;
  const _RepCounterPill({required this.count});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 22, vertical: 8),
      decoration: BoxDecoration(
        color: Colors.black.withValues(alpha: 0.6),
        borderRadius: BorderRadius.circular(28),
        border: Border.all(
            color: Colors.white.withValues(alpha: 0.25), width: 1),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Text(
            '$count',
            style: const TextStyle(
              color: Colors.white,
              fontSize: 38,
              fontWeight: FontWeight.w900,
              fontFeatures: [FontFeature.tabularFigures()],
              height: 1.0,
            ),
          ),
          const SizedBox(width: 8),
          const Padding(
            padding: EdgeInsets.only(bottom: 4),
            child: Text(
              'REPS',
              style: TextStyle(
                color: Colors.white70,
                fontSize: 12,
                fontWeight: FontWeight.w700,
                letterSpacing: 2,
              ),
            ),
          ),
        ],
      ),
    );
  }
}


/// Small status pill showing the current rep phase from the geometric layer.
class _PhasePill extends StatelessWidget {
  final String phase;
  const _PhasePill({required this.phase});

  @override
  Widget build(BuildContext context) {
    final color = switch (phase) {
      'DESCENDING' => Colors.orangeAccent,
      'ASCENDING'  => Colors.lightBlueAccent,
      _            => Colors.white70,
    };
    // Friendly label
    final label = switch (phase) {
      'STANDING'   => 'READY',
      'DESCENDING' => 'GOING DOWN',
      'ASCENDING'  => 'COMING UP',
      _            => phase,
    };
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
      decoration: BoxDecoration(
        color: Colors.black.withValues(alpha: 0.55),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: color.withValues(alpha: 0.5)),
      ),
      child: Text(
        label,
        style: TextStyle(
            color: color,
            fontSize: 11,
            fontWeight: FontWeight.bold,
            letterSpacing: 1),
      ),
    );
  }
}


/// Severity-coloured pill that shows a single active form flag's label
/// (e.g. "Small Knee Caving"). Sized for horizontal scroll layout.
class _FlagChip extends StatelessWidget {
  final ActiveFlag flag;
  const _FlagChip({required this.flag});

  @override
  Widget build(BuildContext context) {
    // Three-tier colour: < 0.55 amber, < 0.85 orange, >= 0.85 red
    final color = flag.severity >= 0.85
        ? Colors.redAccent.shade400
        : flag.severity >= 0.55
            ? Colors.orangeAccent.shade400
            : Colors.amberAccent.shade400;
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.22),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: color, width: 1.5),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(Icons.warning_amber_rounded, color: color, size: 14),
          const SizedBox(width: 5),
          Text(
            flag.label,
            style: TextStyle(
              color: color,
              fontSize: 12,
              fontWeight: FontWeight.bold,
            ),
          ),
        ],
      ),
    );
  }
}
