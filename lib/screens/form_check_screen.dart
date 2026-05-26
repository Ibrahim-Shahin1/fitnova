import 'dart:async';
import 'dart:io';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:provider/provider.dart';

import '../models/form_models.dart';
import '../providers/form_session_provider.dart';
import '../services/form_session_service.dart';

/// Live squat form analysis (D-05 backend).
///
/// The PyTorch backend produces NO pose/skeleton/quality — only periodic KIE/KFE
/// detections. The server fires a `rep_result` on a sliding-window clock (~every
/// few seconds once enough frames buffer), NOT once per actual rep — so these are
/// surfaced as periodic "form checks", not a rep counter (honesty: the model does
/// not detect rep boundaries live).
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

  bool _cameraReady = false;
  bool _sessionActive = false;
  String? _cameraError;

  // Periodic JPEG capture via takePicture() (NOT startImageStream — that delivers
  // YUV on Android which the backend's cv2.imdecode can't decode).
  Timer? _frameTimer;
  bool _capturingFrame = false;
  static const _frameIntervalMs = 100; // ~10 fps capture
  int _framesSent = 0;
  int _framesFailed = 0;

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

    // Prefer front camera for self-recording; fall back to first available.
    final camera = cameras.firstWhere(
      (c) => c.lensDirection == CameraLensDirection.front,
      orElse: () => cameras.first,
    );

    final controller = CameraController(
      camera,
      ResolutionPreset.medium,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.jpeg,
    );

    try {
      await controller.initialize();
      if (mounted) {
        setState(() {
          _cameraController = controller;
          _cameraReady = true;
        });
      }
    } catch (e) {
      if (mounted) setState(() => _cameraError = 'Camera error: $e');
    }
  }

  // ── Session control ──────────────────────────────────────────────────────────

  void _startSession() {
    final provider = context.read<FormSessionProvider>();
    provider.startSession();

    final stream = _formService.connect(widget.exerciseHint);
    _wsSub = stream.listen(
      _onServerMessage,
      onError: (e) => provider.setError(e.toString()),
    );

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
    if (_capturingFrame) return;
    if (_cameraController!.value.isTakingPicture) return;

    _capturingFrame = true;
    try {
      final xfile = await _cameraController!.takePicture();
      final bytes = await File(xfile.path).readAsBytes();
      _formService.sendFrame(bytes, DateTime.now().millisecondsSinceEpoch);
      _framesSent += 1;
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

    switch (type) {
      case 'rep_result':
        // A periodic form check fired (NOT a per-rep boundary).
        provider.addLiveRep(FormRep.fromJson(msg));
        break;
      case 'session_summary':
        provider.setReport(FormReport.fromSession(msg));
        Navigator.of(context).pushReplacementNamed('/form-results');
        break;
      case 'error':
        provider.setError(msg['message'] as String? ?? 'Unknown error');
        break;
      // 'session_started' and any unknown types are ignored.
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
          _buildFullScreenCamera(),
          // Top-center: form-check counter (periodic checks, NOT reps).
          Positioned(
            top: kToolbarHeight + MediaQuery.of(context).padding.top + 12,
            left: 0,
            right: 0,
            child: Center(
              child: Consumer<FormSessionProvider>(
                builder: (ctx, provider, _) =>
                    _FormChecksPill(count: provider.formCheckCount),
              ),
            ),
          ),
          // Below counter: latest form-check result banner.
          Positioned(
            top: kToolbarHeight + MediaQuery.of(context).padding.top + 72,
            left: 12,
            right: 12,
            child: Consumer<FormSessionProvider>(
              builder: (ctx, provider, _) => _LatestCheckBanner(
                rep: provider.lastRep,
                active: _sessionActive,
              ),
            ),
          ),
          // Bottom: floating controls.
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

  Widget _buildFloatingControls() {
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
          Text(
            _sessionActive
                ? 'Checking your form every few seconds…'
                : 'Stand side-on, full body in frame, then start.',
            textAlign: TextAlign.center,
            style: const TextStyle(color: Colors.white70, fontSize: 13),
          ),
          const SizedBox(height: 14),
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
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper widgets
// ─────────────────────────────────────────────────────────────────────────────

/// Top pill: number of periodic form checks completed this session.
/// Labelled "CHECKS" (not "REPS") — the backend fires on a clock, not per rep.
class _FormChecksPill extends StatelessWidget {
  final int count;
  const _FormChecksPill({required this.count});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 22, vertical: 8),
      decoration: BoxDecoration(
        color: Colors.black.withValues(alpha: 0.6),
        borderRadius: BorderRadius.circular(28),
        border: Border.all(color: Colors.white.withValues(alpha: 0.25), width: 1),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Text(
            '$count',
            style: const TextStyle(
              color: Colors.white,
              fontSize: 34,
              fontWeight: FontWeight.w900,
              fontFeatures: [FontFeature.tabularFigures()],
              height: 1.0,
            ),
          ),
          const SizedBox(width: 8),
          const Padding(
            padding: EdgeInsets.only(bottom: 4),
            child: Text(
              'CHECKS',
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

/// Banner showing the most recent form check's KIE/KFE result.
class _LatestCheckBanner extends StatelessWidget {
  final FormRep? rep;
  final bool active;
  const _LatestCheckBanner({required this.rep, required this.active});

  @override
  Widget build(BuildContext context) {
    if (!active && rep == null) return const SizedBox.shrink();

    if (rep == null) {
      return _wrap(
        color: Colors.white24,
        child: const Text(
          'Analyzing your form…',
          style: TextStyle(color: Colors.white, fontWeight: FontWeight.w600),
        ),
      );
    }

    final detected = rep!.detectedErrors;
    if (detected.isEmpty) {
      return _wrap(
        color: Colors.greenAccent.shade400,
        child: const Text(
          'Form looks clean',
          style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold),
        ),
      );
    }

    return _wrap(
      color: Colors.redAccent.shade400,
      child: Wrap(
        spacing: 6,
        runSpacing: 4,
        alignment: WrapAlignment.center,
        children: detected.map((e) => _ErrorChip(error: e)).toList(),
      ),
    );
  }

  Widget _wrap({required Color color, required Widget child}) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
      decoration: BoxDecoration(
        color: Colors.black.withValues(alpha: 0.6),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: color, width: 1.5),
      ),
      child: Center(child: child),
    );
  }
}

/// Severity-coloured chip for a single detected error (no percentages — D-05).
class _ErrorChip extends StatelessWidget {
  final FormError error;
  const _ErrorChip({required this.error});

  @override
  Widget build(BuildContext context) {
    final color = switch (error.severityWord) {
      'strong' => Colors.redAccent.shade400,
      'moderate' => Colors.orangeAccent.shade400,
      _ => Colors.amberAccent.shade400,
    };
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
            '${error.label} (${error.severityWord})',
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
    final bgColor =
        isActive ? Colors.redAccent.shade400 : Colors.greenAccent.shade400;
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
