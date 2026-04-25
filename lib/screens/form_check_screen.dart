import 'dart:async';
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
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

  // Frame throttle (~10fps)
  DateTime _lastFrameSent = DateTime.fromMillisecondsSinceEpoch(0);
  static const _frameIntervalMs = 100; // 10 fps

  @override
  void initState() {
    super.initState();
    _initCamera();
  }

  @override
  void dispose() {
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
      ResolutionPreset.medium,   // 640×480 — good quality/bandwidth balance
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

  // ── Session control ────────────────────────────────────────────────────────

  void _startSession() {
    final provider = context.read<FormSessionProvider>();
    provider.startSession();

    final stream = _formService.connect(widget.exerciseHint);
    _wsSub = stream.listen(
      _onServerMessage,
      onError: (e) => provider.setError(e.toString()),
    );

    // Start sending camera frames
    _cameraController?.startImageStream(_onCameraFrame);
    setState(() => _sessionActive = true);
  }

  void _stopSession() {
    _cameraController?.stopImageStream();
    context.read<FormSessionProvider>().setEnding();
    _formService.endSession();
    setState(() => _sessionActive = false);
  }

  void _onCameraFrame(CameraImage image) {
    final now = DateTime.now();
    if (now.difference(_lastFrameSent).inMilliseconds < _frameIntervalMs) return;
    _lastFrameSent = now;

    // CameraImage with JPEG format → send plane[0] bytes directly
    final jpegBytes = Uint8List.fromList(image.planes[0].bytes);
    _formService.sendFrame(jpegBytes, now.millisecondsSinceEpoch);
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
    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        backgroundColor: Colors.black,
        foregroundColor: Colors.white,
        title: Text(
          widget.exerciseHint != null
              ? widget.exerciseHint!
                  .replaceAll('_', ' ')
                  .split(' ')
                  .map((w) => w.isEmpty ? w : '${w[0].toUpperCase()}${w.substring(1)}')
                  .join(' ')
              : 'Form Check',
          style: const TextStyle(fontSize: 16),
        ),
      ),
      body: Stack(
        children: [
          Column(
            children: [
              // ── Camera preview (70% height) ────────────────────────────────
              Expanded(
                flex: 7,
                child: _buildCameraView(),
              ),
              // ── Live metrics bar ───────────────────────────────────────────
              Expanded(
                flex: 3,
                child: _buildMetricsBar(),
              ),
            ],
          ),
          // ── Mismatch warning banner ────────────────────────────────────────
          Positioned(
            top: 0,
            left: 0,
            right: 0,
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
        ],
      ),
    );
  }

  Widget _buildCameraView() {
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

    return Consumer<FormSessionProvider>(
      builder: (context, provider, _) {
        final jointErrorMap = buildJointErrorMap(provider.liveJointErrors);
        return Stack(
          fit: StackFit.expand,
          children: [
            // Camera preview
            CameraPreview(_cameraController!),
            // Skeleton overlay
            if (provider.liveLandmarks != null)
              CustomPaint(
                painter: SkeletonPainter(
                  landmarks: provider.liveLandmarks,
                  jointErrors: jointErrorMap,
                ),
              ),
            // Exercise label top-left
            Positioned(
              top: 12,
              left: 12,
              child: Container(
                padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                decoration: BoxDecoration(
                  color: Colors.black54,
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Text(
                  provider.liveExercise.replaceAll('_', ' '),
                  style: const TextStyle(color: Colors.white, fontSize: 13),
                ),
              ),
            ),
          ],
        );
      },
    );
  }

  Widget _buildMetricsBar() {
    return Consumer<FormSessionProvider>(
      builder: (context, provider, _) {
        final quality = provider.liveQuality;
        final qualityColor = quality >= 0.7
            ? Colors.green
            : quality >= 0.4
                ? Colors.amber
                : Colors.red;

        return Container(
          color: const Color(0xFF1A1A2E),
          padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.spaceEvenly,
            children: [
              // Quality bar
              Row(
                children: [
                  const Text('Form:', style: TextStyle(color: Colors.white70, fontSize: 13)),
                  const SizedBox(width: 8),
                  Expanded(
                    child: ClipRRect(
                      borderRadius: BorderRadius.circular(4),
                      child: LinearProgressIndicator(
                        value: quality,
                        backgroundColor: Colors.white12,
                        valueColor: AlwaysStoppedAnimation<Color>(qualityColor),
                        minHeight: 10,
                      ),
                    ),
                  ),
                  const SizedBox(width: 8),
                  Text(
                    '${(quality * 100).round()}%',
                    style: TextStyle(
                        color: qualityColor,
                        fontWeight: FontWeight.bold,
                        fontSize: 13),
                  ),
                ],
              ),
              // Reps counter + button row
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Row(
                    children: [
                      const Icon(Icons.repeat, color: Colors.white54, size: 18),
                      const SizedBox(width: 6),
                      Text(
                        'Reps: ${provider.liveRepCount}',
                        style: const TextStyle(color: Colors.white, fontSize: 14),
                      ),
                    ],
                  ),
                  if (!_sessionActive)
                    ElevatedButton.icon(
                      onPressed: _startSession,
                      icon: const Icon(Icons.play_arrow),
                      label: const Text('Start'),
                      style: ElevatedButton.styleFrom(
                        backgroundColor: Colors.green,
                        foregroundColor: Colors.white,
                      ),
                    )
                  else
                    ElevatedButton.icon(
                      onPressed: _stopSession,
                      icon: const Icon(Icons.stop),
                      label: const Text('Stop'),
                      style: ElevatedButton.styleFrom(
                        backgroundColor: Colors.red,
                        foregroundColor: Colors.white,
                      ),
                    ),
                ],
              ),
            ],
          ),
        );
      },
    );
  }
}
