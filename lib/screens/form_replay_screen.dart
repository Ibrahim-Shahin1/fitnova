import 'dart:convert';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:video_player/video_player.dart';

import '../models/form_models.dart';
import '../providers/form_session_provider.dart';

/// Single-rep upload result, shown as a replay:
///   - plays the uploaded clip back,
///   - shows the EXACT 112x112 frames the model analyzed ("what the model sees"),
///   - shows the exact KIE / KFE confidence + verdict for the rep.
///
/// No skeleton / rep-counting — the clip is one rep (the model's training
/// distribution). All numbers are the model's real outputs (D-05).
class FormReplayScreen extends StatefulWidget {
  final String videoPath;
  final FormReport report;

  const FormReplayScreen({
    super.key,
    required this.videoPath,
    required this.report,
  });

  @override
  State<FormReplayScreen> createState() => _FormReplayScreenState();
}

class _FormReplayScreenState extends State<FormReplayScreen> {
  VideoPlayerController? _ctrl;
  bool _ready = false;
  String? _error;

  @override
  void initState() {
    super.initState();
    _init();
  }

  Future<void> _init() async {
    try {
      final c = VideoPlayerController.file(File(widget.videoPath));
      await c.initialize();
      await c.setLooping(true);
      await c.play();
      if (mounted) setState(() { _ctrl = c; _ready = true; });
    } catch (e) {
      if (mounted) setState(() => _error = 'Could not play video: $e');
    }
  }

  @override
  void dispose() {
    _ctrl?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final report = widget.report;
    final rep = report.rep;
    final errors = rep?.errors ?? const <FormError>[];

    return Scaffold(
      backgroundColor: const Color(0xFF0F0F1E),
      appBar: AppBar(
        backgroundColor: const Color(0xFF1A1A2E),
        foregroundColor: Colors.white,
        title: Text('${report.exerciseDisplay} — Analysis'),
        automaticallyImplyLeading: false,
      ),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          _videoCard(),
          const SizedBox(height: 20),

          _sectionHeader('What the model sees',
              'The exact 112x112 frames fed to the model (knee-aware crop).'),
          const SizedBox(height: 10),
          _modelViewStrip(report.modelViewFrames),
          const SizedBox(height: 20),

          _sectionHeader('Detected errors', 'Exact model confidence per error.'),
          const SizedBox(height: 10),
          ...errors.map((e) => _ErrorResultCard(error: e)),
          const SizedBox(height: 16),

          _feedbackCard(report.sessionFeedback),
          const SizedBox(height: 24),

          ElevatedButton.icon(
            onPressed: () {
              context.read<FormSessionProvider>().reset();
              Navigator.of(context).popUntil(
                (r) => r.isFirst || r.settings.name == '/plan',
              );
            },
            icon: const Icon(Icons.arrow_back),
            label: const Text('Done'),
            style: ElevatedButton.styleFrom(
              backgroundColor: const Color(0xFF38BDF8),
              foregroundColor: Colors.white,
              minimumSize: const Size.fromHeight(48),
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
            ),
          ),
        ],
      ),
    );
  }

  Widget _videoCard() {
    Widget child;
    if (_error != null) {
      child = Padding(
        padding: const EdgeInsets.all(24),
        child: Text(_error!, style: const TextStyle(color: Colors.redAccent)),
      );
    } else if (!_ready || _ctrl == null) {
      child = const SizedBox(
        height: 200,
        child: Center(child: CircularProgressIndicator()),
      );
    } else {
      final ar = _ctrl!.value.aspectRatio;
      child = AspectRatio(
        aspectRatio: ar == 0 ? 9 / 16 : ar,
        child: GestureDetector(
          onTap: () => setState(() {
            _ctrl!.value.isPlaying ? _ctrl!.pause() : _ctrl!.play();
          }),
          child: Stack(
            alignment: Alignment.center,
            children: [
              VideoPlayer(_ctrl!),
              if (!_ctrl!.value.isPlaying)
                const Icon(Icons.play_circle_fill, color: Colors.white70, size: 64),
            ],
          ),
        ),
      );
    }
    return ClipRRect(borderRadius: BorderRadius.circular(12), child: child);
  }

  Widget _modelViewStrip(List<String> frames) {
    if (frames.isEmpty) {
      return const Text('(model-view frames unavailable)',
          style: TextStyle(color: Colors.white38, fontSize: 12));
    }
    return SizedBox(
      height: 110,
      child: ListView.separated(
        scrollDirection: Axis.horizontal,
        itemCount: frames.length,
        separatorBuilder: (_, __) => const SizedBox(width: 8),
        itemBuilder: (_, i) {
          try {
            final bytes = base64Decode(frames[i]);
            return ClipRRect(
              borderRadius: BorderRadius.circular(8),
              child: Image.memory(bytes,
                  width: 110, height: 110, fit: BoxFit.cover, gaplessPlayback: true),
            );
          } catch (_) {
            return const SizedBox(width: 110, height: 110);
          }
        },
      ),
    );
  }

  Widget _sectionHeader(String title, String subtitle) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(title,
            style: const TextStyle(
                color: Colors.white, fontSize: 16, fontWeight: FontWeight.bold)),
        const SizedBox(height: 2),
        Text(subtitle, style: const TextStyle(color: Colors.white54, fontSize: 12)),
      ],
    );
  }

  Widget _feedbackCard(String feedback) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFF1A1A2E),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: const Color(0xFF38BDF8).withValues(alpha: 0.4)),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Icon(Icons.fitness_center, color: Color(0xFF38BDF8), size: 20),
          const SizedBox(width: 10),
          Expanded(
            child: Text(
              feedback.isNotEmpty ? feedback : 'Analysis complete.',
              style: const TextStyle(color: Colors.white70, fontSize: 14, height: 1.5),
            ),
          ),
        ],
      ),
    );
  }
}

/// One error's result: big exact confidence + label + DETECTED/clear badge.
class _ErrorResultCard extends StatelessWidget {
  final FormError error;
  const _ErrorResultCard({required this.error});

  @override
  Widget build(BuildContext context) {
    final color = error.detected
        ? (error.severityWord == 'strong'
            ? Colors.redAccent.shade400
            : error.severityWord == 'moderate'
                ? Colors.orangeAccent.shade400
                : Colors.amberAccent.shade400)
        : Colors.green;

    return Container(
      margin: const EdgeInsets.only(bottom: 10),
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: const Color(0xFF1A1A2E),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: color.withValues(alpha: 0.5)),
      ),
      child: Row(
        children: [
          SizedBox(
            width: 64,
            child: Text(
              error.confidence.toStringAsFixed(2),
              style: TextStyle(
                  color: color, fontSize: 26, fontWeight: FontWeight.w900, height: 1.0),
            ),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(error.label,
                    style: const TextStyle(
                        color: Colors.white, fontSize: 16, fontWeight: FontWeight.w600)),
                const SizedBox(height: 2),
                Text('${error.type} · confidence ${error.confidence.toStringAsFixed(3)}',
                    style: const TextStyle(color: Colors.white54, fontSize: 12)),
              ],
            ),
          ),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
            decoration: BoxDecoration(
              color: color.withValues(alpha: 0.18),
              borderRadius: BorderRadius.circular(20),
              border: Border.all(color: color, width: 1.5),
            ),
            child: Text(
              error.detected ? error.severityWord.toUpperCase() : 'CLEAR',
              style: TextStyle(color: color, fontSize: 13, fontWeight: FontWeight.bold),
            ),
          ),
        ],
      ),
    );
  }
}
