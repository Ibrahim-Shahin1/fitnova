import 'package:flutter/material.dart';
import 'package:video_player/video_player.dart';
import '../config/api_config.dart';
import '../models/fitness_plan.dart';

class ExerciseTile extends StatelessWidget {
  final Exercise exercise;

  const ExerciseTile({super.key, required this.exercise});

  @override
  Widget build(BuildContext context) {
    return Card(
      margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 6),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              exercise.exerciseName,
              style: Theme.of(context).textTheme.titleSmall?.copyWith(
                    fontWeight: FontWeight.bold,
                  ),
            ),
            const SizedBox(height: 8),
            Row(
              children: [
                _chip(Icons.repeat, '${exercise.sets} x ${exercise.reps}'),
                const SizedBox(width: 12),
                _chip(Icons.timer, '${exercise.restSeconds}s rest'),
              ],
            ),
            if (exercise.mediaUrl != null) ...[
              const SizedBox(height: 6),
              TextButton.icon(
                icon: const Icon(Icons.play_circle_outline, size: 18),
                label: const Text('Watch demo'),
                style: TextButton.styleFrom(
                  padding: EdgeInsets.zero,
                  tapTargetSize: MaterialTapTargetSize.shrinkWrap,
                ),
                onPressed: () {
                  showDialog(
                    context: context,
                    builder: (_) => _VideoDemoDialog(
                      exerciseName: exercise.exerciseName,
                      videoUrl: '${ApiConfig.baseUrl}${exercise.mediaUrl}',
                    ),
                  );
                },
              ),
            ],
            if (exercise.coachingCue.isNotEmpty) ...[
              const SizedBox(height: 8),
              Text(
                exercise.coachingCue,
                style: Theme.of(context).textTheme.bodySmall?.copyWith(
                      fontStyle: FontStyle.italic,
                      color: Colors.grey[600],
                    ),
              ),
            ],
          ],
        ),
      ),
    );
  }

  Widget _chip(IconData icon, String label) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Icon(icon, size: 16, color: Colors.grey[600]),
        const SizedBox(width: 4),
        Text(label, style: TextStyle(color: Colors.grey[700], fontSize: 13)),
      ],
    );
  }
}

// ── Looping video demo dialog ─────────────────────────────────────────────────

class _VideoDemoDialog extends StatefulWidget {
  final String exerciseName;
  final String videoUrl;

  const _VideoDemoDialog({
    required this.exerciseName,
    required this.videoUrl,
  });

  @override
  State<_VideoDemoDialog> createState() => _VideoDemoDialogState();
}

class _VideoDemoDialogState extends State<_VideoDemoDialog> {
  late final VideoPlayerController _controller;
  bool _ready = false;

  @override
  void initState() {
    super.initState();
    _controller = VideoPlayerController.networkUrl(Uri.parse(widget.videoUrl))
      ..setVolume(0)
      ..setLooping(true)
      ..initialize().then((_) {
        if (mounted) {
          setState(() => _ready = true);
          _controller.play();
        }
      });
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Dialog(
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 16, 16, 8),
            child: Text(
              widget.exerciseName,
              style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 16),
              textAlign: TextAlign.center,
            ),
          ),
          ClipRRect(
            borderRadius:
                const BorderRadius.vertical(bottom: Radius.circular(16)),
            child: SizedBox(
              height: 260,
              width: double.infinity,
              child: _ready
                  ? AspectRatio(
                      aspectRatio: _controller.value.aspectRatio,
                      child: VideoPlayer(_controller),
                    )
                  : const Center(child: CircularProgressIndicator()),
            ),
          ),
          const SizedBox(height: 8),
        ],
      ),
    );
  }
}
