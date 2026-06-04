import 'package:flutter/material.dart';
import 'package:video_player/video_player.dart';

import '../config/api_config.dart';
import '../models/fitness_plan.dart';
import '../theme/app_spacing.dart';

class ExerciseTile extends StatelessWidget {
  final Exercise exercise;

  const ExerciseTile({super.key, required this.exercise});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final cs = theme.colorScheme;

    return Card(
      margin: const EdgeInsets.symmetric(
        horizontal: AppSpacing.md,
        vertical: AppSpacing.xs,
      ),
      child: Padding(
        padding: const EdgeInsets.all(AppSpacing.md),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              exercise.exerciseName,
              style: theme.textTheme.titleMedium?.copyWith(
                fontWeight: FontWeight.w700,
              ),
            ),
            const SizedBox(height: AppSpacing.sm),
            Row(
              children: [
                _chip(context, Icons.repeat, '${exercise.sets} x ${exercise.reps}'),
                const SizedBox(width: AppSpacing.md),
                _chip(context, Icons.timer, '${exercise.restSeconds}s rest'),
              ],
            ),
            if (exercise.mediaUrl != null) ...[
              const SizedBox(height: AppSpacing.xs),
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
            const SizedBox(height: AppSpacing.xs),
            TextButton.icon(
              icon: Icon(Icons.videocam_outlined, size: 18, color: cs.secondary),
              label: Text(
                'Check My Form',
                style: TextStyle(color: cs.secondary),
              ),
              style: TextButton.styleFrom(
                padding: EdgeInsets.zero,
                tapTargetSize: MaterialTapTargetSize.shrinkWrap,
              ),
              onPressed: () {
                Navigator.of(context).pushNamed('/benchmark/browse');
              },
            ),
            if (exercise.coachingCue.isNotEmpty) ...[
              const SizedBox(height: AppSpacing.sm),
              Text(
                exercise.coachingCue,
                style: theme.textTheme.bodySmall?.copyWith(
                  fontStyle: FontStyle.italic,
                  color: cs.onSurfaceVariant,
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }

  Widget _chip(BuildContext context, IconData icon, String label) {
    final muted = Theme.of(context).colorScheme.onSurfaceVariant;
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Icon(icon, size: 16, color: muted),
        const SizedBox(width: AppSpacing.xs),
        Text(
          label,
          style: TextStyle(color: muted, fontSize: 13),
        ),
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
    final theme = Theme.of(context);
    return Dialog(
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(AppRadius.rl),
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Padding(
            padding: const EdgeInsets.fromLTRB(
              AppSpacing.md,
              AppSpacing.md,
              AppSpacing.md,
              AppSpacing.sm,
            ),
            child: Text(
              widget.exerciseName,
              style: theme.textTheme.titleMedium?.copyWith(
                fontWeight: FontWeight.w700,
              ),
              textAlign: TextAlign.center,
            ),
          ),
          ClipRRect(
            borderRadius: BorderRadius.vertical(
              bottom: Radius.circular(AppRadius.rl),
            ),
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
          const SizedBox(height: AppSpacing.sm),
        ],
      ),
    );
  }
}
