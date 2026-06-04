import 'package:flutter/material.dart';
import 'package:video_player/video_player.dart';

import '../../config/api_config.dart';
import '../../theme/app_spacing.dart';

/// A looping, muted video-demo popup for a plan exercise. [mediaUrl] is the
/// backend-relative path (e.g. /static/exercise_videos/squat/demo.mp4); the
/// backend base URL is prepended here. Fails gracefully if the clip is missing.
class ExerciseDemoDialog extends StatefulWidget {
  const ExerciseDemoDialog({
    super.key,
    required this.exerciseName,
    required this.mediaUrl,
  });

  final String exerciseName;
  final String mediaUrl;

  static Future<void> show(
    BuildContext context, {
    required String exerciseName,
    required String mediaUrl,
  }) {
    return showDialog<void>(
      context: context,
      builder: (_) =>
          ExerciseDemoDialog(exerciseName: exerciseName, mediaUrl: mediaUrl),
    );
  }

  @override
  State<ExerciseDemoDialog> createState() => _ExerciseDemoDialogState();
}

class _ExerciseDemoDialogState extends State<ExerciseDemoDialog> {
  VideoPlayerController? _controller;
  bool _ready = false;
  bool _failed = false;

  @override
  void initState() {
    super.initState();
    final url = '${ApiConfig.baseUrl}${widget.mediaUrl}';
    _controller = VideoPlayerController.networkUrl(Uri.parse(url))
      ..setVolume(0)
      ..setLooping(true)
      ..initialize().then((_) {
        if (!mounted) return;
        setState(() => _ready = true);
        _controller?.play();
      }).catchError((_) {
        if (mounted) setState(() => _failed = true);
      });
  }

  @override
  void dispose() {
    _controller?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final cs = theme.colorScheme;
    return Dialog(
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(AppRadius.rl),
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Padding(
            padding: const EdgeInsets.fromLTRB(
              AppSpacing.md, AppSpacing.md, AppSpacing.md, AppSpacing.sm),
            child: Text(
              widget.exerciseName,
              textAlign: TextAlign.center,
              style:
                  theme.textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w700),
            ),
          ),
          ClipRRect(
            borderRadius:
                BorderRadius.vertical(bottom: Radius.circular(AppRadius.rl)),
            child: SizedBox(
              height: 260,
              width: double.infinity,
              child: _failed
                  ? Center(
                      child: Padding(
                        padding: const EdgeInsets.all(AppSpacing.lg),
                        child: Text('Demo unavailable for this exercise.',
                            textAlign: TextAlign.center,
                            style: theme.textTheme.bodyMedium
                                ?.copyWith(color: cs.onSurfaceVariant)),
                      ),
                    )
                  : _ready && _controller != null
                      ? AspectRatio(
                          aspectRatio: _controller!.value.aspectRatio,
                          child: VideoPlayer(_controller!),
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
