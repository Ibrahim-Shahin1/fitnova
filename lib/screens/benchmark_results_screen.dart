import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:video_player/video_player.dart';

import '../models/form_models.dart';
import '../providers/benchmark_provider.dart';
import '../services/api_service.dart';
import '../theme/app_colors.dart';
import '../theme/app_spacing.dart';
import '../widgets/benchmark/error_result_row.dart';
import '../widgets/ui/app_button.dart';
import '../widgets/ui/app_empty_state.dart';
import '../widgets/ui/app_error_state.dart';
import '../widgets/ui/app_loader.dart';

class BenchmarkResultsScreen extends StatefulWidget {
  const BenchmarkResultsScreen({super.key});

  @override
  State<BenchmarkResultsScreen> createState() => _BenchmarkResultsScreenState();
}

class _BenchmarkResultsScreenState extends State<BenchmarkResultsScreen> {
  VideoPlayerController? _ctrl;
  bool _videoReady = false;
  String? _videoError;

  @override
  void initState() {
    super.initState();
    // context.read is safe in initState; watch/listen:true must not be used here.
    final provider = context.read<BenchmarkProvider>();
    if (provider.selectedExercise != 'shallow') {
      _initVideo(provider.selectedExercise, provider.selectedClip?.clipId ?? '');
    }
  }

  Future<void> _initVideo(String exercise, String clipId) async {
    try {
      final uri = Uri.parse(ApiService.benchmarkMediaUrl(exercise, clipId));
      final c = VideoPlayerController.networkUrl(uri);
      await c.initialize();
      await c.setLooping(true);
      await c.play();
      if (mounted) {
        setState(() {
          _ctrl = c;
          _videoReady = true;
        });
      }
    } catch (_) {
      if (mounted) {
        setState(() => _videoError = 'Could not load this clip\'s video.');
      }
    }
  }

  @override
  void dispose() {
    _ctrl?.dispose();
    super.dispose();
  }

  Widget _buildMediaCard(
    BenchmarkProvider provider,
    AppColors ext,
  ) {
    final exercise = provider.selectedExercise;
    final reps = provider.analysisResult ?? [];

    if (exercise == 'shallow') {
      final thumbnail = reps.isNotEmpty ? reps[0].thumbnail : null;
      if (thumbnail != null) {
        try {
          final bytes = base64Decode(thumbnail);
          return ClipRRect(
            borderRadius: BorderRadius.circular(AppRadius.rl),
            child: Image.memory(
              bytes,
              width: double.infinity,
              fit: BoxFit.cover,
            ),
          );
        } catch (_) {
          // malformed base64 — fall through to placeholder
        }
      }
      return Container(
        height: 200,
        decoration: BoxDecoration(
          color: ext.surfaceVariant,
          borderRadius: BorderRadius.circular(AppRadius.rl),
        ),
        child: Icon(Icons.image_not_supported, color: ext.mutedText, size: 48),
      );
    }

    if (_videoError != null) {
      return Container(
        height: 200,
        decoration: BoxDecoration(
          color: ext.surfaceVariant,
          borderRadius: BorderRadius.circular(AppRadius.rl),
        ),
        child: Center(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Text(
                'Video unavailable',
                style: Theme.of(context).textTheme.titleMedium,
              ),
              const SizedBox(height: AppSpacing.xs),
              Text(
                'Could not load this clip\'s video.',
                style: Theme.of(context).textTheme.bodySmall?.copyWith(
                      color: ext.mutedText,
                    ),
              ),
            ],
          ),
        ),
      );
    }

    if (!_videoReady) {
      return Container(
        height: 200,
        decoration: BoxDecoration(
          color: ext.surfaceVariant,
          borderRadius: BorderRadius.circular(AppRadius.rl),
        ),
        child: const Center(child: AppLoader.medium()),
      );
    }

    final ar = _ctrl!.value.aspectRatio;
    return ClipRRect(
      borderRadius: BorderRadius.circular(AppRadius.rl),
      child: AspectRatio(
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
                const Icon(
                  Icons.play_circle_fill,
                  color: Colors.white70,
                  size: 64,
                ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildRepThumbnail(String? thumbnail, AppColors ext) {
    if (thumbnail != null) {
      try {
        return ClipRRect(
          borderRadius: BorderRadius.circular(AppRadius.rm),
          child: Image.memory(
            base64Decode(thumbnail),
            width: 56,
            height: 56,
            fit: BoxFit.cover,
          ),
        );
      } catch (_) {
        // malformed base64 — fall through to placeholder
      }
    }
    return Container(
      width: 56,
      height: 56,
      decoration: BoxDecoration(
        color: ext.surfaceVariant,
        borderRadius: BorderRadius.circular(AppRadius.rm),
      ),
      child: Icon(Icons.image_not_supported, color: ext.mutedText, size: 24),
    );
  }

  @override
  Widget build(BuildContext context) {
    final provider = context.watch<BenchmarkProvider>();
    final theme = Theme.of(context);
    final cs = theme.colorScheme;
    final ext = theme.extension<AppColors>()!;

    final exercise = provider.selectedExercise;
    final clip = provider.selectedClip;
    final clipId = clip?.clipId ?? '';

    final exerciseLabel = switch (exercise) {
      'squat' => 'Squat',
      'ohp' => 'OHP',
      'shallow' => 'Shallow',
      _ => exercise,
    };

    if (provider.analysisState == BenchmarkAnalysisState.analyzing) {
      return Scaffold(
        appBar: AppBar(title: Text('$exerciseLabel — $clipId')),
        body: const Center(
          child: AppLoader.large(label: 'Analyzing clip…'),
        ),
      );
    }

    if (provider.analysisState == BenchmarkAnalysisState.error) {
      return Scaffold(
        appBar: AppBar(title: Text('$exerciseLabel — $clipId')),
        body: AppErrorState(
          icon: Icons.error_outline,
          title: 'Analysis failed',
          message: provider.analysisError,
          actionLabel: 'Try again',
          onAction: () => provider.retriggerAnalysis(),
        ),
      );
    }

    if (provider.analysisState == BenchmarkAnalysisState.noReps ||
        provider.analysisResult?.isEmpty == true) {
      return Scaffold(
        appBar: AppBar(title: Text('$exerciseLabel — $clipId')),
        body: AppEmptyState(
          icon: Icons.videocam_off_outlined,
          title: 'No reps detected',
          message:
              'The model did not find any complete reps in this clip.',
          actionLabel: 'Back to Browse',
          onAction: () => Navigator.of(context).pop(),
        ),
      );
    }

    final reps = provider.analysisResult ?? [];

    return Scaffold(
      appBar: AppBar(title: Text('$exerciseLabel — $clipId')),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(AppSpacing.md),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            _buildMediaCard(provider, ext),
            const SizedBox(height: AppSpacing.lg),
            Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Icon(Icons.info_outline, size: 16, color: cs.primary),
                const SizedBox(width: AppSpacing.sm),
                Expanded(
                  child: Text(
                    'These are Fitness-AQA test clips — the same clips the model was evaluated on. '
                    'The validated F1 numbers apply here. Off-domain phone video is not reliable.',
                    style: theme.textTheme.bodySmall?.copyWith(
                      color: ext.mutedText,
                    ),
                  ),
                ),
              ],
            ),
            const SizedBox(height: AppSpacing.md),
            ...reps.map((rep) => _buildRepCard(rep, ext, theme)),
            const SizedBox(height: AppSpacing.xl),
            AppButton(
              label: 'Back to Browse',
              variant: AppButtonVariant.ghost,
              expand: true,
              onPressed: () => Navigator.of(context).pop(),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildRepCard(FormRep rep, AppColors ext, ThemeData theme) {
    return Padding(
      padding: const EdgeInsets.only(bottom: AppSpacing.md),
      child: Container(
        decoration: BoxDecoration(
          color: theme.colorScheme.surfaceContainerLowest,
          borderRadius: BorderRadius.circular(AppRadius.rl),
          border: Border.all(color: ext.outline, width: 1),
        ),
        padding: const EdgeInsets.all(AppSpacing.md),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                _buildRepThumbnail(rep.thumbnail, ext),
                const SizedBox(width: AppSpacing.md),
                Expanded(
                  child: Text(
                    'Rep ${rep.repIndex + 1} of ${rep.totalReps}',
                    style: theme.textTheme.titleMedium,
                  ),
                ),
              ],
            ),
            if (rep.errors.isNotEmpty) ...[
              const SizedBox(height: AppSpacing.md),
              ...rep.errors.map(
                (e) => Padding(
                  padding: const EdgeInsets.only(bottom: AppSpacing.sm),
                  child: ErrorResultRow(error: e),
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }
}
