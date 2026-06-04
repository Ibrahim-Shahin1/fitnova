import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../models/benchmark_models.dart';
import '../providers/benchmark_provider.dart';
import '../services/api_service.dart';
import '../theme/app_colors.dart';
import '../theme/app_spacing.dart';
import '../widgets/ui/app_empty_state.dart';

// The model's DEPTH score is P(reached depth): high = deep, low = too shallow.
// (Verified empirically against the released crops; the positive/label-1 class is
// the deeper squat, despite the dataset README's "erroneous" wording.) "Reached
// depth" when score >= threshold; "too shallow" (the fault) below it.
class BenchmarkFilmstripScreen extends StatefulWidget {
  const BenchmarkFilmstripScreen({super.key});

  @override
  State<BenchmarkFilmstripScreen> createState() =>
      _BenchmarkFilmstripScreenState();
}

class _BenchmarkFilmstripScreenState extends State<BenchmarkFilmstripScreen> {
  int _selected = 0;

  @override
  void initState() {
    super.initState();
    final frames = context.read<BenchmarkProvider>().filmstripFrames;
    if (frames.isNotEmpty) {
      var maxIdx = 0;
      var maxScore = frames[0].score['DEPTH'] ?? 0.0;
      for (var i = 1; i < frames.length; i++) {
        final s = frames[i].score['DEPTH'] ?? 0.0;
        if (s > maxScore) {
          maxScore = s;
          maxIdx = i;
        }
      }
      _selected = maxIdx;
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final cs = theme.colorScheme;
    final ext = theme.extension<AppColors>()!;
    final provider = context.watch<BenchmarkProvider>();
    final frames = provider.filmstripFrames;
    final thr = kBrowseThresholds['shallow']!['DEPTH']!;

    if (frames.isEmpty) {
      return Scaffold(
        appBar: AppBar(title: const Text('Shallow')),
        body: AppEmptyState(
          icon: Icons.image_not_supported_outlined,
          title: 'No frames for this rep',
          message: 'Go back and pick another squat.',
          actionLabel: 'Back',
          onAction: () => Navigator.of(context).pop(),
        ),
      );
    }

    final sel = _selected.clamp(0, frames.length - 1);
    final frame = frames[sel];
    final score = frame.score['DEPTH'] ?? 0.0;

    return Scaffold(
      appBar: AppBar(title: Text('Shallow — ${provider.filmstripRep}')),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(AppSpacing.md),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            ClipRRect(
              borderRadius: BorderRadius.circular(AppRadius.rl),
              child: Container(
                color: ext.surfaceVariant,
                height: 320,
                width: double.infinity,
                child: Image.network(
                  ApiService.benchmarkMediaUrl('shallow', frame.clipId),
                  fit: BoxFit.contain,
                  errorBuilder: (_, __, ___) => Icon(
                    Icons.image_not_supported,
                    color: ext.mutedText,
                    size: 48,
                  ),
                ),
              ),
            ),
            const SizedBox(height: AppSpacing.md),
            _Filmstrip(
              frames: frames,
              selected: sel,
              threshold: thr,
              cs: cs,
              ext: ext,
              onSelect: (i) => setState(() => _selected = i),
            ),
            const SizedBox(height: AppSpacing.lg),
            Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Icon(Icons.info_outline, size: 16, color: cs.primary),
                const SizedBox(width: AppSpacing.sm),
                Expanded(
                  child: Text(
                    'These are Fitness-AQA test clips — the same clips the model '
                    'was evaluated on. The validated F1 numbers apply here. '
                    'Off-domain phone video is not reliable.',
                    style: theme.textTheme.bodySmall
                        ?.copyWith(color: ext.mutedText),
                  ),
                ),
              ],
            ),
            const SizedBox(height: AppSpacing.md),
            Text(
              'Frame ${sel + 1} of ${frames.length}',
              style: theme.textTheme.titleMedium,
            ),
            const SizedBox(height: AppSpacing.xs),
            Text(
              'Score = the model\'s confidence the squat reached depth. '
              '≥ ${thr.toStringAsFixed(3)} = deep enough; below = too shallow. '
              'Scrub the strip to follow it across the descent.',
              style: theme.textTheme.bodySmall?.copyWith(color: ext.mutedText),
            ),
            const SizedBox(height: AppSpacing.sm),
            _DepthResult(
              score: score,
              threshold: thr,
              groundTruth: frame.groundTruth['DEPTH'],
              cs: cs,
              ext: ext,
            ),
            const SizedBox(height: AppSpacing.xl),
          ],
        ),
      ),
    );
  }
}

class _DepthResult extends StatelessWidget {
  final double score;
  final double threshold;
  final int? groundTruth;
  final ColorScheme cs;
  final AppColors ext;

  const _DepthResult({
    required this.score,
    required this.threshold,
    required this.groundTruth,
    required this.cs,
    required this.ext,
  });

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final reached = score >= threshold;
    final color = reached ? ext.success : ext.error;

    return Container(
      decoration: BoxDecoration(
        color: theme.colorScheme.surfaceContainer,
        borderRadius: BorderRadius.circular(AppRadius.rl),
        border: Border.all(color: ext.outline, width: 1),
      ),
      padding: const EdgeInsets.all(AppSpacing.md),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Expanded(
                child: Text(benchmarkErrorLabel('DEPTH'),
                    style: theme.textTheme.titleMedium),
              ),
              const SizedBox(width: AppSpacing.sm),
              _Pill(
                text: reached ? 'DEPTH REACHED' : 'TOO SHALLOW',
                color: color,
              ),
              if (groundTruth != null) ...[
                const SizedBox(width: AppSpacing.sm),
                _GtPill(reached: reached, groundTruth: groundTruth!, ext: ext),
              ],
            ],
          ),
          const SizedBox(height: AppSpacing.xs),
          Text(score.toStringAsFixed(3), style: theme.textTheme.displayMedium),
          const SizedBox(height: AppSpacing.xs),
          Text(
            'Reached-depth confidence: ${score.toStringAsFixed(3)}   '
            'Threshold: ${threshold.toStringAsFixed(3)}',
            style: theme.textTheme.bodySmall?.copyWith(color: ext.mutedText),
          ),
          const SizedBox(height: AppSpacing.sm),
          _DepthBar(score: score, threshold: threshold, color: color, cs: cs, ext: ext),
        ],
      ),
    );
  }
}

class _DepthBar extends StatelessWidget {
  final double score;
  final double threshold;
  final Color color;
  final ColorScheme cs;
  final AppColors ext;

  const _DepthBar({
    required this.score,
    required this.threshold,
    required this.color,
    required this.cs,
    required this.ext,
  });

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (context, c) {
        final w = c.maxWidth;
        final markerLeft = (threshold.clamp(0.0, 1.0) * w).clamp(0.0, w - 2);
        return SizedBox(
          height: 16,
          child: Stack(
            alignment: Alignment.centerLeft,
            children: [
              Container(
                height: 8,
                decoration: BoxDecoration(
                  color: ext.outline,
                  borderRadius: BorderRadius.circular(AppRadius.pill),
                ),
              ),
              FractionallySizedBox(
                widthFactor: score.clamp(0.0, 1.0),
                child: Container(
                  height: 8,
                  decoration: BoxDecoration(
                    color: color,
                    borderRadius: BorderRadius.circular(AppRadius.pill),
                  ),
                ),
              ),
              Positioned(
                left: markerLeft,
                child: Container(width: 2, height: 16, color: cs.primary),
              ),
            ],
          ),
        );
      },
    );
  }
}

class _Pill extends StatelessWidget {
  final String text;
  final Color color;
  const _Pill({required this.text, required this.color});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Container(
      padding: const EdgeInsets.symmetric(vertical: AppSpacing.sm, horizontal: 12),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.15),
        borderRadius: BorderRadius.circular(AppRadius.pill),
        border: Border.all(color: color, width: 1),
      ),
      child: Text(text,
          style: theme.textTheme.labelMedium?.copyWith(color: color)),
    );
  }
}

class _GtPill extends StatelessWidget {
  final bool reached;
  final int groundTruth;
  final AppColors ext;
  const _GtPill({required this.reached, required this.groundTruth, required this.ext});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final gtReached = groundTruth == 1;
    final match = reached == gtReached;
    final color = match ? ext.success : ext.error;
    final label =
        '${match ? '✓' : '✗'} Benchmark: ${gtReached ? 'deep' : 'shallow'}';
    return Container(
      padding: const EdgeInsets.symmetric(vertical: AppSpacing.sm, horizontal: 12),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.15),
        borderRadius: BorderRadius.circular(AppRadius.pill),
        border: Border.all(color: color, width: 1),
      ),
      child: Text(label,
          style: theme.textTheme.labelMedium?.copyWith(color: color)),
    );
  }
}

class _Filmstrip extends StatelessWidget {
  final List<BenchmarkClip> frames;
  final int selected;
  final double threshold;
  final ColorScheme cs;
  final AppColors ext;
  final void Function(int) onSelect;

  const _Filmstrip({
    required this.frames,
    required this.selected,
    required this.threshold,
    required this.cs,
    required this.ext,
    required this.onSelect,
  });

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return SizedBox(
      height: 96,
      child: ListView.builder(
        scrollDirection: Axis.horizontal,
        itemCount: frames.length,
        itemBuilder: (context, i) {
          final f = frames[i];
          final s = f.score['DEPTH'] ?? 0.0;
          final reached = s >= threshold;
          final isSel = i == selected;
          final borderColor =
              isSel ? cs.primary : (reached ? ext.success : ext.error);
          return GestureDetector(
            onTap: () => onSelect(i),
            child: Container(
              width: 64,
              margin: const EdgeInsets.only(right: AppSpacing.sm),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Container(
                    decoration: BoxDecoration(
                      borderRadius: BorderRadius.circular(AppRadius.rm),
                      border: Border.all(
                        color: borderColor,
                        width: isSel ? 3 : 1.5,
                      ),
                    ),
                    child: ClipRRect(
                      borderRadius: BorderRadius.circular(AppRadius.rm),
                      child: Image.network(
                        ApiService.benchmarkMediaUrl('shallow', f.clipId),
                        width: 56,
                        height: 56,
                        fit: BoxFit.cover,
                        errorBuilder: (_, __, ___) => Container(
                          width: 56,
                          height: 56,
                          color: ext.surfaceVariant,
                          child: Icon(Icons.image_not_supported,
                              color: ext.mutedText, size: 18),
                        ),
                      ),
                    ),
                  ),
                  const SizedBox(height: AppSpacing.xs),
                  Text(
                    s.toStringAsFixed(2),
                    style: theme.textTheme.bodySmall?.copyWith(
                      color: reached ? ext.success : ext.error,
                      fontWeight: isSel ? FontWeight.w700 : FontWeight.w400,
                    ),
                  ),
                ],
              ),
            ),
          );
        },
      ),
    );
  }
}
