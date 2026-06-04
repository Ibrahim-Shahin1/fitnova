import 'package:flutter/material.dart';

import '../../models/benchmark_models.dart';
import '../../models/form_models.dart';
import '../../theme/app_colors.dart';
import '../../theme/app_spacing.dart';
import 'confidence_bar.dart';

class ErrorResultRow extends StatelessWidget {
  final FormError error;

  const ErrorResultRow({super.key, required this.error});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final ext = theme.extension<AppColors>()!;

    final badgeColor = error.detected ? ext.error : ext.success;
    final badgeText = error.detected ? 'DETECTED' : 'CLEAR';

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
                child: Text(
                  benchmarkErrorLabel(error.type),
                  style: theme.textTheme.titleMedium,
                ),
              ),
              const SizedBox(width: AppSpacing.sm),
              _Badge(
                text: badgeText,
                color: badgeColor,
              ),
              _GtBadge(error: error, ext: ext),
            ],
          ),
          const SizedBox(height: AppSpacing.xs),
          Text(
            error.confidence.toStringAsFixed(3),
            style: theme.textTheme.displayMedium,
          ),
          const SizedBox(height: AppSpacing.xs),
          Text(
            'Confidence: ${error.confidence.toStringAsFixed(3)}   '
            'Threshold: ${error.threshold.toStringAsFixed(3)}',
            style: theme.textTheme.bodySmall?.copyWith(color: ext.mutedText),
          ),
          const SizedBox(height: AppSpacing.sm),
          ConfidenceBar(
            confidence: error.confidence,
            threshold: error.threshold,
            detected: error.detected,
          ),
        ],
      ),
    );
  }
}

class _Badge extends StatelessWidget {
  final String text;
  final Color color;

  const _Badge({required this.text, required this.color});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Container(
      padding: const EdgeInsets.symmetric(
        vertical: AppSpacing.sm,
        horizontal: 12,
      ),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.15),
        borderRadius: BorderRadius.circular(AppRadius.pill),
        border: Border.all(color: color, width: 1),
      ),
      child: Text(
        text,
        style: theme.textTheme.labelMedium?.copyWith(color: color),
      ),
    );
  }
}

class _GtBadge extends StatelessWidget {
  final FormError error;
  final AppColors ext;

  const _GtBadge({required this.error, required this.ext});

  @override
  Widget build(BuildContext context) {
    if (error.groundTruth == null) return const SizedBox.shrink();

    final theme = Theme.of(context);
    final detectedInt = error.detected ? 1 : 0;
    final isMatch = detectedInt == error.groundTruth;
    final color = isMatch ? ext.success : ext.error;
    final verdict = benchmarkVerdict(error.type, error.groundTruth!);
    final label = isMatch ? '✓ Benchmark: $verdict' : '✗ Benchmark: $verdict';

    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        const SizedBox(width: AppSpacing.sm),
        Container(
          padding: const EdgeInsets.symmetric(
            vertical: AppSpacing.sm,
            horizontal: 12,
          ),
          decoration: BoxDecoration(
            color: color.withValues(alpha: 0.15),
            borderRadius: BorderRadius.circular(AppRadius.pill),
            border: Border.all(color: color, width: 1),
          ),
          child: Text(
            label,
            style: theme.textTheme.labelMedium?.copyWith(color: color),
          ),
        ),
      ],
    );
  }
}
