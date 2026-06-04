import 'package:flutter/material.dart';

import '../../models/benchmark_models.dart';
import '../../services/api_service.dart';
import '../../theme/app_colors.dart';
import '../../theme/app_spacing.dart';
import '../ui/app_card.dart';

class BenchmarkClipRow extends StatelessWidget {
  final BenchmarkClip clip;
  final String exercise;
  final bool? isCorrect;
  final VoidCallback onTap;

  const BenchmarkClipRow({
    super.key,
    required this.clip,
    required this.exercise,
    required this.isCorrect,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final cs = theme.colorScheme;
    final ext = theme.extension<AppColors>()!;

    final gtSummary = clip.groundTruth.entries
        .map((e) =>
            '${benchmarkErrorChip(e.key)}: ${benchmarkVerdict(e.key, e.value)}')
        .join(' · ');

    return AppCard(
      onTap: onTap,
      padding: const EdgeInsets.all(AppSpacing.md),
      child: Row(
        children: [
          if (exercise == 'shallow')
            ClipRRect(
              borderRadius: BorderRadius.circular(AppRadius.rm),
              child: Image.network(
                ApiService.benchmarkMediaUrl('shallow', clip.clipId),
                width: 48,
                height: 48,
                fit: BoxFit.cover,
                errorBuilder: (_, __, ___) => Container(
                  width: 48,
                  height: 48,
                  color: ext.surfaceVariant,
                  child: Icon(Icons.image_not_supported,
                      color: ext.mutedText, size: 20),
                ),
              ),
            )
          else
            Icon(Icons.video_file_outlined, color: cs.primary, size: 24),
          const SizedBox(width: AppSpacing.sm),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(clip.clipId, style: theme.textTheme.titleMedium),
                const SizedBox(height: AppSpacing.xs),
                Text(
                  'Benchmark — $gtSummary',
                  style: theme.textTheme.bodySmall?.copyWith(
                    color: ext.mutedText,
                  ),
                ),
              ],
            ),
          ),
          if (isCorrect != null) ...[
            Container(
              width: 8,
              height: 8,
              decoration: BoxDecoration(
                color: isCorrect! ? ext.success : ext.error,
                borderRadius: BorderRadius.circular(AppRadius.pill),
              ),
            ),
          ],
          const SizedBox(width: AppSpacing.sm),
          Icon(Icons.chevron_right, color: ext.mutedText, size: 16),
        ],
      ),
    );
  }
}
