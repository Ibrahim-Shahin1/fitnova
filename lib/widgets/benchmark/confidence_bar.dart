import 'package:flutter/material.dart';

import '../../theme/app_colors.dart';
import '../../theme/app_spacing.dart';

class ConfidenceBar extends StatelessWidget {
  final double confidence;
  final double threshold;
  final bool detected;

  const ConfidenceBar({
    super.key,
    required this.confidence,
    required this.threshold,
    required this.detected,
  });

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final ext = theme.extension<AppColors>()!;

    final Color fillColor;
    if (detected) {
      fillColor = ext.error;
    } else if ((threshold - confidence).abs() <= 0.05) {
      fillColor = ext.warning;
    } else {
      fillColor = ext.success;
    }

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        LayoutBuilder(
          builder: (context, constraints) {
            final trackWidth = constraints.maxWidth;
            final markerLeft = (threshold * trackWidth) - 1;

            return SizedBox(
              height: 16,
              child: Stack(
                clipBehavior: Clip.none,
                children: [
                  Positioned(
                    top: 4,
                    left: 0,
                    right: 0,
                    child: Container(
                      height: 8,
                      decoration: BoxDecoration(
                        color: ext.outline,
                        borderRadius:
                            BorderRadius.circular(AppRadius.pill),
                      ),
                    ),
                  ),
                  Positioned(
                    top: 4,
                    left: 0,
                    right: 0,
                    child: FractionallySizedBox(
                      widthFactor: confidence.clamp(0.0, 1.0),
                      alignment: Alignment.centerLeft,
                      child: Container(
                        height: 8,
                        decoration: BoxDecoration(
                          color: fillColor,
                          borderRadius:
                              BorderRadius.circular(AppRadius.pill),
                        ),
                      ),
                    ),
                  ),
                  Positioned(
                    top: 0,
                    left: markerLeft,
                    child: Container(
                      width: 2,
                      height: 16,
                      color: ext.primary,
                    ),
                  ),
                ],
              ),
            );
          },
        ),
        const SizedBox(height: AppSpacing.xs),
        LayoutBuilder(
          builder: (context, constraints) {
            final maxWidth = constraints.maxWidth;
            final labelLeft =
                (threshold * maxWidth).clamp(0.0, maxWidth - 60.0);

            return Stack(
              children: [
                Align(
                  alignment: Alignment.centerLeft,
                  child: Text(
                    confidence.toStringAsFixed(3),
                    style: theme.textTheme.bodySmall?.copyWith(
                      color: ext.mutedText,
                    ),
                  ),
                ),
                Positioned(
                  left: labelLeft,
                  child: Text(
                    'T: ${threshold.toStringAsFixed(3)}',
                    style: theme.textTheme.labelMedium?.copyWith(
                      color: ext.primary,
                    ),
                  ),
                ),
                Align(
                  alignment: Alignment.centerRight,
                  child: Text(
                    '1.0',
                    style: theme.textTheme.bodySmall?.copyWith(
                      color: ext.mutedText,
                    ),
                  ),
                ),
              ],
            );
          },
        ),
      ],
    );
  }
}
