import 'package:flutter/material.dart';

import '../../theme/app_spacing.dart';

/// Branded progress indicator. Defaults to `colorScheme.primary`.
///
/// Use the named constructors for the standard sizes:
///   - `AppLoader.small()`  — 16 px (inline, button-sized)
///   - `AppLoader.medium()` — 24 px (default)
///   - `AppLoader.large()`  — 36 px (centered on a screen)
class AppLoader extends StatelessWidget {
  final double size;
  final double strokeWidth;
  final String? label;
  final Color? color;

  const AppLoader({
    super.key,
    this.size = 24,
    this.strokeWidth = 3,
    this.label,
    this.color,
  });

  const AppLoader.small({super.key, this.label, this.color})
      : size = 16,
        strokeWidth = 2;

  const AppLoader.medium({super.key, this.label, this.color})
      : size = 24,
        strokeWidth = 3;

  const AppLoader.large({super.key, this.label, this.color})
      : size = 36,
        strokeWidth = 4;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final c = color ?? theme.colorScheme.primary;

    final spinner = SizedBox(
      width: size,
      height: size,
      child: CircularProgressIndicator(
        strokeWidth: strokeWidth,
        valueColor: AlwaysStoppedAnimation<Color>(c),
      ),
    );

    if (label == null) return spinner;

    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        spinner,
        const SizedBox(height: AppSpacing.md),
        Text(
          label!,
          style: theme.textTheme.bodyMedium?.copyWith(
            color: theme.colorScheme.onSurfaceVariant,
          ),
        ),
      ],
    );
  }
}
