import 'package:flutter/material.dart';

import '../../theme/app_colors.dart';
import '../../theme/app_spacing.dart';

enum AppButtonVariant { primary, secondary, ghost, danger }

enum AppButtonSize { sm, md, lg }

/// FitNova's standard button.
///
/// Variants:
///   - `primary`   filled, brand cyan
///   - `secondary` filled, deep green (form-correction CTAs)
///   - `ghost`     no fill, on-surface text
///   - `danger`    filled, error red
class AppButton extends StatelessWidget {
  final String label;
  final VoidCallback? onPressed;
  final AppButtonVariant variant;
  final AppButtonSize size;
  final IconData? icon;
  final bool isLoading;
  final bool expand;

  const AppButton({
    super.key,
    required this.label,
    required this.onPressed,
    this.variant = AppButtonVariant.primary,
    this.size = AppButtonSize.md,
    this.icon,
    this.isLoading = false,
    this.expand = false,
  });

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final cs = theme.colorScheme;
    final ext = theme.extension<AppColors>()!;

    final (bg, fg, border) = switch (variant) {
      AppButtonVariant.primary => (cs.primary, cs.onPrimary, null),
      AppButtonVariant.secondary => (cs.secondary, cs.onSecondary, null),
      AppButtonVariant.ghost => (Colors.transparent, cs.onSurface, BorderSide(color: cs.outline, width: 1.5)),
      AppButtonVariant.danger => (ext.error, ext.onError, null),
    };

    final (height, hPad, textStyle) = switch (size) {
      AppButtonSize.sm => (40.0, AppSpacing.md, theme.textTheme.labelMedium),
      AppButtonSize.md => (48.0, AppSpacing.lg, theme.textTheme.labelLarge),
      AppButtonSize.lg => (56.0, AppSpacing.xl, theme.textTheme.labelLarge?.copyWith(fontSize: 16)),
    };

    final Widget child;
    if (isLoading) {
      child = SizedBox(
        height: 18,
        width: 18,
        child: CircularProgressIndicator(
          strokeWidth: 2,
          valueColor: AlwaysStoppedAnimation<Color>(fg),
        ),
      );
    } else if (icon != null) {
      child = Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 18),
          const SizedBox(width: AppSpacing.sm),
          Text(label),
        ],
      );
    } else {
      child = Text(label);
    }

    final button = Material(
      color: bg,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(AppRadius.rl),
        side: border ?? BorderSide.none,
      ),
      child: InkWell(
        onTap: (isLoading || onPressed == null) ? null : onPressed,
        borderRadius: BorderRadius.circular(AppRadius.rl),
        child: AnimatedContainer(
          duration: AppDuration.fast,
          height: height,
          padding: EdgeInsets.symmetric(horizontal: hPad),
          alignment: Alignment.center,
          child: DefaultTextStyle.merge(
            style: (textStyle ?? const TextStyle()).copyWith(color: fg),
            child: IconTheme.merge(
              data: IconThemeData(color: fg),
              child: child,
            ),
          ),
        ),
      ),
    );

    final styled = Opacity(
      opacity: onPressed == null && !isLoading ? 0.5 : 1,
      child: button,
    );

    return expand ? SizedBox(width: double.infinity, child: styled) : styled;
  }
}
