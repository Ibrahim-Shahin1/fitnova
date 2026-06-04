import 'package:flutter/material.dart';

import '../../theme/app_spacing.dart';

/// Standard FitNova surface card.
///
/// Defaults to a `surface`-colored card with the standard elevation. Pass
/// `onTap` to make it tappable (a Material ripple is wired in automatically).
/// Pass `accentColor` to outline the card in that color — used for tinted
/// CTAs (e.g. cyan for the planner card, green for the form-correction card).
class AppCard extends StatelessWidget {
  final Widget child;
  final EdgeInsetsGeometry padding;
  final VoidCallback? onTap;

  /// When set, draws a 1.5 px outline in this color (at 0.5 alpha) around
  /// the card. Common pattern for action cards that want to read as the
  /// color of whatever they trigger.
  final Color? accentColor;

  /// When true, uses `surfaceContainer` instead of `surface` — useful for
  /// nested cards where the parent is already on `surface`.
  final bool elevated;

  const AppCard({
    super.key,
    required this.child,
    this.padding = const EdgeInsets.all(AppSpacing.md),
    this.onTap,
    this.accentColor,
    this.elevated = false,
  });

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;

    final BorderSide side = accentColor != null
        ? BorderSide(color: accentColor!.withValues(alpha: 0.5), width: 1.5)
        : BorderSide.none;

    final shape = RoundedRectangleBorder(
      borderRadius: BorderRadius.circular(AppRadius.rl),
      side: side,
    );

    return Material(
      color: elevated ? cs.surfaceContainer : cs.surface,
      shape: shape,
      clipBehavior: Clip.antiAlias,
      child: InkWell(
        onTap: onTap,
        child: Padding(
          padding: padding,
          child: child,
        ),
      ),
    );
  }
}
