import 'package:flutter/material.dart';

/// Brand color tokens for FitNova.
///
/// `AppColors.light` and `AppColors.dark` hold the full token set for each
/// theme. The standard Material tokens (`primary`, `surface`, etc.) are mapped
/// into Flutter's `ColorScheme` inside `AppTheme`. Custom tokens such as
/// `success`, `warning`, and `mutedText` ride along as a `ThemeExtension` so
/// screens can read them via `Theme.of(context).extension<AppColors>()`.
@immutable
class AppColors extends ThemeExtension<AppColors> {
  final Color primary;
  final Color onPrimary;
  final Color secondary;
  final Color onSecondary;

  /// Page-level background (Scaffold).
  final Color background;
  final Color onBackground;

  /// Default card / elevated surface.
  final Color surface;
  final Color onSurface;

  /// Slightly elevated surface (e.g. nested cards, sheets).
  final Color surfaceContainer;
  final Color surfaceVariant;

  final Color outline;
  final Color outlineVariant;

  /// Lower-contrast text on surface (hints, captions, helper text).
  final Color mutedText;

  final Color success;
  final Color warning;
  final Color error;
  final Color onError;

  const AppColors({
    required this.primary,
    required this.onPrimary,
    required this.secondary,
    required this.onSecondary,
    required this.background,
    required this.onBackground,
    required this.surface,
    required this.onSurface,
    required this.surfaceContainer,
    required this.surfaceVariant,
    required this.outline,
    required this.outlineVariant,
    required this.mutedText,
    required this.success,
    required this.warning,
    required this.error,
    required this.onError,
  });

  static const AppColors light = AppColors(
    primary: Color(0xFF0EA5E9),
    onPrimary: Color(0xFFFFFFFF),
    secondary: Color(0xFF1F8E63),
    onSecondary: Color(0xFFFFFFFF),
    background: Color(0xFFF7F7FA),
    onBackground: Color(0xFF0E0F14),
    surface: Color(0xFFFFFFFF),
    onSurface: Color(0xFF0E0F14),
    surfaceContainer: Color(0xFFEFEFF4),
    surfaceVariant: Color(0xFFE4E4ED),
    outline: Color(0xFFC9CAD3),
    outlineVariant: Color(0xFFE4E4ED),
    mutedText: Color(0xFF5A5C68),
    success: Color(0xFF22A06B),
    warning: Color(0xFFF5A524),
    error: Color(0xFFE5484D),
    onError: Color(0xFFFFFFFF),
  );

  static const AppColors dark = AppColors(
    primary: Color(0xFF38BDF8),
    onPrimary: Color(0xFF0E0F14),
    secondary: Color(0xFF3DBE8B),
    onSecondary: Color(0xFF0E0F14),
    background: Color(0xFF0E0F14),
    onBackground: Color(0xFFF2F3F7),
    surface: Color(0xFF15171F),
    onSurface: Color(0xFFF2F3F7),
    surfaceContainer: Color(0xFF1B1E27),
    surfaceVariant: Color(0xFF252834),
    outline: Color(0xFF3A3D49),
    outlineVariant: Color(0xFF252834),
    mutedText: Color(0xFFA6A9B5),
    success: Color(0xFF3DBE8B),
    warning: Color(0xFFFFB454),
    error: Color(0xFFFF6369),
    onError: Color(0xFF0E0F14),
  );

  @override
  AppColors copyWith({
    Color? primary,
    Color? onPrimary,
    Color? secondary,
    Color? onSecondary,
    Color? background,
    Color? onBackground,
    Color? surface,
    Color? onSurface,
    Color? surfaceContainer,
    Color? surfaceVariant,
    Color? outline,
    Color? outlineVariant,
    Color? mutedText,
    Color? success,
    Color? warning,
    Color? error,
    Color? onError,
  }) {
    return AppColors(
      primary: primary ?? this.primary,
      onPrimary: onPrimary ?? this.onPrimary,
      secondary: secondary ?? this.secondary,
      onSecondary: onSecondary ?? this.onSecondary,
      background: background ?? this.background,
      onBackground: onBackground ?? this.onBackground,
      surface: surface ?? this.surface,
      onSurface: onSurface ?? this.onSurface,
      surfaceContainer: surfaceContainer ?? this.surfaceContainer,
      surfaceVariant: surfaceVariant ?? this.surfaceVariant,
      outline: outline ?? this.outline,
      outlineVariant: outlineVariant ?? this.outlineVariant,
      mutedText: mutedText ?? this.mutedText,
      success: success ?? this.success,
      warning: warning ?? this.warning,
      error: error ?? this.error,
      onError: onError ?? this.onError,
    );
  }

  @override
  AppColors lerp(ThemeExtension<AppColors>? other, double t) {
    if (other is! AppColors) return this;
    return AppColors(
      primary: Color.lerp(primary, other.primary, t)!,
      onPrimary: Color.lerp(onPrimary, other.onPrimary, t)!,
      secondary: Color.lerp(secondary, other.secondary, t)!,
      onSecondary: Color.lerp(onSecondary, other.onSecondary, t)!,
      background: Color.lerp(background, other.background, t)!,
      onBackground: Color.lerp(onBackground, other.onBackground, t)!,
      surface: Color.lerp(surface, other.surface, t)!,
      onSurface: Color.lerp(onSurface, other.onSurface, t)!,
      surfaceContainer: Color.lerp(surfaceContainer, other.surfaceContainer, t)!,
      surfaceVariant: Color.lerp(surfaceVariant, other.surfaceVariant, t)!,
      outline: Color.lerp(outline, other.outline, t)!,
      outlineVariant: Color.lerp(outlineVariant, other.outlineVariant, t)!,
      mutedText: Color.lerp(mutedText, other.mutedText, t)!,
      success: Color.lerp(success, other.success, t)!,
      warning: Color.lerp(warning, other.warning, t)!,
      error: Color.lerp(error, other.error, t)!,
      onError: Color.lerp(onError, other.onError, t)!,
    );
  }
}
