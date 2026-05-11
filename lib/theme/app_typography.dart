import 'package:flutter/material.dart';

/// Typography for FitNova. Built on **Sora** (vendored in `assets/fonts/`).
///
/// Sora was chosen for its strong 700/800 weights — display sizes (rep counts,
/// scores) need to read as athletic and chunky, and Sora's numerals are heavy.
class AppTypography {
  AppTypography._();

  static const String fontFamily = 'Sora';

  /// Build a `TextTheme` keyed off the supplied `ColorScheme`. Body text uses
  /// `cs.onSurface`; muted styles use `cs.onSurfaceVariant`.
  static TextTheme textTheme(ColorScheme cs) {
    final onSurface = cs.onSurface;
    final muted = cs.onSurfaceVariant;

    TextStyle base({
      required double size,
      required FontWeight weight,
      double height = 1.4,
      double letterSpacing = 0,
      Color? color,
    }) {
      return TextStyle(
        fontFamily: fontFamily,
        fontSize: size,
        fontWeight: weight,
        height: height,
        letterSpacing: letterSpacing,
        color: color ?? onSurface,
      );
    }

    return TextTheme(
      // Display — big numerics (score, rep counter)
      displayLarge: base(size: 48, weight: FontWeight.w800, height: 1.1),
      displayMedium: base(size: 36, weight: FontWeight.w700, height: 1.15),
      displaySmall: base(size: 30, weight: FontWeight.w700, height: 1.2),

      // Headline — screen titles, hero text
      headlineLarge: base(size: 28, weight: FontWeight.w700, height: 1.25),
      headlineMedium: base(size: 22, weight: FontWeight.w700, height: 1.3),
      headlineSmall: base(size: 20, weight: FontWeight.w600, height: 1.3),

      // Title — card titles, section headers
      titleLarge: base(size: 18, weight: FontWeight.w600, height: 1.35),
      titleMedium: base(size: 16, weight: FontWeight.w600, height: 1.4),
      titleSmall: base(size: 14, weight: FontWeight.w600, height: 1.4),

      // Body — primary reading text
      bodyLarge: base(size: 15, weight: FontWeight.w400, height: 1.5),
      bodyMedium: base(size: 14, weight: FontWeight.w400, height: 1.5),
      bodySmall: base(size: 12, weight: FontWeight.w400, height: 1.5, color: muted),

      // Label — buttons, chips, tabs
      labelLarge: base(size: 14, weight: FontWeight.w600, letterSpacing: 0.3),
      labelMedium: base(size: 12, weight: FontWeight.w600, letterSpacing: 0.4),
      labelSmall: base(size: 11, weight: FontWeight.w600, letterSpacing: 0.5, color: muted),
    );
  }
}
