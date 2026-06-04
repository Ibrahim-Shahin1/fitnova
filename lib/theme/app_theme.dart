import 'package:flutter/material.dart';

import 'app_colors.dart';
import 'app_spacing.dart';
import 'app_typography.dart';

/// Composes `ThemeData` for both modes from the design tokens. Screens should
/// read colors via `Theme.of(context).colorScheme` and custom tokens
/// (success/warning/etc.) via `Theme.of(context).extension<AppColors>()!`.
class AppTheme {
  AppTheme._();

  static ThemeData light = _build(AppColors.light, Brightness.light);
  static ThemeData dark = _build(AppColors.dark, Brightness.dark);

  static ThemeData _build(AppColors c, Brightness brightness) {
    final scheme = ColorScheme.fromSeed(
      seedColor: c.primary,
      brightness: brightness,
    ).copyWith(
      primary: c.primary,
      onPrimary: c.onPrimary,
      secondary: c.secondary,
      onSecondary: c.onSecondary,
      error: c.error,
      onError: c.onError,
      surface: c.background,
      onSurface: c.onBackground,
      surfaceContainerLowest: c.surface,
      surfaceContainerLow: c.surface,
      surfaceContainer: c.surfaceContainer,
      surfaceContainerHigh: c.surfaceVariant,
      surfaceContainerHighest: c.surfaceVariant,
      onSurfaceVariant: c.mutedText,
      outline: c.outline,
      outlineVariant: c.outlineVariant,
    );

    final textTheme = AppTypography.textTheme(scheme);

    return ThemeData(
      useMaterial3: true,
      brightness: brightness,
      colorScheme: scheme,
      textTheme: textTheme,
      fontFamily: AppTypography.fontFamily,
      scaffoldBackgroundColor: c.background,
      extensions: [c],

      appBarTheme: AppBarTheme(
        backgroundColor: c.background,
        foregroundColor: c.onBackground,
        surfaceTintColor: Colors.transparent,
        elevation: 0,
        scrolledUnderElevation: 0,
        centerTitle: false,
        titleTextStyle: textTheme.titleLarge?.copyWith(color: c.onBackground),
      ),

      cardTheme: CardTheme(
        color: c.surface,
        surfaceTintColor: Colors.transparent,
        elevation: AppElevation.e1,
        margin: EdgeInsets.zero,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(AppRadius.rl),
        ),
      ),

      filledButtonTheme: FilledButtonThemeData(
        style: FilledButton.styleFrom(
          minimumSize: const Size.fromHeight(48),
          padding: const EdgeInsets.symmetric(
            horizontal: AppSpacing.lg,
            vertical: AppSpacing.md,
          ),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(AppRadius.rl),
          ),
          textStyle: textTheme.labelLarge,
        ),
      ),

      elevatedButtonTheme: ElevatedButtonThemeData(
        style: ElevatedButton.styleFrom(
          minimumSize: const Size.fromHeight(48),
          padding: const EdgeInsets.symmetric(
            horizontal: AppSpacing.lg,
            vertical: AppSpacing.md,
          ),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(AppRadius.rl),
          ),
          textStyle: textTheme.labelLarge,
        ),
      ),

      outlinedButtonTheme: OutlinedButtonThemeData(
        style: OutlinedButton.styleFrom(
          minimumSize: const Size.fromHeight(48),
          padding: const EdgeInsets.symmetric(
            horizontal: AppSpacing.lg,
            vertical: AppSpacing.md,
          ),
          side: BorderSide(color: c.outline, width: 1.5),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(AppRadius.rl),
          ),
          textStyle: textTheme.labelLarge,
        ),
      ),

      textButtonTheme: TextButtonThemeData(
        style: TextButton.styleFrom(
          padding: const EdgeInsets.symmetric(
            horizontal: AppSpacing.md,
            vertical: AppSpacing.sm,
          ),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(AppRadius.rm),
          ),
          textStyle: textTheme.labelLarge,
        ),
      ),

      inputDecorationTheme: InputDecorationTheme(
        filled: true,
        fillColor: c.surface,
        contentPadding: const EdgeInsets.symmetric(
          horizontal: AppSpacing.md,
          vertical: AppSpacing.md,
        ),
        hintStyle: textTheme.bodyMedium?.copyWith(color: c.mutedText),
        labelStyle: textTheme.bodyMedium?.copyWith(color: c.mutedText),
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(AppRadius.rm),
          borderSide: BorderSide(color: c.outline),
        ),
        enabledBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(AppRadius.rm),
          borderSide: BorderSide(color: c.outline),
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(AppRadius.rm),
          borderSide: BorderSide(color: c.primary, width: 2),
        ),
        errorBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(AppRadius.rm),
          borderSide: BorderSide(color: c.error),
        ),
        focusedErrorBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(AppRadius.rm),
          borderSide: BorderSide(color: c.error, width: 2),
        ),
      ),

      bottomSheetTheme: BottomSheetThemeData(
        backgroundColor: c.surface,
        surfaceTintColor: Colors.transparent,
        elevation: AppElevation.e2,
        shape: const RoundedRectangleBorder(
          borderRadius: BorderRadius.vertical(
            top: Radius.circular(AppRadius.rxl),
          ),
        ),
      ),

      dialogTheme: DialogTheme(
        backgroundColor: c.surface,
        surfaceTintColor: Colors.transparent,
        elevation: AppElevation.e2,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(AppRadius.rl),
        ),
      ),

      chipTheme: ChipThemeData(
        backgroundColor: c.surfaceContainer,
        selectedColor: c.primary,
        labelStyle: textTheme.labelMedium?.copyWith(color: c.onSurface),
        secondaryLabelStyle: textTheme.labelMedium?.copyWith(color: c.onPrimary),
        padding: const EdgeInsets.symmetric(
          horizontal: AppSpacing.md,
          vertical: AppSpacing.xs,
        ),
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(AppRadius.pill),
        ),
        side: BorderSide.none,
      ),

      dividerTheme: DividerThemeData(
        color: c.outlineVariant,
        thickness: 1,
        space: 1,
      ),

      snackBarTheme: SnackBarThemeData(
        backgroundColor: c.onSurface,
        contentTextStyle: textTheme.bodyMedium?.copyWith(color: c.surface),
        behavior: SnackBarBehavior.floating,
        elevation: AppElevation.e2,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(AppRadius.rm),
        ),
      ),

      iconTheme: IconThemeData(color: c.onBackground, size: 24),

      progressIndicatorTheme: ProgressIndicatorThemeData(
        color: c.primary,
      ),
    );
  }
}
