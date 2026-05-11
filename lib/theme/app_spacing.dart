import 'package:flutter/animation.dart';

/// Spacing scale (logical pixels). Use these for padding, margin, gaps —
/// avoid free-form `EdgeInsets.all(13)` etc. so the visual rhythm stays
/// consistent across screens.
class AppSpacing {
  AppSpacing._();

  static const double xs = 4;
  static const double sm = 8;
  static const double md = 16;
  static const double lg = 24;
  static const double xl = 32;
  static const double xxl = 48;
  static const double xxxl = 64;
}

/// Corner radius scale. The bold-fitness aesthetic favors chunky radii;
/// `rl` is the default for cards and buttons.
class AppRadius {
  AppRadius._();

  static const double rs = 8;
  static const double rm = 12;
  static const double rl = 16;
  static const double rxl = 24;

  /// Use for fully-rounded pills (chips, tags, FABs).
  static const double pill = 999;
}

/// Material elevation tokens. Dark mode prefers tinted surface containers
/// over drop shadows — see `AppTheme`.
class AppElevation {
  AppElevation._();

  static const double e0 = 0;
  static const double e1 = 2;
  static const double e2 = 6;
  static const double e3 = 12;
}

/// Standard motion durations.
class AppDuration {
  AppDuration._();

  static const Duration fast = Duration(milliseconds: 150);
  static const Duration base = Duration(milliseconds: 250);
  static const Duration slow = Duration(milliseconds: 400);
}

/// Standard easing curves.
class AppCurves {
  AppCurves._();

  static const Curve enter = Curves.easeOutCubic;
  static const Curve exit = Curves.easeIn;
}
