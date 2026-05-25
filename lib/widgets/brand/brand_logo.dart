import 'package:flutter/material.dart';
import 'package:flutter_svg/flutter_svg.dart';

/// The full FITNOVA "split bar" wordmark. Use sparingly — splash, auth screens.
///
/// Picks the white-ink (`_dark`) or dark-ink (`_light`) variant automatically
/// from the theme brightness. Pass [onDark] to force one (e.g. the splash,
/// which has a fixed dark backdrop regardless of theme).
class FitNovaWordmark extends StatelessWidget {
  const FitNovaWordmark({super.key, this.height, this.onDark});

  final double? height;
  final bool? onDark;

  @override
  Widget build(BuildContext context) {
    final dark = onDark ?? Theme.of(context).brightness == Brightness.dark;
    return SvgPicture.asset(
      dark ? 'assets/logo/wordmark_dark.svg' : 'assets/logo/wordmark_light.svg',
      height: height,
      fit: BoxFit.contain,
    );
  }
}

/// The compact Barbell-F mark. Use constantly — app bars, nav, tight spots.
///
/// Picks the white-ink (`_dark`) or dark-ink (`_light`) variant from the theme
/// brightness, or force it with [onDark].
class FitNovaMark extends StatelessWidget {
  const FitNovaMark({super.key, this.size, this.onDark});

  final double? size;
  final bool? onDark;

  @override
  Widget build(BuildContext context) {
    final dark = onDark ?? Theme.of(context).brightness == Brightness.dark;
    return SvgPicture.asset(
      dark ? 'assets/logo/logo_dark.svg' : 'assets/logo/logo_light.svg',
      width: size,
      height: size,
      fit: BoxFit.contain,
    );
  }
}
