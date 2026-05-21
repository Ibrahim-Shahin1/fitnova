import 'package:flutter/material.dart';

import '../../theme/app_spacing.dart';
import '../../widgets/ui/app_button.dart';

/// Animated brand splash shown to signed-out users: the logo fades in at full
/// size, then scales down and lifts up, then a tagline + "Let's Begin" button
/// fade in. Tapping the button hands control to the parent (→ sign-in).
class WelcomeScreen extends StatefulWidget {
  const WelcomeScreen({super.key, required this.onBegin});

  final VoidCallback onBegin;

  @override
  State<WelcomeScreen> createState() => _WelcomeScreenState();
}

class _WelcomeScreenState extends State<WelcomeScreen>
    with SingleTickerProviderStateMixin {
  /// Fixed brand backdrop (matches the dark logo lockup), independent of theme.
  static const Color _bg = Color(0xFF0B1020);

  late final AnimationController _c;
  late final Animation<double> _logoOpacity;
  late final Animation<double> _logoScale;
  late final Animation<double> _logoShiftY;
  late final Animation<double> _ctaOpacity;
  late final Animation<double> _ctaShiftY;

  @override
  void initState() {
    super.initState();
    _c = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 2200),
    );

    _logoOpacity = CurvedAnimation(
      parent: _c,
      curve: const Interval(0.0, 0.35, curve: Curves.easeOut),
    );

    // Intro pop (1.15→1.0), hold, then shrink (1.0→0.78).
    _logoScale = TweenSequence<double>([
      TweenSequenceItem(
        tween: Tween(begin: 1.15, end: 1.0)
            .chain(CurveTween(curve: Curves.easeOutCubic)),
        weight: 45,
      ),
      TweenSequenceItem(tween: ConstantTween(1.0), weight: 5),
      TweenSequenceItem(
        tween: Tween(begin: 1.0, end: 0.78)
            .chain(CurveTween(curve: Curves.easeOutCubic)),
        weight: 30,
      ),
      TweenSequenceItem(tween: ConstantTween(0.78), weight: 20),
    ]).animate(_c);

    // Lift the logo up during the shrink phase.
    _logoShiftY = TweenSequence<double>([
      TweenSequenceItem(tween: ConstantTween(0.0), weight: 50),
      TweenSequenceItem(
        tween: Tween(begin: 0.0, end: -56.0)
            .chain(CurveTween(curve: Curves.easeOutCubic)),
        weight: 30,
      ),
      TweenSequenceItem(tween: ConstantTween(-56.0), weight: 20),
    ]).animate(_c);

    _ctaOpacity = CurvedAnimation(
      parent: _c,
      curve: const Interval(0.78, 1.0, curve: Curves.easeOut),
    );
    _ctaShiftY = Tween(begin: 28.0, end: 0.0).animate(
      CurvedAnimation(
        parent: _c,
        curve: const Interval(0.78, 1.0, curve: Curves.easeOutCubic),
      ),
    );

    _c.forward();
  }

  @override
  void dispose() {
    _c.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: _bg,
      body: SafeArea(
        child: AnimatedBuilder(
          animation: _c,
          builder: (context, _) {
            return Stack(
              children: [
                Align(
                  alignment: const Alignment(0, -0.1),
                  child: Transform.translate(
                    offset: Offset(0, _logoShiftY.value),
                    child: Opacity(
                      opacity: _logoOpacity.value,
                      child: Transform.scale(
                        scale: _logoScale.value,
                        child: const Padding(
                          padding:
                              EdgeInsets.symmetric(horizontal: AppSpacing.xxl),
                          child: Image(
                            image: AssetImage('assets/logo/wordmark.png'),
                            fit: BoxFit.contain,
                          ),
                        ),
                      ),
                    ),
                  ),
                ),
                Align(
                  alignment: const Alignment(0, 0.78),
                  child: Opacity(
                    opacity: _ctaOpacity.value,
                    child: Transform.translate(
                      offset: Offset(0, _ctaShiftY.value),
                      child: Padding(
                        padding:
                            const EdgeInsets.symmetric(horizontal: AppSpacing.xl),
                        child: Column(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            const Text(
                              'Train smarter. Lift with confidence.',
                              textAlign: TextAlign.center,
                              style: TextStyle(
                                color: Colors.white70,
                                fontSize: 15,
                                height: 1.4,
                              ),
                            ),
                            const SizedBox(height: AppSpacing.lg),
                            AppButton(
                              label: "Let's Begin",
                              onPressed: widget.onBegin,
                              expand: true,
                              size: AppButtonSize.lg,
                            ),
                          ],
                        ),
                      ),
                    ),
                  ),
                ),
              ],
            );
          },
        ),
      ),
    );
  }
}
