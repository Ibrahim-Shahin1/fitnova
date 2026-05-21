import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../../providers/profile_provider.dart';
import 'profile_setup_screen.dart';
import 'walkthrough_screen.dart';

/// Signed-in-but-not-onboarded experience: profile setup → walkthrough.
/// Finishing flips `onboarding_completed`, and AuthGate reactively routes home.
class OnboardingFlow extends StatefulWidget {
  const OnboardingFlow({super.key});

  @override
  State<OnboardingFlow> createState() => _OnboardingFlowState();
}

class _OnboardingFlowState extends State<OnboardingFlow> {
  int _step = 0; // 0 = profile setup, 1 = walkthrough

  Future<void> _finish() async {
    final provider = context.read<ProfileProvider>();
    final profile = provider.profile;
    if (profile != null) {
      await provider.save(profile.copyWith(onboardingCompleted: true));
      // AuthGate routes to home once onboardingCompleted flips.
    }
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedSwitcher(
      duration: const Duration(milliseconds: 300),
      child: _step == 0
          ? ProfileSetupScreen(
              key: const ValueKey('setup'),
              onDone: () => setState(() => _step = 1),
            )
          : WalkthroughScreen(
              key: const ValueKey('walk'),
              onFinish: _finish,
            ),
    );
  }
}
