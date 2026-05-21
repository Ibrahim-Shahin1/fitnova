import 'package:flutter/material.dart';

import 'sign_in_screen.dart';
import 'welcome_screen.dart';

/// Signed-out experience: the animated welcome splash, swapping to the sign-in
/// screen when the user taps "Let's Begin". Kept as a single subtree (not pushed
/// routes) so that when auth state flips to signed-in, AuthGate cleanly replaces
/// the whole flow with the app home — nothing left stacked on top.
class AuthFlow extends StatefulWidget {
  const AuthFlow({super.key});

  @override
  State<AuthFlow> createState() => _AuthFlowState();
}

class _AuthFlowState extends State<AuthFlow> {
  bool _showSignIn = false;

  @override
  Widget build(BuildContext context) {
    return AnimatedSwitcher(
      duration: const Duration(milliseconds: 350),
      child: _showSignIn
          ? const SignInScreen(key: ValueKey('signin'))
          : WelcomeScreen(
              key: const ValueKey('welcome'),
              onBegin: () => setState(() => _showSignIn = true),
            ),
    );
  }
}
