import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../../providers/auth_provider.dart';
import '../../providers/profile_provider.dart';
import '../../widgets/brand/brand_logo.dart';
import '../onboarding/onboarding_flow.dart';
import 'auth_flow.dart';
import 'new_password_screen.dart';

/// Top-level reactive router.
///
///   recovering password → NewPasswordScreen
///   not signed in       → AuthFlow (welcome → sign-in)
///   signed in, loading   → branded splash
///   signed in, !onboarded → OnboardingFlow
///   signed in, onboarded → home (placeholder until the shell unit)
class AuthGate extends StatelessWidget {
  const AuthGate({super.key});

  @override
  Widget build(BuildContext context) {
    final auth = context.watch<AuthProvider>();

    if (auth.isRecoveringPassword) {
      return const NewPasswordScreen();
    }
    if (!auth.isAuthenticated) {
      return const AuthFlow();
    }

    final profile = context.watch<ProfileProvider>();
    if (profile.error != null) {
      return _ProfileLoadError(
        onRetry: () {
          final uid = auth.user?.id;
          if (uid != null) context.read<ProfileProvider>().load(uid);
        },
      );
    }
    if (profile.loading || profile.profile == null) {
      return const _BrandSplash();
    }
    if (!profile.profile!.onboardingCompleted) {
      return const OnboardingFlow();
    }
    return _SignedInPlaceholder(email: auth.email);
  }
}

class _BrandSplash extends StatelessWidget {
  const _BrandSplash();

  @override
  Widget build(BuildContext context) {
    return const Scaffold(
      backgroundColor: Color(0xFF0B1020),
      body: Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            FitNovaMark(onDark: true, size: 96),
            SizedBox(height: 28),
            SizedBox(
              width: 22,
              height: 22,
              child: CircularProgressIndicator(strokeWidth: 2),
            ),
          ],
        ),
      ),
    );
  }
}

class _ProfileLoadError extends StatelessWidget {
  const _ProfileLoadError({required this.onRetry});

  final VoidCallback onRetry;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              const Icon(Icons.cloud_off_outlined, size: 48),
              const SizedBox(height: 12),
              const Text("Couldn't load your profile.",
                  textAlign: TextAlign.center),
              const SizedBox(height: 16),
              FilledButton(onPressed: onRetry, child: const Text('Retry')),
            ],
          ),
        ),
      ),
    );
  }
}

class _SignedInPlaceholder extends StatelessWidget {
  const _SignedInPlaceholder({required this.email});

  final String? email;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('FitNova')),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const Icon(Icons.check_circle, color: Colors.green, size: 48),
            const SizedBox(height: 12),
            Text('Signed in as ${email ?? "unknown"}'),
            const SizedBox(height: 16),
            FilledButton(
              onPressed: () => context.read<AuthProvider>().signOut(),
              child: const Text('Sign out'),
            ),
          ],
        ),
      ),
    );
  }
}
