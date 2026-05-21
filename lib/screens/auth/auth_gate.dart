import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../../providers/auth_provider.dart';
import 'sign_in_screen.dart';

/// Top-level router: shows authenticated vs unauthenticated UI based on the
/// Supabase session. Unauthenticated → sign-in. Authenticated → placeholder for
/// now; the bottom-nav shell replaces it in a later unit.
class AuthGate extends StatelessWidget {
  const AuthGate({super.key});

  @override
  Widget build(BuildContext context) {
    final auth = context.watch<AuthProvider>();
    if (auth.isAuthenticated) {
      return _SignedInPlaceholder(email: auth.email);
    }
    return const SignInScreen();
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
