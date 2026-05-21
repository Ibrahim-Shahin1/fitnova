import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../../providers/auth_provider.dart';

/// Top-level router: shows authenticated vs unauthenticated UI based on the
/// Supabase session. Both branches are scaffolding placeholders for now — real
/// sign-in screens arrive in the next unit, the bottom-nav shell after that.
class AuthGate extends StatelessWidget {
  const AuthGate({super.key});

  @override
  Widget build(BuildContext context) {
    final auth = context.watch<AuthProvider>();
    if (auth.isAuthenticated) {
      return _SignedInPlaceholder(email: auth.email);
    }
    return const _SignedOutPlaceholder();
  }
}

class _SignedOutPlaceholder extends StatelessWidget {
  const _SignedOutPlaceholder();

  @override
  Widget build(BuildContext context) {
    return const Scaffold(
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Text('FitNova',
                style: TextStyle(fontSize: 32, fontWeight: FontWeight.w800)),
            SizedBox(height: 8),
            Text('Not signed in'),
            SizedBox(height: 4),
            Text('(sign-in screen — next unit)',
                style: TextStyle(color: Colors.grey)),
          ],
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
