import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../../providers/auth_provider.dart';
import '../../services/auth_service.dart';
import '../../theme/app_spacing.dart';
import '../../widgets/ui/app_button.dart';

/// Shown after sign-up: prompts the user to confirm via the emailed link.
///
/// If the deep link confirms the account while this screen is open (auth flips
/// to signed-in), it pops the auth stack so AuthGate can route to home.
class EmailVerifyPendingScreen extends StatefulWidget {
  const EmailVerifyPendingScreen({super.key, required this.email});

  final String email;

  @override
  State<EmailVerifyPendingScreen> createState() =>
      _EmailVerifyPendingScreenState();
}

class _EmailVerifyPendingScreenState extends State<EmailVerifyPendingScreen> {
  bool _resending = false;

  Future<void> _resend() async {
    setState(() => _resending = true);
    try {
      await AuthService.resendConfirmation(widget.email);
      _toast('Confirmation email re-sent to ${widget.email}');
    } catch (_) {
      _toast("Couldn't resend right now. Try again shortly.");
    } finally {
      if (mounted) setState(() => _resending = false);
    }
  }

  void _toast(String msg) {
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(msg)));
  }

  @override
  Widget build(BuildContext context) {
    // If the email link confirms while we're here, leave the auth stack so
    // AuthGate (the reactive root) can show the app home.
    if (context.watch<AuthProvider>().isAuthenticated) {
      WidgetsBinding.instance.addPostFrameCallback((_) {
        if (mounted) Navigator.of(context).popUntil((r) => r.isFirst);
      });
    }

    final theme = Theme.of(context);
    return Scaffold(
      appBar: AppBar(),
      body: SafeArea(
        child: Center(
          child: SingleChildScrollView(
            padding: const EdgeInsets.all(AppSpacing.lg),
            child: ConstrainedBox(
              constraints: const BoxConstraints(maxWidth: 420),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Icon(Icons.mark_email_unread_outlined,
                      size: 56, color: theme.colorScheme.primary),
                  const SizedBox(height: AppSpacing.lg),
                  Text(
                    'Check your email',
                    textAlign: TextAlign.center,
                    style: theme.textTheme.headlineSmall
                        ?.copyWith(fontWeight: FontWeight.w700),
                  ),
                  const SizedBox(height: AppSpacing.sm),
                  Text(
                    'We sent a confirmation link to ${widget.email}. '
                    'Open it to activate your account, then come back and sign in.',
                    textAlign: TextAlign.center,
                    style: theme.textTheme.bodyMedium
                        ?.copyWith(color: theme.colorScheme.onSurfaceVariant),
                  ),
                  const SizedBox(height: AppSpacing.xl),
                  AppButton(
                    label: 'Resend email',
                    onPressed: _resending ? null : _resend,
                    isLoading: _resending,
                    variant: AppButtonVariant.ghost,
                    expand: true,
                  ),
                  const SizedBox(height: AppSpacing.sm),
                  AppButton(
                    label: 'Back to sign in',
                    onPressed: () =>
                        Navigator.of(context).popUntil((r) => r.isFirst),
                    expand: true,
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}
