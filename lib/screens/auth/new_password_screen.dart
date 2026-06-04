import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:supabase_flutter/supabase_flutter.dart';

import '../../providers/auth_provider.dart';
import '../../theme/app_spacing.dart';
import '../../widgets/ui/app_button.dart';
import '../../widgets/ui/app_text_field.dart';

/// Shown by AuthGate while AuthProvider is in password-recovery mode (entered
/// via the reset deep link). Setting the password clears recovery mode and the
/// active session routes the user to home.
class NewPasswordScreen extends StatefulWidget {
  const NewPasswordScreen({super.key});

  @override
  State<NewPasswordScreen> createState() => _NewPasswordScreenState();
}

class _NewPasswordScreenState extends State<NewPasswordScreen> {
  final _formKey = GlobalKey<FormState>();
  final _password = TextEditingController();
  final _confirm = TextEditingController();
  bool _obscure = true;
  bool _loading = false;

  @override
  void dispose() {
    _password.dispose();
    _confirm.dispose();
    super.dispose();
  }

  Future<void> _submit() async {
    if (!_formKey.currentState!.validate()) return;
    setState(() => _loading = true);
    try {
      await context.read<AuthProvider>().completePasswordReset(_password.text);
      // Success → recovery mode clears + session active → AuthGate → home.
    } on AuthException catch (e) {
      if (mounted) _showError(e.message);
    } catch (_) {
      if (mounted) _showError("Couldn't update your password. Try again.");
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  void _showError(String msg) => ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(msg),
          backgroundColor: Theme.of(context).colorScheme.error,
        ),
      );

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Scaffold(
      body: SafeArea(
        child: Center(
          child: SingleChildScrollView(
            padding: const EdgeInsets.all(AppSpacing.lg),
            child: ConstrainedBox(
              constraints: const BoxConstraints(maxWidth: 420),
              child: Form(
                key: _formKey,
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Text(
                      'Set a new password',
                      textAlign: TextAlign.center,
                      style: theme.textTheme.headlineSmall
                          ?.copyWith(fontWeight: FontWeight.w700),
                    ),
                    const SizedBox(height: AppSpacing.xl),
                    AppTextField(
                      controller: _password,
                      label: 'New password',
                      prefixIcon: Icons.lock_outline,
                      obscure: _obscure,
                      textInputAction: TextInputAction.next,
                      suffix: IconButton(
                        icon: Icon(_obscure
                            ? Icons.visibility_off_outlined
                            : Icons.visibility_outlined),
                        onPressed: () => setState(() => _obscure = !_obscure),
                      ),
                      validator: (v) => (v == null || v.length < 6)
                          ? 'At least 6 characters'
                          : null,
                    ),
                    const SizedBox(height: AppSpacing.md),
                    AppTextField(
                      controller: _confirm,
                      label: 'Confirm new password',
                      prefixIcon: Icons.lock_outline,
                      obscure: _obscure,
                      textInputAction: TextInputAction.done,
                      onSubmitted: (_) => _submit(),
                      validator: (v) =>
                          (v != _password.text) ? 'Passwords do not match' : null,
                    ),
                    const SizedBox(height: AppSpacing.lg),
                    AppButton(
                      label: 'Update password',
                      onPressed: _loading ? null : _submit,
                      isLoading: _loading,
                      expand: true,
                      size: AppButtonSize.lg,
                    ),
                  ],
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }
}
