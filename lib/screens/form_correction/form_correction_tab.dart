import 'package:flutter/material.dart';

import '../../theme/app_spacing.dart';
import '../../widgets/ui/app_button.dart';

/// Form Correction tab — the entry point into the existing camera/video
/// form-analysis flow. "Start" launches the unchanged exercise-selection →
/// guidelines → live-check / upload → results pipeline via its named route.
class FormCorrectionTab extends StatelessWidget {
  const FormCorrectionTab({super.key});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final cs = theme.colorScheme;

    return Scaffold(
      appBar: AppBar(title: const Text('Form Correction')),
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(AppSpacing.lg),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              const SizedBox(height: AppSpacing.xl),
              Icon(Icons.videocam_outlined, size: 64, color: cs.secondary),
              const SizedBox(height: AppSpacing.lg),
              Text(
                'Check your form',
                textAlign: TextAlign.center,
                style: theme.textTheme.headlineSmall
                    ?.copyWith(fontWeight: FontWeight.w700),
              ),
              const SizedBox(height: AppSpacing.sm),
              Text(
                'Record yourself live or upload a clip of a lift, and get '
                'feedback on specific form errors — grounded in a trained model.',
                textAlign: TextAlign.center,
                style: theme.textTheme.bodyMedium
                    ?.copyWith(color: cs.onSurfaceVariant),
              ),
              const Spacer(),
              AppButton(
                label: 'Start form check',
                icon: Icons.camera_alt_outlined,
                variant: AppButtonVariant.secondary,
                size: AppButtonSize.lg,
                expand: true,
                onPressed: () =>
                    Navigator.of(context).pushNamed('/exercise-select'),
              ),
              const SizedBox(height: AppSpacing.md),
              Text(
                'You\'ll pick an exercise and a camera setup next.',
                textAlign: TextAlign.center,
                style: theme.textTheme.labelSmall
                    ?.copyWith(color: cs.onSurfaceVariant),
              ),
              const SizedBox(height: AppSpacing.lg),
            ],
          ),
        ),
      ),
    );
  }
}
