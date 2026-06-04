import 'package:flutter/material.dart';

import '../../theme/app_spacing.dart';
import '../../widgets/ui/app_button.dart';

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
              Icon(Icons.analytics_outlined, size: 64, color: cs.secondary),
              const SizedBox(height: AppSpacing.lg),
              Text(
                'Benchmark Inspector',
                textAlign: TextAlign.center,
                style: theme.textTheme.headlineSmall
                    ?.copyWith(fontWeight: FontWeight.w700),
              ),
              const SizedBox(height: AppSpacing.sm),
              Text(
                'Browse benchmark clips from the Fitness-AQA test split, analyze '
                'one, and inspect the model\'s exact confidence per error.',
                textAlign: TextAlign.center,
                style: theme.textTheme.bodyMedium
                    ?.copyWith(color: cs.onSurfaceVariant),
              ),
              const Spacer(),
              AppButton(
                label: 'Browse benchmark clips',
                icon: Icons.folder_open_outlined,
                variant: AppButtonVariant.secondary,
                size: AppButtonSize.lg,
                expand: true,
                onPressed: () =>
                    Navigator.of(context).pushNamed('/benchmark/browse'),
              ),
              const SizedBox(height: AppSpacing.lg),
            ],
          ),
        ),
      ),
    );
  }
}
