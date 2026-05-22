import 'package:flutter/material.dart';

import '../../theme/app_spacing.dart';
import '../../widgets/ui/app_button.dart';
import 'coach_chat_screen.dart';

/// Fitness Planning — the primary feature. The AI coach is the way you build
/// and manage your plan. (The plan view that shows what the coach built lands
/// in the next unit.)
class PlanningTab extends StatelessWidget {
  const PlanningTab({super.key});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final cs = theme.colorScheme;

    return Scaffold(
      appBar: AppBar(title: const Text('Fitness Planning')),
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(AppSpacing.lg),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              const SizedBox(height: AppSpacing.xl),
              Icon(Icons.auto_awesome, size: 64, color: cs.primary),
              const SizedBox(height: AppSpacing.lg),
              Text(
                'Build your plan with your coach',
                textAlign: TextAlign.center,
                style: theme.textTheme.headlineSmall
                    ?.copyWith(fontWeight: FontWeight.w700),
              ),
              const SizedBox(height: AppSpacing.sm),
              Text(
                'Your AI coach builds a personalized plan from your profile, '
                'works around injuries and equipment, and adjusts it whenever '
                'you ask.',
                textAlign: TextAlign.center,
                style: theme.textTheme.bodyMedium
                    ?.copyWith(color: cs.onSurfaceVariant),
              ),
              const Spacer(),
              AppButton(
                label: 'Talk to your coach',
                icon: Icons.chat_bubble_outline,
                size: AppButtonSize.lg,
                expand: true,
                onPressed: () => Navigator.of(context).push(
                  MaterialPageRoute(builder: (_) => const CoachChatScreen()),
                ),
              ),
              const SizedBox(height: AppSpacing.lg),
            ],
          ),
        ),
      ),
    );
  }
}
