import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../../models/active_plan.dart';
import '../../providers/plan_provider.dart';
import '../../theme/app_spacing.dart';
import '../../widgets/plan/plan_schedule.dart';
import '../../widgets/ui/app_button.dart';
import 'coach_chat_screen.dart';

/// Fitness Planning — shows the active plan the coach built (full 7-day detail),
/// or, if there's none yet, routes the user to the coach to create one. The
/// coach is always reachable to adjust the plan; we reload on return.
class PlanningTab extends StatelessWidget {
  const PlanningTab({super.key});

  @override
  Widget build(BuildContext context) {
    return ChangeNotifierProvider(
      create: (_) => PlanProvider()..load(),
      child: const _PlanningView(),
    );
  }
}

class _PlanningView extends StatelessWidget {
  const _PlanningView();

  Future<void> _openCoach(BuildContext context) async {
    final prov = context.read<PlanProvider>();
    await Navigator.of(context).push(
      MaterialPageRoute(builder: (_) => const CoachChatScreen()),
    );
    await prov.load(); // reflect any plan the coach created/edited
  }

  @override
  Widget build(BuildContext context) {
    final prov = context.watch<PlanProvider>();
    return Scaffold(
      appBar: AppBar(
        title: const Text('Fitness Planning'),
        actions: [
          if (prov.plan != null)
            IconButton(
              tooltip: 'Talk to coach',
              icon: const Icon(Icons.chat_bubble_outline),
              onPressed: () => _openCoach(context),
            ),
        ],
      ),
      body: prov.loading
          ? const Center(child: CircularProgressIndicator())
          : prov.plan == null
              ? _CoachCta(onOpen: () => _openCoach(context))
              : _PlanView(plan: prov.plan!, onAdjust: () => _openCoach(context)),
    );
  }
}

class _CoachCta extends StatelessWidget {
  const _CoachCta({required this.onOpen});

  final VoidCallback onOpen;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final cs = theme.colorScheme;
    return SafeArea(
      child: Padding(
        padding: const EdgeInsets.all(AppSpacing.lg),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            const SizedBox(height: AppSpacing.xl),
            Icon(Icons.auto_awesome, size: 64, color: cs.primary),
            const SizedBox(height: AppSpacing.lg),
            Text('Build your plan with your coach',
                textAlign: TextAlign.center,
                style: theme.textTheme.headlineSmall
                    ?.copyWith(fontWeight: FontWeight.w700)),
            const SizedBox(height: AppSpacing.sm),
            Text(
              'Your AI coach builds a personalized plan from your profile, works '
              'around injuries and equipment, and adjusts it whenever you ask.',
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
              onPressed: onOpen,
            ),
            const SizedBox(height: AppSpacing.lg),
          ],
        ),
      ),
    );
  }
}

class _PlanView extends StatelessWidget {
  const _PlanView({required this.plan, required this.onAdjust});

  final ActivePlan plan;
  final VoidCallback onAdjust;

  @override
  Widget build(BuildContext context) {
    return ListView(
      padding: const EdgeInsets.all(AppSpacing.lg),
      children: [
        PlanScheduleView(plan: plan),
        const SizedBox(height: AppSpacing.sm),
        AppButton(
          label: 'Adjust with coach',
          icon: Icons.chat_bubble_outline,
          variant: AppButtonVariant.ghost,
          expand: true,
          onPressed: onAdjust,
        ),
        const SizedBox(height: AppSpacing.lg),
      ],
    );
  }
}
