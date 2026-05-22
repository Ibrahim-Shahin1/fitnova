import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../../models/active_plan.dart';
import '../../providers/plan_provider.dart';
import '../../theme/app_spacing.dart';
import '../../widgets/ui/app_button.dart';
import '../../widgets/ui/app_card.dart';
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
    final theme = Theme.of(context);
    final cs = theme.colorScheme;
    return ListView(
      padding: const EdgeInsets.all(AppSpacing.lg),
      children: [
        Text(plan.programTitle,
            style: theme.textTheme.headlineSmall
                ?.copyWith(fontWeight: FontWeight.w800)),
        const SizedBox(height: AppSpacing.xs),
        Text('${plan.trainingDayCount} training days · 7-day split',
            style: theme.textTheme.bodySmall
                ?.copyWith(color: cs.onSurfaceVariant)),
        if (plan.personalizationNotes != null &&
            plan.personalizationNotes!.isNotEmpty) ...[
          const SizedBox(height: AppSpacing.md),
          AppCard(
            accentColor: cs.primary,
            child: Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Icon(Icons.auto_awesome, size: 18, color: cs.primary),
                const SizedBox(width: AppSpacing.sm),
                Expanded(
                  child: Text(plan.personalizationNotes!,
                      style: theme.textTheme.bodySmall?.copyWith(height: 1.35)),
                ),
              ],
            ),
          ),
        ],
        const SizedBox(height: AppSpacing.lg),
        for (final day in plan.days)
          Padding(
            padding: const EdgeInsets.only(bottom: AppSpacing.md),
            child: _DayCard(day: day),
          ),
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

class _DayCard extends StatelessWidget {
  const _DayCard({required this.day});

  final PlanDay day;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final cs = theme.colorScheme;
    return AppCard(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Text('DAY ${day.dayNumber}',
                  style: theme.textTheme.labelMedium?.copyWith(
                    color: cs.primary,
                    fontWeight: FontWeight.w800,
                    letterSpacing: 0.5,
                  )),
              const SizedBox(width: AppSpacing.sm),
              Expanded(
                child: Text(day.focus,
                    style: theme.textTheme.titleMedium
                        ?.copyWith(fontWeight: FontWeight.w700)),
              ),
              if (day.isRestDay)
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
                  decoration: BoxDecoration(
                    color: cs.surfaceContainerHighest,
                    borderRadius: BorderRadius.circular(AppRadius.pill),
                  ),
                  child: Text('REST',
                      style: theme.textTheme.labelSmall?.copyWith(
                        color: cs.onSurfaceVariant,
                        fontWeight: FontWeight.w700,
                      )),
                ),
            ],
          ),
          if (!day.isRestDay && day.exercises.isNotEmpty) ...[
            const Divider(height: AppSpacing.lg),
            for (final ex in day.exercises) _ExerciseRow(ex: ex),
          ] else if (day.isRestDay) ...[
            const SizedBox(height: AppSpacing.xs),
            Text('Recovery / mobility',
                style: theme.textTheme.bodySmall
                    ?.copyWith(color: cs.onSurfaceVariant)),
          ],
        ],
      ),
    );
  }
}

class _ExerciseRow extends StatelessWidget {
  const _ExerciseRow({required this.ex});

  final PlanExercise ex;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final cs = theme.colorScheme;
    final cue = ex.coachingCue;
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 6),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Expanded(
                child: Text(ex.name,
                    style: theme.textTheme.bodyMedium
                        ?.copyWith(fontWeight: FontWeight.w600)),
              ),
              const SizedBox(width: AppSpacing.sm),
              Text('${ex.sets} × ${ex.reps}',
                  style: theme.textTheme.bodyMedium?.copyWith(
                    color: cs.primary,
                    fontWeight: FontWeight.w700,
                  )),
            ],
          ),
          Text(
            cue != null && cue.isNotEmpty
                ? 'Rest ${ex.restSeconds}s · $cue'
                : 'Rest ${ex.restSeconds}s',
            style: theme.textTheme.bodySmall?.copyWith(color: cs.onSurfaceVariant),
          ),
        ],
      ),
    );
  }
}
