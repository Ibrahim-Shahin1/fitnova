import 'package:flutter/material.dart';

import '../../models/active_plan.dart';
import '../../theme/app_spacing.dart';
import '../ui/app_card.dart';

/// The canonical "clean schedule" rendering of an active plan — an optional
/// header (title, training-day count, personalization notes) followed by one
/// card per day. Non-scrolling (a [Column]) so it can sit inside either the
/// Planning tab's [ListView] or the coach chat's one-time reveal bubble, so both
/// surfaces render the plan identically.
class PlanScheduleView extends StatelessWidget {
  const PlanScheduleView({super.key, required this.plan, this.showHeader = true});

  final ActivePlan plan;
  final bool showHeader;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final cs = theme.colorScheme;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      mainAxisSize: MainAxisSize.min,
      children: [
        if (showHeader) ...[
          Text(plan.programTitle,
              style: theme.textTheme.titleLarge
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
                        style:
                            theme.textTheme.bodySmall?.copyWith(height: 1.35)),
                  ),
                ],
              ),
            ),
          ],
          const SizedBox(height: AppSpacing.lg),
        ],
        for (final day in plan.days)
          Padding(
            padding: const EdgeInsets.only(bottom: AppSpacing.md),
            child: _DayCard(day: day),
          ),
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
