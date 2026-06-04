import 'package:flutter/material.dart';

import '../../models/meal_plan.dart';
import '../../screens/nutrition/recipe_detail_screen.dart';
import '../../theme/app_spacing.dart';
import '../ui/app_card.dart';

/// Renders a full meal plan: targets header, coaching notes, then per-day
/// expandable cards with meals + per-meal macros. Used both in the post-generate
/// success view and the Nutrition tab's saved-plan view.
class MealPlanView extends StatelessWidget {
  const MealPlanView({super.key, required this.plan});

  final MealPlan plan;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final cs = theme.colorScheme;
    final t = plan.targets;
    return ListView(
      padding: const EdgeInsets.all(AppSpacing.lg),
      children: [
        Text(plan.planTitle,
            style: theme.textTheme.titleLarge?.copyWith(fontWeight: FontWeight.w800)),
        const SizedBox(height: AppSpacing.xs),
        Wrap(spacing: AppSpacing.sm, runSpacing: AppSpacing.xs, children: [
          _Pill(text: '${t.targetCalories} kcal/day', color: cs.primary),
          _Pill(text: 'BMI ${t.bmi.toStringAsFixed(1)}', color: cs.secondary),
          if (plan.qualityScore != null)
            _Pill(text: '${plan.qualityScore}/10', color: cs.tertiary),
        ]),
        const SizedBox(height: AppSpacing.sm),
        // Deterministic verification badge — proves the crew checked every meal
        // against the calorie/macro/diet/allergen constraints.
        if (plan.mealsChecked > 0)
          Container(
            padding: const EdgeInsets.symmetric(
                horizontal: AppSpacing.sm, vertical: AppSpacing.xs),
            decoration: BoxDecoration(
              color: (plan.verifiedClean ? cs.secondary : cs.error)
                  .withValues(alpha: 0.12),
              borderRadius: BorderRadius.circular(AppRadius.rm),
            ),
            child: Row(mainAxisSize: MainAxisSize.min, children: [
              Icon(plan.verifiedClean ? Icons.verified : Icons.warning_amber,
                  size: 15, color: plan.verifiedClean ? cs.secondary : cs.error),
              const SizedBox(width: AppSpacing.xs),
              Flexible(
                child: Text(
                  plan.verifiedClean
                      ? '${plan.mealsChecked} meals checked · all constraints satisfied'
                      : '${plan.violationCount} constraint issue(s) in ${plan.mealsChecked} meals',
                  style: theme.textTheme.labelSmall?.copyWith(
                      color: plan.verifiedClean ? cs.secondary : cs.error,
                      fontWeight: FontWeight.w700),
                ),
              ),
            ]),
          ),
        const SizedBox(height: AppSpacing.md),

        // Macro target summary
        AppCard(
          accentColor: cs.primary,
          child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            Text('Daily targets', style: theme.textTheme.titleSmall?.copyWith(fontWeight: FontWeight.w800)),
            const SizedBox(height: AppSpacing.sm),
            Row(mainAxisAlignment: MainAxisAlignment.spaceBetween, children: [
              _Macro(label: 'Protein', value: '${t.proteinG}g'),
              _Macro(label: 'Carbs', value: '${t.carbsG}g'),
              _Macro(label: 'Fat', value: '${t.fatG}g'),
              _Macro(label: 'Match', value: '${plan.avgCalorieMatchPct.toStringAsFixed(0)}%'),
            ]),
            if (plan.targetsRationale.isNotEmpty) ...[
              const Divider(height: AppSpacing.lg),
              Text(plan.targetsRationale,
                  style: theme.textTheme.bodySmall?.copyWith(color: cs.onSurfaceVariant)),
            ],
          ]),
        ),

        if (plan.coachingNotes.isNotEmpty) ...[
          const SizedBox(height: AppSpacing.md),
          AppCard(
            elevated: true,
            child: Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
              Icon(Icons.tips_and_updates, size: 18, color: cs.primary),
              const SizedBox(width: AppSpacing.sm),
              Expanded(child: Text(plan.coachingNotes, style: theme.textTheme.bodySmall)),
            ]),
          ),
        ],

        const SizedBox(height: AppSpacing.lg),
        Text('Your week', style: theme.textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w700)),
        const SizedBox(height: AppSpacing.sm),
        for (final day in plan.days) _DayCard(day: day),
        const SizedBox(height: AppSpacing.xl),
      ],
    );
  }
}

class _DayCard extends StatelessWidget {
  const _DayCard({required this.day});
  final MealDay day;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final cs = theme.colorScheme;
    return Padding(
      padding: const EdgeInsets.only(bottom: AppSpacing.sm),
      child: AppCard(
        padding: EdgeInsets.zero,
        child: Theme(
          data: theme.copyWith(dividerColor: Colors.transparent),
          child: ExpansionTile(
            initiallyExpanded: day.dayNumber == 1,
            tilePadding: const EdgeInsets.symmetric(horizontal: AppSpacing.md),
            title: Row(children: [
              CircleAvatar(
                radius: 14,
                backgroundColor: cs.primary.withValues(alpha: 0.14),
                child: Text('${day.dayNumber}',
                    style: theme.textTheme.labelMedium
                        ?.copyWith(color: cs.primary, fontWeight: FontWeight.w800)),
              ),
              const SizedBox(width: AppSpacing.sm),
              Expanded(
                child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                  Text(day.theme.isNotEmpty ? day.theme : 'Day ${day.dayNumber}',
                      style: theme.textTheme.titleSmall?.copyWith(fontWeight: FontWeight.w700)),
                  Text('${day.totalCalories} kcal · P${day.proteinG} C${day.carbsG} F${day.fatG}',
                      style: theme.textTheme.labelSmall?.copyWith(color: cs.onSurfaceVariant)),
                ]),
              ),
            ]),
            children: [
              for (final m in day.meals) _MealRow(meal: m),
              const SizedBox(height: AppSpacing.sm),
            ],
          ),
        ),
      ),
    );
  }
}

class _MealRow extends StatelessWidget {
  const _MealRow({required this.meal});
  final Meal meal;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final cs = theme.colorScheme;
    return InkWell(
      onTap: () => Navigator.of(context).push(
        MaterialPageRoute(builder: (_) => RecipeDetailScreen(meal: meal)),
      ),
      child: Padding(
        padding:
            const EdgeInsets.fromLTRB(AppSpacing.md, AppSpacing.xs, AppSpacing.md, AppSpacing.xs),
        child: Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
          SizedBox(
            width: 78,
            child: Text(meal.slot[0].toUpperCase() + meal.slot.substring(1),
                style: theme.textTheme.labelMedium
                    ?.copyWith(color: cs.primary, fontWeight: FontWeight.w700)),
          ),
          Expanded(
            child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
              Text(meal.name, style: theme.textTheme.bodyMedium),
              Text('${meal.calories.round()} kcal · '
                  'P${meal.proteinG.round()} C${meal.carbsG.round()} F${meal.fatG.round()}'
                  '${meal.minutes > 0 ? ' · ${meal.minutes}min' : ''}',
                  style: theme.textTheme.labelSmall?.copyWith(color: cs.onSurfaceVariant)),
            ]),
          ),
          Icon(Icons.chevron_right, size: 18, color: cs.onSurfaceVariant),
        ]),
      ),
    );
  }
}

/// Bottom sheet for one recipe: ingredients + an on-demand
/// "Generate cooking instructions" action backed by the fine-tuned DistilGPT-2.
class _Pill extends StatelessWidget {
  const _Pill({required this.text, required this.color});
  final String text;
  final Color color;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: AppSpacing.sm, vertical: 4),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.14),
        borderRadius: BorderRadius.circular(AppRadius.pill),
      ),
      child: Text(text,
          style: Theme.of(context).textTheme.labelSmall
              ?.copyWith(color: color, fontWeight: FontWeight.w800)),
    );
  }
}

class _Macro extends StatelessWidget {
  const _Macro({required this.label, required this.value});
  final String label;
  final String value;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Column(children: [
      Text(value, style: theme.textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w800)),
      Text(label,
          style: theme.textTheme.labelSmall
              ?.copyWith(color: theme.colorScheme.onSurfaceVariant)),
    ]);
  }
}
