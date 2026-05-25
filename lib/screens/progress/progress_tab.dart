import 'package:flutter/material.dart';
import 'package:intl/intl.dart';
import 'package:provider/provider.dart';

import '../../models/workout_log.dart';
import '../../providers/progress_provider.dart';
import '../../theme/app_spacing.dart';
import '../../widgets/progress/e1rm_chart.dart';
import '../../widgets/progress/volume_chart.dart';
import '../../widgets/ui/app_card.dart';

/// Progress — the tracking home. Reads logged sets and derives estimated-1RM
/// progression, PRs, weekly volume, and this-week-vs-last-week.
class ProgressTab extends StatelessWidget {
  const ProgressTab({super.key});

  @override
  Widget build(BuildContext context) {
    return ChangeNotifierProvider(
      create: (_) => ProgressProvider()..load(),
      child: const _ProgressView(),
    );
  }
}

class _ProgressView extends StatelessWidget {
  const _ProgressView();

  @override
  Widget build(BuildContext context) {
    final prov = context.watch<ProgressProvider>();
    return Scaffold(
      appBar: AppBar(
        title: const Text('Progress'),
        actions: [
          IconButton(
            tooltip: 'Refresh',
            icon: const Icon(Icons.refresh),
            onPressed: prov.loading ? null : prov.refresh,
          ),
        ],
      ),
      body: prov.loading && !prov.hasData
          ? const Center(child: CircularProgressIndicator())
          : RefreshIndicator(
              onRefresh: prov.refresh,
              child: !prov.hasData
                  ? const _Empty()
                  : _Content(prov: prov),
            ),
    );
  }
}

class _Empty extends StatelessWidget {
  const _Empty();

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final cs = theme.colorScheme;
    return ListView(
      padding: const EdgeInsets.all(AppSpacing.lg),
      children: [
        const SizedBox(height: AppSpacing.xl * 2),
        Icon(Icons.insights_outlined, size: 64, color: cs.primary),
        const SizedBox(height: AppSpacing.lg),
        Text('No workouts logged yet',
            textAlign: TextAlign.center,
            style: theme.textTheme.titleLarge
                ?.copyWith(fontWeight: FontWeight.w700)),
        const SizedBox(height: AppSpacing.sm),
        Text(
          'Open your plan in Fitness Planning and tap an exercise to log your '
          'sets. Your strength trend, PRs and weekly volume show up here.',
          textAlign: TextAlign.center,
          style: theme.textTheme.bodyMedium?.copyWith(color: cs.onSurfaceVariant),
        ),
      ],
    );
  }
}

class _Content extends StatelessWidget {
  const _Content({required this.prov});

  final ProgressProvider prov;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final exercises = prov.exercisesWithStrengthData;
    final selected = prov.selectedExercise;

    return ListView(
      padding: const EdgeInsets.all(AppSpacing.lg),
      children: [
        _sectionTitle(theme, 'This week'),
        const SizedBox(height: AppSpacing.sm),
        _WeekSnapshot(thisWeek: prov.thisWeek, lastWeek: prov.lastWeek),
        const SizedBox(height: AppSpacing.lg),

        if (selected != null) ...[
          _sectionTitle(theme, 'Strength progression'),
          const SizedBox(height: AppSpacing.sm),
          _ExercisePicker(
            exercises: exercises,
            selected: selected,
            onSelect: prov.select,
          ),
          const SizedBox(height: AppSpacing.sm),
          AppCard(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                _DeltaLine(delta: prov.progressDelta(selected)),
                const SizedBox(height: AppSpacing.sm),
                E1rmChart(points: prov.e1rmSeries(selected)),
                const SizedBox(height: AppSpacing.xs),
                Text('Estimated 1RM (kg) — weight × (1 + reps/30)',
                    style: theme.textTheme.labelSmall
                        ?.copyWith(color: theme.colorScheme.onSurfaceVariant)),
              ],
            ),
          ),
          const SizedBox(height: AppSpacing.lg),
        ],

        _sectionTitle(theme, 'Personal records'),
        const SizedBox(height: AppSpacing.sm),
        _PrList(prs: prov.personalRecords()),
        const SizedBox(height: AppSpacing.lg),

        _sectionTitle(theme, 'Weekly volume'),
        const SizedBox(height: AppSpacing.sm),
        AppCard(child: VolumeChart(weeks: prov.weeklyVolume())),
        const SizedBox(height: AppSpacing.lg),

        _sectionTitle(theme, 'Recent sets'),
        const SizedBox(height: AppSpacing.sm),
        _RecentList(logs: prov.logs.take(20).toList()),
        const SizedBox(height: AppSpacing.lg),
      ],
    );
  }

  Widget _sectionTitle(ThemeData theme, String t) => Text(t,
      style: theme.textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w700));
}

String _fmtVol(double v) =>
    v >= 1000 ? '${(v / 1000).toStringAsFixed(1)}k' : v.round().toString();

class _WeekSnapshot extends StatelessWidget {
  const _WeekSnapshot({required this.thisWeek, required this.lastWeek});

  final WeekStats thisWeek;
  final WeekStats lastWeek;

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        Expanded(
          child: _StatTile(
            label: 'Workouts',
            value: '${thisWeek.workouts}',
            delta: thisWeek.workouts - lastWeek.workouts,
            icon: Icons.event_available,
          ),
        ),
        const SizedBox(width: AppSpacing.sm),
        Expanded(
          child: _StatTile(
            label: 'Sets',
            value: '${thisWeek.sets}',
            delta: thisWeek.sets - lastWeek.sets,
            icon: Icons.repeat,
          ),
        ),
        const SizedBox(width: AppSpacing.sm),
        Expanded(
          child: _StatTile(
            label: 'Volume kg',
            value: _fmtVol(thisWeek.volume),
            deltaPct: lastWeek.volume > 0
                ? ((thisWeek.volume - lastWeek.volume) / lastWeek.volume) * 100
                : null,
            icon: Icons.monitor_weight_outlined,
          ),
        ),
      ],
    );
  }
}

class _StatTile extends StatelessWidget {
  const _StatTile({
    required this.label,
    required this.value,
    required this.icon,
    this.delta,
    this.deltaPct,
  });

  final String label;
  final String value;
  final IconData icon;
  final int? delta;
  final double? deltaPct;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final cs = theme.colorScheme;

    String? deltaText;
    Color deltaColor = cs.onSurfaceVariant;
    if (deltaPct != null) {
      final up = deltaPct! >= 0;
      deltaText = '${up ? '▲' : '▼'} ${deltaPct!.abs().toStringAsFixed(0)}%';
      deltaColor = up ? Colors.green : cs.error;
    } else if (delta != null && delta != 0) {
      final up = delta! > 0;
      deltaText = '${up ? '▲' : '▼'} ${delta!.abs()}';
      deltaColor = up ? Colors.green : cs.error;
    }

    return AppCard(
      elevated: true,
      padding: const EdgeInsets.symmetric(
          vertical: AppSpacing.md, horizontal: AppSpacing.sm),
      child: Column(
        children: [
          Icon(icon, color: cs.primary, size: 22),
          const SizedBox(height: AppSpacing.xs),
          Text(value,
              style: theme.textTheme.titleLarge
                  ?.copyWith(fontWeight: FontWeight.w800)),
          Text(label,
              textAlign: TextAlign.center,
              style:
                  theme.textTheme.labelSmall?.copyWith(color: cs.onSurfaceVariant)),
          if (deltaText != null) ...[
            const SizedBox(height: 2),
            Text(deltaText,
                style: theme.textTheme.labelSmall
                    ?.copyWith(color: deltaColor, fontWeight: FontWeight.w700)),
          ],
        ],
      ),
    );
  }
}

class _ExercisePicker extends StatelessWidget {
  const _ExercisePicker({
    required this.exercises,
    required this.selected,
    required this.onSelect,
  });

  final List<String> exercises;
  final String selected;
  final ValueChanged<String> onSelect;

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      height: 38,
      child: ListView.separated(
        scrollDirection: Axis.horizontal,
        itemCount: exercises.length,
        separatorBuilder: (_, __) => const SizedBox(width: AppSpacing.sm),
        itemBuilder: (context, i) {
          final name = exercises[i];
          return ChoiceChip(
            label: Text(name),
            selected: name == selected,
            onSelected: (_) => onSelect(name),
          );
        },
      ),
    );
  }
}

class _DeltaLine extends StatelessWidget {
  const _DeltaLine({required this.delta});

  final ({double deltaKg, double pct, int days})? delta;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final cs = theme.colorScheme;
    if (delta == null) {
      return Text('Building your trend — log a few sessions.',
          style: theme.textTheme.bodySmall?.copyWith(color: cs.onSurfaceVariant));
    }
    final up = delta!.deltaKg >= 0;
    final color = up ? Colors.green : cs.error;
    final sign = up ? '+' : '−';
    return Row(
      children: [
        Icon(up ? Icons.trending_up : Icons.trending_down, color: color, size: 20),
        const SizedBox(width: AppSpacing.sm),
        Expanded(
          child: Text(
            'e1RM $sign${delta!.deltaKg.abs().toStringAsFixed(1)} kg '
            '($sign${delta!.pct.abs().toStringAsFixed(0)}%) over ${delta!.days} days',
            style: theme.textTheme.bodyMedium
                ?.copyWith(fontWeight: FontWeight.w700, color: color),
          ),
        ),
      ],
    );
  }
}

class _PrList extends StatelessWidget {
  const _PrList({required this.prs});

  final List<PrEntry> prs;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final cs = theme.colorScheme;
    if (prs.isEmpty) {
      return AppCard(
        child: Text('Log weight × reps to start setting records.',
            style: theme.textTheme.bodyMedium?.copyWith(color: cs.onSurfaceVariant)),
      );
    }
    return Column(
      children: [
        for (final pr in prs.take(6))
          Padding(
            padding: const EdgeInsets.only(bottom: AppSpacing.sm),
            child: AppCard(
              padding: const EdgeInsets.symmetric(
                  horizontal: AppSpacing.md, vertical: AppSpacing.sm),
              child: Row(
                children: [
                  Text('🏆', style: theme.textTheme.titleMedium),
                  const SizedBox(width: AppSpacing.sm),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(pr.exercise,
                            style: theme.textTheme.bodyMedium
                                ?.copyWith(fontWeight: FontWeight.w700)),
                        Text(
                          '${pr.weightKg.toStringAsFixed(pr.weightKg % 1 == 0 ? 0 : 1)} kg × ${pr.reps}  ·  ${DateFormat('MMM d').format(pr.date)}',
                          style: theme.textTheme.bodySmall
                              ?.copyWith(color: cs.onSurfaceVariant),
                        ),
                      ],
                    ),
                  ),
                  Text('${pr.e1rm.toStringAsFixed(0)} kg',
                      style: theme.textTheme.titleMedium?.copyWith(
                          color: cs.primary, fontWeight: FontWeight.w800)),
                  Text('  e1RM',
                      style: theme.textTheme.labelSmall
                          ?.copyWith(color: cs.onSurfaceVariant)),
                ],
              ),
            ),
          ),
      ],
    );
  }
}

class _RecentList extends StatelessWidget {
  const _RecentList({required this.logs});

  final List<WorkoutLog> logs;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final cs = theme.colorScheme;
    if (logs.isEmpty) {
      return AppCard(
        child: Text('Your logged sets will appear here.',
            style: theme.textTheme.bodyMedium?.copyWith(color: cs.onSurfaceVariant)),
      );
    }
    return Column(
      children: [
        for (final l in logs)
          Padding(
            padding: const EdgeInsets.only(bottom: AppSpacing.xs),
            child: AppCard(
              padding: const EdgeInsets.symmetric(
                  horizontal: AppSpacing.md, vertical: AppSpacing.sm),
              child: Row(
                children: [
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(l.exerciseName,
                            style: theme.textTheme.bodyMedium
                                ?.copyWith(fontWeight: FontWeight.w600)),
                        Text(
                          DateFormat('EEE, MMM d · h:mm a').format(l.performedAt),
                          style: theme.textTheme.labelSmall
                              ?.copyWith(color: cs.onSurfaceVariant),
                        ),
                      ],
                    ),
                  ),
                  Text(_setSummary(l),
                      style: theme.textTheme.bodyMedium?.copyWith(
                          color: cs.primary, fontWeight: FontWeight.w700)),
                ],
              ),
            ),
          ),
      ],
    );
  }

  String _setSummary(WorkoutLog l) {
    final parts = <String>[];
    if (l.weightKg != null) {
      parts.add(
          '${l.weightKg!.toStringAsFixed(l.weightKg! % 1 == 0 ? 0 : 1)} kg');
    }
    if (l.repsCompleted != null) parts.add('× ${l.repsCompleted}');
    if (parts.isEmpty && l.durationSeconds != null) {
      parts.add('${l.durationSeconds}s');
    }
    final base = parts.join(' ');
    return l.rpe != null ? '$base · RPE ${l.rpe!.toStringAsFixed(l.rpe! % 1 == 0 ? 0 : 1)}' : base;
  }
}
