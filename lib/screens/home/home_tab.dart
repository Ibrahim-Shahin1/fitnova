import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../../providers/profile_provider.dart';
import '../../theme/app_spacing.dart';
import '../../widgets/brand/brand_logo.dart';
import '../../widgets/ui/app_button.dart';
import '../../widgets/ui/app_card.dart';

/// Home — the at-a-glance dashboard and default tab.
///
/// Progress widgets show empty states until the plan + set-logging features
/// feed them real data; the greeting + date are live now.
class HomeTab extends StatelessWidget {
  const HomeTab({super.key, required this.onOpenPlanning});

  /// Switch the shell to the Planning tab (e.g. from the "set up plan" CTA).
  final VoidCallback onOpenPlanning;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final cs = theme.colorScheme;
    final name = context.watch<ProfileProvider>().profile?.displayName;
    final first = (name == null || name.trim().isEmpty)
        ? null
        : name.trim().split(RegExp(r'\s+')).first;

    return Scaffold(
      appBar: AppBar(title: const FitNovaMark(size: 26)),
      body: ListView(
        padding: const EdgeInsets.all(AppSpacing.lg),
        children: [
          Text(
            first != null ? '${_greeting()}, $first' : _greeting(),
            style: theme.textTheme.headlineSmall
                ?.copyWith(fontWeight: FontWeight.w700),
          ),
          const SizedBox(height: AppSpacing.xs),
          Text(_today(),
              style: theme.textTheme.bodyMedium
                  ?.copyWith(color: cs.onSurfaceVariant)),
          const SizedBox(height: AppSpacing.lg),

          // Today's workout (empty until a plan exists)
          AppCard(
            accentColor: cs.primary,
            onTap: onOpenPlanning,
            child: Row(
              children: [
                Icon(Icons.event_note_outlined, color: cs.primary, size: 28),
                const SizedBox(width: AppSpacing.md),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text("Today's workout",
                          style: theme.textTheme.titleMedium
                              ?.copyWith(fontWeight: FontWeight.w700)),
                      const SizedBox(height: 2),
                      Text('No active plan yet — generate one to get started.',
                          style: theme.textTheme.bodySmall
                              ?.copyWith(color: cs.onSurfaceVariant)),
                    ],
                  ),
                ),
                Icon(Icons.chevron_right, color: cs.onSurfaceVariant),
              ],
            ),
          ),
          const SizedBox(height: AppSpacing.lg),

          _sectionTitle(theme, 'This week'),
          const SizedBox(height: AppSpacing.sm),
          const Row(
            children: [
              Expanded(
                  child: _StatCard(
                      label: 'Workouts', value: '—', icon: Icons.fitness_center)),
              SizedBox(width: AppSpacing.sm),
              Expanded(
                  child: _StatCard(
                      label: 'Sets', value: '—', icon: Icons.repeat)),
              SizedBox(width: AppSpacing.sm),
              Expanded(
                  child: _StatCard(
                      label: 'Streak',
                      value: '—',
                      icon: Icons.local_fire_department_outlined)),
            ],
          ),
          const SizedBox(height: AppSpacing.lg),

          _sectionTitle(theme, 'Progress'),
          const SizedBox(height: AppSpacing.sm),
          AppCard(
            child: SizedBox(
              height: 150,
              child: Center(
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Icon(Icons.show_chart, size: 40, color: cs.onSurfaceVariant),
                    const SizedBox(height: AppSpacing.sm),
                    Text('Log workouts to see your progress',
                        style: theme.textTheme.bodyMedium
                            ?.copyWith(color: cs.onSurfaceVariant)),
                  ],
                ),
              ),
            ),
          ),
          const SizedBox(height: AppSpacing.lg),

          AppButton(
            label: 'Set up my plan',
            icon: Icons.add,
            expand: true,
            onPressed: onOpenPlanning,
          ),
        ],
      ),
    );
  }

  Widget _sectionTitle(ThemeData theme, String text) => Text(
        text,
        style: theme.textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w700),
      );

  String _greeting() {
    final h = DateTime.now().hour;
    if (h < 12) return 'Good morning';
    if (h < 17) return 'Good afternoon';
    return 'Good evening';
  }

  String _today() {
    final now = DateTime.now();
    const days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
    const months = [
      'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
    ];
    return '${days[now.weekday - 1]}, ${months[now.month - 1]} ${now.day}';
  }
}

class _StatCard extends StatelessWidget {
  const _StatCard({required this.label, required this.value, required this.icon});

  final String label;
  final String value;
  final IconData icon;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final cs = theme.colorScheme;
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
              style: theme.textTheme.labelSmall
                  ?.copyWith(color: cs.onSurfaceVariant)),
        ],
      ),
    );
  }
}
