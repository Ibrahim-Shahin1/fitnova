import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../providers/user_provider.dart';
import '../theme/app_spacing.dart';
import '../theme/theme_controller.dart';
import '../widgets/ui/app_card.dart';

/// Post-registration mode chooser.
/// Lets the user jump directly into either:
///   - Workout Planner (goal → chat → plan)
///   - Form Correction (exercise selection → live camera or upload)
///
/// Also hosts the app-wide theme toggle (System / Light / Dark) since this
/// is the universal hub reachable by both planner and form-correction users.
class ModeSelectScreen extends StatelessWidget {
  const ModeSelectScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final user = context.read<UserProvider>();
    final theme = Theme.of(context);
    final cs = theme.colorScheme;

    return Scaffold(
      appBar: AppBar(
        title: Text('Hi ${user.name}'),
        automaticallyImplyLeading: false,
        actions: [
          IconButton(
            tooltip: 'Theme',
            icon: const Icon(Icons.brightness_6_outlined),
            onPressed: () => _showThemeSheet(context),
          ),
          const SizedBox(width: AppSpacing.xs),
        ],
      ),
      body: Padding(
        padding: const EdgeInsets.all(AppSpacing.lg),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const SizedBox(height: AppSpacing.sm),
            Text(
              'What would you like to do?',
              style: theme.textTheme.headlineMedium,
            ),
            const SizedBox(height: AppSpacing.lg),
            _ModeCard(
              icon: Icons.event_note,
              title: 'Workout Planner',
              subtitle: 'Pick a goal · chat · get a personalised week plan',
              accentColor: cs.primary,
              onTap: () => Navigator.pushReplacementNamed(context, '/home'),
            ),
            const SizedBox(height: AppSpacing.md),
            _ModeCard(
              icon: Icons.videocam,
              title: 'Form Correction',
              subtitle:
                  'Live camera or upload a clip · get rep-by-rep feedback',
              accentColor: cs.secondary,
              onTap: () =>
                  Navigator.pushReplacementNamed(context, '/exercise-select'),
            ),
          ],
        ),
      ),
    );
  }
}

class _ModeCard extends StatelessWidget {
  final IconData icon;
  final String title;
  final String subtitle;
  final Color accentColor;
  final VoidCallback onTap;

  const _ModeCard({
    required this.icon,
    required this.title,
    required this.subtitle,
    required this.accentColor,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final cs = theme.colorScheme;

    return AppCard(
      padding: const EdgeInsets.all(AppSpacing.lg),
      onTap: onTap,
      accentColor: accentColor,
      child: Row(
        children: [
          Container(
            width: 56,
            height: 56,
            decoration: BoxDecoration(
              color: accentColor.withValues(alpha: 0.18),
              borderRadius: BorderRadius.circular(AppRadius.rm),
            ),
            child: Icon(icon, color: accentColor, size: 28),
          ),
          const SizedBox(width: AppSpacing.md),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(title, style: theme.textTheme.titleLarge),
                const SizedBox(height: AppSpacing.xs),
                Text(
                  subtitle,
                  style: theme.textTheme.bodyMedium?.copyWith(
                    color: cs.onSurfaceVariant,
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(width: AppSpacing.sm),
          Icon(
            Icons.arrow_forward_ios,
            color: cs.onSurfaceVariant,
            size: 16,
          ),
        ],
      ),
    );
  }
}

void _showThemeSheet(BuildContext context) {
  showModalBottomSheet<void>(
    context: context,
    showDragHandle: true,
    builder: (_) => const _ThemeSheet(),
  );
}

class _ThemeSheet extends StatelessWidget {
  const _ThemeSheet();

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Consumer<ThemeController>(
      builder: (context, controller, _) {
        Future<void> pick(ThemeMode mode) async {
          await controller.setMode(mode);
          if (context.mounted) Navigator.pop(context);
        }

        return SafeArea(
          top: false,
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Padding(
                padding: const EdgeInsets.symmetric(
                  horizontal: AppSpacing.lg,
                  vertical: AppSpacing.sm,
                ),
                child: Text(
                  'Appearance',
                  style: theme.textTheme.titleLarge,
                ),
              ),
              const SizedBox(height: AppSpacing.xs),
              _ThemeOption(
                icon: Icons.smartphone_outlined,
                label: 'System',
                subtitle: 'Match your device',
                value: ThemeMode.system,
                groupValue: controller.mode,
                onTap: () => pick(ThemeMode.system),
              ),
              _ThemeOption(
                icon: Icons.light_mode_outlined,
                label: 'Light',
                value: ThemeMode.light,
                groupValue: controller.mode,
                onTap: () => pick(ThemeMode.light),
              ),
              _ThemeOption(
                icon: Icons.dark_mode_outlined,
                label: 'Dark',
                value: ThemeMode.dark,
                groupValue: controller.mode,
                onTap: () => pick(ThemeMode.dark),
              ),
              const SizedBox(height: AppSpacing.md),
            ],
          ),
        );
      },
    );
  }
}

class _ThemeOption extends StatelessWidget {
  final IconData icon;
  final String label;
  final String? subtitle;
  final ThemeMode value;
  final ThemeMode groupValue;
  final VoidCallback onTap;

  const _ThemeOption({
    required this.icon,
    required this.label,
    this.subtitle,
    required this.value,
    required this.groupValue,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final cs = theme.colorScheme;
    final isSelected = value == groupValue;

    return ListTile(
      leading: Icon(icon, color: isSelected ? cs.primary : cs.onSurfaceVariant),
      title: Text(label, style: theme.textTheme.titleMedium),
      subtitle: subtitle != null
          ? Text(
              subtitle!,
              style: theme.textTheme.bodySmall?.copyWith(
                color: cs.onSurfaceVariant,
              ),
            )
          : null,
      trailing: Radio<ThemeMode>(
        value: value,
        groupValue: groupValue,
        onChanged: (_) => onTap(),
      ),
      onTap: onTap,
    );
  }
}
