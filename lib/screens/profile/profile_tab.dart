import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../../models/user_profile.dart';
import '../../providers/auth_provider.dart';
import '../../providers/profile_provider.dart';
import '../../theme/app_spacing.dart';
import '../../theme/theme_controller.dart';
import '../../widgets/ui/app_button.dart';
import '../../widgets/ui/app_card.dart';
import 'edit_profile_screen.dart';

/// Profile — the user's personal hub: identity, body stats, training profile,
/// and settings (theme, sign out). Body stats + training profile are editable
/// via the Edit screen. Progress charts (weight trend, PRs) arrive with logging.
class ProfileTab extends StatelessWidget {
  const ProfileTab({super.key});

  @override
  Widget build(BuildContext context) {
    final profile = context.watch<ProfileProvider>().profile;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Profile'),
        actions: [
          if (profile != null)
            IconButton(
              tooltip: 'Edit profile',
              icon: const Icon(Icons.edit_outlined),
              onPressed: () => Navigator.of(context).push(
                MaterialPageRoute(builder: (_) => const EditProfileScreen()),
              ),
            ),
        ],
      ),
      body: profile == null
          ? const Center(child: CircularProgressIndicator())
          : ListView(
              padding: const EdgeInsets.all(AppSpacing.lg),
              children: [
                _Header(profile: profile),
                const SizedBox(height: AppSpacing.lg),

                _SectionTitle('Body stats'),
                AppCard(
                  child: Column(
                    children: [
                      _InfoRow('Height', _height(profile)),
                      _InfoRow('Weight', _weight(profile)),
                      _InfoRow('BMI', _bmi(profile)),
                      _InfoRow('Age', profile.age?.toString() ?? '—'),
                    ],
                  ),
                ),
                const SizedBox(height: AppSpacing.lg),

                _SectionTitle('Training profile'),
                AppCard(
                  child: Column(
                    children: [
                      _InfoRow('Experience', _level(profile.experienceLevel)),
                      _InfoRow('Focus', _pretty(profile.trainingFocus)),
                      _InfoRow(
                          'Frequency',
                          profile.workoutFrequency != null
                              ? '${profile.workoutFrequency}×/week'
                              : '—'),
                      _InfoRow(
                          'Session',
                          profile.sessionDurationHours != null
                              ? '${_trim(profile.sessionDurationHours!)} h'
                              : '—'),
                      if (profile.equipment.isNotEmpty)
                        _ChipsRow('Equipment', profile.equipment),
                      if (profile.injuries.isNotEmpty)
                        _ChipsRow('Injuries', profile.injuries),
                    ],
                  ),
                ),
                const SizedBox(height: AppSpacing.lg),

                _SectionTitle('Settings'),
                AppCard(child: const _ThemeSetting()),
                const SizedBox(height: AppSpacing.lg),

                AppButton(
                  label: 'Sign out',
                  variant: AppButtonVariant.ghost,
                  icon: Icons.logout_outlined,
                  expand: true,
                  onPressed: () => context.read<AuthProvider>().signOut(),
                ),
              ],
            ),
    );
  }
}

// ── Header ────────────────────────────────────────────────────────────────

class _Header extends StatelessWidget {
  const _Header({required this.profile});

  final UserProfile profile;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final cs = theme.colorScheme;
    final name = (profile.displayName?.trim().isNotEmpty ?? false)
        ? profile.displayName!.trim()
        : 'Athlete';
    final initials = name
        .split(RegExp(r'\s+'))
        .take(2)
        .map((w) => w.isEmpty ? '' : w[0].toUpperCase())
        .join();

    return Row(
      children: [
        CircleAvatar(
          radius: 32,
          backgroundColor: cs.primaryContainer,
          child: Text(
            initials.isEmpty ? 'A' : initials,
            style: theme.textTheme.titleLarge?.copyWith(
              color: cs.onPrimaryContainer,
              fontWeight: FontWeight.w800,
            ),
          ),
        ),
        const SizedBox(width: AppSpacing.md),
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(name,
                  style: theme.textTheme.titleLarge
                      ?.copyWith(fontWeight: FontWeight.w700)),
              if (profile.trainingFocus != null)
                Text(_pretty(profile.trainingFocus),
                    style: theme.textTheme.bodyMedium
                        ?.copyWith(color: cs.onSurfaceVariant)),
            ],
          ),
        ),
      ],
    );
  }
}

// ── Settings ────────────────────────────────────────────────────────────────

class _ThemeSetting extends StatelessWidget {
  const _ThemeSetting();

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final mode = context.watch<ThemeController>().mode;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text('Theme', style: theme.textTheme.bodyMedium),
        const SizedBox(height: AppSpacing.sm),
        SizedBox(
          width: double.infinity,
          child: SegmentedButton<ThemeMode>(
            showSelectedIcon: false,
            segments: const [
              ButtonSegment(value: ThemeMode.system, label: Text('Auto')),
              ButtonSegment(value: ThemeMode.light, label: Text('Light')),
              ButtonSegment(value: ThemeMode.dark, label: Text('Dark')),
            ],
            selected: {mode},
            onSelectionChanged: (s) =>
                context.read<ThemeController>().setMode(s.first),
          ),
        ),
      ],
    );
  }
}

// ── Small building blocks ────────────────────────────────────────────────────

class _SectionTitle extends StatelessWidget {
  const _SectionTitle(this.text);

  final String text;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Padding(
      padding: const EdgeInsets.only(bottom: AppSpacing.sm),
      child: Text(text,
          style:
              theme.textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w700)),
    );
  }
}

class _InfoRow extends StatelessWidget {
  const _InfoRow(this.label, this.value);

  final String label;
  final String value;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final cs = theme.colorScheme;
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 6),
      child: Row(
        children: [
          Expanded(
            child: Text(label,
                style: theme.textTheme.bodyMedium
                    ?.copyWith(color: cs.onSurfaceVariant)),
          ),
          Text(value,
              style:
                  theme.textTheme.bodyMedium?.copyWith(fontWeight: FontWeight.w600)),
        ],
      ),
    );
  }
}

class _ChipsRow extends StatelessWidget {
  const _ChipsRow(this.label, this.items);

  final String label;
  final List<String> items;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final cs = theme.colorScheme;
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 6),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(label,
              style: theme.textTheme.bodyMedium
                  ?.copyWith(color: cs.onSurfaceVariant)),
          const SizedBox(height: AppSpacing.sm),
          Wrap(
            spacing: 6,
            runSpacing: 6,
            children: items
                .map((e) => Chip(
                      label: Text(_pretty(e)),
                      visualDensity: VisualDensity.compact,
                      materialTapTargetSize: MaterialTapTargetSize.shrinkWrap,
                    ))
                .toList(),
          ),
        ],
      ),
    );
  }
}

// ── Formatting helpers ───────────────────────────────────────────────────────

String _level(int? l) => switch (l) {
      1 => 'Beginner',
      2 => 'Intermediate',
      3 => 'Advanced',
      _ => '—',
    };

String _pretty(String? raw) {
  if (raw == null || raw.isEmpty) return '—';
  return raw
      .replaceAll('_', ' ')
      .split(' ')
      .map((w) => w.isEmpty ? w : '${w[0].toUpperCase()}${w.substring(1)}')
      .join(' ');
}

String _trim(double v) =>
    v == v.roundToDouble() ? v.toStringAsFixed(0) : v.toStringAsFixed(1);

String _bmi(UserProfile p) {
  final h = p.heightCm;
  final w = p.weightKg;
  if (h == null || w == null || h <= 0) return '—';
  final m = h / 100;
  return (w / (m * m)).toStringAsFixed(1);
}

String _weight(UserProfile p) {
  final w = p.weightKg;
  return w == null ? '—' : '${w.toStringAsFixed(1)} kg';
}

String _height(UserProfile p) {
  final h = p.heightCm;
  return h == null ? '—' : '${h.toStringAsFixed(0)} cm';
}
