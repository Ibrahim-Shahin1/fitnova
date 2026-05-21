import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../../models/user_profile.dart';
import '../../providers/auth_provider.dart';
import '../../providers/profile_provider.dart';
import '../../theme/app_spacing.dart';
import '../../widgets/ui/app_button.dart';
import '../../widgets/ui/app_text_field.dart';

const _focusOptions = ['powerbuilding', 'powerlifting', 'hypertrophy', 'general'];
const _injuryOptions = [
  'lower_back', 'knees', 'shoulders', 'elbows', 'wrists', 'hips', 'ankles', 'neck',
];

/// Step 1 of onboarding: collect the user's physical profile + training focus.
/// Saves to Supabase (onboarding_completed stays false until the walkthrough).
class ProfileSetupScreen extends StatefulWidget {
  const ProfileSetupScreen({super.key, required this.onDone});

  final VoidCallback onDone;

  @override
  State<ProfileSetupScreen> createState() => _ProfileSetupScreenState();
}

class _ProfileSetupScreenState extends State<ProfileSetupScreen> {
  final _formKey = GlobalKey<FormState>();
  final _name = TextEditingController();
  final _age = TextEditingController();
  final _height = TextEditingController();
  final _weight = TextEditingController();
  String _gender = 'Male';
  String _focus = 'general';
  final Set<String> _injuries = {};
  bool _saving = false;

  @override
  void initState() {
    super.initState();
    final p = context.read<ProfileProvider>().profile;
    _name.text = p?.displayName ?? '';
    if (p?.age != null) _age.text = '${p!.age}';
    if (p?.heightCm != null) _height.text = '${p!.heightCm}';
    if (p?.weightKg != null) _weight.text = '${p!.weightKg}';
    _gender = p?.gender ?? 'Male';
    _focus = p?.trainingFocus ?? 'general';
    _injuries.addAll(p?.injuries ?? const []);
  }

  @override
  void dispose() {
    _name.dispose();
    _age.dispose();
    _height.dispose();
    _weight.dispose();
    super.dispose();
  }

  Future<void> _continue() async {
    if (!_formKey.currentState!.validate()) return;
    setState(() => _saving = true);
    final provider = context.read<ProfileProvider>();
    final base = provider.profile ??
        UserProfile(id: context.read<AuthProvider>().user!.id);
    final updated = base.copyWith(
      displayName: _name.text.trim(),
      age: int.tryParse(_age.text.trim()),
      gender: _gender,
      heightCm: double.tryParse(_height.text.trim()),
      weightKg: double.tryParse(_weight.text.trim()),
      trainingFocus: _focus,
      injuries: _injuries.toList(),
      onboardingCompleted: false, // flips true after the walkthrough
    );
    try {
      await provider.save(updated);
      if (mounted) widget.onDone();
    } catch (_) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text("Couldn't save your profile. Try again.")),
        );
      }
    } finally {
      if (mounted) setState(() => _saving = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Scaffold(
      appBar: AppBar(title: const Text('Your profile')),
      body: SafeArea(
        child: Form(
          key: _formKey,
          child: ListView(
            padding: const EdgeInsets.all(AppSpacing.lg),
            children: [
              Text('Tell us about you',
                  style: theme.textTheme.headlineSmall
                      ?.copyWith(fontWeight: FontWeight.w700)),
              const SizedBox(height: AppSpacing.xs),
              Text('We use this to personalize your plan and keep it safe.',
                  style: theme.textTheme.bodyMedium
                      ?.copyWith(color: theme.colorScheme.onSurfaceVariant)),
              const SizedBox(height: AppSpacing.xl),
              AppTextField(
                controller: _name,
                label: 'Display name',
                prefixIcon: Icons.person_outline,
                textInputAction: TextInputAction.next,
                validator: (v) =>
                    (v == null || v.trim().isEmpty) ? 'Enter a name' : null,
              ),
              const SizedBox(height: AppSpacing.md),
              AppTextField(
                controller: _age,
                label: 'Age',
                prefixIcon: Icons.cake_outlined,
                keyboardType: TextInputType.number,
                textInputAction: TextInputAction.next,
                validator: (v) {
                  final n = int.tryParse(v ?? '');
                  return (n == null || n < 10 || n > 100)
                      ? 'Enter a valid age'
                      : null;
                },
              ),
              const SizedBox(height: AppSpacing.md),
              Text('Gender', style: theme.textTheme.labelLarge),
              const SizedBox(height: AppSpacing.xs),
              SegmentedButton<String>(
                segments: const [
                  ButtonSegment(value: 'Male', label: Text('Male')),
                  ButtonSegment(value: 'Female', label: Text('Female')),
                ],
                selected: {_gender},
                onSelectionChanged: (s) => setState(() => _gender = s.first),
              ),
              const SizedBox(height: AppSpacing.md),
              Row(
                children: [
                  Expanded(
                    child: AppTextField(
                      controller: _height,
                      label: 'Height (cm)',
                      prefixIcon: Icons.height,
                      keyboardType: TextInputType.number,
                      textInputAction: TextInputAction.next,
                      validator: (v) {
                        final n = double.tryParse(v ?? '');
                        return (n == null || n < 80 || n > 250) ? 'cm?' : null;
                      },
                    ),
                  ),
                  const SizedBox(width: AppSpacing.md),
                  Expanded(
                    child: AppTextField(
                      controller: _weight,
                      label: 'Weight (kg)',
                      prefixIcon: Icons.monitor_weight_outlined,
                      keyboardType: TextInputType.number,
                      validator: (v) {
                        final n = double.tryParse(v ?? '');
                        return (n == null || n < 20 || n > 400) ? 'kg?' : null;
                      },
                    ),
                  ),
                ],
              ),
              const SizedBox(height: AppSpacing.md),
              DropdownButtonFormField<String>(
                value: _focus,
                decoration: const InputDecoration(
                  labelText: 'Training focus',
                  prefixIcon: Icon(Icons.fitness_center),
                ),
                items: _focusOptions
                    .map((f) =>
                        DropdownMenuItem(value: f, child: Text(_titleCase(f))))
                    .toList(),
                onChanged: (v) => setState(() => _focus = v ?? 'general'),
              ),
              const SizedBox(height: AppSpacing.lg),
              Text('Any injuries to work around? (optional)',
                  style: theme.textTheme.labelLarge),
              const SizedBox(height: AppSpacing.sm),
              Wrap(
                spacing: AppSpacing.sm,
                runSpacing: AppSpacing.xs,
                children: _injuryOptions.map((inj) {
                  final selected = _injuries.contains(inj);
                  return FilterChip(
                    label: Text(_titleCase(inj.replaceAll('_', ' '))),
                    selected: selected,
                    onSelected: (on) => setState(
                        () => on ? _injuries.add(inj) : _injuries.remove(inj)),
                  );
                }).toList(),
              ),
              const SizedBox(height: AppSpacing.xl),
              AppButton(
                label: 'Continue',
                onPressed: _saving ? null : _continue,
                isLoading: _saving,
                expand: true,
                size: AppButtonSize.lg,
              ),
            ],
          ),
        ),
      ),
    );
  }

  static String _titleCase(String s) =>
      s.isEmpty ? s : s[0].toUpperCase() + s.substring(1);
}
