import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../../providers/profile_provider.dart';
import '../../theme/app_spacing.dart';
import '../../widgets/ui/app_button.dart';
import '../../widgets/ui/app_text_field.dart';

const _focusOptions = ['bodybuilding', 'powerbuilding', 'powerlifting', 'cardio', 'general'];
const _injuryOptions = [
  'lower_back', 'knees', 'shoulders', 'elbows', 'wrists', 'hips', 'ankles', 'neck',
];
const _equipmentOptions = [
  'barbell', 'dumbbells', 'machines', 'cables',
  'kettlebell', 'bodyweight', 'resistance_bands', 'pull_up_bar',
];

/// Edit the signed-in user's body stats + training profile. Pre-filled from the
/// current profile; saves via ProfileProvider. The onboarding flag is preserved
/// (copyWith leaves it untouched), so editing never re-triggers onboarding.
class EditProfileScreen extends StatefulWidget {
  const EditProfileScreen({super.key});

  @override
  State<EditProfileScreen> createState() => _EditProfileScreenState();
}

class _EditProfileScreenState extends State<EditProfileScreen> {
  final _formKey = GlobalKey<FormState>();
  final _name = TextEditingController();
  final _age = TextEditingController();
  final _height = TextEditingController();
  final _weight = TextEditingController();
  String _gender = 'Male';
  String _focus = 'general';
  int _experience = 1;
  int _frequency = 3;
  final Set<String> _injuries = {};
  final Set<String> _equipment = {};
  bool _saving = false;

  @override
  void initState() {
    super.initState();
    final p = context.read<ProfileProvider>().profile;
    _name.text = p?.displayName ?? '';
    if (p?.age != null) _age.text = '${p!.age}';
    if (p?.heightCm != null) _height.text = _num(p!.heightCm!);
    if (p?.weightKg != null) _weight.text = _num(p!.weightKg!);
    _gender = p?.gender ?? 'Male';
    _focus = _focusOptions.contains(p?.trainingFocus) ? p!.trainingFocus! : 'general';
    _experience = p?.experienceLevel ?? 1;
    _frequency = p?.workoutFrequency ?? 3;
    _injuries.addAll(p?.injuries ?? const []);
    _equipment.addAll(p?.equipment ?? const []);
  }

  static String _num(double v) =>
      v == v.roundToDouble() ? v.toStringAsFixed(0) : v.toString();

  @override
  void dispose() {
    _name.dispose();
    _age.dispose();
    _height.dispose();
    _weight.dispose();
    super.dispose();
  }

  Future<void> _save() async {
    if (!_formKey.currentState!.validate()) return;
    final provider = context.read<ProfileProvider>();
    final base = provider.profile;
    if (base == null) return;
    final messenger = ScaffoldMessenger.of(context);
    final navigator = Navigator.of(context);

    setState(() => _saving = true);
    final updated = base.copyWith(
      displayName: _name.text.trim(),
      age: int.tryParse(_age.text.trim()),
      gender: _gender,
      heightCm: double.tryParse(_height.text.trim()),
      weightKg: double.tryParse(_weight.text.trim()),
      trainingFocus: _focus,
      experienceLevel: _experience,
      workoutFrequency: _frequency,
      injuries: _injuries.toList(),
      equipment: _equipment.toList(),
    );

    try {
      await provider.save(updated);
      if (!mounted) return;
      navigator.pop();
      messenger.showSnackBar(const SnackBar(content: Text('Profile updated')));
    } catch (_) {
      if (!mounted) return;
      setState(() => _saving = false);
      messenger.showSnackBar(
          const SnackBar(content: Text("Couldn't save. Try again.")));
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Scaffold(
      appBar: AppBar(title: const Text('Edit profile')),
      body: SafeArea(
        child: Form(
          key: _formKey,
          child: ListView(
            padding: const EdgeInsets.all(AppSpacing.lg),
            children: [
              _label(theme, 'Body stats'),
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
                showSelectedIcon: false,
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
                      textInputAction: TextInputAction.next,
                      validator: (v) {
                        final n = double.tryParse(v ?? '');
                        return (n == null || n < 20 || n > 400) ? 'kg?' : null;
                      },
                    ),
                  ),
                ],
              ),
              const SizedBox(height: AppSpacing.xl),

              _label(theme, 'Training profile'),
              DropdownButtonFormField<int>(
                value: _experience,
                decoration: const InputDecoration(
                  labelText: 'Experience',
                  prefixIcon: Icon(Icons.school_outlined),
                ),
                items: const [
                  DropdownMenuItem(value: 1, child: Text('Beginner')),
                  DropdownMenuItem(value: 2, child: Text('Intermediate')),
                  DropdownMenuItem(value: 3, child: Text('Advanced')),
                ],
                onChanged: (v) => setState(() => _experience = v ?? 1),
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
                        DropdownMenuItem(value: f, child: Text(_title(f))))
                    .toList(),
                onChanged: (v) => setState(() => _focus = v ?? 'general'),
              ),
              const SizedBox(height: AppSpacing.md),
              DropdownButtonFormField<int>(
                value: _frequency,
                decoration: const InputDecoration(
                  labelText: 'Workouts per week',
                  prefixIcon: Icon(Icons.calendar_today_outlined),
                ),
                items: [
                  for (var i = 1; i <= 7; i++)
                    DropdownMenuItem(value: i, child: Text('$i / week')),
                ],
                onChanged: (v) => setState(() => _frequency = v ?? 3),
              ),
              const SizedBox(height: AppSpacing.lg),
              Text('Equipment', style: theme.textTheme.labelLarge),
              const SizedBox(height: AppSpacing.sm),
              _chips(_equipmentOptions, _equipment),
              const SizedBox(height: AppSpacing.lg),
              Text('Injuries to work around', style: theme.textTheme.labelLarge),
              const SizedBox(height: AppSpacing.sm),
              _chips(_injuryOptions, _injuries),
              const SizedBox(height: AppSpacing.xl),

              AppButton(
                label: 'Save changes',
                onPressed: _saving ? null : _save,
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

  Widget _label(ThemeData theme, String text) => Padding(
        padding: const EdgeInsets.only(bottom: AppSpacing.md),
        child: Text(text,
            style: theme.textTheme.titleMedium
                ?.copyWith(fontWeight: FontWeight.w700)),
      );

  Widget _chips(List<String> options, Set<String> selected) => Wrap(
        spacing: AppSpacing.sm,
        runSpacing: AppSpacing.xs,
        children: options.map((opt) {
          final on = selected.contains(opt);
          return FilterChip(
            label: Text(_title(opt.replaceAll('_', ' '))),
            selected: on,
            onSelected: (v) => setState(
                () => v ? selected.add(opt) : selected.remove(opt)),
          );
        }).toList(),
      );

  static String _title(String s) => s
      .split(' ')
      .map((w) => w.isEmpty ? w : '${w[0].toUpperCase()}${w.substring(1)}')
      .join(' ');
}
