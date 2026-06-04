import 'dart:async';

import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../../models/meal_plan.dart';
import '../../models/user_profile.dart';
import '../../providers/nutrition_provider.dart';
import '../../providers/profile_provider.dart';
import '../../services/nutrition_service.dart';
import '../../theme/app_spacing.dart';
import '../../widgets/ui/app_button.dart';
import '../../widgets/ui/app_card.dart';
import 'generate_meal_plan_screen.dart';

/// Nutrition intake. Body stats (age, sex, height, weight) come from the user's
/// PROFILE (filled at onboarding) — shown read-only, never re-entered. Only asks
/// what's genuinely meal-specific and not in the profile: nutrition goal, activity
/// level (pre-filled from training frequency), dietary prefs, allergens, meals/day.
class NutritionIntakeScreen extends StatefulWidget {
  const NutritionIntakeScreen({super.key});

  @override
  State<NutritionIntakeScreen> createState() => _NutritionIntakeScreenState();
}

class _NutritionIntakeScreenState extends State<NutritionIntakeScreen> {
  // Body stats resolved from the profile (with safe fallbacks if incomplete).
  late double _weightKg;
  late double _heightCm;
  late int _age;
  late String _sex;
  bool _profileComplete = false;

  late Map<String, dynamic> _form; // meal-specific inputs only
  final _allergenController = TextEditingController();
  NutritionTargets? _preview;
  Timer? _debounce;
  bool _previewLoading = false;

  static const _goals = ['lose', 'maintain', 'gain'];
  static const _goalLabels = {'lose': 'Lose fat', 'maintain': 'Maintain', 'gain': 'Gain muscle'};
  static const _activities = ['sedentary', 'light', 'moderate', 'active', 'athlete'];
  static const _diets = ['vegetarian', 'vegan', 'gluten-free', 'dairy-free', 'low-carb'];

  @override
  void initState() {
    super.initState();
    final saved = context.read<NutritionProvider>().intake;
    final UserProfile? p = context.read<ProfileProvider>().profile;

    _weightKg = p?.weightKg ?? (saved['weight_kg'] as num?)?.toDouble() ?? 75;
    _heightCm = p?.heightCm ?? (saved['height_cm'] as num?)?.toDouble() ?? 175;
    _age = p?.age ?? (saved['age'] as num?)?.toInt() ?? 25;
    _sex = _normSex(p?.gender) ?? (saved['sex'] as String?) ?? 'male';
    _profileComplete =
        p?.weightKg != null && p?.heightCm != null && p?.age != null && p?.gender != null;

    _form = {
      'activity_level': saved['activity_level'] ?? _activityFromFrequency(p?.workoutFrequency),
      'goal': saved['goal'] ?? _goalFromFocus(p?.trainingFocus),
      'meals_per_day': saved['meals_per_day'] ?? 4,
      'days': saved['days'] ?? 7,
      'diet': List<String>.from(saved['diet'] ?? const []),
      'exclude_ingredients': List<String>.from(saved['exclude_ingredients'] ?? const []),
    };
    _refreshPreview();
  }

  @override
  void dispose() {
    _debounce?.cancel();
    _allergenController.dispose();
    super.dispose();
  }

  static String? _normSex(String? g) {
    if (g == null) return null;
    final s = g.toLowerCase();
    if (s.startsWith('m')) return 'male';
    if (s.startsWith('f')) return 'female';
    return null;
  }

  static String _activityFromFrequency(int? freq) {
    if (freq == null) return 'moderate';
    if (freq <= 2) return 'light';
    if (freq <= 4) return 'moderate';
    if (freq <= 6) return 'active';
    return 'athlete';
  }

  static String _goalFromFocus(String? focus) {
    final s = (focus ?? '').toLowerCase();
    if (s.contains('build') || s.contains('power') || s.contains('mass') || s.contains('hyper')) {
      return 'gain';
    }
    if (s.contains('cardio') || s.contains('lean') || s.contains('loss') || s.contains('cut')) {
      return 'lose';
    }
    return 'maintain';
  }

  void _set(String key, dynamic value) {
    setState(() => _form[key] = value);
    _refreshPreview();
  }

  void _refreshPreview() {
    _debounce?.cancel();
    _debounce = Timer(const Duration(milliseconds: 400), () async {
      setState(() => _previewLoading = true);
      try {
        final t = await NutritionService.fetchTargets({
          'weight_kg': _weightKg,
          'height_cm': _heightCm,
          'age': _age,
          'sex': _sex,
          'activity_level': _form['activity_level'],
          'goal': _form['goal'],
        });
        if (mounted) setState(() => _preview = t);
      } catch (_) {
        if (mounted) setState(() => _preview = null);
      } finally {
        if (mounted) setState(() => _previewLoading = false);
      }
    });
  }

  void _addAllergen() {
    final raw = _allergenController.text.trim().toLowerCase();
    if (raw.isEmpty) return;
    final list = (_form['exclude_ingredients'] as List).cast<String>();
    for (final part in raw.split(',')) {
      final pp = part.trim();
      if (pp.isNotEmpty && !list.contains(pp)) list.add(pp);
    }
    _allergenController.clear();
    setState(() {});
  }

  Future<void> _generate() async {
    final payload = {
      'weight_kg': _weightKg,
      'height_cm': _heightCm,
      'age': _age,
      'sex': _sex,
      ..._form,
    };
    await context.read<NutritionProvider>().saveIntake(payload);
    if (!mounted) return;
    final prov = context.read<NutritionProvider>();
    final ok = await Navigator.of(context).push<bool>(
      MaterialPageRoute(
        builder: (_) => ChangeNotifierProvider<NutritionProvider>.value(
          value: prov,
          child: GenerateMealPlanScreen(params: payload),
        ),
      ),
    );
    if (ok == true && mounted) Navigator.of(context).pop(true);
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final cs = theme.colorScheme;
    final allergens = (_form['exclude_ingredients'] as List).cast<String>();
    return Scaffold(
      appBar: AppBar(title: const Text('Build your meal plan')),
      body: ListView(
        padding: const EdgeInsets.all(AppSpacing.lg),
        children: [
          _ProfileCard(
            weightKg: _weightKg,
            heightCm: _heightCm,
            age: _age,
            sex: _sex,
            complete: _profileComplete,
          ),
          const SizedBox(height: AppSpacing.lg),

          Text('Your nutrition goal',
              style: theme.textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w700)),
          const SizedBox(height: AppSpacing.sm),
          _ChoiceChips(
            options: _goals,
            labels: _goalLabels,
            selected: _form['goal'] as String,
            onSelect: (v) => _set('goal', v),
          ),

          const SizedBox(height: AppSpacing.md),
          Text('Activity level', style: theme.textTheme.titleSmall),
          Text('Pre-filled from your training frequency — adjust if needed.',
              style: theme.textTheme.bodySmall?.copyWith(color: cs.onSurfaceVariant)),
          const SizedBox(height: AppSpacing.sm),
          _ChoiceChips(
            options: _activities,
            selected: _form['activity_level'] as String,
            onSelect: (v) => _set('activity_level', v),
          ),

          const SizedBox(height: AppSpacing.md),
          _MultiChips(
            label: 'Dietary preferences (optional)',
            options: _diets,
            selected: (_form['diet'] as List).cast<String>(),
            onToggle: (v) {
              final list = (_form['diet'] as List).cast<String>();
              setState(() => list.contains(v) ? list.remove(v) : list.add(v));
            },
          ),

          const SizedBox(height: AppSpacing.md),
          Text('Allergens / ingredients to avoid (optional)',
              style: theme.textTheme.titleSmall),
          const SizedBox(height: AppSpacing.sm),
          Row(children: [
            Expanded(
              child: TextField(
                controller: _allergenController,
                textInputAction: TextInputAction.done,
                onSubmitted: (_) => _addAllergen(),
                decoration: InputDecoration(
                  hintText: 'e.g. peanut, shellfish',
                  isDense: true,
                  filled: true,
                  fillColor: cs.surfaceContainerHighest,
                  border: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(AppRadius.rm),
                    borderSide: BorderSide.none,
                  ),
                ),
              ),
            ),
            const SizedBox(width: AppSpacing.sm),
            IconButton.filledTonal(onPressed: _addAllergen, icon: const Icon(Icons.add)),
          ]),
          if (allergens.isNotEmpty) ...[
            const SizedBox(height: AppSpacing.sm),
            Wrap(
              spacing: AppSpacing.sm,
              children: [
                for (final a in allergens)
                  InputChip(
                    label: Text(a),
                    onDeleted: () => setState(() => allergens.remove(a)),
                  ),
              ],
            ),
          ],

          const SizedBox(height: AppSpacing.md),
          _MealsPerDay(
            value: _form['meals_per_day'] as int,
            onChanged: (v) => setState(() => _form['meals_per_day'] = v),
          ),

          const SizedBox(height: AppSpacing.lg),
          _TargetPreview(preview: _preview, loading: _previewLoading),
          const SizedBox(height: AppSpacing.lg),

          AppButton(
            label: 'Build my meal plan',
            icon: Icons.auto_awesome,
            size: AppButtonSize.lg,
            expand: true,
            onPressed: _generate,
          ),
          const SizedBox(height: AppSpacing.lg),
        ],
      ),
    );
  }
}

class _ProfileCard extends StatelessWidget {
  const _ProfileCard({
    required this.weightKg,
    required this.heightCm,
    required this.age,
    required this.sex,
    required this.complete,
  });

  final double weightKg;
  final double heightCm;
  final int age;
  final String sex;
  final bool complete;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final cs = theme.colorScheme;
    final bmi = weightKg / ((heightCm / 100) * (heightCm / 100));
    return AppCard(
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Row(children: [
          Icon(Icons.person, size: 18, color: cs.primary),
          const SizedBox(width: AppSpacing.sm),
          Text('From your profile',
              style: theme.textTheme.titleSmall?.copyWith(fontWeight: FontWeight.w800)),
          const Spacer(),
          if (!complete)
            Text('complete it in Profile',
                style: theme.textTheme.labelSmall?.copyWith(color: cs.error)),
        ]),
        const SizedBox(height: AppSpacing.sm),
        Wrap(spacing: AppSpacing.md, runSpacing: AppSpacing.xs, children: [
          _kv(theme, 'Sex', sex[0].toUpperCase() + sex.substring(1)),
          _kv(theme, 'Age', '$age'),
          _kv(theme, 'Height', '${heightCm.round()} cm'),
          _kv(theme, 'Weight', '${weightKg.round()} kg'),
          _kv(theme, 'BMI', bmi.toStringAsFixed(1)),
        ]),
      ]),
    );
  }

  Widget _kv(ThemeData theme, String k, String v) => Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(k, style: theme.textTheme.labelSmall
              ?.copyWith(color: theme.colorScheme.onSurfaceVariant)),
          Text(v, style: theme.textTheme.titleSmall?.copyWith(fontWeight: FontWeight.w700)),
        ],
      );
}

class _ChoiceChips extends StatelessWidget {
  const _ChoiceChips({
    required this.options,
    required this.selected,
    required this.onSelect,
    this.labels,
  });
  final List<String> options;
  final String selected;
  final ValueChanged<String> onSelect;
  final Map<String, String>? labels;

  @override
  Widget build(BuildContext context) {
    return Wrap(
      spacing: AppSpacing.sm,
      children: [
        for (final o in options)
          ChoiceChip(
            label: Text(labels?[o] ?? o[0].toUpperCase() + o.substring(1)),
            selected: selected == o,
            onSelected: (_) => onSelect(o),
          ),
      ],
    );
  }
}

class _MultiChips extends StatelessWidget {
  const _MultiChips({
    required this.label,
    required this.options,
    required this.selected,
    required this.onToggle,
  });
  final String label;
  final List<String> options;
  final List<String> selected;
  final ValueChanged<String> onToggle;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(label, style: theme.textTheme.titleSmall),
        const SizedBox(height: AppSpacing.sm),
        Wrap(
          spacing: AppSpacing.sm,
          children: [
            for (final o in options)
              FilterChip(
                label: Text(o),
                selected: selected.contains(o),
                onSelected: (_) => onToggle(o),
              ),
          ],
        ),
      ],
    );
  }
}

class _MealsPerDay extends StatelessWidget {
  const _MealsPerDay({required this.value, required this.onChanged});
  final int value;
  final ValueChanged<int> onChanged;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Row(
      children: [
        Text('Meals per day', style: theme.textTheme.titleSmall),
        const Spacer(),
        SegmentedButton<int>(
          segments: const [
            ButtonSegment(value: 3, label: Text('3')),
            ButtonSegment(value: 4, label: Text('4')),
          ],
          selected: {value},
          onSelectionChanged: (s) => onChanged(s.first),
        ),
      ],
    );
  }
}

class _TargetPreview extends StatelessWidget {
  const _TargetPreview({required this.preview, required this.loading});
  final NutritionTargets? preview;
  final bool loading;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final cs = theme.colorScheme;
    return AppCard(
      accentColor: cs.primary,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(Icons.local_fire_department, color: cs.primary, size: 20),
              const SizedBox(width: AppSpacing.sm),
              Text('Your daily targets',
                  style: theme.textTheme.titleSmall?.copyWith(fontWeight: FontWeight.w800)),
              const Spacer(),
              if (loading)
                const SizedBox(width: 16, height: 16, child: CircularProgressIndicator(strokeWidth: 2)),
            ],
          ),
          const SizedBox(height: AppSpacing.md),
          if (preview == null && !loading)
            Text('Adjust your inputs to see targets.',
                style: theme.textTheme.bodySmall?.copyWith(color: cs.onSurfaceVariant))
          else if (preview != null) ...[
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                _Stat(label: 'Calories', value: '${preview!.targetCalories}', sub: 'kcal/day'),
                _Stat(label: 'BMR', value: '${preview!.bmr}', sub: 'kcal'),
                _Stat(label: 'TDEE', value: '${preview!.tdee}', sub: 'kcal'),
              ],
            ),
            const Divider(height: AppSpacing.lg),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                _Stat(label: 'Protein', value: '${preview!.proteinG}g', accent: true),
                _Stat(label: 'Carbs', value: '${preview!.carbsG}g', accent: true),
                _Stat(label: 'Fat', value: '${preview!.fatG}g', accent: true),
              ],
            ),
          ],
        ],
      ),
    );
  }
}

class _Stat extends StatelessWidget {
  const _Stat({required this.label, required this.value, this.sub, this.accent = false});
  final String label;
  final String value;
  final String? sub;
  final bool accent;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final cs = theme.colorScheme;
    return Column(
      children: [
        Text(value,
            style: theme.textTheme.titleMedium?.copyWith(
                fontWeight: FontWeight.w800, color: accent ? cs.primary : null)),
        Text(label, style: theme.textTheme.labelSmall?.copyWith(color: cs.onSurfaceVariant)),
        if (sub != null)
          Text(sub!, style: theme.textTheme.labelSmall?.copyWith(color: cs.onSurfaceVariant)),
      ],
    );
  }
}
