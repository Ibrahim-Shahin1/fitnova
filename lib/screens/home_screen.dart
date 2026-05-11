import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../providers/user_provider.dart';
import '../theme/app_spacing.dart';
import '../widgets/goal_card.dart';

class HomeScreen extends StatelessWidget {
  const HomeScreen({super.key});

  // Curated bold-fitness palette — distinct, vibrant on both light and dark
  // surfaces. Cyan stays reserved for the brand primary, so none of these
  // collide with app chrome.
  static const _cBuildMuscle = Color(0xFFEF4444); // red
  static const _cGetStronger = Color(0xFF7C3AED); // purple
  static const _cPowerbuilding = Color(0xFFFF6B35); // electric orange
  static const _cLoseWeight = Color(0xFFF59E0B); // amber
  static const _cGetFlexible = Color(0xFF10B981); // emerald
  static const _cGetFitFast = Color(0xFFEC4899); // pink

  @override
  Widget build(BuildContext context) {
    final user = context.read<UserProvider>();
    final theme = Theme.of(context);
    final cs = theme.colorScheme;

    return Scaffold(
      appBar: AppBar(
        title: Text('Welcome, ${user.name}'),
        automaticallyImplyLeading: false,
      ),
      body: Padding(
        padding: const EdgeInsets.all(AppSpacing.lg),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'What do you want to achieve?',
              style: theme.textTheme.headlineMedium?.copyWith(
                fontWeight: FontWeight.w700,
              ),
            ),
            const SizedBox(height: AppSpacing.xs),
            Text(
              'Pick a goal and our AI will create a plan just for you.',
              style: theme.textTheme.bodyMedium?.copyWith(
                color: cs.onSurfaceVariant,
              ),
            ),
            const SizedBox(height: AppSpacing.lg),
            Expanded(
              child: GridView.count(
                crossAxisCount: 2,
                mainAxisSpacing: AppSpacing.md,
                crossAxisSpacing: AppSpacing.md,
                childAspectRatio: 0.9,
                children: [
                  GoalCard(
                    icon: Icons.fitness_center,
                    title: 'Build Muscle',
                    subtitle: 'Hypertrophy & size',
                    color: _cBuildMuscle,
                    onTap: () => _selectGoal(context, 'Strength', focus: 'hypertrophy'),
                  ),
                  GoalCard(
                    icon: Icons.shield,
                    title: 'Get Stronger',
                    subtitle: 'Powerlifting focus',
                    color: _cGetStronger,
                    onTap: () => _selectGoal(context, 'Strength', focus: 'powerlifting'),
                  ),
                  GoalCard(
                    icon: Icons.bolt,
                    title: 'Powerbuilding',
                    subtitle: 'Strength + size',
                    color: _cPowerbuilding,
                    onTap: () => _selectGoal(context, 'Strength', focus: 'powerbuilding'),
                  ),
                  GoalCard(
                    icon: Icons.local_fire_department,
                    title: 'Lose Weight',
                    subtitle: 'Cardio & fat burn',
                    color: _cLoseWeight,
                    onTap: () => _selectGoal(context, 'Cardio'),
                  ),
                  GoalCard(
                    icon: Icons.self_improvement,
                    title: 'Get Flexible',
                    subtitle: 'Yoga & mobility',
                    color: _cGetFlexible,
                    onTap: () => _selectGoal(context, 'Yoga'),
                  ),
                  GoalCard(
                    icon: Icons.flash_on,
                    title: 'Get Fit Fast',
                    subtitle: 'High-intensity',
                    color: _cGetFitFast,
                    onTap: () => _selectGoal(context, 'HIIT'),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  void _selectGoal(BuildContext context, String workoutType, {String? focus}) {
    context.read<UserProvider>().setWorkoutType(workoutType, focus: focus);
    Navigator.pushNamed(context, '/chat');
  }
}
