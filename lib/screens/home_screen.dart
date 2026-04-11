import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/user_provider.dart';
import '../widgets/goal_card.dart';

class HomeScreen extends StatelessWidget {
  const HomeScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final user = context.read<UserProvider>();

    return Scaffold(
      appBar: AppBar(
        title: Text('Welcome, ${user.name}'),
        automaticallyImplyLeading: false,
      ),
      body: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'What do you want to achieve?',
              style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                    fontWeight: FontWeight.bold,
                  ),
            ),
            const SizedBox(height: 8),
            Text(
              'Pick a goal and our AI will create a plan just for you.',
              style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                    color: Colors.grey[600],
                  ),
            ),
            const SizedBox(height: 24),
            Expanded(
              child: GridView.count(
                crossAxisCount: 2,
                mainAxisSpacing: 16,
                crossAxisSpacing: 16,
                childAspectRatio: 0.9,
                children: [
                  GoalCard(
                    icon: Icons.fitness_center,
                    title: 'Build Muscle',
                    subtitle: 'Hypertrophy & size',
                    color: Colors.deepOrange,
                    onTap: () => _selectGoal(context, 'Strength', focus: 'hypertrophy'),
                  ),
                  GoalCard(
                    icon: Icons.shield,
                    title: 'Get Stronger',
                    subtitle: 'Powerlifting focus',
                    color: Colors.indigo,
                    onTap: () => _selectGoal(context, 'Strength', focus: 'powerlifting'),
                  ),
                  GoalCard(
                    icon: Icons.bolt,
                    title: 'Powerbuilding',
                    subtitle: 'Strength + size',
                    color: Colors.deepPurple,
                    onTap: () => _selectGoal(context, 'Strength', focus: 'powerbuilding'),
                  ),
                  GoalCard(
                    icon: Icons.local_fire_department,
                    title: 'Lose Weight',
                    subtitle: 'Cardio & fat burn',
                    color: Colors.red,
                    onTap: () => _selectGoal(context, 'Cardio'),
                  ),
                  GoalCard(
                    icon: Icons.self_improvement,
                    title: 'Get Flexible',
                    subtitle: 'Yoga & mobility',
                    color: Colors.teal,
                    onTap: () => _selectGoal(context, 'Yoga'),
                  ),
                  GoalCard(
                    icon: Icons.flash_on,
                    title: 'Get Fit Fast',
                    subtitle: 'High-intensity',
                    color: Colors.amber.shade700,
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
