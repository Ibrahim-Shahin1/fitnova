import 'package:flutter/material.dart';

import '../../widgets/ui/app_empty_state.dart';

/// Fitness Planning — the primary feature. Placeholder for now; the real
/// dashboard (active plan, AI coach, workout history) is built in later units.
class PlanningTab extends StatelessWidget {
  const PlanningTab({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Fitness Planning')),
      body: const AppEmptyState(
        icon: Icons.fitness_center_outlined,
        title: 'Your training plan',
        message:
            'Your weekly plan, AI coach, and workout history will live here.',
      ),
    );
  }
}
