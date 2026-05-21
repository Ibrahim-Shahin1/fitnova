import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../../providers/auth_provider.dart';
import '../../widgets/brand/brand_logo.dart';
import '../../widgets/ui/app_empty_state.dart';

/// Fitness Planning home — the primary feature and default tab.
///
/// Placeholder for now: the real dashboard (active plan, AI coach, workout
/// history) is built in later units. Sign-out lives here until there's a
/// dedicated settings screen.
class PlanningTab extends StatelessWidget {
  const PlanningTab({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const FitNovaMark(size: 28),
        actions: [
          IconButton(
            tooltip: 'Sign out',
            icon: const Icon(Icons.logout_outlined),
            onPressed: () => context.read<AuthProvider>().signOut(),
          ),
        ],
      ),
      body: const AppEmptyState(
        icon: Icons.fitness_center_outlined,
        title: 'Your training plan',
        message:
            'Your weekly plan, AI coach, and workout history will live here.',
      ),
    );
  }
}
