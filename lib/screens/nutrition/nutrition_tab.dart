import 'package:flutter/material.dart';

import '../../widgets/ui/app_empty_state.dart';

/// Nutrition tab — a future feature. Placeholder for now.
class NutritionTab extends StatelessWidget {
  const NutritionTab({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Nutrition')),
      body: const AppEmptyState(
        icon: Icons.restaurant_outlined,
        title: 'Nutrition guidance',
        message: 'Meal and macro guidance is coming in a future update.',
      ),
    );
  }
}
