import 'package:flutter/material.dart';

import '../fitness_planning/planning_tab.dart';
import '../form_correction/form_correction_tab.dart';
import '../nutrition/nutrition_tab.dart';

/// The signed-in home: a 3-tab bottom-nav shell. Tabs are kept alive (and keep
/// their scroll/state) via IndexedStack. Fitness Planning is the default tab.
class AppShell extends StatefulWidget {
  const AppShell({super.key});

  @override
  State<AppShell> createState() => _AppShellState();
}

class _AppShellState extends State<AppShell> {
  int _index = 0;

  static const List<Widget> _tabs = [
    PlanningTab(),
    FormCorrectionTab(),
    NutritionTab(),
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: IndexedStack(index: _index, children: _tabs),
      bottomNavigationBar: NavigationBar(
        selectedIndex: _index,
        onDestinationSelected: (i) => setState(() => _index = i),
        destinations: const [
          NavigationDestination(
            icon: Icon(Icons.fitness_center_outlined),
            selectedIcon: Icon(Icons.fitness_center),
            label: 'Planning',
          ),
          NavigationDestination(
            icon: Icon(Icons.videocam_outlined),
            selectedIcon: Icon(Icons.videocam),
            label: 'Form',
          ),
          NavigationDestination(
            icon: Icon(Icons.restaurant_outlined),
            selectedIcon: Icon(Icons.restaurant),
            label: 'Nutrition',
          ),
        ],
      ),
    );
  }
}
