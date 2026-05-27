import 'package:flutter/material.dart';

import '../fitness_planning/planning_tab.dart';
import '../home/home_tab.dart';
import '../nutrition/nutrition_tab.dart';
import '../profile/profile_tab.dart';
import '../progress/progress_tab.dart';

/// The signed-in home: a 6-tab bottom-nav shell. Tabs keep their state via
/// IndexedStack. Home (the dashboard) is the default landing tab.
class AppShell extends StatefulWidget {
  const AppShell({super.key});

  @override
  State<AppShell> createState() => _AppShellState();
}

class _AppShellState extends State<AppShell> {
  int _index = 0;

  void _goTo(int i) => setState(() => _index = i);

  @override
  Widget build(BuildContext context) {
    // Built per-frame (not const) so Home can hold a tab-switch callback.
    // IndexedStack preserves each tab's State by position across rebuilds.
    final tabs = <Widget>[
      HomeTab(onOpenPlanning: () => _goTo(1)),
      const PlanningTab(),
      const ProgressTab(),
      const NutritionTab(),
      const ProfileTab(),
    ];

    return Scaffold(
      body: IndexedStack(index: _index, children: tabs),
      bottomNavigationBar: NavigationBar(
        selectedIndex: _index,
        onDestinationSelected: _goTo,
        destinations: const [
          NavigationDestination(
            icon: Icon(Icons.home_outlined),
            selectedIcon: Icon(Icons.home),
            label: 'Home',
          ),
          NavigationDestination(
            icon: Icon(Icons.fitness_center_outlined),
            selectedIcon: Icon(Icons.fitness_center),
            label: 'Planning',
          ),
          NavigationDestination(
            icon: Icon(Icons.insights_outlined),
            selectedIcon: Icon(Icons.insights),
            label: 'Progress',
          ),
          NavigationDestination(
            icon: Icon(Icons.restaurant_outlined),
            selectedIcon: Icon(Icons.restaurant),
            label: 'Nutrition',
          ),
          NavigationDestination(
            icon: Icon(Icons.person_outline),
            selectedIcon: Icon(Icons.person),
            label: 'Profile',
          ),
        ],
      ),
    );
  }
}
