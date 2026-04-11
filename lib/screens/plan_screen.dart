import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/user_provider.dart';
import '../widgets/exercise_tile.dart';

class PlanScreen extends StatelessWidget {
  const PlanScreen({super.key});

  static const _dayLabels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];

  @override
  Widget build(BuildContext context) {
    final user = context.read<UserProvider>();
    final plan = user.currentPlan!;

    return DefaultTabController(
      length: 7,
      child: Scaffold(
        appBar: AppBar(
          title: Text(plan.programTitle, overflow: TextOverflow.ellipsis),
          automaticallyImplyLeading: false,
          bottom: TabBar(
            isScrollable: false,
            tabs: List.generate(
              7,
              (i) => Tab(text: _dayLabels[i]),
            ),
          ),
        ),
        body: Column(
          children: [
            if (plan.personalizationNotes.isNotEmpty)
              Container(
                width: double.infinity,
                padding: const EdgeInsets.all(16),
                color: Theme.of(context)
                    .colorScheme
                    .primaryContainer
                    .withValues(alpha: 0.3),
                child: Text(
                  plan.personalizationNotes,
                  style: Theme.of(context).textTheme.bodyMedium,
                ),
              ),
            Expanded(
              child: TabBarView(
                children: List.generate(7, (i) {
                  final dayKey = 'day_${i + 1}';
                  final day = plan.weeklyPlan[dayKey];
                  if (day == null) {
                    return const Center(child: Text('No data'));
                  }
                  if (day.isRestDay) {
                    return Center(
                      child: Column(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          Icon(
                            Icons.self_improvement,
                            size: 64,
                            color: Colors.grey[400],
                          ),
                          const SizedBox(height: 16),
                          Text(
                            'Rest & Recovery',
                            style: Theme.of(context)
                                .textTheme
                                .headlineSmall
                                ?.copyWith(color: Colors.grey[600]),
                          ),
                          const SizedBox(height: 8),
                          Text(
                            day.focus,
                            style: TextStyle(color: Colors.grey[500]),
                          ),
                        ],
                      ),
                    );
                  }
                  return ListView(
                    padding: const EdgeInsets.symmetric(vertical: 12),
                    children: [
                      Padding(
                        padding: const EdgeInsets.symmetric(
                            horizontal: 16, vertical: 8),
                        child: Text(
                          day.focus,
                          style:
                              Theme.of(context).textTheme.titleMedium?.copyWith(
                                    fontWeight: FontWeight.bold,
                                  ),
                        ),
                      ),
                      ...day.exercises.map(
                        (ex) => ExerciseTile(exercise: ex),
                      ),
                    ],
                  );
                }),
              ),
            ),
            SafeArea(
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: FilledButton.icon(
                  onPressed: () {
                    Navigator.pushNamedAndRemoveUntil(
                      context,
                      '/home',
                      (route) => false,
                    );
                  },
                  icon: const Icon(Icons.arrow_back),
                  label: const Text('Choose Another Goal'),
                  style: FilledButton.styleFrom(
                    minimumSize: const Size(double.infinity, 48),
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
