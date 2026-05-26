import 'package:flutter/material.dart';
import '../models/exercise_meta.dart';
import '../services/api_service.dart';

/// Exercise picker for the form-correction flow.
///
/// This milestone targets the three Fitness-AQA exercises only — Squat,
/// Overhead Press, Barbell Row. The backend still serves the full 27-exercise
/// SSOT at /api/exercises (shared with consistency tests + the recommender),
/// so we filter to the three CLIENT-SIDE here rather than editing the JSON.
/// Only Squat is implemented (OHP = Phase 6, Barbell Row = Phase 7); the other
/// two render as disabled "Coming soon" tiles so they never hit the squat-only
/// backend.
class ExerciseSelectionScreen extends StatefulWidget {
  const ExerciseSelectionScreen({super.key});

  @override
  State<ExerciseSelectionScreen> createState() =>
      _ExerciseSelectionScreenState();
}

class _ExerciseSelectionScreenState extends State<ExerciseSelectionScreen> {
  late Future<List<ExerciseMeta>> _exercisesFuture;

  // The three Fitness-AQA exercises, in display order.
  static const _allowOrder = <String>[
    'squat',
    'dumbbell_overhead_shoulder_press', // relabelled "Overhead Press" below
    'barbell_row',
  ];
  // Only Squat is wired to the backend this milestone.
  static const _enabledKeys = <String>{'squat'};
  // Cosmetic relabels so tiles read as the Fitness-AQA exercise names.
  static const _displayOverride = <String, String>{
    'dumbbell_overhead_shoulder_press': 'Overhead Press',
  };

  @override
  void initState() {
    super.initState();
    _exercisesFuture = _loadExercises();
  }

  Future<List<ExerciseMeta>> _loadExercises() async {
    final data = await ApiService.fetchExercises();
    final out = <ExerciseMeta>[];
    for (final key in _allowOrder) {
      final json = data[key];
      if (json != null) {
        out.add(ExerciseMeta.fromJson(key, json as Map<String, dynamic>));
      }
    }
    return out;
  }

  void _retry() {
    setState(() {
      _exercisesFuture = _loadExercises();
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Choose Exercise'),
      ),
      body: FutureBuilder<List<ExerciseMeta>>(
        future: _exercisesFuture,
        builder: (context, snapshot) {
          if (snapshot.connectionState == ConnectionState.waiting) {
            return const Center(child: CircularProgressIndicator());
          }

          if (snapshot.hasError) {
            return Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Text(
                    'Failed to load exercises',
                    style: Theme.of(context).textTheme.bodyLarge,
                  ),
                  const SizedBox(height: 16),
                  ElevatedButton(
                    onPressed: _retry,
                    child: const Text('Retry'),
                  ),
                ],
              ),
            );
          }

          final exercises = snapshot.data ?? [];
          return ListView(
            padding: const EdgeInsets.all(16),
            children: [
              Text(
                'Form analysis currently supports Squat. Overhead Press and '
                'Barbell Row are coming in upcoming phases.',
                style: Theme.of(context).textTheme.bodySmall?.copyWith(
                      color: Theme.of(context).colorScheme.onSurfaceVariant,
                    ),
              ),
              const SizedBox(height: 12),
              ...exercises.map(
                (meta) => _ExerciseTile(
                  meta: meta,
                  displayName: _displayOverride[meta.name] ?? meta.displayName,
                  enabled: _enabledKeys.contains(meta.name),
                ),
              ),
            ],
          );
        },
      ),
    );
  }
}

class _ExerciseTile extends StatelessWidget {
  final ExerciseMeta meta;
  final String displayName;
  final bool enabled;

  const _ExerciseTile({
    required this.meta,
    required this.displayName,
    required this.enabled,
  });

  IconData _iconForView(String view) {
    switch (view) {
      case 'side':
        return Icons.videocam;
      case 'front':
        return Icons.person;
      default:
        return Icons.view_in_ar;
    }
  }

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;

    return Opacity(
      opacity: enabled ? 1.0 : 0.55,
      child: Card(
        margin: const EdgeInsets.only(bottom: 10),
        child: InkWell(
          onTap: enabled
              ? () => Navigator.of(context)
                  .pushNamed('/guidelines', arguments: meta)
              : () => ScaffoldMessenger.of(context).showSnackBar(
                    SnackBar(
                      content: Text('$displayName form analysis is coming soon.'),
                      duration: const Duration(seconds: 2),
                    ),
                  ),
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: Row(
              children: [
                Icon(_iconForView(meta.cameraView), size: 32, color: cs.primary),
                const SizedBox(width: 16),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        displayName,
                        style: Theme.of(context).textTheme.titleMedium,
                      ),
                      const SizedBox(height: 2),
                      Text(
                        '${meta.cameraView[0].toUpperCase()}${meta.cameraView.substring(1)} view',
                        style: Theme.of(context).textTheme.bodySmall?.copyWith(
                              color: cs.onSurfaceVariant,
                            ),
                      ),
                    ],
                  ),
                ),
                if (enabled)
                  Icon(Icons.chevron_right, color: cs.onSurfaceVariant)
                else
                  Chip(
                    label: const Text('Coming soon', style: TextStyle(fontSize: 11)),
                    padding: EdgeInsets.zero,
                    materialTapTargetSize: MaterialTapTargetSize.shrinkWrap,
                    visualDensity: VisualDensity.compact,
                  ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
