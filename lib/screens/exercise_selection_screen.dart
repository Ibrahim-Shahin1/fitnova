import 'package:flutter/material.dart';
import '../models/exercise_meta.dart';
import '../services/api_service.dart';

class ExerciseSelectionScreen extends StatefulWidget {
  const ExerciseSelectionScreen({super.key});

  @override
  State<ExerciseSelectionScreen> createState() =>
      _ExerciseSelectionScreenState();
}

class _ExerciseSelectionScreenState extends State<ExerciseSelectionScreen> {
  late Future<List<ExerciseMeta>> _exercisesFuture;

  @override
  void initState() {
    super.initState();
    _exercisesFuture = _loadExercises();
  }

  Future<List<ExerciseMeta>> _loadExercises() async {
    final data = await ApiService.fetchExercises();
    final exercises = <ExerciseMeta>[];
    data.forEach((name, json) {
      exercises.add(ExerciseMeta.fromJson(name, json as Map<String, dynamic>));
    });
    exercises.sort((a, b) => a.idx.compareTo(b.idx));
    return exercises;
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
          final sideExercises =
              exercises.where((e) => e.cameraView == 'side').toList();
          final frontExercises =
              exercises.where((e) => e.cameraView == 'front').toList();
          final eitherExercises =
              exercises.where((e) => e.cameraView == 'either').toList();

          return ListView(
            padding: const EdgeInsets.symmetric(vertical: 8),
            children: [
              if (sideExercises.isNotEmpty) ...[
                _SectionHeader('Side-view Exercises'),
                _ExerciseGrid(exercises: sideExercises),
              ],
              if (frontExercises.isNotEmpty) ...[
                _SectionHeader('Front-view Exercises'),
                _ExerciseGrid(exercises: frontExercises),
              ],
              if (eitherExercises.isNotEmpty) ...[
                _SectionHeader('Either View'),
                _ExerciseGrid(exercises: eitherExercises),
              ],
            ],
          );
        },
      ),
    );
  }
}

class _SectionHeader extends StatelessWidget {
  final String title;

  const _SectionHeader(this.title);

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 16, 16, 8),
      child: Text(
        title,
        style: Theme.of(context).textTheme.titleMedium?.copyWith(
              fontWeight: FontWeight.bold,
            ),
      ),
    );
  }
}

class _ExerciseGrid extends StatelessWidget {
  final List<ExerciseMeta> exercises;

  const _ExerciseGrid({required this.exercises});

  @override
  Widget build(BuildContext context) {
    return GridView.count(
      crossAxisCount: 2,
      shrinkWrap: true,
      physics: const NeverScrollableScrollPhysics(),
      childAspectRatio: 1.4,
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
      mainAxisSpacing: 8,
      crossAxisSpacing: 8,
      children: exercises
          .map((meta) => _ExerciseCard(meta: meta))
          .toList(),
    );
  }
}

class _ExerciseCard extends StatelessWidget {
  final ExerciseMeta meta;

  const _ExerciseCard({required this.meta});

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
    return Card(
      child: InkWell(
        onTap: () {
          Navigator.of(context).pushNamed('/guidelines', arguments: meta);
        },
        child: Padding(
          padding: const EdgeInsets.all(12),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(
                _iconForView(meta.cameraView),
                size: 32,
                color: Theme.of(context).colorScheme.primary,
              ),
              const SizedBox(height: 8),
              Text(
                meta.displayName,
                textAlign: TextAlign.center,
                style: Theme.of(context).textTheme.labelLarge,
              ),
              const SizedBox(height: 6),
              Chip(
                label: Text(
                  meta.cameraView.replaceFirst(meta.cameraView[0],
                      meta.cameraView[0].toUpperCase()),
                  style: const TextStyle(fontSize: 11),
                ),
                padding: EdgeInsets.zero,
                materialTapTargetSize: MaterialTapTargetSize.shrinkWrap,
              ),
            ],
          ),
        ),
      ),
    );
  }
}
