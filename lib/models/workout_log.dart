/// One logged set, as returned by /api/logs.
class WorkoutLog {
  const WorkoutLog({
    required this.id,
    required this.exerciseName,
    required this.setNumber,
    this.repsCompleted,
    this.weightKg,
    this.durationSeconds,
    this.rpe,
    this.notes,
    required this.performedAt,
    this.planExerciseId,
  });

  final String id;
  final String exerciseName;
  final int setNumber;
  final int? repsCompleted;
  final double? weightKg;
  final int? durationSeconds;
  final double? rpe;
  final String? notes;
  final DateTime performedAt;
  final String? planExerciseId;

  /// Epley estimated 1-rep-max (kg): weight × (1 + reps/30). Null if either is
  /// missing/zero. The standard, defensible strength-progress signal.
  double? get estimated1RM {
    final w = weightKg;
    final r = repsCompleted;
    if (w == null || w <= 0 || r == null || r <= 0) return null;
    return w * (1 + r / 30.0);
  }

  /// Set volume (kg) = weight × reps. Null if either is missing.
  double? get volume {
    final w = weightKg;
    final r = repsCompleted;
    if (w == null || r == null) return null;
    return w * r;
  }

  factory WorkoutLog.fromJson(Map<String, dynamic> m) => WorkoutLog(
        id: (m['id'] ?? '').toString(),
        exerciseName: (m['exercise_name'] ?? '') as String,
        setNumber: (m['set_number'] ?? 1) as int,
        repsCompleted: (m['reps_completed'] as num?)?.toInt(),
        weightKg: (m['weight_kg'] as num?)?.toDouble(),
        durationSeconds: (m['duration_seconds'] as num?)?.toInt(),
        rpe: (m['rpe'] as num?)?.toDouble(),
        notes: m['notes'] as String?,
        performedAt:
            DateTime.tryParse((m['performed_at'] ?? '').toString())?.toLocal() ??
                DateTime.now(),
        planExerciseId: m['plan_exercise_id']?.toString(),
      );
}
