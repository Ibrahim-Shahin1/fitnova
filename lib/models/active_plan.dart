/// The persisted active plan as returned by GET /api/plan/active.
class ActivePlan {
  const ActivePlan({
    required this.programTitle,
    this.personalizationNotes,
    required this.source,
    required this.days,
  });

  final String programTitle;
  final String? personalizationNotes;
  final String source;
  final List<PlanDay> days;

  int get trainingDayCount => days.where((d) => !d.isRestDay).length;

  factory ActivePlan.fromJson(Map<String, dynamic> m) => ActivePlan(
        programTitle: (m['program_title'] ?? 'Your plan') as String,
        personalizationNotes: m['personalization_notes'] as String?,
        source: (m['source'] ?? '') as String,
        days: ((m['days'] as List?) ?? const [])
            .map((d) => PlanDay.fromJson(d as Map<String, dynamic>))
            .toList(),
      );
}

class PlanDay {
  PlanDay({
    required this.id,
    required this.dayNumber,
    required this.focus,
    required this.isRestDay,
    required this.exercises,
    this.isCompleted = false,
  });

  final String id;
  final int dayNumber;
  final String focus;
  final bool isRestDay;
  final List<PlanExercise> exercises;

  /// Mutable so completion can be toggled optimistically.
  bool isCompleted;

  factory PlanDay.fromJson(Map<String, dynamic> m) => PlanDay(
        id: (m['id'] ?? '').toString(),
        dayNumber: (m['day_number'] ?? 0) as int,
        focus: (m['focus'] ?? '') as String,
        isRestDay: (m['is_rest_day'] ?? false) as bool,
        isCompleted: (m['is_completed'] ?? false) as bool,
        exercises: ((m['exercises'] as List?) ?? const [])
            .map((e) => PlanExercise.fromJson(e as Map<String, dynamic>))
            .toList(),
      );
}

class PlanExercise {
  PlanExercise({
    required this.id,
    required this.name,
    required this.sets,
    required this.reps,
    required this.restSeconds,
    this.coachingCue,
    this.mediaUrl,
    this.isCompleted = false,
  });

  final String id;
  final String name;
  final int sets;
  final String reps;
  final int restSeconds;
  final String? coachingCue;

  /// Backend-relative demo clip path (e.g. /static/exercise_videos/.../demo.mp4),
  /// or null when no demo is matched.
  final String? mediaUrl;

  /// Mutable so completion can be toggled optimistically.
  bool isCompleted;

  factory PlanExercise.fromJson(Map<String, dynamic> m) => PlanExercise(
        id: (m['id'] ?? '').toString(),
        name: (m['exercise_name'] ?? '') as String,
        sets: (m['sets'] ?? 0) as int,
        reps: (m['reps'] ?? '').toString(),
        restSeconds: (m['rest_seconds'] ?? 0) as int,
        coachingCue: m['coaching_cue'] as String?,
        mediaUrl: m['media_url'] as String?,
        isCompleted: (m['is_completed'] ?? false) as bool,
      );
}
