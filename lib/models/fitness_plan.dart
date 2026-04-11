class Exercise {
  final String exerciseName;
  final int sets;
  final String reps;
  final int restSeconds;
  final String coachingCue;
  final String? mediaUrl;

  Exercise({
    required this.exerciseName,
    required this.sets,
    required this.reps,
    required this.restSeconds,
    required this.coachingCue,
    this.mediaUrl,
  });

  factory Exercise.fromJson(Map<String, dynamic> json) {
    return Exercise(
      exerciseName: json['exercise_name'] as String,
      sets: json['sets'] as int,
      reps: json['reps'] as String,
      restSeconds: json['rest_seconds'] as int,
      coachingCue: json['coaching_cue'] as String,
      mediaUrl: json['media_url'] as String?,
    );
  }
}

class DayPlan {
  final int dayNumber;
  final String focus;
  final bool isRestDay;
  final List<Exercise> exercises;

  DayPlan({
    required this.dayNumber,
    required this.focus,
    required this.isRestDay,
    required this.exercises,
  });

  factory DayPlan.fromJson(Map<String, dynamic> json) {
    return DayPlan(
      dayNumber: json['day_number'] as int,
      focus: json['focus'] as String,
      isRestDay: json['is_rest_day'] as bool,
      exercises: (json['exercises'] as List)
          .map((e) => Exercise.fromJson(e as Map<String, dynamic>))
          .toList(),
    );
  }
}

class FitnessPlan {
  final int programId;
  final String programTitle;
  final int similarUserId;
  final String personalizationNotes;
  final String source;
  final Map<String, DayPlan> weeklyPlan;

  FitnessPlan({
    required this.programId,
    required this.programTitle,
    required this.similarUserId,
    required this.personalizationNotes,
    required this.source,
    required this.weeklyPlan,
  });

  factory FitnessPlan.fromJson(Map<String, dynamic> json) {
    final planMap = json['weekly_plan'] as Map<String, dynamic>;
    final weekly = planMap.map(
      (key, value) =>
          MapEntry(key, DayPlan.fromJson(value as Map<String, dynamic>)),
    );
    return FitnessPlan(
      programId: json['program_id'] as int,
      programTitle: json['program_title'] as String,
      similarUserId: json['similar_user_id'] as int,
      personalizationNotes: json['personalization_notes'] as String,
      source: json['source'] as String,
      weeklyPlan: weekly,
    );
  }
}
