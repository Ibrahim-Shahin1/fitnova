/// Maps a row of the Supabase `public.profiles` table.
///
/// Detail fields are nullable — the row is created blank by the `handle_new_user`
/// trigger on sign-up and filled in during onboarding.
class UserProfile {
  const UserProfile({
    required this.id,
    this.displayName,
    this.age,
    this.gender,
    this.heightCm,
    this.weightKg,
    this.injuries = const [],
    this.trainingFocus,
    this.yearsTraining,
    this.equipment = const [],
    this.experienceLevel,
    this.sessionDurationHours,
    this.workoutFrequency,
    this.unitPreference = 'metric',
    this.onboardingCompleted = false,
  });

  final String id;
  final String? displayName;
  final int? age;
  final String? gender;
  final double? heightCm;
  final double? weightKg;
  final List<String> injuries;
  final String? trainingFocus;
  final int? yearsTraining;
  final List<String> equipment;
  final int? experienceLevel;
  final double? sessionDurationHours;
  final int? workoutFrequency;
  final String unitPreference;
  final bool onboardingCompleted;

  factory UserProfile.fromMap(Map<String, dynamic> m) {
    return UserProfile(
      id: m['id'] as String,
      displayName: m['display_name'] as String?,
      age: _toInt(m['age']),
      gender: m['gender'] as String?,
      heightCm: _toDouble(m['height_cm']),
      weightKg: _toDouble(m['weight_kg']),
      injuries: _toStringList(m['injuries']),
      trainingFocus: m['training_focus'] as String?,
      yearsTraining: _toInt(m['years_training']),
      equipment: _toStringList(m['equipment']),
      experienceLevel: _toInt(m['experience_level']),
      sessionDurationHours: _toDouble(m['session_duration_hours']),
      workoutFrequency: _toInt(m['workout_frequency']),
      unitPreference: (m['unit_preference'] as String?) ?? 'metric',
      onboardingCompleted: (m['onboarding_completed'] as bool?) ?? false,
    );
  }

  /// Client-writable fields only (id is the key; created_at/updated_at are
  /// managed by the database).
  Map<String, dynamic> toUpdateMap() => {
        'display_name': displayName,
        'age': age,
        'gender': gender,
        'height_cm': heightCm,
        'weight_kg': weightKg,
        'injuries': injuries,
        'training_focus': trainingFocus,
        'years_training': yearsTraining,
        'equipment': equipment,
        'experience_level': experienceLevel,
        'session_duration_hours': sessionDurationHours,
        'workout_frequency': workoutFrequency,
        'unit_preference': unitPreference,
        'onboarding_completed': onboardingCompleted,
      };

  UserProfile copyWith({
    String? displayName,
    int? age,
    String? gender,
    double? heightCm,
    double? weightKg,
    List<String>? injuries,
    String? trainingFocus,
    int? yearsTraining,
    List<String>? equipment,
    int? experienceLevel,
    double? sessionDurationHours,
    int? workoutFrequency,
    String? unitPreference,
    bool? onboardingCompleted,
  }) {
    return UserProfile(
      id: id,
      displayName: displayName ?? this.displayName,
      age: age ?? this.age,
      gender: gender ?? this.gender,
      heightCm: heightCm ?? this.heightCm,
      weightKg: weightKg ?? this.weightKg,
      injuries: injuries ?? this.injuries,
      trainingFocus: trainingFocus ?? this.trainingFocus,
      yearsTraining: yearsTraining ?? this.yearsTraining,
      equipment: equipment ?? this.equipment,
      experienceLevel: experienceLevel ?? this.experienceLevel,
      sessionDurationHours: sessionDurationHours ?? this.sessionDurationHours,
      workoutFrequency: workoutFrequency ?? this.workoutFrequency,
      unitPreference: unitPreference ?? this.unitPreference,
      onboardingCompleted: onboardingCompleted ?? this.onboardingCompleted,
    );
  }

  static int? _toInt(dynamic v) => v == null ? null : (v as num).toInt();
  static double? _toDouble(dynamic v) => v == null ? null : (v as num).toDouble();
  static List<String> _toStringList(dynamic v) =>
      v == null ? const <String>[] : (v as List).map((e) => e.toString()).toList();
}
