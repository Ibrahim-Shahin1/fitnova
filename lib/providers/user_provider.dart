import 'package:flutter/foundation.dart';
import '../models/fitness_plan.dart';

class UserProvider extends ChangeNotifier {
  // ── Registration fields ──────────────────────────────────────────────────
  String name = '';
  int age = 30;
  String gender = 'Male';
  double heightCm = 170.0;
  double weightKg = 70.0;
  List<String> injuries = [];

  double get bmi => weightKg / ((heightCm / 100) * (heightCm / 100));

  // ── Goal selection ───────────────────────────────────────────────────────
  String? workoutType;
  String? trainingFocus; // powerbuilding, powerlifting, hypertrophy, null

  // ── Optional profile enrichment ─────────────────────────────────────────
  int? yearsTraining;
  List<String> equipment = [];

  // ── Chat extraction ──────────────────────────────────────────────────────
  int? experienceLevel;
  double? sessionDurationHours;
  int? workoutFrequency;

  // ── Generated plan ───────────────────────────────────────────────────────
  FitnessPlan? currentPlan;

  void setProfile({
    required String name,
    required int age,
    required String gender,
    required double heightCm,
    required double weightKg,
  }) {
    this.name = name;
    this.age = age;
    this.gender = gender;
    this.heightCm = heightCm;
    this.weightKg = weightKg;
    notifyListeners();
  }

  void setInjuries(List<String> value) {
    injuries = value;
    notifyListeners();
  }

  void setWorkoutType(String type, {String? focus}) {
    workoutType = type;
    trainingFocus = focus;
    // Clear previous chat extraction when switching goals
    experienceLevel = null;
    sessionDurationHours = null;
    workoutFrequency = null;
    currentPlan = null;
    notifyListeners();
  }

  void updateExtracted(Map<String, dynamic> extracted) {
    if (extracted.containsKey('experience_level')) {
      experienceLevel = extracted['experience_level'] as int;
    }
    if (extracted.containsKey('session_duration_hours')) {
      sessionDurationHours = (extracted['session_duration_hours'] as num).toDouble();
    }
    if (extracted.containsKey('workout_frequency')) {
      workoutFrequency = extracted['workout_frequency'] as int;
    }
    notifyListeners();
  }

  void setPlan(FitnessPlan plan) {
    currentPlan = plan;
    notifyListeners();
  }

  bool get isReadyForPlan =>
      experienceLevel != null &&
      sessionDurationHours != null &&
      workoutFrequency != null;

  /// Build the full profile dict for POST /generate-plan.
  Map<String, dynamic> get generatePlanPayload => {
    'experience_level': experienceLevel!,
    'workout_type': workoutType!,
    'session_duration_hours': sessionDurationHours!,
    'workout_frequency': workoutFrequency!,
    'age': age,
    'gender': gender,
    'bmi': double.parse(bmi.toStringAsFixed(1)),
    'injuries': injuries,
    if (trainingFocus != null) 'training_focus': trainingFocus,
    if (yearsTraining != null) 'years_training': yearsTraining,
    if (equipment.isNotEmpty) 'equipment': equipment,
  };

  /// Build user_context dict for POST /chat.
  Map<String, dynamic> get chatUserContext => {
    'workout_type': workoutType!,
    'age': age,
    'gender': gender,
    'bmi': double.parse(bmi.toStringAsFixed(1)),
    'injuries': injuries,
    if (trainingFocus != null) 'training_focus': trainingFocus,
  };
}
