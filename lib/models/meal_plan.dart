// Nutrition meal-plan models — mirror the backend /api/nutrition response shape.
// Plain data classes with fromJson factories (app convention; no serialization libs).

class NutritionTargets {
  const NutritionTargets({
    required this.bmi,
    required this.bmiCategory,
    required this.bmr,
    required this.tdee,
    required this.targetCalories,
    required this.proteinG,
    required this.carbsG,
    required this.fatG,
    required this.goal,
    required this.activityLevel,
  });

  final double bmi;
  final String bmiCategory;
  final int bmr;
  final int tdee;
  final int targetCalories;
  final int proteinG;
  final int carbsG;
  final int fatG;
  final String goal;
  final String activityLevel;

  factory NutritionTargets.fromJson(Map<String, dynamic> j) => NutritionTargets(
        bmi: (j['bmi'] as num).toDouble(),
        bmiCategory: j['bmi_category'] as String? ?? '',
        bmr: (j['bmr'] as num?)?.toInt() ?? 0,
        tdee: (j['tdee'] as num?)?.toInt() ?? 0,
        targetCalories: (j['target_calories'] as num?)?.toInt() ?? 0,
        proteinG: (j['protein_g'] as num?)?.toInt() ?? 0,
        carbsG: (j['carbs_g'] as num?)?.toInt() ?? 0,
        fatG: (j['fat_g'] as num?)?.toInt() ?? 0,
        goal: j['goal'] as String? ?? '',
        activityLevel: j['activity_level'] as String? ?? '',
      );
}

class Meal {
  const Meal({
    required this.slot,
    required this.name,
    required this.calories,
    required this.proteinG,
    required this.carbsG,
    required this.fatG,
    required this.minutes,
    required this.recipeId,
    required this.ingredients,
    required this.calorieLevel,
    required this.steps,
  });

  final String slot;
  final String name;
  final double calories;
  final double proteinG;
  final double carbsG;
  final double fatG;
  final int minutes;
  final int recipeId;
  final List<String> ingredients;
  final int calorieLevel; // 0 low, 1 medium, 2 high
  final List<String> steps; // ORIGINAL Food.com dataset steps

  static const _calWords = {0: 'low', 1: 'medium', 2: 'high'};
  String get calorieLevelWord => _calWords[calorieLevel] ?? 'medium';

  factory Meal.fromJson(Map<String, dynamic> j) => Meal(
        slot: j['slot'] as String? ?? '',
        name: j['name'] as String? ?? 'Recipe',
        calories: (j['calories'] as num?)?.toDouble() ?? 0,
        proteinG: (j['protein_g'] as num?)?.toDouble() ?? 0,
        carbsG: (j['carbs_g'] as num?)?.toDouble() ?? 0,
        fatG: (j['fat_g'] as num?)?.toDouble() ?? 0,
        minutes: (j['minutes'] as num?)?.toInt() ?? 0,
        recipeId: (j['recipe_id'] as num?)?.toInt() ?? 0,
        ingredients:
            ((j['ingredients'] as List?) ?? const []).map((e) => '$e').toList(),
        calorieLevel: (j['calorie_level'] as num?)?.toInt() ?? 1,
        steps: ((j['steps'] as List?) ?? const []).map((e) => '$e').toList(),
      );
}

class MealDay {
  const MealDay({
    required this.dayNumber,
    required this.theme,
    required this.meals,
    required this.totalCalories,
    required this.proteinG,
    required this.carbsG,
    required this.fatG,
    required this.calorieMatchPct,
  });

  final int dayNumber;
  final String theme;
  final List<Meal> meals;
  final int totalCalories;
  final int proteinG;
  final int carbsG;
  final int fatG;
  final double calorieMatchPct;

  factory MealDay.fromJson(Map<String, dynamic> j) {
    final totals = (j['totals'] as Map?)?.cast<String, dynamic>() ?? const {};
    return MealDay(
      dayNumber: (j['day_number'] as num?)?.toInt() ?? 0,
      theme: j['theme'] as String? ?? '',
      meals: ((j['meals'] as List?) ?? const [])
          .map((m) => Meal.fromJson((m as Map).cast<String, dynamic>()))
          .toList(),
      totalCalories: (totals['calories'] as num?)?.toInt() ?? 0,
      proteinG: (totals['protein_g'] as num?)?.toInt() ?? 0,
      carbsG: (totals['carbs_g'] as num?)?.toInt() ?? 0,
      fatG: (totals['fat_g'] as num?)?.toInt() ?? 0,
      calorieMatchPct: (j['calorie_match_pct'] as num?)?.toDouble() ?? 0,
    );
  }
}

class MealPlan {
  const MealPlan({
    required this.planTitle,
    required this.targets,
    required this.targetsRationale,
    required this.coachingNotes,
    required this.qualityScore,
    required this.avgCalorieMatchPct,
    required this.source,
    required this.mealsChecked,
    required this.violationCount,
    required this.days,
    required this.diet,
    required this.excludedIngredients,
  });

  final String planTitle;
  final NutritionTargets targets;
  final String targetsRationale;
  final String coachingNotes;
  final int? qualityScore;
  final double avgCalorieMatchPct;
  final String source;
  final int mealsChecked;
  final int violationCount;
  final List<MealDay> days;
  final List<String> diet;
  final List<String> excludedIngredients;

  bool get verifiedClean => mealsChecked > 0 && violationCount == 0;

  factory MealPlan.fromJson(Map<String, dynamic> j) {
    final q = (j['quality_report'] as Map?)?.cast<String, dynamic>() ?? const {};
    final v = (q['verification'] as Map?)?.cast<String, dynamic>() ?? const {};
    return MealPlan(
      planTitle: j['plan_title'] as String? ?? 'Meal Plan',
      targets: NutritionTargets.fromJson(
          (j['targets'] as Map).cast<String, dynamic>()),
      targetsRationale: j['targets_rationale'] as String? ?? '',
      coachingNotes: j['coaching_notes'] as String? ?? '',
      qualityScore: (q['score'] as num?)?.toInt(),
      avgCalorieMatchPct: (q['avg_calorie_match_pct'] as num?)?.toDouble() ?? 0,
      source: j['source'] as String? ?? 'deterministic',
      mealsChecked: (v['meals_checked'] as num?)?.toInt() ?? 0,
      violationCount: ((v['violations'] as List?)?.length) ?? 0,
      days: ((j['days'] as List?) ?? const [])
          .map((d) => MealDay.fromJson((d as Map).cast<String, dynamic>()))
          .toList(),
      diet: ((j['diet'] as List?) ?? const []).map((e) => '$e').toList(),
      excludedIngredients:
          ((j['excluded_ingredients'] as List?) ?? const []).map((e) => '$e').toList(),
    );
  }
}
