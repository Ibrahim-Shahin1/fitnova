import 'dart:convert';

import 'package:flutter/foundation.dart';
import 'package:shared_preferences/shared_preferences.dart';

import '../models/meal_plan.dart';

/// Holds the user's nutrition intake inputs + the last generated meal plan,
/// persisted locally (SharedPreferences) so it survives app restarts.
///
/// Persistence is SCOPED PER USER (keys are suffixed with the auth user id) so a
/// new account never sees a previous session's plan on the same device. The
/// nutrition backend endpoints are stateless, so this local store is the only
/// place a plan lives; keeping it user-scoped is what prevents cross-user leaks.
class NutritionProvider extends ChangeNotifier {
  NutritionProvider({this.userId});

  /// Auth user id; null only if somehow unauthenticated (falls back to 'guest').
  final String? userId;

  // Legacy un-scoped keys (pre-fix). Deleted on load so old global plans don't
  // leak into new accounts.
  static const _legacyPlanKey = 'nutrition_plan_json';
  static const _legacyIntakeKey = 'nutrition_intake_json';

  String get _scope => userId ?? 'guest';
  String get _planKey => 'nutrition_plan_$_scope';
  String get _intakeKey => 'nutrition_intake_$_scope';

  MealPlan? _plan;
  Map<String, dynamic>? _planRaw; // raw json, kept so plan-chat can mutate it
  Map<String, dynamic> _intake = _defaultIntake();
  bool _loading = true;

  MealPlan? get plan => _plan;
  Map<String, dynamic>? get planRaw => _planRaw;
  Map<String, dynamic> get intake => _intake;
  bool get loading => _loading;
  bool get hasPlan => _plan != null;

  static Map<String, dynamic> _defaultIntake() => {
        'activity_level': 'moderate',
        'goal': 'maintain',
        'days': 7,
        'meals_per_day': 4,
        'diet': <String>[],
        'exclude_ingredients': <String>[],
      };

  Future<void> load() async {
    final prefs = await SharedPreferences.getInstance();

    // One-time cleanup: remove the old un-scoped keys so a stale plan saved
    // before per-user scoping can't show up for a different/new account.
    await prefs.remove(_legacyPlanKey);
    await prefs.remove(_legacyIntakeKey);

    final intakeStr = prefs.getString(_intakeKey);
    if (intakeStr != null) {
      try {
        _intake = {..._defaultIntake(), ...jsonDecode(intakeStr) as Map<String, dynamic>};
      } catch (_) {/* keep defaults */}
    }
    final planStr = prefs.getString(_planKey);
    if (planStr != null) {
      try {
        _plan = MealPlan.fromJson(jsonDecode(planStr) as Map<String, dynamic>);
      } catch (_) {/* ignore corrupt cache */}
    }
    _loading = false;
    notifyListeners();
  }

  Future<void> saveIntake(Map<String, dynamic> intake) async {
    _intake = {..._intake, ...intake};
    notifyListeners();
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_intakeKey, jsonEncode(_intake));
  }

  Future<void> setPlan(MealPlan plan, Map<String, dynamic> rawJson) async {
    _plan = plan;
    _planRaw = rawJson;
    notifyListeners();
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_planKey, jsonEncode(rawJson));
  }

  /// Apply a plan returned by plan-chat (already constraint-verified server-side).
  Future<void> applyPlanJson(Map<String, dynamic> rawJson) async {
    _planRaw = rawJson;
    _plan = MealPlan.fromJson(rawJson);
    notifyListeners();
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_planKey, jsonEncode(rawJson));
  }

  Future<void> clearPlan() async {
    _plan = null;
    _planRaw = null;
    notifyListeners();
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove(_planKey);
  }
}
