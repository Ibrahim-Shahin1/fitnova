import 'package:flutter/foundation.dart';

import '../models/active_plan.dart';
import '../services/plan_service.dart';

/// Holds the user's active plan for the Planning tab. Reloaded on open and after
/// the coach changes the plan.
class PlanProvider extends ChangeNotifier {
  ActivePlan? _plan;
  bool _loading = false;
  Object? _error;

  ActivePlan? get plan => _plan;
  bool get loading => _loading;
  Object? get error => _error;

  Future<void> load() async {
    _loading = true;
    _error = null;
    notifyListeners();
    try {
      _plan = await PlanService.fetchActive();
    } catch (e) {
      _error = e;
    } finally {
      _loading = false;
      notifyListeners();
    }
  }
}
