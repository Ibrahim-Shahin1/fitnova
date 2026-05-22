import 'dart:convert';

import 'package:http/http.dart' as http;

import '../config/api_config.dart';
import '../models/active_plan.dart';
import 'supabase_service.dart';

/// Reads the user's active plan from the backend (authed with the Supabase token).
class PlanService {
  const PlanService._();

  static Map<String, String> _headers() {
    final token = SupabaseService.client.auth.currentSession?.accessToken;
    return {
      'Content-Type': 'application/json',
      if (token != null) 'Authorization': 'Bearer $token',
    };
  }

  /// Fetch the active plan, or null if the user has none yet.
  static Future<ActivePlan?> fetchActive() async {
    final url = Uri.parse('${ApiConfig.baseUrl}/api/plan/active');
    final resp = await http.get(url, headers: _headers());
    if (resp.statusCode != 200) {
      throw Exception('Failed to load plan: ${resp.statusCode}');
    }
    final data = jsonDecode(resp.body) as Map<String, dynamic>;
    final plan = data['plan'];
    if (plan == null) return null;
    return ActivePlan.fromJson(plan as Map<String, dynamic>);
  }
}
