import 'dart:convert';

import 'package:http/http.dart' as http;

import '../config/api_config.dart';
import '../models/workout_log.dart';
import 'supabase_service.dart';

/// Talks to the backend logging endpoints (`/api/logs`) with the user's
/// Supabase access token attached as a Bearer header.
class LogService {
  const LogService._();

  static Map<String, String> _headers() {
    final token = SupabaseService.client.auth.currentSession?.accessToken;
    return {
      'Content-Type': 'application/json',
      if (token != null) 'Authorization': 'Bearer $token',
    };
  }

  /// Log one set. Returns the stored row.
  static Future<WorkoutLog> createLog({
    required String exerciseName,
    required int setNumber,
    int? repsCompleted,
    double? weightKg,
    int? durationSeconds,
    double? rpe,
    String? notes,
    String? planExerciseId,
  }) async {
    final url = Uri.parse('${ApiConfig.baseUrl}/api/logs');
    final body = <String, dynamic>{
      'exercise_name': exerciseName,
      'set_number': setNumber,
      if (repsCompleted != null) 'reps_completed': repsCompleted,
      if (weightKg != null) 'weight_kg': weightKg,
      if (durationSeconds != null) 'duration_seconds': durationSeconds,
      if (rpe != null) 'rpe': rpe,
      if (notes != null && notes.isNotEmpty) 'notes': notes,
      if (planExerciseId != null) 'plan_exercise_id': planExerciseId,
    };
    final resp = await http.post(url, headers: _headers(), body: jsonEncode(body));
    if (resp.statusCode != 200) {
      throw Exception('Log failed: ${resp.statusCode} ${resp.body}');
    }
    final data = jsonDecode(resp.body) as Map<String, dynamic>;
    return WorkoutLog.fromJson(data['log'] as Map<String, dynamic>);
  }

  /// Fetch the user's logs within [days] (newest first), optionally filtered.
  static Future<List<WorkoutLog>> fetchLogs({
    int days = 120,
    String? exerciseName,
  }) async {
    final qp = {
      'days': '$days',
      if (exerciseName != null && exerciseName.isNotEmpty)
        'exercise_name': exerciseName,
    };
    final url = Uri.parse('${ApiConfig.baseUrl}/api/logs')
        .replace(queryParameters: qp);
    final resp = await http.get(url, headers: _headers());
    if (resp.statusCode != 200) {
      throw Exception('Failed to load logs: ${resp.statusCode}');
    }
    final data = jsonDecode(resp.body) as Map<String, dynamic>;
    return ((data['logs'] as List?) ?? const [])
        .cast<Map<String, dynamic>>()
        .map(WorkoutLog.fromJson)
        .toList();
  }

  /// Delete one of the user's logs.
  static Future<void> deleteLog(String id) async {
    final url = Uri.parse('${ApiConfig.baseUrl}/api/logs/$id');
    final resp = await http.delete(url, headers: _headers());
    if (resp.statusCode != 200) {
      throw Exception('Delete failed: ${resp.statusCode}');
    }
  }
}
