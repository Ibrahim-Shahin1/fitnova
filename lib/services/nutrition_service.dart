import 'dart:async';
import 'dart:convert';

import 'package:http/http.dart' as http;

import '../config/api_config.dart';
import '../models/meal_plan.dart';

/// REST + SSE client for the nutrition feature (/api/nutrition/*).
/// Stateless like the rest of ApiService — no auth header (the nutrition
/// endpoints are public, mirroring the legacy /generate-plan + /chat).
class NutritionService {
  /// POST /api/nutrition/targets — instant BMI + calorie/macro targets.
  static Future<NutritionTargets> fetchTargets(
      Map<String, dynamic> body) async {
    final url = Uri.parse('${ApiConfig.baseUrl}/api/nutrition/targets');
    final res = await http.post(
      url,
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode(body),
    );
    if (res.statusCode != 200) {
      throw Exception('Targets failed: ${res.statusCode} ${res.body}');
    }
    return NutritionTargets.fromJson(
        jsonDecode(res.body) as Map<String, dynamic>);
  }

  /// POST /api/nutrition/generate — full plan (blocking; used as a fallback).
  static Future<MealPlan> generate(Map<String, dynamic> body) async {
    final url = Uri.parse('${ApiConfig.baseUrl}/api/nutrition/generate');
    final res = await http.post(
      url,
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode(body),
    );
    if (res.statusCode != 200) {
      throw Exception('Generation failed: ${res.statusCode} ${res.body}');
    }
    return MealPlan.fromJson(jsonDecode(res.body) as Map<String, dynamic>);
  }

  /// POST /api/nutrition/recipe-instructions — generate cooking instructions for
  /// one recipe via the fine-tuned DistilGPT-2 (the N2 model). Throws on failure
  /// (503 if the model isn't installed server-side).
  static Future<String> recipeInstructions({
    required String name,
    required List<String> ingredients,
    String calorieLevel = 'medium',
  }) async {
    final url =
        Uri.parse('${ApiConfig.baseUrl}/api/nutrition/recipe-instructions');
    final res = await http.post(
      url,
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({
        'name': name,
        'ingredients': ingredients,
        'calorie_level': calorieLevel,
      }),
    );
    if (res.statusCode == 503) {
      throw Exception("The recipe-writing model isn't available on the server.");
    }
    if (res.statusCode != 200) {
      throw Exception('Instructions failed: ${res.statusCode} ${res.body}');
    }
    return (jsonDecode(res.body) as Map<String, dynamic>)['instructions']
            as String? ??
        '';
  }

  /// POST /api/nutrition/generate/stream — SSE per-agent progress.
  /// Yields {event: 'started'|'agent'|'plan'|'error', ...}.
  static Stream<Map<String, dynamic>> generateStream(
      Map<String, dynamic> body) async* {
    final url = Uri.parse('${ApiConfig.baseUrl}/api/nutrition/generate/stream');
    final request = http.Request('POST', url)
      ..headers['Content-Type'] = 'application/json'
      ..headers['Accept'] = 'text/event-stream'
      ..body = jsonEncode(body);

    final streamed = await http.Client().send(request);
    if (streamed.statusCode != 200) {
      final b = await streamed.stream.bytesToString();
      throw Exception('Stream failed: ${streamed.statusCode} $b');
    }
    final lines = streamed.stream
        .transform(utf8.decoder)
        .transform(const LineSplitter());
    await for (final line in lines) {
      if (line.startsWith('data: ')) {
        final payload = line.substring(6);
        if (payload.trim().isEmpty) continue;
        yield jsonDecode(payload) as Map<String, dynamic>;
      }
    }
  }

  /// POST /api/nutrition/recipe-chat — converse with GPT-4o-mini about one recipe
  /// (substitutions, scaling, technique). [recipe] carries name/ingredients/steps/
  /// ai_instructions for grounding; [messages] is the running history.
  static Future<String> recipeChat({
    required Map<String, dynamic> recipe,
    required List<Map<String, String>> messages,
  }) async {
    final url = Uri.parse('${ApiConfig.baseUrl}/api/nutrition/recipe-chat');
    final res = await http.post(
      url,
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({'recipe': recipe, 'messages': messages}),
    );
    if (res.statusCode == 503) {
      throw Exception('The recipe chat assistant is unavailable right now.');
    }
    if (res.statusCode != 200) {
      throw Exception('Chat failed: ${res.statusCode} ${res.body}');
    }
    return (jsonDecode(res.body) as Map<String, dynamic>)['reply'] as String? ?? '';
  }

  /// POST /api/nutrition/plan-chat — modify an EXISTING plan by chat (swap a meal,
  /// or add a constraint like "no fish" and re-pick every violating meal). Returns
  /// {reply, plan?} — `plan` (a full MealPlan json) is present only when it changed.
  static Future<({String reply, Map<String, dynamic>? plan})> planChat({
    required Map<String, dynamic> plan,
    required String message,
  }) async {
    final url = Uri.parse('${ApiConfig.baseUrl}/api/nutrition/plan-chat');
    final res = await http.post(
      url,
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({'plan': plan, 'message': message}),
    );
    if (res.statusCode == 503) {
      throw Exception('The plan assistant is unavailable right now.');
    }
    if (res.statusCode != 200) {
      throw Exception('Plan chat failed: ${res.statusCode} ${res.body}');
    }
    final body = jsonDecode(res.body) as Map<String, dynamic>;
    return (
      reply: body['reply'] as String? ?? '',
      plan: body['plan'] as Map<String, dynamic>?,
    );
  }
}
