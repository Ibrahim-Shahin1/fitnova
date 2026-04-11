import 'dart:convert';
import 'package:http/http.dart' as http;
import '../config/api_config.dart';
import '../models/chat_models.dart';
import '../models/fitness_plan.dart';

class ApiService {
  static Future<ChatResponse> sendChat({
    required List<ChatMessage> conversation,
    required Map<String, dynamic> userContext,
  }) async {
    final url = Uri.parse('${ApiConfig.baseUrl}/chat');
    final response = await http.post(
      url,
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({
        'conversation': conversation.map((m) => m.toJson()).toList(),
        'user_context': userContext,
      }),
    );

    if (response.statusCode != 200) {
      throw Exception('Chat failed: ${response.statusCode} ${response.body}');
    }

    return ChatResponse.fromJson(
      jsonDecode(response.body) as Map<String, dynamic>,
    );
  }

  static Future<FitnessPlan> generatePlan(
      Map<String, dynamic> profilePayload) async {
    final url = Uri.parse('${ApiConfig.baseUrl}/generate-plan');
    final response = await http.post(
      url,
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode(profilePayload),
    );

    if (response.statusCode != 200) {
      throw Exception(
          'Plan generation failed: ${response.statusCode} ${response.body}');
    }

    return FitnessPlan.fromJson(
      jsonDecode(response.body) as Map<String, dynamic>,
    );
  }
}
