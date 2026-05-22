import 'dart:convert';

import 'package:http/http.dart' as http;

import '../config/api_config.dart';
import '../models/coach_message.dart';
import 'supabase_service.dart';

/// Talks to the backend coach endpoints (`/api/chat/*`) with the user's
/// Supabase access token attached as a Bearer header.
class ConversationService {
  const ConversationService._();

  static Map<String, String> _headers() {
    final token = SupabaseService.client.auth.currentSession?.accessToken;
    return {
      'Content-Type': 'application/json',
      if (token != null) 'Authorization': 'Bearer $token',
    };
  }

  /// Load the full coach transcript for the signed-in user.
  static Future<List<CoachMessage>> fetchMessages() async {
    final url = Uri.parse('${ApiConfig.baseUrl}/api/chat/messages');
    final resp = await http.get(url, headers: _headers());
    if (resp.statusCode != 200) {
      throw Exception('Failed to load messages: ${resp.statusCode}');
    }
    final data = jsonDecode(resp.body) as Map<String, dynamic>;
    final msgs = ((data['messages'] as List?) ?? const [])
        .cast<Map<String, dynamic>>();
    return msgs.map(CoachMessage.fromJson).toList();
  }

  /// Send a message to the coach; returns its reply (the tool-loop runs server
  /// side, so this can take a while — generous timeout).
  static Future<CoachReply> sendMessage(String content) async {
    final url = Uri.parse('${ApiConfig.baseUrl}/api/chat/messages');
    final resp = await http
        .post(url, headers: _headers(), body: jsonEncode({'content': content}))
        .timeout(const Duration(seconds: 120));
    if (resp.statusCode != 200) {
      throw Exception('Coach failed: ${resp.statusCode} ${resp.body}');
    }
    return CoachReply.fromJson(jsonDecode(resp.body) as Map<String, dynamic>);
  }
}
