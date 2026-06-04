import 'dart:convert';
import 'dart:typed_data';
import 'package:http/http.dart' as http;
import '../config/api_config.dart';
import '../models/benchmark_models.dart';
import '../models/chat_models.dart';
import '../models/fitness_plan.dart';
import '../models/form_models.dart';

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

  static Future<Map<String, dynamic>> fetchExercises() async {
    final url = Uri.parse('${ApiConfig.baseUrl}/api/exercises');
    final response = await http.get(url);
    if (response.statusCode != 200) {
      throw Exception('Failed to load exercises: ${response.statusCode}');
    }
    return jsonDecode(response.body) as Map<String, dynamic>;
  }

  static Future<List<FormRep>> analyzeClip(
    String exercise,
    String clipId, {
    Uint8List? fileBytes,
  }) async {
    final url = Uri.parse('${ApiConfig.baseUrl}/analyze-form-video');
    final request = http.MultipartRequest('POST', url)
      ..fields['exercise'] = exercise
      ..fields['clip_id'] = clipId;

    if (fileBytes != null) {
      request.files.add(
        http.MultipartFile.fromBytes('file', fileBytes,
            filename: '$clipId.mp4'),
      );
    }

    final streamed = await request.send().timeout(const Duration(minutes: 4));
    final response = await http.Response.fromStream(streamed);
    if (response.statusCode != 200) {
      throw Exception(
          'Analysis failed: ${response.statusCode} ${response.body}');
    }
    final json = jsonDecode(response.body) as Map<String, dynamic>;
    return (json['reps'] as List<dynamic>)
        .map((r) => FormRep.fromJson(r as Map<String, dynamic>))
        .toList();
  }

  static Future<List<BenchmarkClip>> getBenchmarkCatalog(
      String exercise) async {
    final url = Uri.parse(
        '${ApiConfig.baseUrl}/api/benchmark/catalog?exercise=$exercise');
    final response = await http.get(url);
    if (response.statusCode != 200) {
      throw Exception(
          'Catalog load failed: ${response.statusCode} ${response.body}');
    }
    final json = jsonDecode(response.body) as Map<String, dynamic>;
    return (json['clips'] as List<dynamic>)
        .map((e) => BenchmarkClip.fromJson(e as Map<String, dynamic>))
        .toList();
  }

  static String benchmarkMediaUrl(String exercise, String clipId) =>
      '${ApiConfig.baseUrl}/api/benchmark/media/$exercise/$clipId';
}
