import 'dart:convert';

import 'package:http/http.dart' as http;

import '../config/api_config.dart';
import 'supabase_service.dart';

/// Opens the SSE stream behind the live 4-agent generation screen and yields
/// decoded event maps as the crew works: started → agent(running/done)… →
/// done{plan_id,quality_report} | error{error}.
class PlanStreamService {
  const PlanStreamService._();

  static Stream<Map<String, dynamic>> generate(
      Map<String, dynamic> params) async* {
    final token = SupabaseService.client.auth.currentSession?.accessToken;
    final req = http.Request(
        'POST', Uri.parse('${ApiConfig.baseUrl}/api/plan/generate/stream'));
    req.headers['Content-Type'] = 'application/json';
    if (token != null) req.headers['Authorization'] = 'Bearer $token';
    req.body = jsonEncode(params);

    final client = http.Client();
    try {
      final resp = await client.send(req).timeout(const Duration(seconds: 240));
      if (resp.statusCode != 200) {
        yield {'event': 'error', 'error': 'Server returned ${resp.statusCode}'};
        return;
      }
      final lines =
          resp.stream.transform(utf8.decoder).transform(const LineSplitter());
      await for (final line in lines) {
        final l = line.trim();
        if (!l.startsWith('data:')) continue;
        final payload = l.substring(5).trim();
        if (payload.isEmpty) continue;
        try {
          yield jsonDecode(payload) as Map<String, dynamic>;
        } catch (_) {
          // skip malformed frame
        }
      }
    } catch (e) {
      yield {'event': 'error', 'error': "Couldn't reach the builder: $e"};
    } finally {
      client.close();
    }
  }
}
