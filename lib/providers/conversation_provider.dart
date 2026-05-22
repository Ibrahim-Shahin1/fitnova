import 'package:flutter/foundation.dart';

import '../models/active_plan.dart';
import '../models/coach_message.dart';
import '../services/conversation_service.dart';
import '../services/plan_service.dart';

/// Drives the coach chat screen: holds the visible transcript, the initial-load
/// state, the "coach is replying" state, and a distinct "building your plan"
/// state used to show a richer loader while the LLM generates a plan. The
/// transcript is the server's source of truth (reloaded on open); the plan
/// reveal card is injected locally and intentionally not persisted.
class ConversationProvider extends ChangeNotifier {
  final List<CoachMessage> _messages = [];
  bool _loading = false;
  bool _sending = false;
  bool _buildingPlan = false;
  Object? _error;

  List<CoachMessage> get messages => List.unmodifiable(_messages);
  bool get loading => _loading;
  bool get sending => _sending;

  /// True while we expect this turn to (re)generate the plan — drives the
  /// plan-building loader instead of the plain "coach is thinking" one.
  bool get buildingPlan => _buildingPlan;
  Object? get error => _error;

  /// Load the transcript, keeping only the user/assistant text turns.
  Future<void> loadHistory() async {
    _loading = true;
    _error = null;
    notifyListeners();
    try {
      final all = await ConversationService.fetchMessages();
      _messages
        ..clear()
        ..addAll(all.where((m) =>
            (m.role == 'user' || m.role == 'assistant') &&
            (m.content?.trim().isNotEmpty ?? false)));
    } catch (e) {
      _error = e;
    } finally {
      _loading = false;
      notifyListeners();
    }
  }

  /// Optimistically append the user's message, await the coach, append the
  /// reply, and — if the coach (re)generated the plan — reveal the full
  /// schedule once.
  Future<void> send(String text) async {
    final expectPlan = _looksLikePlanBuild(text);
    _messages.add(CoachMessage(role: 'user', content: text));
    _sending = true;
    _buildingPlan = expectPlan;
    _error = null;
    notifyListeners();
    try {
      final reply = await ConversationService.sendMessage(text);
      if (reply.assistantMessage.trim().isNotEmpty) {
        _messages.add(
          CoachMessage(role: 'assistant', content: reply.assistantMessage),
        );
      }
      if (reply.planGenerated) {
        final ActivePlan? plan = await _fetchPlanQuietly();
        if (plan != null) {
          _messages.add(CoachMessage(role: 'assistant', plan: plan));
        }
      }
    } catch (e) {
      _error = e;
      _messages.add(const CoachMessage(
        role: 'assistant',
        content:
            "Sorry — I couldn't reach the coach. Check your connection and try again.",
      ));
    } finally {
      _sending = false;
      _buildingPlan = false;
      notifyListeners();
    }
  }

  Future<ActivePlan?> _fetchPlanQuietly() async {
    try {
      return await PlanService.fetchActive();
    } catch (_) {
      // The reveal is best-effort; the Planning tab still shows the new plan.
      return null;
    }
  }

  /// Heuristic: does this message read like a request to build or change the
  /// plan? Only controls which loader is shown — the actual reveal is gated on
  /// the server's authoritative `plan_generated` flag, so a wrong guess here is
  /// cosmetic.
  bool _looksLikePlanBuild(String text) {
    final t = text.toLowerCase().trim();
    const phrases = [
      'generate', 'build me', 'build a', 'build my', 'make me', 'make it',
      'create a plan', 'create my plan', 'new plan', 'new program', 'redo',
      'regenerate', 're-generate', 'rebuild', 'remake', 'rewrite',
      'change the plan', 'change my plan', 'switch to', 'day plan',
      'day program', 'day split', 'days a week', 'days not', 'plan for me',
    ];
    for (final p in phrases) {
      if (t.contains(p)) return true;
    }
    if (RegExp(r'\b[1-7]\s*-?\s*days?\b').hasMatch(t)) return true;

    // A short "yes" right after the coach asked a question is a go-ahead.
    final lastCoach = _lastAssistantText();
    if (lastCoach != null && lastCoach.trimRight().endsWith('?')) {
      const yeses = [
        'yes', 'yep', 'yeah', 'sure', 'ok', 'okay', 'go', 'go ahead',
        'go for it', 'do it', 'sounds good', "let's do it", 'lets do it',
        'perfect', 'yes please', 'please do', 'build it', 'generate it',
      ];
      for (final y in yeses) {
        if (t == y || t == '$y.' || t == '$y!' || t.startsWith('$y ')) {
          return true;
        }
      }
    }
    return false;
  }

  String? _lastAssistantText() {
    for (var i = _messages.length - 1; i >= 0; i--) {
      final m = _messages[i];
      if (m.role == 'assistant' && (m.content?.isNotEmpty ?? false)) {
        return m.content;
      }
    }
    return null;
  }
}
