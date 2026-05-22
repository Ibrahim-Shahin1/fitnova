import 'package:flutter/foundation.dart';

import '../models/coach_message.dart';
import '../services/conversation_service.dart';
import '../services/plan_service.dart';

/// Drives the coach chat: the visible transcript, load/sending state, and the
/// "ready to generate" signal. Plan generation itself happens on the dedicated
/// live builder screen; once that finishes, [onPlanGenerated] reveals the new
/// plan inline.
class ConversationProvider extends ChangeNotifier {
  final List<CoachMessage> _messages = [];
  bool _loading = false;
  bool _sending = false;
  Object? _error;
  Map<String, dynamic>? _pendingParams;

  List<CoachMessage> get messages => List.unmodifiable(_messages);
  bool get loading => _loading;
  bool get sending => _sending;
  Object? get error => _error;

  /// Non-null once the coach has gathered enough and readied a plan — the UI
  /// then shows the "Generate Plan" button. Carries the override params.
  Map<String, dynamic>? get pendingPlanParams => _pendingParams;
  bool get readyToGenerate => _pendingParams != null;

  /// Load the transcript; for a fresh conversation, let the coach speak first.
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
      if (_messages.isEmpty) {
        final opener = await ConversationService.startConversation();
        if (opener != null && opener.trim().isNotEmpty) {
          _messages.add(CoachMessage(role: 'assistant', content: opener));
        }
      }
    } catch (e) {
      _error = e;
    } finally {
      _loading = false;
      notifyListeners();
    }
  }

  Future<void> send(String text) async {
    _messages.add(CoachMessage(role: 'user', content: text));
    _sending = true;
    _error = null;
    notifyListeners();
    try {
      final reply = await ConversationService.sendMessage(text);
      if (reply.assistantMessage.trim().isNotEmpty) {
        _messages.add(
          CoachMessage(role: 'assistant', content: reply.assistantMessage),
        );
      }
      _pendingParams = reply.readyToGenerate ? reply.planParams : null;
    } catch (e) {
      _error = e;
      _messages.add(const CoachMessage(
        role: 'assistant',
        content:
            "Sorry — I couldn't reach the coach. Check your connection and try again.",
      ));
    } finally {
      _sending = false;
      notifyListeners();
    }
  }

  /// Called when the live builder finishes: clear the pending button and reveal
  /// the freshly generated plan inline once.
  Future<void> onPlanGenerated() async {
    _pendingParams = null;
    notifyListeners();
    try {
      final plan = await PlanService.fetchActive();
      if (plan != null) {
        _messages.add(CoachMessage(role: 'assistant', plan: plan));
        notifyListeners();
      }
    } catch (_) {
      // best-effort; the Planning tab still shows the new plan
    }
  }
}
