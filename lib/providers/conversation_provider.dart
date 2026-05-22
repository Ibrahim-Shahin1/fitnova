import 'package:flutter/foundation.dart';

import '../models/coach_message.dart';
import '../services/conversation_service.dart';

/// Drives the coach chat screen: holds the visible transcript, the initial-load
/// state, and the "coach is replying" state. Scoped to the chat screen; the
/// transcript is the server's source of truth (reloaded on open).
class ConversationProvider extends ChangeNotifier {
  final List<CoachMessage> _messages = [];
  bool _loading = false;
  bool _sending = false;
  Object? _error;

  List<CoachMessage> get messages => List.unmodifiable(_messages);
  bool get loading => _loading;
  bool get sending => _sending;
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

  /// Optimistically append the user's message, await the coach, append the reply.
  Future<void> send(String text) async {
    _messages.add(CoachMessage(role: 'user', content: text));
    _sending = true;
    _error = null;
    notifyListeners();
    try {
      final reply = await ConversationService.sendMessage(text);
      _messages.add(
        CoachMessage(role: 'assistant', content: reply.assistantMessage),
      );
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
}
