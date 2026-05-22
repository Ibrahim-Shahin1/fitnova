import 'active_plan.dart';

/// A coach chat message from the backend transcript, or a locally-injected
/// plan reveal. When [plan] is set, this entry is the one-time post-generation
/// schedule card (not a server-persisted text turn).
class CoachMessage {
  const CoachMessage({required this.role, this.content, this.toolName, this.plan});

  final String role; // 'user' | 'assistant' | 'tool'
  final String? content;
  final String? toolName;
  final ActivePlan? plan;

  bool get isUser => role == 'user';
  bool get isPlanReveal => plan != null;

  factory CoachMessage.fromJson(Map<String, dynamic> m) => CoachMessage(
        role: (m['role'] ?? '') as String,
        content: m['content'] as String?,
        toolName: m['tool_name'] as String?,
      );
}

/// The coach's reply to a sent message.
class CoachReply {
  const CoachReply({
    required this.assistantMessage,
    this.toolsCalled = const [],
    this.planGenerated = false,
  });

  final String assistantMessage;
  final List<String> toolsCalled;

  /// True when the coach generated/replaced the plan this turn — the client
  /// then fetches the active plan and reveals the full schedule once.
  final bool planGenerated;

  factory CoachReply.fromJson(Map<String, dynamic> m) => CoachReply(
        assistantMessage: (m['assistant_message'] ?? '') as String,
        toolsCalled: ((m['tool_invocations'] as List?) ?? const [])
            .map((t) => ((t as Map)['tool'] ?? '').toString())
            .toList(),
        planGenerated: (m['plan_generated'] ?? false) as bool,
      );
}
