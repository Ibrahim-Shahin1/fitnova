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
    this.readyToGenerate = false,
    this.planParams = const {},
  });

  final String assistantMessage;

  /// True once the coach has gathered enough and called prepare_plan — the app
  /// surfaces a "Generate Plan" button that opens the live 4-agent builder.
  final bool readyToGenerate;

  /// Override params the coach captured (frequency, equipment, injuries…), to be
  /// sent to the streaming generate endpoint. Empty means "use the profile".
  final Map<String, dynamic> planParams;

  factory CoachReply.fromJson(Map<String, dynamic> m) => CoachReply(
        assistantMessage: (m['assistant_message'] ?? '') as String,
        readyToGenerate: (m['ready_to_generate'] ?? false) as bool,
        planParams:
            ((m['plan_params'] as Map?) ?? const {}).cast<String, dynamic>(),
      );
}
