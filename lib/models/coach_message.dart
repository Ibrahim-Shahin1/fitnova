/// A coach chat message from the backend transcript.
class CoachMessage {
  const CoachMessage({required this.role, this.content, this.toolName});

  final String role; // 'user' | 'assistant' | 'tool'
  final String? content;
  final String? toolName;

  bool get isUser => role == 'user';

  factory CoachMessage.fromJson(Map<String, dynamic> m) => CoachMessage(
        role: (m['role'] ?? '') as String,
        content: m['content'] as String?,
        toolName: m['tool_name'] as String?,
      );
}

/// The coach's reply to a sent message.
class CoachReply {
  const CoachReply({required this.assistantMessage, this.toolsCalled = const []});

  final String assistantMessage;
  final List<String> toolsCalled;

  factory CoachReply.fromJson(Map<String, dynamic> m) => CoachReply(
        assistantMessage: (m['assistant_message'] ?? '') as String,
        toolsCalled: ((m['tool_invocations'] as List?) ?? const [])
            .map((t) => ((t as Map)['tool'] ?? '').toString())
            .toList(),
      );
}
