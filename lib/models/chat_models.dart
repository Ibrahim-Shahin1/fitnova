class ChatMessage {
  final String role;
  final String content;

  ChatMessage({required this.role, required this.content});

  Map<String, dynamic> toJson() => {'role': role, 'content': content};

  factory ChatMessage.fromJson(Map<String, dynamic> json) {
    return ChatMessage(
      role: json['role'] as String,
      content: json['content'] as String,
    );
  }
}

class ChatResponse {
  final String status;
  final String message;
  final Map<String, dynamic> extracted;

  ChatResponse({
    required this.status,
    required this.message,
    required this.extracted,
  });

  factory ChatResponse.fromJson(Map<String, dynamic> json) {
    return ChatResponse(
      status: json['status'] as String,
      message: json['message'] as String,
      extracted: Map<String, dynamic>.from(json['extracted'] as Map),
    );
  }

  bool get isReady => status == 'ready';
}
