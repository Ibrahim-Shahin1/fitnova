import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';

import 'package:web_socket_channel/web_socket_channel.dart';

import '../config/api_config.dart';

/// WebSocket client for real-time form analysis sessions.
class FormSessionService {
  WebSocketChannel? _channel;
  StreamController<dynamic>? _controller;

  bool get isConnected => _channel != null;

  // ── Connect ────────────────────────────────────────────────────────────────

  Stream<dynamic> connect(String? exerciseHint) {
    _controller = StreamController<dynamic>.broadcast();

    final uri = Uri.parse('${ApiConfig.wsUrl}/ws/form-session');
    _channel = WebSocketChannel.connect(uri);

    // Start session
    _channel!.sink.add(jsonEncode({
      'type': 'start_session',
      if (exerciseHint != null) 'selected_exercise': exerciseHint,
    }));

    // Forward all server messages to the stream
    _channel!.stream.listen(
      (msg) {
        try {
          final decoded = jsonDecode(msg as String);
          _controller?.add(decoded);
        } catch (_) {
          _controller?.add(msg);
        }
      },
      onError: (err) => _controller?.addError(err),
      onDone: () => _controller?.close(),
    );

    return _controller!.stream;
  }

  // ── Send a frame ───────────────────────────────────────────────────────────

  void sendFrame(Uint8List jpegBytes, int timestampMs) {
    if (_channel == null) return;
    _channel!.sink.add(jsonEncode({
      'type': 'frame',
      'data': base64Encode(jpegBytes),
      'timestamp_ms': timestampMs,
    }));
  }

  // ── End session ────────────────────────────────────────────────────────────

  void endSession() {
    _channel?.sink.add(jsonEncode({'type': 'end_session'}));
  }

  // ── Disconnect ─────────────────────────────────────────────────────────────

  void disconnect() {
    _channel?.sink.close();
    _channel = null;
    _controller?.close();
    _controller = null;
  }
}
