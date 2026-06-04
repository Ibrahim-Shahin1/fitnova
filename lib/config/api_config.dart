import 'dart:io' show Platform;
import 'package:flutter/foundation.dart' show kIsWeb;

class ApiConfig {
  // BlueStacks demo: `adb reverse tcp:8000 tcp:8000` maps the device's
  // localhost:8000 to the host backend, so Android uses 127.0.0.1 here (the
  // standard emulator's 10.0.2.2 does not apply under BlueStacks + reverse).
  static String get baseUrl {
    if (kIsWeb) return 'http://localhost:8000';
    if (Platform.isAndroid) return 'http://127.0.0.1:8000';
    return 'http://localhost:8000';
  }

  static String get wsUrl {
    if (kIsWeb) return 'ws://localhost:8000';
    if (Platform.isAndroid) return 'ws://127.0.0.1:8000';
    return 'ws://localhost:8000';
  }
}
