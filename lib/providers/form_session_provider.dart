import 'package:flutter/foundation.dart';

// Live WebSocket form-session state — reserved for the future live camera path.
// That path is currently disabled; this provider is kept registered so the
// dependency graph stays intact when the live feature is rebuilt.
class FormSessionProvider extends ChangeNotifier {}
