import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';

/// Drives the app's `ThemeMode`. Defaults to `ThemeMode.system` and persists
/// the user's choice via `SharedPreferences` so subsequent launches respect it.
///
/// Call `await init()` before `runApp()` to avoid a first-frame flash when
/// the user has previously selected a mode that differs from the system mode.
class ThemeController extends ChangeNotifier {
  static const String _prefsKey = 'theme_mode';

  ThemeMode _mode = ThemeMode.system;
  ThemeMode get mode => _mode;

  Future<void> init() async {
    final prefs = await SharedPreferences.getInstance();
    final stored = prefs.getString(_prefsKey);
    _mode = _parseMode(stored);
    notifyListeners();
  }

  Future<void> setMode(ThemeMode mode) async {
    if (_mode == mode) return;
    _mode = mode;
    notifyListeners();
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_prefsKey, _modeToString(mode));
  }

  static ThemeMode _parseMode(String? stored) {
    switch (stored) {
      case 'light':
        return ThemeMode.light;
      case 'dark':
        return ThemeMode.dark;
      case 'system':
      default:
        return ThemeMode.system;
    }
  }

  static String _modeToString(ThemeMode mode) {
    switch (mode) {
      case ThemeMode.light:
        return 'light';
      case ThemeMode.dark:
        return 'dark';
      case ThemeMode.system:
        return 'system';
    }
  }
}
