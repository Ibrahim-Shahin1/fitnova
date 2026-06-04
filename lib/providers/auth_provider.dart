import 'dart:async';

import 'package:flutter/foundation.dart';
import 'package:supabase_flutter/supabase_flutter.dart';

import '../services/supabase_service.dart';

/// Exposes the current Supabase auth session to the widget tree and reacts to
/// sign-in / sign-out / token-refresh events so the UI re-routes automatically.
class AuthProvider extends ChangeNotifier {
  AuthProvider() {
    _session = SupabaseService.client.auth.currentSession;
    _sub = SupabaseService.client.auth.onAuthStateChange.listen((data) {
      _session = data.session;
      if (data.event == AuthChangeEvent.passwordRecovery) {
        _recovering = true;
      }
      notifyListeners();
    });
  }

  Session? _session;
  bool _recovering = false;
  late final StreamSubscription<AuthState> _sub;

  Session? get session => _session;
  User? get user => _session?.user;
  bool get isAuthenticated => _session != null;
  bool get isRecoveringPassword => _recovering;
  String? get email => _session?.user.email;

  Future<void> signOut() async {
    _recovering = false;
    await SupabaseService.client.auth.signOut();
  }

  /// Set a new password during a recovery session, then exit recovery mode.
  Future<void> completePasswordReset(String newPassword) async {
    await SupabaseService.client.auth.updateUser(
      UserAttributes(password: newPassword),
    );
    _recovering = false;
    notifyListeners();
  }

  @override
  void dispose() {
    _sub.cancel();
    super.dispose();
  }
}
