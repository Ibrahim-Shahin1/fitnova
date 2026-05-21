import 'package:supabase_flutter/supabase_flutter.dart';

import 'supabase_service.dart';

/// Wrapper over the Supabase auth calls used by the auth screens.
class AuthService {
  const AuthService._();

  /// Deep-link target for confirmation / recovery emails.
  /// The matching intent-filter / URL scheme is registered in Unit 6.
  static const String redirectUrl = 'io.fitnova.app://auth-callback';

  static GoTrueClient get _auth => SupabaseService.client.auth;

  static Future<AuthResponse> signIn({
    required String email,
    required String password,
  }) {
    return _auth.signInWithPassword(email: email, password: password);
  }

  static Future<AuthResponse> signUp({
    required String email,
    required String password,
  }) {
    return _auth.signUp(
      email: email,
      password: password,
      emailRedirectTo: redirectUrl,
    );
  }

  static Future<void> sendPasswordReset(String email) {
    return _auth.resetPasswordForEmail(email, redirectTo: redirectUrl);
  }
}
