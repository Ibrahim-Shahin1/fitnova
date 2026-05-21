import 'package:supabase_flutter/supabase_flutter.dart';

import '../config/supabase_config.dart';

/// Thin wrapper around the Supabase client singleton.
class SupabaseService {
  const SupabaseService._();

  /// Initialize Supabase. Call once in main() before runApp().
  static Future<void> initialize() async {
    await Supabase.initialize(
      url: SupabaseConfig.url,
      anonKey: SupabaseConfig.anonKey,
    );
  }

  /// The shared Supabase client (auth + Postgres + storage).
  static SupabaseClient get client => Supabase.instance.client;
}
