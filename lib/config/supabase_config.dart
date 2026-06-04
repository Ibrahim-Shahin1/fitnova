/// Supabase client configuration for the Flutter app.
///
/// The URL + anon key are PUBLIC by design — the anon key is shipped to every
/// client and row-level security (RLS) is what protects data. They are
/// compile-time constants with optional --dart-define overrides so the same
/// build can target a different project without code changes.
class SupabaseConfig {
  const SupabaseConfig._();

  static const String url = String.fromEnvironment(
    'SUPABASE_URL',
    defaultValue: 'https://deduivgbiupcjvuntnjn.supabase.co',
  );

  static const String anonKey = String.fromEnvironment(
    'SUPABASE_ANON_KEY',
    defaultValue:
        'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImRlZHVpdmdiaXVwY2p2dW50bmpuIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzkzMjgwOTAsImV4cCI6MjA5NDkwNDA5MH0.Jk9yKoZrqBPgyH0rCfiDlYCLJqtWnfGFqzeoPryF6Y8',
  );
}
