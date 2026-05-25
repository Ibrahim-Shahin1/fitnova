import '../models/user_profile.dart';
import 'supabase_service.dart';

/// Reads/writes the current user's `profiles` row directly against Supabase
/// (RLS scopes every query to auth.uid(), so no user_id is trusted from input).
class ProfileService {
  const ProfileService._();

  /// Fetch a profile row by user id, or null if it somehow doesn't exist yet.
  static Future<UserProfile?> fetch(String userId) async {
    final row = await SupabaseService.client
        .from('profiles')
        .select()
        .eq('id', userId)
        .maybeSingle();
    return row == null ? null : UserProfile.fromMap(row);
  }

  /// Update the row (the trigger created it on sign-up) and return the result.
  static Future<UserProfile> update(UserProfile profile) async {
    final row = await SupabaseService.client
        .from('profiles')
        .update(profile.toUpdateMap())
        .eq('id', profile.id)
        .select()
        .single();
    return UserProfile.fromMap(row);
  }
}
