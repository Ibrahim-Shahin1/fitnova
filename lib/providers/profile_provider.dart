import 'package:flutter/foundation.dart';

import '../models/user_profile.dart';
import '../services/profile_service.dart';

/// Holds the signed-in user's profile. Loaded after auth, cleared on sign-out.
class ProfileProvider extends ChangeNotifier {
  UserProfile? _profile;
  bool _loading = false;
  Object? _error;
  String? _syncedUserId;

  UserProfile? get profile => _profile;
  bool get loading => _loading;
  Object? get error => _error;
  bool get onboardingCompleted => _profile?.onboardingCompleted ?? false;

  /// Called by the proxy provider when the auth user changes: load the profile
  /// for a newly signed-in user, clear it on sign-out. The load is deferred via
  /// a microtask so we never notifyListeners() during a build.
  void syncWithAuth(String? userId) {
    if (userId == _syncedUserId) return;
    _syncedUserId = userId;
    if (userId == null) {
      _profile = null;
      _error = null;
      _loading = false;
    } else {
      Future.microtask(() => load(userId));
    }
  }

  Future<void> load(String userId) async {
    _loading = true;
    _error = null;
    notifyListeners();
    try {
      _profile = await ProfileService.fetch(userId);
    } catch (e) {
      _error = e;
    } finally {
      _loading = false;
      notifyListeners();
    }
  }

  /// Persist an updated profile and keep the local copy in sync.
  Future<void> save(UserProfile updated) async {
    _profile = await ProfileService.update(updated);
    notifyListeners();
  }

  /// Clear local state on sign-out to avoid leaking one user's data to the next.
  void clear() {
    _profile = null;
    _error = null;
    _loading = false;
    notifyListeners();
  }
}
