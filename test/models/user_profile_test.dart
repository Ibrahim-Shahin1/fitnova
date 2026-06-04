import 'package:fitnova_application/models/user_profile.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  group('UserProfile', () {
    test('fromMap parses a full Supabase row', () {
      final p = UserProfile.fromMap({
        'id': 'abc-123',
        'display_name': 'Ibrahim',
        'age': 23,
        'gender': 'Male',
        'height_cm': 180.5,
        'weight_kg': 82.0,
        'injuries': ['lower_back', 'knees'],
        'training_focus': 'powerbuilding',
        'years_training': 4,
        'equipment': ['barbell', 'dumbbell'],
        'experience_level': 3,
        'session_duration_hours': 1.5,
        'workout_frequency': 5,
        'unit_preference': 'metric',
        'onboarding_completed': true,
      });
      expect(p.id, 'abc-123');
      expect(p.displayName, 'Ibrahim');
      expect(p.age, 23);
      expect(p.heightCm, 180.5);
      expect(p.injuries, ['lower_back', 'knees']);
      expect(p.trainingFocus, 'powerbuilding');
      expect(p.onboardingCompleted, isTrue);
    });

    test('fromMap tolerates nulls and int-typed numerics', () {
      final p = UserProfile.fromMap({
        'id': 'u1',
        'height_cm': 175, // int where a double is expected
        'session_duration_hours': 1,
        'injuries': null,
        'equipment': null,
      });
      expect(p.displayName, isNull);
      expect(p.heightCm, 175.0);
      expect(p.sessionDurationHours, 1.0);
      expect(p.injuries, isEmpty);
      expect(p.equipment, isEmpty);
      expect(p.unitPreference, 'metric'); // default
      expect(p.onboardingCompleted, isFalse); // default
    });

    test('toUpdateMap omits id and round-trips through fromMap', () {
      const p = UserProfile(
        id: 'u2',
        displayName: 'Test',
        age: 30,
        unitPreference: 'imperial',
        onboardingCompleted: true,
      );
      final m = p.toUpdateMap();
      expect(m.containsKey('id'), isFalse);
      expect(m['display_name'], 'Test');
      expect(m['unit_preference'], 'imperial');
      final back = UserProfile.fromMap({...m, 'id': 'u2'});
      expect(back.displayName, 'Test');
      expect(back.age, 30);
      expect(back.unitPreference, 'imperial');
    });

    test('copyWith overrides only the provided fields', () {
      const p = UserProfile(id: 'u3', displayName: 'A', age: 25);
      final q = p.copyWith(age: 26, onboardingCompleted: true);
      expect(q.id, 'u3');
      expect(q.displayName, 'A');
      expect(q.age, 26);
      expect(q.onboardingCompleted, isTrue);
    });
  });
}
