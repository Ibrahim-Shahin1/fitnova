import 'package:fitnova_application/models/form_models.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  group('FormError', () {
    test('fromJson parses full payload', () {
      final e = FormError.fromJson({
        'type': 'KIE',
        'detected': true,
        'confidence': 0.72,
        'threshold': 0.614,
        'ground_truth': 1,
        'intervals': [
          [0.0, 2.5]
        ],
      });
      expect(e.type, 'KIE');
      expect(e.confidence, closeTo(0.72, 0.001));
      expect(e.threshold, closeTo(0.614, 0.001));
      expect(e.groundTruth, 1);
      expect(e.detected, isTrue);
      expect(e.intervals, [
        [0.0, 2.5]
      ]);
    });

    test('fromJson returns null groundTruth when key absent', () {
      final e = FormError.fromJson({
        'type': 'KFE',
        'detected': false,
        'confidence': 0.3,
        'threshold': 0.385,
      });
      // fabrication-prevention: absent ground_truth must be null, never 0
      expect(e.groundTruth, isNull);
    });
  });

  group('FormRep', () {
    test('fromJson parses full payload', () {
      final r = FormRep.fromJson({
        'exercise': 'squat',
        'rep_index': 0,
        'total_reps': 1,
        'thumbnail': 'aGVsbG8=',
        'errors': [
          {
            'type': 'KIE',
            'detected': true,
            'confidence': 0.72,
            'threshold': 0.614,
            'ground_truth': 1,
          }
        ],
      });
      expect(r.repIndex, 0);
      expect(r.totalReps, 1);
      expect(r.thumbnail, 'aGVsbG8=');
      expect(r.errors.length, 1);
    });

    test('fromJson returns null thumbnail when key absent', () {
      final r = FormRep.fromJson({
        'exercise': 'squat',
        'rep_index': 0,
        'total_reps': 1,
        'errors': [],
      });
      expect(r.thumbnail, isNull);
    });
  });
}
