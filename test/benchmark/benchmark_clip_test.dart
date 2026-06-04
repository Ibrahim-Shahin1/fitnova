import 'package:fitnova_application/models/benchmark_models.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  group('BenchmarkClip', () {
    test('fromJson parses clip_id, groundTruth, and score from snake_case keys', () {
      final clip = BenchmarkClip.fromJson({
        'clip_id': 'squat_001',
        'ground_truth': {'KIE': 1, 'KFE': 0},
        'score': {'KIE': 0.72, 'KFE': 0.34},
      });
      expect(clip.clipId, 'squat_001');
      expect(clip.groundTruth['KIE'], 1);
      expect(clip.groundTruth['KFE'], 0);
      expect(clip.score['KIE'], closeTo(0.72, 0.001));
      expect(clip.score['KFE'], closeTo(0.34, 0.001));
    });

    test('fromJson groundTruth values are int (not double)', () {
      final clip = BenchmarkClip.fromJson({
        'clip_id': 'c2',
        'ground_truth': {'KIE': 1, 'KFE': 0},
        'score': {'KIE': 0.50, 'KFE': 0.20},
      });
      expect(clip.groundTruth['KIE'], isA<int>());
      expect(clip.groundTruth['KFE'], isA<int>());
    });

    test('fromJson score values are double', () {
      final clip = BenchmarkClip.fromJson({
        'clip_id': 'c3',
        'ground_truth': {'ELBOWS': 1, 'KNEES': 0},
        'score': {'ELBOWS': 0.357, 'KNEES': 0.200},
      });
      expect(clip.score['ELBOWS'], isA<double>());
      expect(clip.score['KNEES'], isA<double>());
      expect(clip.score['ELBOWS'], closeTo(0.357, 0.001));
    });

    test('fromJson handles single-error shallow clip', () {
      final clip = BenchmarkClip.fromJson({
        'clip_id': 'shallow_042',
        'ground_truth': {'DEPTH': 1},
        'score': {'DEPTH': 0.80},
      });
      expect(clip.clipId, 'shallow_042');
      expect(clip.groundTruth['DEPTH'], 1);
      expect(clip.score['DEPTH'], closeTo(0.80, 0.001));
    });
  });
}
