import 'package:fitnova_application/models/benchmark_models.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  group('browse filter', () {
    // ── benchmarkClipIsDetected ────────────────────────────────────────────

    test('benchmarkClipIsDetected: KIE score above threshold is detected', () {
      final clip = BenchmarkClip(
        clipId: 'test_001',
        groundTruth: {'KIE': 1, 'KFE': 0},
        score: {'KIE': 0.70, 'KFE': 0.20},
      );
      expect(benchmarkClipIsDetected('squat', 'KIE', clip), isTrue);
    });

    test('benchmarkClipIsDetected: KFE score below threshold is not detected', () {
      final clip = BenchmarkClip(
        clipId: 'test_001',
        groundTruth: {'KIE': 1, 'KFE': 0},
        score: {'KIE': 0.70, 'KFE': 0.20},
      );
      expect(benchmarkClipIsDetected('squat', 'KFE', clip), isFalse);
    });

    test('benchmarkClipIsDetected: OHP ELBOWS at threshold is detected', () {
      final clip = BenchmarkClip(
        clipId: 'ohp_001',
        groundTruth: {'ELBOWS': 1, 'KNEES': 0},
        score: {'ELBOWS': 0.357, 'KNEES': 0.200},
      );
      // exactly at threshold (>=) → detected
      expect(benchmarkClipIsDetected('ohp', 'ELBOWS', clip), isTrue);
    });

    test('benchmarkClipIsDetected: Shallow DEPTH below threshold is not detected', () {
      final clip = BenchmarkClip(
        clipId: 'shallow_001',
        groundTruth: {'DEPTH': 0},
        score: {'DEPTH': 0.30},
      );
      expect(benchmarkClipIsDetected('shallow', 'DEPTH', clip), isFalse);
    });

    // ── benchmarkErrorIsCorrect ────────────────────────────────────────────

    test('benchmarkErrorIsCorrect: KIE detected=true, GT=1 → correct', () {
      final clip = BenchmarkClip(
        clipId: 'test_002',
        groundTruth: {'KIE': 1, 'KFE': 0},
        score: {'KIE': 0.70, 'KFE': 0.20},
      );
      expect(benchmarkErrorIsCorrect('squat', 'KIE', clip), isTrue);
    });

    test('benchmarkErrorIsCorrect: KFE detected=false, GT=0 → correct', () {
      final clip = BenchmarkClip(
        clipId: 'test_002',
        groundTruth: {'KIE': 1, 'KFE': 0},
        score: {'KIE': 0.70, 'KFE': 0.20},
      );
      expect(benchmarkErrorIsCorrect('squat', 'KFE', clip), isTrue);
    });

    test('benchmarkErrorIsCorrect: KIE detected=true, GT=0 → wrong', () {
      final clip = BenchmarkClip(
        clipId: 'test_003',
        groundTruth: {'KIE': 0, 'KFE': 0},
        score: {'KIE': 0.70, 'KFE': 0.20},
      );
      expect(benchmarkErrorIsCorrect('squat', 'KIE', clip), isFalse);
    });

    // ── benchmarkClipIsCorrect ─────────────────────────────────────────────

    test('benchmarkClipIsCorrect: ALL errors correct → clip correct', () {
      // KIE: detected(0.70>=0.614)=true, GT=1 ✓; KFE: detected(0.20>=0.385)=false, GT=0 ✓
      final clip = BenchmarkClip(
        clipId: 'test_004',
        groundTruth: {'KIE': 1, 'KFE': 0},
        score: {'KIE': 0.70, 'KFE': 0.20},
      );
      expect(benchmarkClipIsCorrect('squat', clip), isTrue);
    });

    test('benchmarkClipIsCorrect: ANY error wrong → clip wrong', () {
      // KIE: detected(0.70>=0.614)=true, GT=0 — WRONG
      final clip = BenchmarkClip(
        clipId: 'test_005',
        groundTruth: {'KIE': 0, 'KFE': 0},
        score: {'KIE': 0.70, 'KFE': 0.20},
      );
      expect(benchmarkClipIsCorrect('squat', clip), isFalse);
    });

    test('benchmarkClipIsCorrect: OHP both errors correct', () {
      // ELBOWS: detected(0.40>=0.357)=true, GT=1 ✓; KNEES: detected(0.20>=0.476)=false, GT=0 ✓
      final clip = BenchmarkClip(
        clipId: 'ohp_002',
        groundTruth: {'ELBOWS': 1, 'KNEES': 0},
        score: {'ELBOWS': 0.40, 'KNEES': 0.20},
      );
      expect(benchmarkClipIsCorrect('ohp', clip), isTrue);
    });

    test('benchmarkClipIsCorrect: Shallow single DEPTH error correct', () {
      // DEPTH: detected(0.50>=0.395)=true, GT=1 ✓
      final clip = BenchmarkClip(
        clipId: 'shallow_003',
        groundTruth: {'DEPTH': 1},
        score: {'DEPTH': 0.50},
      );
      expect(benchmarkClipIsCorrect('shallow', clip), isTrue);
    });

    test('benchmarkClipIsCorrect: Shallow DEPTH wrong', () {
      // DEPTH: detected(0.50>=0.395)=true, GT=0 → WRONG
      final clip = BenchmarkClip(
        clipId: 'shallow_004',
        groundTruth: {'DEPTH': 0},
        score: {'DEPTH': 0.50},
      );
      expect(benchmarkClipIsCorrect('shallow', clip), isFalse);
    });
  });
}
