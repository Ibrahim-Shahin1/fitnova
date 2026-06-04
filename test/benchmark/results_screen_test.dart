import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:provider/provider.dart';
import 'package:video_player/video_player.dart';

import 'package:fitnova_application/models/benchmark_models.dart';
import 'package:fitnova_application/models/form_models.dart';
import 'package:fitnova_application/providers/benchmark_provider.dart';
import 'package:fitnova_application/screens/benchmark_results_screen.dart';
import 'package:fitnova_application/theme/app_theme.dart';
import 'package:fitnova_application/widgets/benchmark/confidence_bar.dart';
import 'package:fitnova_application/widgets/benchmark/error_result_row.dart';
import 'package:fitnova_application/widgets/ui/app_empty_state.dart';
import 'package:fitnova_application/widgets/ui/app_error_state.dart';

// 1x1 transparent PNG — valid base64 for testing Image.memory path.
const _kValidBase64Png =
    'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk'
    '+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==';

BenchmarkClip _squatClip() => const BenchmarkClip(
      clipId: 'squat_test_001',
      groundTruth: {'KIE': 1, 'KFE': 0},
      score: {'KIE': 0.72, 'KFE': 0.20},
    );

BenchmarkClip _shallowClip() => const BenchmarkClip(
      clipId: 'shallow_test_001',
      groundTruth: {'DEPTH': 1},
      score: {'DEPTH': 0.72},
    );

Widget _buildScreen(BenchmarkProvider provider) =>
    ChangeNotifierProvider<BenchmarkProvider>.value(
      value: provider,
      child: MaterialApp(
        theme: AppTheme.light,
        home: const BenchmarkResultsScreen(),
      ),
    );

void main() {
  group('BenchmarkResultsScreen', () {
    testWidgets(
        'no-reps: shows AppEmptyState with zero ErrorResultRow/ConfidenceBar (UI-05)',
        (tester) async {
      final provider = BenchmarkProvider()
        ..selectedExercise = 'squat'
        ..selectedClip = _squatClip()
        ..analysisState = BenchmarkAnalysisState.noReps
        ..analysisResult = [];

      await tester.pumpWidget(_buildScreen(provider));
      await tester.pump();

      expect(find.byType(AppEmptyState), findsOneWidget);
      expect(find.byType(ErrorResultRow), findsNothing);
      expect(find.byType(ConfidenceBar), findsNothing);
    });

    testWidgets(
        'error: shows AppErrorState with zero ErrorResultRow/ConfidenceBar (UI-05)',
        (tester) async {
      final provider = BenchmarkProvider()
        ..selectedExercise = 'squat'
        ..selectedClip = _squatClip()
        ..analysisState = BenchmarkAnalysisState.error
        ..analysisError = 'boom';

      await tester.pumpWidget(_buildScreen(provider));
      await tester.pump();

      expect(find.byType(AppErrorState), findsOneWidget);
      expect(find.byType(ErrorResultRow), findsNothing);
      expect(find.byType(ConfidenceBar), findsNothing);
    });

    testWidgets(
        'normal done: shows ErrorResultRow and inline caveat (UI-02/UI-04)',
        (tester) async {
      final rep = FormRep(
        exercise: 'shallow',
        repIndex: 0,
        totalReps: 1,
        thumbnail: _kValidBase64Png,
        errors: const [
          FormError(
            type: 'DEPTH',
            detected: true,
            confidence: 0.72,
            threshold: 0.614,
            groundTruth: 1,
          ),
        ],
      );

      // Use shallow to avoid VideoPlayerController network init in tests.
      final provider = BenchmarkProvider()
        ..selectedExercise = 'shallow'
        ..selectedClip = _shallowClip()
        ..analysisState = BenchmarkAnalysisState.done
        ..analysisResult = [rep];

      await tester.pumpWidget(_buildScreen(provider));
      await tester.pump();

      expect(find.byType(ErrorResultRow), findsAtLeastNWidgets(1));
      expect(
        find.textContaining('These are Fitness-AQA test clips'),
        findsOneWidget,
      );
    });

    testWidgets(
        'GT-null: no benchmark badge text rendered when groundTruth is null (UI-02)',
        (tester) async {
      final rep = FormRep(
        exercise: 'shallow',
        repIndex: 0,
        totalReps: 1,
        errors: const [
          FormError(
            type: 'KIE',
            detected: false,
            confidence: 0.30,
            threshold: 0.614,
            // groundTruth omitted — null, no badge should render
          ),
        ],
      );

      final provider = BenchmarkProvider()
        ..selectedExercise = 'shallow'
        ..selectedClip = _shallowClip()
        ..analysisState = BenchmarkAnalysisState.done
        ..analysisResult = [rep];

      await tester.pumpWidget(_buildScreen(provider));
      await tester.pump();

      expect(find.textContaining('Benchmark:'), findsNothing);
    });

    testWidgets(
        'shallow done: no VideoPlayer in tree — uses Image.memory path (UI-03)',
        (tester) async {
      final rep = FormRep(
        exercise: 'shallow',
        repIndex: 0,
        totalReps: 1,
        thumbnail: _kValidBase64Png,
        errors: const [
          FormError(
            type: 'DEPTH',
            detected: true,
            confidence: 0.72,
            threshold: 0.395,
            groundTruth: 1,
          ),
        ],
      );

      final provider = BenchmarkProvider()
        ..selectedExercise = 'shallow'
        ..selectedClip = _shallowClip()
        ..analysisState = BenchmarkAnalysisState.done
        ..analysisResult = [rep];

      await tester.pumpWidget(_buildScreen(provider));
      await tester.pump();

      // Shallow renders Image.memory crop, not VideoPlayer.
      expect(find.byType(VideoPlayer), findsNothing);
    });
  });
}
