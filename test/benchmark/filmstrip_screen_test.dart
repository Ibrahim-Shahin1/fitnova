import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:provider/provider.dart';

import 'package:fitnova_application/models/benchmark_models.dart';
import 'package:fitnova_application/providers/benchmark_provider.dart';
import 'package:fitnova_application/screens/benchmark_filmstrip_screen.dart';
import 'package:fitnova_application/theme/app_theme.dart';

BenchmarkClip _clip(String id, double depth, int gt) => BenchmarkClip(
      clipId: id,
      groundTruth: {'DEPTH': gt},
      score: {'DEPTH': depth},
    );

Widget _wrap(BenchmarkProvider p) => ChangeNotifierProvider<BenchmarkProvider>.value(
      value: p,
      child: MaterialApp(
        theme: AppTheme.light,
        home: const BenchmarkFilmstripScreen(),
      ),
    );

void main() {
  testWidgets('filmstrip defaults to the deepest frame and reads high score as DEEP',
      (tester) async {
    final provider = BenchmarkProvider()
      ..filmstripFrames = [
        _clip('900_1_10', 0.98, 1), // deepest (max score) -> default selected
        _clip('900_1_14', 0.40, 1),
        _clip('900_1_18', 0.02, 0),
      ]
      ..filmstripRep = '900_1';

    await tester.pumpWidget(_wrap(provider));
    await tester.pump();

    // Defaults to the deepest frame = highest DEPTH score (0.98 at index 0).
    expect(find.text('Frame 1 of 3'), findsOneWidget);
    // High score is read as DEEP (good), not as a detected error.
    expect(find.text('DEPTH REACHED'), findsOneWidget);
    expect(find.text('TOO SHALLOW'), findsNothing);
    // Benchmark label 1 = deep; model reached depth -> match.
    expect(find.textContaining('Benchmark: deep'), findsOneWidget);
    // In-domain caveat present.
    expect(find.textContaining('Fitness-AQA test clips'), findsOneWidget);
    // Per-frame thumbnail score labels.
    expect(find.text('0.98'), findsOneWidget);
    expect(find.text('0.02'), findsOneWidget);
  });

  testWidgets('a below-threshold frame reads as TOO SHALLOW', (tester) async {
    final provider = BenchmarkProvider()
      ..filmstripFrames = [_clip('901_1_20', 0.05, 0)]
      ..filmstripRep = '901_1';

    await tester.pumpWidget(_wrap(provider));
    await tester.pump();

    expect(find.text('TOO SHALLOW'), findsOneWidget);
    expect(find.text('DEPTH REACHED'), findsNothing);
    // Benchmark label 0 = shallow; model also shallow -> match.
    expect(find.textContaining('Benchmark: shallow'), findsOneWidget);
  });

  testWidgets('empty filmstrip shows a clear empty state', (tester) async {
    final provider = BenchmarkProvider()..filmstripFrames = const [];
    await tester.pumpWidget(_wrap(provider));
    await tester.pump();

    expect(find.textContaining('No frames'), findsOneWidget);
    expect(find.text('DEPTH REACHED'), findsNothing);
    expect(find.text('TOO SHALLOW'), findsNothing);
  });
}
