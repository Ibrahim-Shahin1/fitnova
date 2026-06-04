import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:fitnova_application/theme/app_theme.dart';
import 'package:fitnova_application/widgets/benchmark/confidence_bar.dart';

void main() {
  group('ConfidenceBar', () {
    testWidgets('renders confidence value and threshold label (detected=true)',
        (tester) async {
      await tester.pumpWidget(
        MaterialApp(
          theme: AppTheme.light,
          home: Scaffold(
            body: ConfidenceBar(
              confidence: 0.72,
              threshold: 0.614,
              detected: true,
            ),
          ),
        ),
      );
      expect(find.text('0.720'), findsOneWidget);
      expect(find.text('T: 0.614'), findsOneWidget);
    });

    testWidgets('renders "1.0" right-side label', (tester) async {
      await tester.pumpWidget(
        MaterialApp(
          theme: AppTheme.light,
          home: Scaffold(
            body: ConfidenceBar(
              confidence: 0.50,
              threshold: 0.50,
              detected: false,
            ),
          ),
        ),
      );
      expect(find.text('1.0'), findsOneWidget);
    });

    testWidgets('renders with detected=false and clear zone (no ambiguity)',
        (tester) async {
      await tester.pumpWidget(
        MaterialApp(
          theme: AppTheme.light,
          home: Scaffold(
            body: ConfidenceBar(
              confidence: 0.20,
              threshold: 0.614,
              detected: false,
            ),
          ),
        ),
      );
      expect(find.text('0.200'), findsOneWidget);
      expect(find.text('T: 0.614'), findsOneWidget);
    });

    testWidgets('renders with detected=false and ambiguous zone', (tester) async {
      await tester.pumpWidget(
        MaterialApp(
          theme: AppTheme.light,
          home: Scaffold(
            body: ConfidenceBar(
              confidence: 0.60,
              threshold: 0.614,
              detected: false,
            ),
          ),
        ),
      );
      expect(find.text('0.600'), findsOneWidget);
    });
  });
}
