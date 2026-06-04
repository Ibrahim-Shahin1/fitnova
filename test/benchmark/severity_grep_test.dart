import 'dart:io';
import 'package:flutter_test/flutter_test.dart';

void main() {
  test('No banned severity/coaching literals appear in lib/', () async {
    const bannedTokens = [
      'severity_word',
      'severityWord',
      '"strong"',
      "'strong'",
      '"moderate"',
      "'moderate'",
      '"mild"',
      "'mild'",
      '"severe"',
      "'severe'",
      '"minor"',
      "'minor'",
      'try to',
      'focus on',
      'you should',
      'good job',
      'well done',
      'Great form',
      'Form looks good',
      '_synthFeedback',
    ];

    final libDir = Directory('lib');
    // Skip nutrition-pathed files: that merged feature legitimately uses
    // 'moderate' as a physical-activity level (sedentary/light/moderate/
    // active/athlete), which is NOT form-severity language. The v1.1 honesty
    // contract governs the form-correction/benchmark UI surface only.
    final dartFiles = libDir
        .listSync(recursive: true)
        .whereType<File>()
        .where((f) =>
            f.path.endsWith('.dart') && !f.path.contains('nutrition'));

    final violations = <String>[];
    for (final file in dartFiles) {
      final content = await file.readAsString();
      for (final token in bannedTokens) {
        if (content.contains(token)) {
          violations.add('${file.path}: contains "$token"');
        }
      }
    }

    expect(violations, isEmpty,
        reason: 'Banned severity/coaching tokens found:\n${violations.join('\n')}');
  });
}
