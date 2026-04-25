import 'package:flutter/material.dart';

/// Dismissible banner shown when the model's prediction stream disagrees with
/// the user's pre-selected exercise. Non-blocking — session continues.
class MismatchBanner extends StatelessWidget {
  final String selected;
  final String predicted;
  final double confidence;
  final VoidCallback onDismiss;

  const MismatchBanner({
    super.key,
    required this.selected,
    required this.predicted,
    required this.confidence,
    required this.onDismiss,
  });

  String _humanise(String snakeCase) =>
      snakeCase.replaceAll('_', ' ').split(' ')
          .map((w) => w.isEmpty ? w : '${w[0].toUpperCase()}${w.substring(1)}')
          .join(' ');

  @override
  Widget build(BuildContext context) {
    return Material(
      color: Colors.amber[100],
      elevation: 2,
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
        child: Row(
          children: [
            Icon(Icons.warning_amber_rounded, color: Colors.amber[800]),
            const SizedBox(width: 10),
            Expanded(
              child: Text(
                'This looks more like ${_humanise(predicted)} than '
                '${_humanise(selected)} '
                '(${(confidence * 100).round()}% confident). '
                'If this is intentional, you can ignore this.',
                style: const TextStyle(fontSize: 13),
              ),
            ),
            IconButton(
              icon: const Icon(Icons.close, size: 18),
              onPressed: onDismiss,
              padding: EdgeInsets.zero,
              constraints: const BoxConstraints(),
            ),
          ],
        ),
      ),
    );
  }
}
