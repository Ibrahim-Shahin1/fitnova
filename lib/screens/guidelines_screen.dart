import 'package:flutter/material.dart';

import '../models/exercise_meta.dart';
import '../theme/app_spacing.dart';

class GuidelinesScreen extends StatelessWidget {
  final ExerciseMeta meta;

  const GuidelinesScreen({super.key, required this.meta});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final cs = theme.colorScheme;

    return Scaffold(
      appBar: AppBar(
        title: Text(meta.displayName),
      ),
      body: ListView(
        padding: const EdgeInsets.all(AppSpacing.md),
        children: [
          // Diagram card
          Card(
            child: SizedBox(
              height: 300,
              child: Padding(
                padding: const EdgeInsets.all(AppSpacing.lg),
                child: _DiagramWidget(meta: meta),
              ),
            ),
          ),
          const SizedBox(height: AppSpacing.md),

          // Setup info card
          Card(
            child: Column(
              children: [
                ListTile(
                  leading: Icon(Icons.videocam, color: cs.primary),
                  title: Text(
                    '${meta.cameraView[0].toUpperCase()}${meta.cameraView.substring(1)} view',
                  ),
                ),
                const Divider(height: 1),
                ListTile(
                  leading: Icon(Icons.straighten, color: cs.primary),
                  title: Text('${meta.distanceM} m away'),
                ),
                const Divider(height: 1),
                ListTile(
                  leading: Icon(Icons.height, color: cs.primary),
                  title: Row(
                    children: [
                      Text('Phone at ${meta.phoneHeightCm} cm'),
                      const SizedBox(width: AppSpacing.sm),
                      Chip(
                        label: Text(
                          meta.orientation[0].toUpperCase() +
                              meta.orientation.substring(1),
                          style: const TextStyle(fontSize: 11),
                        ),
                        padding: EdgeInsets.zero,
                        materialTapTargetSize:
                            MaterialTapTargetSize.shrinkWrap,
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(height: AppSpacing.md),

          // Guidelines text
          Card(
            child: Padding(
              padding: const EdgeInsets.all(AppSpacing.md),
              child: Text(
                meta.guidelines,
                style: theme.textTheme.bodyMedium,
              ),
            ),
          ),
          const SizedBox(height: AppSpacing.md),

          // Key errors
          if (meta.keyErrors.isNotEmpty) ...[
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: AppSpacing.xs),
              child: Text(
                'Key Areas to Monitor',
                style: theme.textTheme.labelLarge,
              ),
            ),
            const SizedBox(height: AppSpacing.sm),
            Wrap(
              spacing: AppSpacing.sm,
              runSpacing: AppSpacing.sm,
              children: meta.keyErrors
                  .map(
                    (error) => Chip(
                      label: Text(
                        error[0].toUpperCase() + error.substring(1),
                        style: const TextStyle(fontSize: 12),
                      ),
                    ),
                  )
                  .toList(),
            ),
            const SizedBox(height: AppSpacing.lg),
          ],

          // CTA Buttons
          ElevatedButton.icon(
            icon: const Icon(Icons.videocam),
            label: const Text('Live Feedback'),
            onPressed: () {
              Navigator.of(context)
                  .pushNamed('/form-check', arguments: meta.name);
            },
          ),
          const SizedBox(height: AppSpacing.sm),
          OutlinedButton.icon(
            icon: const Icon(Icons.upload_file),
            label: const Text('Upload Video'),
            onPressed: () {
              Navigator.of(context)
                  .pushNamed('/video-upload', arguments: meta);
            },
          ),
          const SizedBox(height: AppSpacing.xl),
        ],
      ),
    );
  }
}

class _DiagramWidget extends StatelessWidget {
  final ExerciseMeta meta;

  const _DiagramWidget({required this.meta});

  @override
  Widget build(BuildContext context) {
    final accent = Theme.of(context).colorScheme.primary;
    if (meta.cameraView == 'either') {
      return Row(
        mainAxisAlignment: MainAxisAlignment.spaceEvenly,
        children: [
          Expanded(
            child: _CameraViewDiagram(
              cameraView: 'side',
              distanceM: meta.distanceM,
              accent: accent,
            ),
          ),
          const SizedBox(width: AppSpacing.md),
          Expanded(
            child: _CameraViewDiagram(
              cameraView: 'front',
              distanceM: meta.distanceM,
              accent: accent,
            ),
          ),
        ],
      );
    } else {
      return _CameraViewDiagram(
        cameraView: meta.cameraView,
        distanceM: meta.distanceM,
        accent: accent,
      );
    }
  }
}

class _CameraViewDiagram extends StatelessWidget {
  final String cameraView;
  final double distanceM;
  final Color accent;

  const _CameraViewDiagram({
    required this.cameraView,
    required this.distanceM,
    required this.accent,
  });

  @override
  Widget build(BuildContext context) {
    return CustomPaint(
      painter: _DiagramPainter(
        cameraView: cameraView,
        distanceM: distanceM,
        accent: accent,
      ),
      size: const Size(double.infinity, double.infinity),
    );
  }
}

class _DiagramPainter extends CustomPainter {
  final String cameraView;
  final double distanceM;
  final Color accent;

  _DiagramPainter({
    required this.cameraView,
    required this.distanceM,
    required this.accent,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = accent
      ..strokeWidth = 2
      ..style = PaintingStyle.stroke;

    final solidPaint = Paint()
      ..color = accent
      ..style = PaintingStyle.fill;

    final textPainter = TextPainter(
      text: TextSpan(
        text: '${distanceM.toStringAsFixed(1)} m',
        style: TextStyle(color: accent, fontSize: 12),
      ),
      textDirection: TextDirection.ltr,
    );
    textPainter.layout();

    if (cameraView == 'side') {
      // Phone on left, figure on right
      final phoneRect = Rect.fromLTWH(8, size.height / 2 - 20, 30, 40);
      canvas.drawRect(phoneRect, paint);
      canvas.drawLine(Offset(18, size.height / 2 - 20),
          Offset(18, size.height / 2 + 20), paint);

      // Stick figure (side view)
      final figureX = size.width - 30;
      final figureY = size.height / 2;
      canvas.drawCircle(Offset(figureX, figureY - 15), 6, solidPaint);
      canvas.drawLine(Offset(figureX, figureY - 9), Offset(figureX, figureY),
          paint);
      canvas.drawLine(Offset(figureX - 8, figureY - 5),
          Offset(figureX + 8, figureY - 5), paint);
      canvas.drawLine(Offset(figureX - 6, figureY),
          Offset(figureX - 10, figureY + 12), paint);
      canvas.drawLine(Offset(figureX + 6, figureY),
          Offset(figureX + 10, figureY + 12), paint);

      // Distance line
      final lineY = size.height / 2 + 30;
      canvas.drawLine(Offset(38, lineY), Offset(size.width - 36, lineY), paint);
      canvas.drawLine(Offset(38, lineY - 4), Offset(38, lineY + 4), paint);
      canvas.drawLine(Offset(size.width - 36, lineY - 4),
          Offset(size.width - 36, lineY + 4), paint);
      textPainter.paint(
          canvas, Offset(size.width / 2 - textPainter.width / 2, lineY + 6));
    } else {
      // Front view: phone at bottom center, figure facing it
      final phoneRect = Rect.fromLTWH(
          size.width / 2 - 15, size.height - 40, 30, 35);
      canvas.drawRect(phoneRect, paint);

      // Stick figure (front view)
      final figureX = size.width / 2;
      final figureY = size.height / 2 - 20;
      canvas.drawCircle(Offset(figureX, figureY), 6, solidPaint);
      canvas.drawLine(Offset(figureX, figureY + 6),
          Offset(figureX, figureY + 18), paint);
      canvas.drawLine(Offset(figureX - 12, figureY + 10),
          Offset(figureX + 12, figureY + 10), paint);
      canvas.drawLine(Offset(figureX - 6, figureY + 18),
          Offset(figureX - 10, figureY + 32), paint);
      canvas.drawLine(Offset(figureX + 6, figureY + 18),
          Offset(figureX + 10, figureY + 32), paint);

      // Distance line (vertical)
      final lineX = size.width / 2 + 40;
      canvas.drawLine(Offset(lineX, figureY + 6),
          Offset(lineX, size.height - 40), paint);
      canvas.drawLine(Offset(lineX - 4, figureY + 6),
          Offset(lineX + 4, figureY + 6), paint);
      canvas.drawLine(Offset(lineX - 4, size.height - 40),
          Offset(lineX + 4, size.height - 40), paint);
      textPainter.paint(canvas, Offset(lineX + 8, size.height / 2 - 8));
    }
  }

  @override
  bool shouldRepaint(_DiagramPainter oldDelegate) {
    return oldDelegate.cameraView != cameraView ||
        oldDelegate.distanceM != distanceM ||
        oldDelegate.accent != accent;
  }
}
