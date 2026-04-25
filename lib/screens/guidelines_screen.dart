import 'package:flutter/material.dart';
import '../models/exercise_meta.dart';

class GuidelinesScreen extends StatelessWidget {
  final ExerciseMeta meta;

  const GuidelinesScreen({super.key, required this.meta});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(meta.displayName),
      ),
      backgroundColor: Colors.grey[50],
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          // Diagram card
          Card(
            child: SizedBox(
              height: 300,
              child: Padding(
                padding: const EdgeInsets.all(24),
                child: _DiagramWidget(meta: meta),
              ),
            ),
          ),
          const SizedBox(height: 16),

          // Setup info card
          Card(
            child: Column(
              children: [
                ListTile(
                  leading: const Icon(Icons.videocam, color: Color(0xFF6C63FF)),
                  title: Text(
                    '${meta.cameraView[0].toUpperCase()}${meta.cameraView.substring(1)} view',
                  ),
                ),
                const Divider(height: 1),
                ListTile(
                  leading:
                      const Icon(Icons.straighten, color: Color(0xFF6C63FF)),
                  title: Text('${meta.distanceM} m away'),
                ),
                const Divider(height: 1),
                ListTile(
                  leading:
                      const Icon(Icons.height, color: Color(0xFF6C63FF)),
                  title: Row(
                    children: [
                      Text('Phone at ${meta.phoneHeightCm} cm'),
                      const SizedBox(width: 8),
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
          const SizedBox(height: 16),

          // Guidelines text
          Card(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Text(
                meta.guidelines,
                style: Theme.of(context).textTheme.bodyMedium,
              ),
            ),
          ),
          const SizedBox(height: 16),

          // Key errors
          if (meta.keyErrors.isNotEmpty) ...[
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 4),
              child: Text(
                'Key Areas to Monitor',
                style: Theme.of(context).textTheme.labelLarge,
              ),
            ),
            const SizedBox(height: 8),
            Wrap(
              spacing: 8,
              runSpacing: 8,
              children: meta.keyErrors
                  .map(
                    (error) => Chip(
                      label: Text(
                        error[0].toUpperCase() + error.substring(1),
                        style: const TextStyle(fontSize: 12),
                      ),
                      backgroundColor: Colors.grey[200],
                    ),
                  )
                  .toList(),
            ),
            const SizedBox(height: 24),
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
          const SizedBox(height: 12),
          OutlinedButton.icon(
            icon: const Icon(Icons.upload_file),
            label: const Text('Upload Video'),
            onPressed: () {
              Navigator.of(context)
                  .pushNamed('/video-upload', arguments: meta);
            },
          ),
          const SizedBox(height: 32),
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
    if (meta.cameraView == 'either') {
      // Side-by-side layout for "either"
      return Row(
        mainAxisAlignment: MainAxisAlignment.spaceEvenly,
        children: [
          Expanded(
            child: _CameraViewDiagram(
              cameraView: 'side',
              distanceM: meta.distanceM,
            ),
          ),
          const SizedBox(width: 16),
          Expanded(
            child: _CameraViewDiagram(
              cameraView: 'front',
              distanceM: meta.distanceM,
            ),
          ),
        ],
      );
    } else {
      return _CameraViewDiagram(
        cameraView: meta.cameraView,
        distanceM: meta.distanceM,
      );
    }
  }
}

class _CameraViewDiagram extends StatelessWidget {
  final String cameraView;
  final double distanceM;

  const _CameraViewDiagram({
    required this.cameraView,
    required this.distanceM,
  });

  @override
  Widget build(BuildContext context) {
    return CustomPaint(
      painter: _DiagramPainter(cameraView: cameraView, distanceM: distanceM),
      size: const Size(double.infinity, double.infinity),
    );
  }
}

class _DiagramPainter extends CustomPainter {
  final String cameraView;
  final double distanceM;

  _DiagramPainter({required this.cameraView, required this.distanceM});

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = const Color(0xFF6C63FF)
      ..strokeWidth = 2
      ..style = PaintingStyle.stroke;

    final solidPaint = Paint()
      ..color = const Color(0xFF6C63FF)
      ..style = PaintingStyle.fill;

    final textPainter = TextPainter(
      text: TextSpan(
        text: '${distanceM.toStringAsFixed(1)} m',
        style: const TextStyle(color: Color(0xFF6C63FF), fontSize: 12),
      ),
      textDirection: TextDirection.ltr,
    );
    textPainter.layout();

    if (cameraView == 'side') {
      // Phone on left, figure on right
      // Phone icon
      final phoneRect = Rect.fromLTWH(8, size.height / 2 - 20, 30, 40);
      canvas.drawRect(phoneRect, paint);
      canvas.drawLine(Offset(18, size.height / 2 - 20),
          Offset(18, size.height / 2 + 20), paint);

      // Stick figure (side view: circle head, line body, lines for limbs)
      final figureX = size.width - 30;
      final figureY = size.height / 2;
      canvas.drawCircle(Offset(figureX, figureY - 15), 6, solidPaint);
      canvas.drawLine(Offset(figureX, figureY - 9), Offset(figureX, figureY),
          paint);
      canvas.drawLine(Offset(figureX - 8, figureY - 5), Offset(figureX + 8, figureY - 5),
          paint);
      canvas.drawLine(Offset(figureX - 6, figureY), Offset(figureX - 10, figureY + 12),
          paint);
      canvas.drawLine(Offset(figureX + 6, figureY), Offset(figureX + 10, figureY + 12),
          paint);

      // Distance line
      final lineY = size.height / 2 + 30;
      canvas.drawLine(Offset(38, lineY), Offset(size.width - 36, lineY), paint);
      canvas.drawLine(Offset(38, lineY - 4), Offset(38, lineY + 4), paint);
      canvas.drawLine(
          Offset(size.width - 36, lineY - 4), Offset(size.width - 36, lineY + 4), paint);
      textPainter.paint(
          canvas, Offset(size.width / 2 - textPainter.width / 2, lineY + 6));
    } else {
      // Front view: phone at bottom center, figure facing it
      // Phone icon (portrait)
      final phoneRect = Rect.fromLTWH(size.width / 2 - 15, size.height - 40, 30, 35);
      canvas.drawRect(phoneRect, paint);

      // Stick figure (front view: circle head, line body, arms out, legs)
      final figureX = size.width / 2;
      final figureY = size.height / 2 - 20;
      canvas.drawCircle(Offset(figureX, figureY), 6, solidPaint);
      canvas.drawLine(Offset(figureX, figureY + 6), Offset(figureX, figureY + 18),
          paint);
      canvas.drawLine(Offset(figureX - 12, figureY + 10), Offset(figureX + 12, figureY + 10),
          paint);
      canvas.drawLine(Offset(figureX - 6, figureY + 18), Offset(figureX - 10, figureY + 32),
          paint);
      canvas.drawLine(Offset(figureX + 6, figureY + 18), Offset(figureX + 10, figureY + 32),
          paint);

      // Distance line (vertical)
      final lineX = size.width / 2 + 40;
      canvas.drawLine(Offset(lineX, figureY + 6), Offset(lineX, size.height - 40), paint);
      canvas.drawLine(
          Offset(lineX - 4, figureY + 6), Offset(lineX + 4, figureY + 6), paint);
      canvas.drawLine(Offset(lineX - 4, size.height - 40), Offset(lineX + 4, size.height - 40),
          paint);
      textPainter.paint(canvas, Offset(lineX + 8, size.height / 2 - 8));
    }
  }

  @override
  bool shouldRepaint(_DiagramPainter oldDelegate) {
    return oldDelegate.cameraView != cameraView ||
        oldDelegate.distanceM != distanceM;
  }
}
