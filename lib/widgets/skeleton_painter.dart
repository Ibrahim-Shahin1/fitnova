import 'package:flutter/material.dart';

/// Skeleton bone connections (pairs of canonical joint indices)
const _connections = [
  [0, 2], [2, 4],   // L arm:  shoulder → elbow → wrist
  [1, 3], [3, 5],   // R arm:  shoulder → elbow → wrist
  [0, 1],           // collar: l_shoulder → r_shoulder
  [0, 6], [1, 7],   // torso sides
  [6, 7],           // pelvis span
  [12, 14], [14, 13], // spine: pelvis → spine_mid → neck
  [6, 8],  [8, 10], // L leg: hip → knee → ankle
  [7, 9],  [9, 11], // R leg: hip → knee → ankle
];

/// Joint-group → canonical joint indices mapping (for colouring)
/// Matches the 10 joint groups from the backend
const _groupToJoints = {
  0: [2],       // L Elbow
  1: [3],       // R Elbow
  2: [0],       // L Shoulder
  3: [1],       // R Shoulder
  4: [8],       // L Knee
  5: [9],       // R Knee
  6: [6],       // L Hip
  7: [7],       // R Hip
  8: [12, 13, 14], // Trunk
  9: [13],      // Neck
};

Color _jointColor(double errorProb) {
  if (errorProb < 0.3) return const Color(0xFF4CAF50); // green
  if (errorProb < 0.6) return const Color(0xFFFFC107); // amber
  return const Color(0xFFF44336);                       // red
}

/// Builds a per-joint (15-joint) error probability list from
/// the 10-group error list returned by the backend.
List<double> buildJointErrorMap(List<double> groupErrors) {
  final result = List<double>.filled(15, 0.0);
  _groupToJoints.forEach((groupIdx, jointIdxs) {
    final err = groupIdx < groupErrors.length ? groupErrors[groupIdx] : 0.0;
    for (final j in jointIdxs) {
      result[j] = (result[j] < err) ? err : result[j]; // max
    }
  });
  return result;
}

/// Draws the 15-joint skeleton on top of the camera preview.
class SkeletonPainter extends CustomPainter {
  /// 15 joints × [x, y, z] in image-normalised coordinates (0-1).
  final List<List<double>>? landmarks;

  /// Per-joint error probabilities (15 values after buildJointErrorMap).
  final List<double> jointErrors;

  /// Whether to mirror the X axis. Front-camera previews are displayed
  /// mirrored to the user (selfie convention) but MediaPipe processes the
  /// raw frame, so the landmarks come back in the un-mirrored frame's
  /// coordinate space. Set this to true for the front camera so the
  /// overlay tracks the user's body, not their reflection.
  final bool mirror;

  const SkeletonPainter({
    required this.landmarks,
    required this.jointErrors,
    this.mirror = false,
  });

  @override
  void paint(Canvas canvas, Size size) {
    if (landmarks == null || landmarks!.length < 15) return;

    final pts = landmarks!.map((j) {
      final nx = mirror ? (1.0 - j[0]) : j[0];
      return Offset(nx * size.width, j[1] * size.height);
    }).toList();

    final bonePaint = Paint()
      ..strokeWidth = 2.5
      ..style = PaintingStyle.stroke;

    // Draw bones
    for (final conn in _connections) {
      final a = conn[0], b = conn[1];
      if (a >= pts.length || b >= pts.length) continue;
      // Bone colour = average of the two endpoint error probabilities
      final avgErr = ((a < jointErrors.length ? jointErrors[a] : 0.0) +
              (b < jointErrors.length ? jointErrors[b] : 0.0)) /
          2.0;
      bonePaint.color = _jointColor(avgErr).withValues(alpha:0.75);
      canvas.drawLine(pts[a], pts[b], bonePaint);
    }

    // Draw joints
    for (int i = 0; i < pts.length; i++) {
      final err = i < jointErrors.length ? jointErrors[i] : 0.0;
      final color = _jointColor(err);
      canvas.drawCircle(pts[i], 5.0, Paint()..color = color);
      canvas.drawCircle(
        pts[i], 5.0,
        Paint()
          ..color = Colors.white.withValues(alpha:0.6)
          ..style = PaintingStyle.stroke
          ..strokeWidth = 1.5,
      );
    }
  }

  @override
  bool shouldRepaint(SkeletonPainter old) =>
      old.landmarks != landmarks ||
      old.jointErrors != jointErrors ||
      old.mirror != mirror;
}
