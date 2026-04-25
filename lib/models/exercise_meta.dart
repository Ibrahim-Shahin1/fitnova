/// Matches backend/config/exercises.json entry shape.
class ExerciseMeta {
  final String name;              // snake_case key (e.g. "squat")
  final int idx;
  final String displayName;
  final String cameraView;        // "side" | "front" | "either"
  final double distanceM;
  final int phoneHeightCm;
  final String orientation;       // "landscape" | "portrait"
  final List<String> keyErrors;
  final String guidelines;

  const ExerciseMeta({
    required this.name,
    required this.idx,
    required this.displayName,
    required this.cameraView,
    required this.distanceM,
    required this.phoneHeightCm,
    required this.orientation,
    required this.keyErrors,
    required this.guidelines,
  });

  factory ExerciseMeta.fromJson(String name, Map<String, dynamic> json) {
    return ExerciseMeta(
      name: name,
      idx: json['idx'] as int,
      displayName: json['display_name'] as String,
      cameraView: json['camera_view'] as String,
      distanceM: (json['distance_m'] as num).toDouble(),
      phoneHeightCm: json['phone_height_cm'] as int,
      orientation: json['orientation'] as String,
      keyErrors: (json['key_errors_detected'] as List<dynamic>)
          .map((e) => e as String).toList(),
      guidelines: json['guidelines'] as String,
    );
  }
}
