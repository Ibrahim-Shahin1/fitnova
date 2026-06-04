// Phase-9 honest per-rep schema — one error detection per FormError, one rep per FormRep.

/// One detected (or not-detected) form error from the Phase-9 backend payload.
class FormError {
  final String type;
  final bool detected;
  final double confidence;
  final double threshold;
  final int? groundTruth;
  final List<List<double>> intervals;

  const FormError({
    required this.type,
    required this.detected,
    required this.confidence,
    required this.threshold,
    this.groundTruth,
    this.intervals = const [],
  });

  factory FormError.fromJson(Map<String, dynamic> json) => FormError(
        type: json['type'] as String? ?? '',
        detected: json['detected'] as bool? ?? false,
        confidence: (json['confidence'] as num? ?? 0.0).toDouble(),
        threshold: (json['threshold'] as num? ?? 0.5).toDouble(),
        groundTruth: json['ground_truth'] as int?,
        intervals: (json['intervals'] as List<dynamic>? ?? [])
            .map((p) => (p as List<dynamic>)
                .map((v) => (v as num).toDouble())
                .toList())
            .toList(),
      );
}

/// One rep's error detections from POST /analyze-form-video.
class FormRep {
  final String exercise;
  final int repIndex;
  final int totalReps;
  final String? thumbnail;
  final List<FormError> errors;

  const FormRep({
    required this.exercise,
    required this.repIndex,
    required this.totalReps,
    this.thumbnail,
    this.errors = const [],
  });

  factory FormRep.fromJson(Map<String, dynamic> json) => FormRep(
        exercise: json['exercise'] as String? ?? 'squat',
        repIndex: json['rep_index'] as int? ?? 0,
        totalReps: json['total_reps'] as int? ?? 1,
        thumbnail: json['thumbnail'] as String?,
        errors: (json['errors'] as List<dynamic>? ?? [])
            .map((e) => FormError.fromJson(e as Map<String, dynamic>))
            .toList(),
      );

  List<FormError> get detectedErrors => errors.where((e) => e.detected).toList();
}
