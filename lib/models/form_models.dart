// Form detection data models — mirrors backend FormFrameResult / FormSessionSummary

/// Categorical form-issue chip emitted by the backend's geometric rule layer.
/// Replaces raw percentage values in the UI.
class ActiveFlag {
  final String name;     // canonical rule id, e.g. "knees_caving"
  final String label;    // human-readable, e.g. "Small Knee Caving"
  final double severity; // 0.0-1.0; useful for sorting / color

  const ActiveFlag({
    required this.name,
    required this.label,
    required this.severity,
  });

  factory ActiveFlag.fromJson(Map<String, dynamic> json) => ActiveFlag(
        name:     json['name']     as String? ?? '',
        label:    json['label']    as String? ?? '',
        severity: (json['severity'] as num? ?? 0.0).toDouble(),
      );
}


class FormFrameResult {
  final int timestampMs;
  final List<List<double>>? landmarks; // 15 joints × [x, y, z] normalised 0-1
  final List<double> jointErrors;      // 10 values, 0.0-1.0
  final double qualityScore;
  final String? qualityLabel;          // e.g. "Good Form"
  final int repCount;
  final String exerciseDetected;
  final double confidence;
  final String status;
  // New fields from the geometric rule layer (additive — old payloads still parse)
  final List<ActiveFlag> activeFlags;
  final String? phase;                 // "STANDING" | "DESCENDING" | "ASCENDING"
  final bool geometricReady;

  const FormFrameResult({
    required this.timestampMs,
    this.landmarks,
    required this.jointErrors,
    required this.qualityScore,
    this.qualityLabel,
    required this.repCount,
    required this.exerciseDetected,
    required this.confidence,
    required this.status,
    this.activeFlags = const [],
    this.phase,
    this.geometricReady = false,
  });

  factory FormFrameResult.fromJson(Map<String, dynamic> json) {
    return FormFrameResult(
      timestampMs: json['timestamp_ms'] as int? ?? 0,
      landmarks: (json['landmarks'] as List<dynamic>?)
          ?.map((j) => (j as List<dynamic>).map((v) => (v as num).toDouble()).toList())
          .toList(),
      jointErrors: (json['joint_errors'] as List<dynamic>? ?? [])
          .map((v) => (v as num).toDouble())
          .toList(),
      qualityScore: (json['quality_score'] as num? ?? 0.5).toDouble(),
      qualityLabel: json['quality_label'] as String?,
      repCount: json['rep_count'] as int? ?? 0,
      exerciseDetected: json['exercise_detected'] as String? ?? 'detecting…',
      confidence: (json['confidence'] as num? ?? 0.0).toDouble(),
      status: json['status'] as String? ?? 'ok',
      activeFlags: (json['active_flags'] as List<dynamic>? ?? [])
          .map((j) => ActiveFlag.fromJson(j as Map<String, dynamic>))
          .toList(),
      phase: json['phase'] as String?,
      geometricReady: json['geometric_ready'] as bool? ?? false,
    );
  }

  bool get hasPose => landmarks != null && status == 'ok';
}


class RepError {
  final String name;
  final double value;
  const RepError({required this.name, required this.value});

  factory RepError.fromJson(Map<String, dynamic> json) => RepError(
        name: json['name'] as String? ?? '',
        value: (json['value'] as num? ?? 0.0).toDouble(),
      );
}


class RepResult {
  final int repIdx;
  final double quality;
  final List<double> jointErrors;
  final List<RepError> topErrors;

  const RepResult({
    required this.repIdx,
    required this.quality,
    required this.jointErrors,
    this.topErrors = const [],
  });

  factory RepResult.fromJson(Map<String, dynamic> json) => RepResult(
        repIdx: json['rep_idx'] as int? ?? 0,
        quality: (json['quality'] as num? ?? 0.5).toDouble(),
        jointErrors: (json['joint_errors'] as List<dynamic>? ?? [])
            .map((v) => (v as num).toDouble())
            .toList(),
        topErrors: (json['top_errors'] as List<dynamic>? ?? [])
            .map((j) => RepError.fromJson(j as Map<String, dynamic>))
            .toList(),
      );

  String get qualityPercent => '${(quality * 100).round()}%';
}


/// Per-frame snapshot captured during a video upload analysis. Lets the
/// replay screen overlay the skeleton + joint errors at each video time.
class FormFrameTimeline {
  final int timestampMs;
  final List<List<double>>? landmarks; // 15 × [x, y, z]
  final List<double> jointErrors;      // 10-vector
  final double qualityScore;
  final String? qualityLabel;
  final int repCount;
  final List<ActiveFlag> activeFlags;
  final String? phase;

  const FormFrameTimeline({
    required this.timestampMs,
    this.landmarks,
    required this.jointErrors,
    required this.qualityScore,
    this.qualityLabel,
    required this.repCount,
    this.activeFlags = const [],
    this.phase,
  });

  factory FormFrameTimeline.fromJson(Map<String, dynamic> json) =>
      FormFrameTimeline(
        timestampMs: (json['timestamp_ms'] as num? ?? 0).toInt(),
        landmarks: (json['landmarks'] as List<dynamic>?)
            ?.map((j) => (j as List<dynamic>)
                .map((v) => (v as num).toDouble())
                .toList())
            .toList(),
        jointErrors: (json['joint_errors'] as List<dynamic>? ?? [])
            .map((v) => (v as num).toDouble())
            .toList(),
        qualityScore: (json['quality_score'] as num? ?? 0.5).toDouble(),
        qualityLabel: json['quality_label'] as String?,
        repCount: (json['rep_count'] as num? ?? 0).toInt(),
        activeFlags: (json['active_flags'] as List<dynamic>? ?? [])
            .map((j) => ActiveFlag.fromJson(j as Map<String, dynamic>))
            .toList(),
        phase: json['phase'] as String?,
      );
}


class FormSessionSummary {
  final String exercise;
  final int totalReps;
  final int durationSeconds;
  final double averageQuality;
  final List<double> perRepScores;
  final List<RepResult> perRepDetails;
  final Map<String, int> commonErrors;
  final String qualityTrend;
  final String llmFeedback;
  final String modelVersion;
  final List<FormFrameTimeline> timeline;

  const FormSessionSummary({
    required this.exercise,
    required this.totalReps,
    required this.durationSeconds,
    required this.averageQuality,
    required this.perRepScores,
    required this.perRepDetails,
    required this.commonErrors,
    required this.qualityTrend,
    required this.llmFeedback,
    required this.modelVersion,
    this.timeline = const [],
  });

  factory FormSessionSummary.fromJson(Map<String, dynamic> json) {
    return FormSessionSummary(
      exercise: json['exercise'] as String? ?? 'Unknown',
      totalReps: json['total_reps'] as int? ?? 0,
      durationSeconds: json['duration_seconds'] as int? ?? 0,
      averageQuality: (json['average_quality'] as num? ?? 0.5).toDouble(),
      perRepScores: (json['per_rep_scores'] as List<dynamic>? ?? [])
          .map((v) => (v as num).toDouble())
          .toList(),
      perRepDetails: (json['per_rep_details'] as List<dynamic>? ?? [])
          .map((j) => RepResult.fromJson(j as Map<String, dynamic>))
          .toList(),
      commonErrors: (json['common_errors'] as Map<String, dynamic>? ?? {})
          .map((k, v) => MapEntry(k, (v as num).toInt())),
      qualityTrend: json['quality_trend'] as String? ?? 'stable',
      llmFeedback: json['llm_feedback'] as String? ?? '',
      modelVersion: json['model_version'] as String? ?? 'v4',
      timeline: (json['timeline'] as List<dynamic>? ?? [])
          .map((j) => FormFrameTimeline.fromJson(j as Map<String, dynamic>))
          .toList(),
    );
  }

  String get qualityPercent => '${(averageQuality * 100).round()}%';

  String get exerciseDisplay =>
      exercise.replaceAll('_', ' ').split(' ').map((w) =>
          w.isEmpty ? w : '${w[0].toUpperCase()}${w.substring(1)}').join(' ');
}
