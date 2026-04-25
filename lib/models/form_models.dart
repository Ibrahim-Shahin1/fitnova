// Form detection data models — mirrors backend FormFrameResult / FormSessionSummary

class FormFrameResult {
  final int timestampMs;
  final List<List<double>>? landmarks; // 15 joints × [x, y, z] normalised 0-1
  final List<double> jointErrors;      // 10 values, 0.0-1.0
  final double qualityScore;
  final int repCount;
  final String exerciseDetected;
  final double confidence;
  final String status;

  const FormFrameResult({
    required this.timestampMs,
    this.landmarks,
    required this.jointErrors,
    required this.qualityScore,
    required this.repCount,
    required this.exerciseDetected,
    required this.confidence,
    required this.status,
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
      repCount: json['rep_count'] as int? ?? 0,
      exerciseDetected: json['exercise_detected'] as String? ?? 'detecting…',
      confidence: (json['confidence'] as num? ?? 0.0).toDouble(),
      status: json['status'] as String? ?? 'ok',
    );
  }

  bool get hasPose => landmarks != null && status == 'ok';
}


class RepResult {
  final int repIdx;
  final double quality;
  final List<double> jointErrors;

  const RepResult({
    required this.repIdx,
    required this.quality,
    required this.jointErrors,
  });

  factory RepResult.fromJson(Map<String, dynamic> json) => RepResult(
        repIdx: json['rep_idx'] as int? ?? 0,
        quality: (json['quality'] as num? ?? 0.5).toDouble(),
        jointErrors: (json['joint_errors'] as List<dynamic>? ?? [])
            .map((v) => (v as num).toDouble())
            .toList(),
      );
}


class FormSessionSummary {
  final String exercise;
  final int totalReps;
  final int durationSeconds;
  final double averageQuality;
  final List<double> perRepScores;
  final Map<String, int> commonErrors;
  final String qualityTrend;
  final String llmFeedback;

  const FormSessionSummary({
    required this.exercise,
    required this.totalReps,
    required this.durationSeconds,
    required this.averageQuality,
    required this.perRepScores,
    required this.commonErrors,
    required this.qualityTrend,
    required this.llmFeedback,
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
      commonErrors: (json['common_errors'] as Map<String, dynamic>? ?? {})
          .map((k, v) => MapEntry(k, (v as num).toInt())),
      qualityTrend: json['quality_trend'] as String? ?? 'stable',
      llmFeedback: json['llm_feedback'] as String? ?? '',
    );
  }

  String get qualityPercent => '${(averageQuality * 100).round()}%';

  String get exerciseDisplay =>
      exercise.replaceAll('_', ' ').split(' ').map((w) =>
          w.isEmpty ? w : '${w[0].toUpperCase()}${w.substring(1)}').join(' ');
}
