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


// ─────────────────────────────────────────────────────────────────────────────
// D-05 schema models — Phase 5 PyTorch Squat backend.
//
// These replace the live/upload path's use of the old MediaPipe-era models
// above (FormFrameResult / FormSessionSummary), which are kept ONLY so the
// now-unreachable FormReplayScreen + /form-replay route still compile.
// Backend contract: 05-CONTEXT.md D-05.
//   - upload  POST /analyze-form-video → UploadResponse{exercise,total_reps,reps[...]}
//   - live    WS  /ws/form-session     → rep_result / session_summary
// No percentages are shown to the user; severity_word is a confidence-derived
// estimate, not a measured result.
// ─────────────────────────────────────────────────────────────────────────────

/// One detected (or not-detected) form error from the D-05 schema.
class FormError {
  final String type;          // "KIE" (knees caving inward) | "KFE" (knees too far forward)
  final bool detected;
  final double confidence;    // 0.0–1.0 sigmoid score (not shown as a % in the UI)
  final String severityWord;  // none | possible | moderate | strong
  final List<List<double>> intervals; // [[startSec, endSec], ...] for detected errors

  const FormError({
    required this.type,
    required this.detected,
    required this.confidence,
    required this.severityWord,
    this.intervals = const [],
  });

  factory FormError.fromJson(Map<String, dynamic> json) => FormError(
        type: json['type'] as String? ?? '',
        detected: json['detected'] as bool? ?? false,
        confidence: (json['confidence'] as num? ?? 0.0).toDouble(),
        severityWord: json['severity_word'] as String? ?? 'none',
        intervals: (json['intervals'] as List<dynamic>? ?? [])
            .map((pair) => (pair as List<dynamic>)
                .map((v) => (v as num).toDouble())
                .toList())
            .toList(),
      );

  /// Human-readable error name for the UI.
  String get label => switch (type) {
        'KIE' => 'Knees caving inward',
        'KFE' => 'Knees too far forward',
        _ => type,
      };

  /// "Knees caving inward — moderate" (detected) or "… — not detected".
  String get summary =>
      detected ? '$label — $severityWord' : '$label — not detected';
}

/// One rep's result: a list of error detections. Upload reps carry no repNumber;
/// live `rep_result` messages carry a 1-based repNumber — a periodic "form check"
/// fired on a sliding-window clock, NOT a true rep count (see PLAN D2).
class FormRep {
  final String exercise;
  final int? repNumber;
  final List<FormError> errors;

  const FormRep({
    required this.exercise,
    this.repNumber,
    this.errors = const [],
  });

  factory FormRep.fromJson(Map<String, dynamic> json) => FormRep(
        exercise: json['exercise'] as String? ?? 'squat',
        repNumber: json['rep_number'] as int?,
        errors: (json['errors'] as List<dynamic>? ?? [])
            .map((e) => FormError.fromJson(e as Map<String, dynamic>))
            .toList(),
      );

  /// Detected errors only (for chip display).
  List<FormError> get detectedErrors =>
      errors.where((e) => e.detected).toList();
}

/// Unified form report rendered by FormResultsScreen — built from either the
/// upload UploadResponse or the live session_summary.
class FormReport {
  final String exercise;
  final int totalReps;          // upload: total_reps (1 for single-rep)
  final List<FormRep> reps;
  final String sessionFeedback; // upload: synthesized client-side
  final List<String> modelViewFrames; // base64 JPEGs the model actually analyzed (upload)
  final double durationS;       // clip duration in seconds (upload)

  const FormReport({
    required this.exercise,
    required this.totalReps,
    required this.reps,
    required this.sessionFeedback,
    this.modelViewFrames = const [],
    this.durationS = 0.0,
  });

  /// The single analyzed rep (single-rep upload), or null if none.
  FormRep? get rep => reps.isNotEmpty ? reps.first : null;

  /// From POST /analyze-form-video → UploadResponse (single-rep).
  factory FormReport.fromUpload(Map<String, dynamic> json) {
    final reps = (json['reps'] as List<dynamic>? ?? [])
        .map((r) => FormRep.fromJson(r as Map<String, dynamic>))
        .toList();
    final mv = json['model_view'] as Map<String, dynamic>?;
    return FormReport(
      exercise: json['exercise'] as String? ?? 'squat',
      totalReps: json['total_reps'] as int? ?? reps.length,
      reps: reps,
      sessionFeedback: _synthFeedback(reps),
      modelViewFrames: ((mv?['frames']) as List<dynamic>? ?? [])
          .map((f) => f as String)
          .toList(),
      durationS: (json['duration_s'] as num? ?? 0.0).toDouble(),
    );
  }

  /// From WS /ws/form-session → session_summary.
  factory FormReport.fromSession(Map<String, dynamic> json) {
    final reps = (json['rep_results'] as List<dynamic>? ?? [])
        .map((r) => FormRep.fromJson(r as Map<String, dynamic>))
        .toList();
    return FormReport(
      exercise: json['exercise'] as String? ?? 'squat',
      totalReps: json['total_reps'] as int? ?? reps.length,
      reps: reps,
      sessionFeedback:
          json['session_feedback'] as String? ?? _synthFeedback(reps),
    );
  }

  String get exerciseDisplay => exercise
      .replaceAll('_', ' ')
      .split(' ')
      .map((w) => w.isEmpty ? w : '${w[0].toUpperCase()}${w.substring(1)}')
      .join(' ');

  /// Count of reps where the given error type was detected.
  int detectedCount(String type) => reps
      .where((r) => r.errors.any((e) => e.type == type && e.detected))
      .length;

  /// Deterministic summary for the upload path (backend sends none). Mirrors the
  /// live session_feedback style — no percentages (D-05).
  static String _synthFeedback(List<FormRep> reps) {
    if (reps.isEmpty) {
      return 'No reps were analyzed — try a longer clip with your full body in frame.';
    }
    final n = reps.length;
    final kie = reps
        .where((r) => r.errors.any((e) => e.type == 'KIE' && e.detected))
        .length;
    final kfe = reps
        .where((r) => r.errors.any((e) => e.type == 'KFE' && e.detected))
        .length;
    final parts = <String>['Analyzed $n rep${n == 1 ? '' : 's'}.'];
    parts.add(kie > 0
        ? 'Knees caving inward on $kie of $n — drive your knees out in line with your toes.'
        : 'Good knee tracking — no inward caving detected.');
    parts.add(kfe > 0
        ? 'Knees travelling too far forward on $kfe of $n — initiate by pushing your hips back.'
        : 'Good depth control — no forward-knee error detected.');
    return parts.join(' ');
  }
}
