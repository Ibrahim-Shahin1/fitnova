// Frozen browse thresholds — match results.pkl val-tuned values exactly.
const Map<String, Map<String, double>> kBrowseThresholds = {
  'squat':   {'KIE': 0.614, 'KFE': 0.385},
  'ohp':     {'ELBOWS': 0.357, 'KNEES': 0.476},
  'shallow': {'DEPTH': 0.395},
};

class BenchmarkClip {
  final String clipId;
  final Map<String, int> groundTruth;
  final Map<String, double> score;

  const BenchmarkClip({
    required this.clipId,
    required this.groundTruth,
    required this.score,
  });

  factory BenchmarkClip.fromJson(Map<String, dynamic> json) => BenchmarkClip(
        clipId: json['clip_id'] as String,
        groundTruth: (json['ground_truth'] as Map<String, dynamic>)
            .map((k, v) => MapEntry(k, (v as num).toInt())),
        score: (json['score'] as Map<String, dynamic>)
            .map((k, v) => MapEntry(k, (v as num).toDouble())),
      );
}

bool benchmarkClipIsDetected(String exercise, String err, BenchmarkClip clip) =>
    clip.score[err]! >= kBrowseThresholds[exercise]![err]!;

bool benchmarkErrorIsCorrect(String exercise, String err, BenchmarkClip clip) =>
    benchmarkClipIsDetected(exercise, err, clip) == (clip.groundTruth[err] == 1);

bool benchmarkClipIsCorrect(String exercise, BenchmarkClip clip) =>
    clip.score.keys.every((err) => benchmarkErrorIsCorrect(exercise, err, clip));

// Plain-language labels for the dataset error codes (UI display only).
String benchmarkErrorChip(String type) => switch (type) {
      'KIE' => 'Knees in',
      'KFE' => 'Knees fwd',
      'ELBOWS' => 'Elbows',
      'KNEES' => 'Knees',
      'DEPTH' => 'Depth',
      _ => type,
    };

String benchmarkErrorLabel(String type) => switch (type) {
      'KIE' => 'Knees caving in',
      'KFE' => 'Knees forward',
      'ELBOWS' => 'Elbows',
      'KNEES' => 'Knees',
      'DEPTH' => 'Squat depth',
      _ => type,
    };

// The benchmark label in words. DEPTH is a depth call (1 = deep); the others
// are fault present/absent (1 = the error is present).
String benchmarkVerdict(String type, int value) => type == 'DEPTH'
    ? (value == 1 ? 'deep' : 'shallow')
    : (value == 1 ? 'present' : 'absent');
