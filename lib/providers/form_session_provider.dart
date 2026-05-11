import 'package:flutter/foundation.dart';

import '../models/form_models.dart';

enum FormSessionState { idle, connecting, active, ending, done, error }

/// Global state for an active or completed form analysis session.
class FormSessionProvider extends ChangeNotifier {
  FormSessionState _state = FormSessionState.idle;
  FormFrameResult? _latestFrame;
  FormSessionSummary? _summary;
  String? _errorMessage;

  // Live metrics (updated each frame)
  double _liveQuality  = 0.5;
  String? _liveQualityLabel;
  int    _liveRepCount = 0;
  String _liveExercise = 'detecting…';
  List<double> _liveJointErrors = List.filled(10, 0.0);
  List<List<double>>? _liveLandmarks;
  List<ActiveFlag> _liveActiveFlags = const [];
  String? _livePhase;
  bool _liveGeometricReady = false;

  // Mismatch warning state
  String? _mismatchPredicted;
  String? _mismatchSelected;
  double? _mismatchConfidence;
  bool _mismatchDismissed = false;

  // ── Getters ────────────────────────────────────────────────────────────────

  FormSessionState   get state           => _state;
  FormFrameResult?   get latestFrame     => _latestFrame;
  FormSessionSummary? get summary        => _summary;
  String?            get errorMessage    => _errorMessage;

  double             get liveQuality     => _liveQuality;
  String?            get liveQualityLabel => _liveQualityLabel;
  int                get liveRepCount    => _liveRepCount;
  String             get liveExercise    => _liveExercise;
  List<double>       get liveJointErrors => _liveJointErrors;
  List<List<double>>? get liveLandmarks  => _liveLandmarks;
  List<ActiveFlag>   get liveActiveFlags => _liveActiveFlags;
  String?            get livePhase       => _livePhase;
  bool               get liveGeometricReady => _liveGeometricReady;

  bool get isActive => _state == FormSessionState.active;
  bool get isDone   => _state == FormSessionState.done;

  bool   get hasMismatchWarning =>
      _mismatchPredicted != null && !_mismatchDismissed;
  String get mismatchPredicted  => _mismatchPredicted ?? '';
  String get mismatchSelected   => _mismatchSelected ?? '';
  double get mismatchConfidence => _mismatchConfidence ?? 0.0;

  // ── State transitions ──────────────────────────────────────────────────────

  void startSession() {
    _state          = FormSessionState.active;
    _latestFrame    = null;
    _summary        = null;
    _errorMessage   = null;
    _liveQuality    = 0.5;
    _liveQualityLabel = null;
    _liveRepCount   = 0;
    _liveExercise   = 'detecting…';
    _liveJointErrors = List.filled(10, 0.0);
    _liveLandmarks  = null;
    _liveActiveFlags = const [];
    _livePhase      = null;
    _liveGeometricReady = false;
    _mismatchPredicted   = null;
    _mismatchSelected    = null;
    _mismatchConfidence  = null;
    _mismatchDismissed   = false;
    notifyListeners();
  }

  void updateFrame(FormFrameResult frame) {
    _latestFrame     = frame;
    _liveQuality     = frame.qualityScore;
    _liveQualityLabel = frame.qualityLabel;
    _liveRepCount    = frame.repCount;
    _liveExercise    = frame.exerciseDetected;
    _liveJointErrors = frame.jointErrors;
    _liveLandmarks   = frame.landmarks;
    _liveActiveFlags = frame.activeFlags;
    _livePhase       = frame.phase;
    _liveGeometricReady = frame.geometricReady;
    notifyListeners();
  }

  void setEnding() {
    _state = FormSessionState.ending;
    notifyListeners();
  }

  void setSummary(FormSessionSummary summary) {
    _summary = summary;
    _state   = FormSessionState.done;
    notifyListeners();
  }

  void setError(String message) {
    _errorMessage = message;
    _state        = FormSessionState.error;
    notifyListeners();
  }

  void reset() {
    _state           = FormSessionState.idle;
    _latestFrame     = null;
    _summary         = null;
    _errorMessage    = null;
    _liveQuality     = 0.5;
    _liveRepCount    = 0;
    _liveExercise    = 'detecting…';
    _liveJointErrors = List.filled(10, 0.0);
    _liveLandmarks   = null;
    _mismatchPredicted   = null;
    _mismatchSelected    = null;
    _mismatchConfidence  = null;
    _mismatchDismissed   = false;
    notifyListeners();
  }

  void setMismatchWarning({
    required String selected,
    required String predicted,
    required double confidence,
  }) {
    _mismatchSelected    = selected;
    _mismatchPredicted   = predicted;
    _mismatchConfidence  = confidence;
    _mismatchDismissed   = false;
    notifyListeners();
  }

  void dismissMismatch() {
    _mismatchDismissed = true;
    notifyListeners();
  }
}
