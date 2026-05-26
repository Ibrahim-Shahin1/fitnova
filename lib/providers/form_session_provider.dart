import 'package:flutter/foundation.dart';

import '../models/form_models.dart';

enum FormSessionState { idle, connecting, active, ending, done, error }

/// Global state for an active or completed form-analysis session (D-05 schema).
///
/// The Phase-5 PyTorch backend emits no pose/landmarks/quality — only per-rep
/// KIE/KFE detections. Live results arrive as periodic "form checks" (a
/// sliding-window cadence, ~every few seconds — NOT a true rep count); the
/// upload path returns one [FormReport] for the whole clip.
class FormSessionProvider extends ChangeNotifier {
  FormSessionState _state = FormSessionState.idle;
  String? _errorMessage;

  // Live state: periodic per-window form checks (not true reps — see model docs).
  final List<FormRep> _liveReps = [];
  FormRep? _lastRep;
  int _formCheckCount = 0;

  // Completed report (live session_summary OR upload UploadResponse).
  FormReport? _report;

  // ── Getters ────────────────────────────────────────────────────────────────

  FormSessionState get state => _state;
  String? get errorMessage => _errorMessage;

  List<FormRep> get liveReps => List.unmodifiable(_liveReps);
  FormRep? get lastRep => _lastRep;
  int get formCheckCount => _formCheckCount;

  FormReport? get report => _report;

  bool get isActive => _state == FormSessionState.active;
  bool get isDone => _state == FormSessionState.done;

  // ── Transitions ──────────────────────────────────────────────────────────────

  void startSession() {
    _state = FormSessionState.active;
    _errorMessage = null;
    _liveReps.clear();
    _lastRep = null;
    _formCheckCount = 0;
    _report = null;
    notifyListeners();
  }

  /// Append a live `rep_result` (a periodic form check).
  void addLiveRep(FormRep rep) {
    _liveReps.add(rep);
    _lastRep = rep;
    _formCheckCount += 1;
    notifyListeners();
  }

  void setEnding() {
    _state = FormSessionState.ending;
    notifyListeners();
  }

  /// Set the completed report (live `session_summary` or upload) and mark done.
  void setReport(FormReport report) {
    _report = report;
    _state = FormSessionState.done;
    notifyListeners();
  }

  void setError(String message) {
    _errorMessage = message;
    _state = FormSessionState.error;
    notifyListeners();
  }

  void reset() {
    _state = FormSessionState.idle;
    _errorMessage = null;
    _liveReps.clear();
    _lastRep = null;
    _formCheckCount = 0;
    _report = null;
    notifyListeners();
  }
}
