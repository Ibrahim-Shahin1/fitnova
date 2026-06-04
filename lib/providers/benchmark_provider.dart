import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;

import '../models/benchmark_models.dart';
import '../models/form_models.dart';
import '../services/api_service.dart';

enum BenchmarkLoadState { idle, loading, loaded, error }

enum BenchmarkAnalysisState { idle, analyzing, done, noReps, error }

class BenchmarkProvider extends ChangeNotifier {
  final Map<String, List<BenchmarkClip>> _clips = {};
  final Map<String, BenchmarkLoadState> _loadState = {};

  Set<String> activeFilters = {'All'};
  String selectedExercise = 'squat';

  BenchmarkAnalysisState analysisState = BenchmarkAnalysisState.idle;
  List<FormRep>? analysisResult;
  String? analysisError;
  BenchmarkClip? selectedClip;

  List<BenchmarkClip> clips(String exercise) => _clips[exercise] ?? [];

  BenchmarkLoadState loadState(String exercise) =>
      _loadState[exercise] ?? BenchmarkLoadState.idle;

  // Returns whether the clip is correctly called for all its errors (browse-screen filter).
  bool clipIsCorrect(String exercise, BenchmarkClip clip) =>
      benchmarkClipIsCorrect(exercise, clip);

  // clip_id format is {video}_{rep}_{frame}; the rep is one squat attempt.
  String _repKey(String clipId) {
    final p = clipId.split('_');
    return p.length >= 2 ? '${p[0]}_${p[1]}' : clipId;
  }

  int _frameNum(String clipId) {
    final p = clipId.split('_');
    return p.length >= 3 ? (int.tryParse(p[2]) ?? 0) : 0;
  }

  // Shallow is a per-frame image benchmark that samples many consecutive frames per
  // rep; collapse to one representative frame per rep (the deepest moment = highest DEPTH
  // score, since the score is P(reached depth)) so the list reads as distinct squat
  // attempts. Other exercises are already one clip per rep and pass through unchanged.
  // The 540-frame F1 is unaffected.
  List<BenchmarkClip> browseClips(String exercise) {
    final all = clips(exercise);
    if (exercise != 'shallow') return all;
    final byRep = <String, BenchmarkClip>{};
    for (final c in all) {
      final key = _repKey(c.clipId);
      final cur = byRep[key];
      if (cur == null ||
          (c.score['DEPTH'] ?? 0.0) > (cur.score['DEPTH'] ?? 0.0)) {
        byRep[key] = c;
      }
    }
    return byRep.values.toList()
      ..sort((a, b) => a.clipId.compareTo(b.clipId));
  }

  // Filmstrip: all frames of one shallow rep, in chronological (descent) order, so the
  // user can scrub the whole movement and watch the per-frame DEPTH score track the squat.
  List<BenchmarkClip> filmstripFrames = const [];
  String filmstripRep = '';

  void selectShallowRep(BenchmarkClip rep) {
    final key = _repKey(rep.clipId);
    filmstripFrames = clips('shallow')
        .where((c) => _repKey(c.clipId) == key)
        .toList()
      ..sort((a, b) => _frameNum(a.clipId).compareTo(_frameNum(b.clipId)));
    filmstripRep = key;
    notifyListeners();
  }

  Future<void> loadCatalog(String exercise) async {
    if (_loadState[exercise] == BenchmarkLoadState.loaded) return;

    _loadState[exercise] = BenchmarkLoadState.loading;
    notifyListeners();

    try {
      final clips = await ApiService.getBenchmarkCatalog(exercise);
      _clips[exercise] = clips;
      _loadState[exercise] = BenchmarkLoadState.loaded;
    } catch (_) {
      _loadState[exercise] = BenchmarkLoadState.error;
    }

    notifyListeners();
  }

  Future<void> startAnalysis(String exercise, BenchmarkClip clip) async {
    selectedExercise = exercise;
    selectedClip = clip;
    analysisState = BenchmarkAnalysisState.analyzing;
    analysisError = null;
    analysisResult = null;
    notifyListeners();

    try {
      final List<FormRep> reps;
      if (exercise != 'shallow') {
        final bytes = (await http.get(
                Uri.parse(ApiService.benchmarkMediaUrl(exercise, clip.clipId))))
            .bodyBytes;
        reps = await ApiService.analyzeClip(exercise, clip.clipId,
            fileBytes: bytes);
      } else {
        reps = await ApiService.analyzeClip('shallow', clip.clipId);
      }

      analysisResult = reps;
      analysisState = reps.isEmpty
          ? BenchmarkAnalysisState.noReps
          : BenchmarkAnalysisState.done;
    } catch (e) {
      analysisError = e.toString();
      analysisState = BenchmarkAnalysisState.error;
    }

    notifyListeners();
  }

  Future<void> retriggerAnalysis() async {
    if (selectedClip == null) return;
    await startAnalysis(selectedExercise, selectedClip!);
  }

  void setActiveFilters(Set<String> filters) {
    activeFilters = filters;
    notifyListeners();
  }

  void clearFilters() {
    activeFilters = {'All'};
    notifyListeners();
  }

  void resetAnalysis() {
    analysisState = BenchmarkAnalysisState.idle;
    analysisResult = null;
    analysisError = null;
    notifyListeners();
  }
}
