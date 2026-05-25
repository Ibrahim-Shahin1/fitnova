import 'package:flutter/foundation.dart';

import '../models/workout_log.dart';
import '../services/log_service.dart';

/// Rolling 7-day window stats.
class WeekStats {
  const WeekStats(this.workouts, this.sets, this.volume);
  final int workouts; // distinct calendar days trained
  final int sets;
  final double volume; // kg (Σ weight×reps)
}

/// A best-ever set for one exercise (by estimated 1RM).
class PrEntry {
  const PrEntry({
    required this.exercise,
    required this.e1rm,
    required this.weightKg,
    required this.reps,
    required this.date,
  });
  final String exercise;
  final double e1rm;
  final double weightKg;
  final int reps;
  final DateTime date;
}

/// Loads the user's workout logs and derives progress metrics — estimated-1RM
/// progression, PRs, weekly volume, and this-week-vs-last-week. All math is
/// standard strength-training stuff computed from real logged sets.
class ProgressProvider extends ChangeNotifier {
  List<WorkoutLog> _logs = []; // newest first (server order)
  bool _loading = false;
  bool _initialized = false;
  Object? _error;
  String? _selected;

  List<WorkoutLog> get logs => List.unmodifiable(_logs);
  bool get loading => _loading;
  bool get hasData => _logs.isNotEmpty;
  Object? get error => _error;
  String? get selectedExercise => _selected;

  Future<void> load() async {
    _loading = true;
    _error = null;
    notifyListeners();
    try {
      _logs = await LogService.fetchLogs(days: 120);
      _initialized = true;
      final ex = exercisesWithStrengthData;
      if (_selected == null || !ex.contains(_selected)) {
        _selected = ex.isNotEmpty ? ex.first : null;
      }
    } catch (e) {
      _error = e;
    } finally {
      _loading = false;
      notifyListeners();
    }
  }

  /// Load once on first view; cheap no-op afterwards (call refresh() to reload).
  Future<void> ensureLoaded() async {
    if (!_initialized && !_loading) await load();
  }

  Future<void> refresh() => load();

  void select(String name) {
    _selected = name;
    notifyListeners();
  }

  /// Distinct exercises that have at least one weight×reps set, most-recent first.
  List<String> get exercisesWithStrengthData {
    final seen = <String>{};
    final out = <String>[];
    for (final l in _logs) {
      if (l.estimated1RM != null && seen.add(l.exerciseName)) {
        out.add(l.exerciseName);
      }
    }
    return out;
  }

  /// Best estimated-1RM per day for [name], oldest → newest.
  List<({DateTime date, double e1rm})> e1rmSeries(String name) {
    final byDay = <DateTime, double>{};
    for (final l in _logs) {
      if (l.exerciseName != name) continue;
      final e = l.estimated1RM;
      if (e == null) continue;
      final d = DateTime(
          l.performedAt.year, l.performedAt.month, l.performedAt.day);
      final cur = byDay[d];
      if (cur == null || e > cur) byDay[d] = e;
    }
    final entries = byDay.entries.toList()
      ..sort((a, b) => a.key.compareTo(b.key));
    return [for (final e in entries) (date: e.key, e1rm: e.value)];
  }

  WeekStats _statsBetween(DateTime start, DateTime end) {
    final days = <DateTime>{};
    var sets = 0;
    var volume = 0.0;
    for (final l in _logs) {
      if (l.performedAt.isAfter(start) && !l.performedAt.isAfter(end)) {
        sets++;
        volume += l.volume ?? 0;
        days.add(DateTime(
            l.performedAt.year, l.performedAt.month, l.performedAt.day));
      }
    }
    return WeekStats(days.length, sets, volume);
  }

  WeekStats get thisWeek {
    final now = DateTime.now();
    return _statsBetween(now.subtract(const Duration(days: 7)), now);
  }

  WeekStats get lastWeek {
    final now = DateTime.now();
    return _statsBetween(
        now.subtract(const Duration(days: 14)), now.subtract(const Duration(days: 7)));
  }

  /// Total volume per rolling 7-day bucket, oldest → newest ([weeks] buckets).
  List<({DateTime weekStart, double volume})> weeklyVolume({int weeks = 8}) {
    final now = DateTime.now();
    final endToday = DateTime(now.year, now.month, now.day)
        .add(const Duration(days: 1)); // exclusive end of today
    final out = <({DateTime weekStart, double volume})>[];
    for (var i = weeks - 1; i >= 0; i--) {
      final end = endToday.subtract(Duration(days: 7 * i));
      final start = end.subtract(const Duration(days: 7));
      var vol = 0.0;
      for (final l in _logs) {
        if (l.performedAt.isAfter(start) && !l.performedAt.isAfter(end)) {
          vol += l.volume ?? 0;
        }
      }
      out.add((weekStart: start, volume: vol));
    }
    return out;
  }

  /// Best set (by e1RM) per exercise, most-recently-achieved first.
  List<PrEntry> personalRecords() {
    final best = <String, PrEntry>{};
    for (final l in _logs) {
      final e = l.estimated1RM;
      if (e == null) continue;
      final cur = best[l.exerciseName];
      if (cur == null || e > cur.e1rm) {
        best[l.exerciseName] = PrEntry(
          exercise: l.exerciseName,
          e1rm: e,
          weightKg: l.weightKg!,
          reps: l.repsCompleted!,
          date: l.performedAt,
        );
      }
    }
    final list = best.values.toList()..sort((a, b) => b.date.compareTo(a.date));
    return list;
  }

  /// Estimated-1RM change for [name] from first to last logged session.
  ({double deltaKg, double pct, int days})? progressDelta(String name) {
    final s = e1rmSeries(name);
    if (s.length < 2) return null;
    final first = s.first.e1rm;
    final last = s.last.e1rm;
    final dKg = last - first;
    final pct = first > 0 ? (dKg / first) * 100 : 0.0;
    final days = s.last.date.difference(s.first.date).inDays;
    return (deltaKg: dKg, pct: pct, days: days);
  }
}
