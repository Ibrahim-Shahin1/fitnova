import 'dart:io';
import 'package:flutter/material.dart';
import 'package:video_player/video_player.dart';

import '../models/form_models.dart';
import '../widgets/skeleton_painter.dart';

/// Replays an uploaded form-check video with the live skeleton + joint
/// errors overlaid, plus a scrubber that marks every detected rep.
///
/// Args (passed through `RouteSettings.arguments` as a map):
///   - `videoPath`: absolute path to the MP4 the user picked from gallery.
///   - `summary`:  the FormSessionSummary returned by `/analyze-form-video`,
///                 containing the per-frame `timeline`.
class FormReplayScreen extends StatefulWidget {
  final String videoPath;
  final FormSessionSummary summary;

  const FormReplayScreen({
    super.key,
    required this.videoPath,
    required this.summary,
  });

  @override
  State<FormReplayScreen> createState() => _FormReplayScreenState();
}

class _FormReplayScreenState extends State<FormReplayScreen> {
  late VideoPlayerController _video;
  bool _ready = false;

  // Index into summary.timeline of the frame closest to the current
  // playback position. Recomputed on every position change.
  int _frameIdx = 0;

  // The list of (timestamp_ms, rep_count_so_far) jumps. Computed once.
  // Each entry is a "rep boundary" — the moment rep_count incremented.
  late final List<int> _repBoundariesMs;

  @override
  void initState() {
    super.initState();
    _repBoundariesMs = _computeRepBoundaries(widget.summary.timeline);
    _video = VideoPlayerController.file(File(widget.videoPath))
      ..initialize().then((_) {
        if (!mounted) return;
        setState(() => _ready = true);
        _video.addListener(_onVideoTick);
      });
  }

  @override
  void dispose() {
    _video.removeListener(_onVideoTick);
    _video.dispose();
    super.dispose();
  }

  /// Find the timeline indices where rep_count jumped up.
  static List<int> _computeRepBoundaries(List<FormFrameTimeline> timeline) {
    final out = <int>[];
    int prev = 0;
    for (final f in timeline) {
      if (f.repCount > prev) {
        out.add(f.timestampMs);
        prev = f.repCount;
      }
    }
    return out;
  }

  void _onVideoTick() {
    if (!_video.value.isInitialized) return;
    final posMs = _video.value.position.inMilliseconds;
    // Binary-search for the closest timeline frame.
    final tl = widget.summary.timeline;
    if (tl.isEmpty) return;
    int lo = 0, hi = tl.length - 1;
    while (lo < hi) {
      final mid = (lo + hi) >> 1;
      if (tl[mid].timestampMs < posMs) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }
    // Pick the closer of `lo` and `lo-1`.
    int idx = lo;
    if (idx > 0) {
      final dPrev = (tl[idx - 1].timestampMs - posMs).abs();
      final dThis = (tl[idx].timestampMs - posMs).abs();
      if (dPrev < dThis) idx = idx - 1;
    }
    if (idx != _frameIdx && mounted) {
      setState(() => _frameIdx = idx);
    }
  }

  void _seekToMs(int ms) {
    _video.seekTo(Duration(milliseconds: ms));
  }

  @override
  Widget build(BuildContext context) {
    final summary = widget.summary;
    return Scaffold(
      backgroundColor: const Color(0xFF0F0F1E),
      appBar: AppBar(
        backgroundColor: const Color(0xFF1A1A2E),
        foregroundColor: Colors.white,
        title: Text('${summary.exerciseDisplay} — Replay'),
      ),
      body: !_ready
          ? const Center(
              child: CircularProgressIndicator(
                valueColor: AlwaysStoppedAnimation<Color>(Color(0xFF38BDF8)),
              ),
            )
          : LayoutBuilder(builder: (context, constraints) {
              // Cap the video at 55% of available height so portrait phone
              // videos (9:16 aspect) don't push the rest of the UI off-screen
              // and trigger Flutter's overflow indicator (yellow/black stripes).
              // Letterbox if the natural aspect doesn't fit.
              final maxVideoHeight = constraints.maxHeight * 0.55;
              return Column(
                children: [
                  // ── Video + skeleton overlay ─────────────────────────────
                  ConstrainedBox(
                    constraints: BoxConstraints(maxHeight: maxVideoHeight),
                    child: _buildVideoArea(),
                  ),
                  // ── Playback controls + rep scrubber ──────────────────────
                  _buildScrubber(),
                  // ── Live current-frame stats ──────────────────────────────
                  _buildLiveStats(),
                  // ── Rep cards (rest of available space) ───────────────────
                  Expanded(child: _buildRepList()),
                ],
              );
            }),
    );
  }

  Widget _buildVideoArea() {
    final tl = widget.summary.timeline;
    final cur = (tl.isEmpty || _frameIdx >= tl.length) ? null : tl[_frameIdx];
    final jointErrorMap =
        cur == null ? <double>[] : buildJointErrorMap(cur.jointErrors);

    // Center inside the constrained box and letterbox using AspectRatio +
    // FittedBox.contain. Tall portrait videos will be height-capped (no overflow);
    // wide landscape videos will fill the width and leave horizontal padding.
    return Center(
      child: AspectRatio(
        aspectRatio: _video.value.aspectRatio,
        child: GestureDetector(
          onTap: () {
            if (_video.value.isPlaying) {
              _video.pause();
            } else {
              _video.play();
            }
            setState(() {});
          },
          child: Stack(
            fit: StackFit.expand,
            children: [
              VideoPlayer(_video),
              if (cur != null && cur.landmarks != null)
                CustomPaint(
                  painter: SkeletonPainter(
                    landmarks: cur.landmarks,
                    jointErrors: jointErrorMap,
                    // Uploaded videos are NOT mirrored — we draw straight.
                    mirror: false,
                  ),
                ),
              // Live rep counter overlay (top-center on the video area).
              // Mirrors the live form-check screen's rep counter pill.
              Positioned(
                top: 12,
                left: 0,
                right: 0,
                child: Center(
                  child: _ReplayRepCounterPill(
                    count: cur?.repCount ?? 0,
                  ),
                ),
              ),
              // Active-flag chip overlay (top-left of video). Shows the
              // most severe rule firing at the current playback frame.
              if (cur != null && cur.activeFlags.isNotEmpty)
                Positioned(
                  top: 12,
                  left: 12,
                  child: _ReplayFlagChip(
                    flag: cur.activeFlags.reduce(
                        (a, b) => a.severity >= b.severity ? a : b),
                  ),
                ),
              // Play/pause indicator
              if (!_video.value.isPlaying)
                const Center(
                  child: Icon(Icons.play_circle_fill,
                      color: Colors.white70, size: 64),
                ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildScrubber() {
    final dur = _video.value.duration;
    final pos = _video.value.position;
    final total = dur.inMilliseconds.clamp(1, 1 << 31);
    final fraction = pos.inMilliseconds / total;

    return Container(
      color: const Color(0xFF1A1A2E),
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      child: Column(
        children: [
          // Custom progress bar that draws rep boundary tick marks.
          SizedBox(
            height: 22,
            child: LayoutBuilder(builder: (ctx, c) {
              final width = c.maxWidth;
              return Stack(
                children: [
                  // Background track
                  Positioned.fill(
                    child: Padding(
                      padding: const EdgeInsets.symmetric(vertical: 8),
                      child: ClipRRect(
                        borderRadius: BorderRadius.circular(2),
                        child: LinearProgressIndicator(
                          value: fraction.clamp(0.0, 1.0),
                          backgroundColor: Colors.white12,
                          valueColor: const AlwaysStoppedAnimation<Color>(
                              Color(0xFF38BDF8)),
                          minHeight: 6,
                        ),
                      ),
                    ),
                  ),
                  // Rep tick marks (orange)
                  for (final ms in _repBoundariesMs)
                    Positioned(
                      left: (ms / total) * width - 1,
                      top: 0,
                      bottom: 0,
                      child: GestureDetector(
                        onTap: () => _seekToMs(ms),
                        child: Container(
                          width: 2,
                          color: Colors.orangeAccent,
                        ),
                      ),
                    ),
                  // Tap-to-seek invisible layer
                  Positioned.fill(
                    child: GestureDetector(
                      behavior: HitTestBehavior.translucent,
                      onTapDown: (d) {
                        final f = (d.localPosition.dx / width).clamp(0.0, 1.0);
                        _seekToMs((f * total).round());
                      },
                    ),
                  ),
                ],
              );
            }),
          ),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              IconButton(
                icon: Icon(
                  _video.value.isPlaying ? Icons.pause : Icons.play_arrow,
                  color: Colors.white,
                ),
                onPressed: () {
                  setState(() {
                    if (_video.value.isPlaying) {
                      _video.pause();
                    } else {
                      _video.play();
                    }
                  });
                },
              ),
              Text(
                '${_fmt(pos)}  /  ${_fmt(dur)}',
                style: const TextStyle(color: Colors.white70, fontSize: 12),
              ),
              Text(
                'Reps: ${widget.summary.totalReps}',
                style: const TextStyle(color: Colors.white70, fontSize: 12),
              ),
            ],
          ),
        ],
      ),
    );
  }

  String _fmt(Duration d) {
    final m = d.inMinutes.remainder(60).toString().padLeft(2, '0');
    final s = d.inSeconds.remainder(60).toString().padLeft(2, '0');
    return '$m:$s';
  }

  Widget _buildLiveStats() {
    final tl = widget.summary.timeline;
    if (tl.isEmpty || _frameIdx >= tl.length) {
      return const SizedBox.shrink();
    }
    final cur = tl[_frameIdx];
    final q = cur.qualityScore;
    final qColor = q >= 0.7
        ? Colors.green
        : q >= 0.4
            ? Colors.amber
            : Colors.red;
    // Highlight any joint group above 0.4
    final lit = <_LitJoint>[];
    const groupNames = [
      'L Elbow', 'R Elbow', 'L Shoulder', 'R Shoulder',
      'L Knee', 'R Knee', 'L Hip', 'R Hip',
      'Trunk', 'Neck',
    ];
    for (var i = 0; i < cur.jointErrors.length && i < groupNames.length; i++) {
      if (cur.jointErrors[i] > 0.4) {
        lit.add(_LitJoint(groupNames[i], cur.jointErrors[i]));
      }
    }
    return Container(
      color: const Color(0xFF1A1A2E),
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      child: Row(
        children: [
          // Quality badge
          Container(
            padding:
                const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
            decoration: BoxDecoration(
              color: qColor.withValues(alpha: 0.2),
              borderRadius: BorderRadius.circular(8),
              border: Border.all(color: qColor, width: 1),
            ),
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                Icon(Icons.assessment, color: qColor, size: 14),
                const SizedBox(width: 4),
                Text('${(q * 100).round()}%',
                    style: TextStyle(
                      color: qColor,
                      fontWeight: FontWeight.bold,
                      fontSize: 12,
                    )),
              ],
            ),
          ),
          const SizedBox(width: 8),
          // Lit joint chips
          Expanded(
            child: SingleChildScrollView(
              scrollDirection: Axis.horizontal,
              child: Row(
                children: lit
                    .map((j) => Padding(
                          padding: const EdgeInsets.only(right: 6),
                          child: Container(
                            padding: const EdgeInsets.symmetric(
                                horizontal: 8, vertical: 3),
                            decoration: BoxDecoration(
                              color: Colors.red.withValues(alpha: 0.15),
                              borderRadius: BorderRadius.circular(6),
                              border: Border.all(
                                  color: Colors.red.withValues(alpha: 0.4)),
                            ),
                            child: Text(
                              '${j.name} ${(j.value * 100).round()}%',
                              style: const TextStyle(
                                color: Colors.white70,
                                fontSize: 11,
                              ),
                            ),
                          ),
                        ))
                    .toList(),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildRepList() {
    final reps = widget.summary.perRepDetails;
    if (reps.isEmpty) {
      return const Center(
        child: Padding(
          padding: EdgeInsets.all(24),
          child: Text(
            'No reps detected in this video.',
            style: TextStyle(color: Colors.white60),
          ),
        ),
      );
    }
    return ListView.builder(
      padding: const EdgeInsets.all(12),
      itemCount: reps.length,
      itemBuilder: (_, i) {
        final rep = reps[i];
        final color = rep.quality >= 0.7
            ? Colors.green
            : rep.quality >= 0.4
                ? Colors.amber
                : Colors.red;
        // Rep boundary timestamp (if available)
        final repMs = i < _repBoundariesMs.length ? _repBoundariesMs[i] : null;
        return Padding(
          padding: const EdgeInsets.only(bottom: 10),
          child: InkWell(
            onTap: repMs == null ? null : () => _seekToMs(repMs),
            borderRadius: BorderRadius.circular(10),
            child: Container(
              padding: const EdgeInsets.symmetric(
                  horizontal: 12, vertical: 10),
              decoration: BoxDecoration(
                color: const Color(0xFF1A1A2E),
                borderRadius: BorderRadius.circular(10),
                border: Border.all(
                    color: color.withValues(alpha: 0.25), width: 1),
              ),
              child: Row(
                children: [
                  Text('Rep ${rep.repIdx}',
                      style: const TextStyle(
                        color: Colors.white,
                        fontSize: 13,
                        fontWeight: FontWeight.w600,
                      )),
                  const SizedBox(width: 12),
                  Expanded(
                    child: ClipRRect(
                      borderRadius: BorderRadius.circular(4),
                      child: LinearProgressIndicator(
                        value: rep.quality,
                        backgroundColor: Colors.white12,
                        valueColor: AlwaysStoppedAnimation<Color>(color),
                        minHeight: 8,
                      ),
                    ),
                  ),
                  const SizedBox(width: 10),
                  SizedBox(
                    width: 38,
                    child: Text(rep.qualityPercent,
                        textAlign: TextAlign.right,
                        style: TextStyle(
                          color: color,
                          fontSize: 12,
                          fontWeight: FontWeight.bold,
                        )),
                  ),
                  if (repMs != null) ...[
                    const SizedBox(width: 8),
                    const Icon(Icons.play_arrow,
                        color: Colors.white54, size: 18),
                  ],
                ],
              ),
            ),
          ),
        );
      },
    );
  }
}

class _LitJoint {
  final String name;
  final double value;
  const _LitJoint(this.name, this.value);
}


/// Live rep counter pill shown on top-center of the replay video. Mirrors
/// the live form-check screen so the user gets consistent UX between live
/// session and uploaded video replay.
class _ReplayRepCounterPill extends StatelessWidget {
  final int count;
  const _ReplayRepCounterPill({required this.count});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 6),
      decoration: BoxDecoration(
        color: Colors.black.withValues(alpha: 0.65),
        borderRadius: BorderRadius.circular(24),
        border: Border.all(
            color: Colors.white.withValues(alpha: 0.3), width: 1),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.end,
        children: [
          Text(
            '$count',
            style: const TextStyle(
              color: Colors.white,
              fontSize: 30,
              fontWeight: FontWeight.w900,
              fontFeatures: [FontFeature.tabularFigures()],
              height: 1.0,
            ),
          ),
          const SizedBox(width: 6),
          const Padding(
            padding: EdgeInsets.only(bottom: 3),
            child: Text(
              'REPS',
              style: TextStyle(
                color: Colors.white70,
                fontSize: 11,
                fontWeight: FontWeight.w700,
                letterSpacing: 1.5,
              ),
            ),
          ),
        ],
      ),
    );
  }
}


/// Active-flag chip overlay used on top-left of the replay video frame.
class _ReplayFlagChip extends StatelessWidget {
  final ActiveFlag flag;
  const _ReplayFlagChip({required this.flag});

  @override
  Widget build(BuildContext context) {
    final color = flag.severity >= 0.85
        ? Colors.redAccent.shade400
        : flag.severity >= 0.55
            ? Colors.orangeAccent.shade400
            : Colors.amberAccent.shade400;
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.22),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: color, width: 1.4),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(Icons.warning_amber_rounded, color: color, size: 12),
          const SizedBox(width: 4),
          Text(
            flag.label,
            style: TextStyle(
              color: color,
              fontSize: 11,
              fontWeight: FontWeight.bold,
            ),
          ),
        ],
      ),
    );
  }
}
