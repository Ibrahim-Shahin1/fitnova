import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../models/form_models.dart';
import '../providers/form_session_provider.dart';

class FormResultsScreen extends StatelessWidget {
  const FormResultsScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final provider = context.watch<FormSessionProvider>();
    final summary  = provider.summary;

    if (summary == null) {
      return const Scaffold(
        body: Center(child: CircularProgressIndicator()),
      );
    }

    return Scaffold(
      backgroundColor: const Color(0xFF0F0F1E),
      appBar: AppBar(
        backgroundColor: const Color(0xFF1A1A2E),
        foregroundColor: Colors.white,
        title: Text('${summary.exerciseDisplay} — Results'),
        automaticallyImplyLeading: false,
      ),
      body: ListView(
        padding: const EdgeInsets.all(20),
        children: [
          // ── Overall score ──────────────────────────────────────────────
          _ScoreCard(summary: summary),
          const SizedBox(height: 20),

          // ── Per-rep breakdown ──────────────────────────────────────────
          if (summary.perRepDetails.isNotEmpty) ...[
            _SectionHeader('Per-Rep Breakdown'),
            const SizedBox(height: 10),
            ...summary.perRepDetails.map((rep) => _RepDetailCard(rep: rep)),
            const SizedBox(height: 20),
          ] else if (summary.perRepScores.isNotEmpty) ...[
            // Legacy fallback: just a row of bars without per-rep error names
            _SectionHeader('Per-Rep Breakdown'),
            const SizedBox(height: 10),
            ...List.generate(summary.perRepScores.length, (i) {
              final score = summary.perRepScores[i];
              return _RepBar(repIndex: i + 1, score: score);
            }),
            const SizedBox(height: 20),
          ],

          // ── Common errors ──────────────────────────────────────────────
          if (summary.commonErrors.isNotEmpty) ...[
            _SectionHeader('Areas to Improve'),
            const SizedBox(height: 10),
            ...summary.commonErrors.entries.map(
              (e) => _ErrorChip(joint: e.key, repCount: e.value),
            ),
            const SizedBox(height: 20),
          ],

          // ── LLM feedback ───────────────────────────────────────────────
          _SectionHeader("Coach's Feedback"),
          const SizedBox(height: 10),
          _FeedbackCard(feedback: summary.llmFeedback),
          const SizedBox(height: 32),

          // ── Back button ────────────────────────────────────────────────
          ElevatedButton.icon(
            onPressed: () {
              provider.reset();
              Navigator.of(context).popUntil((r) => r.isFirst ||
                  r.settings.name == '/plan');
            },
            icon: const Icon(Icons.arrow_back),
            label: const Text('Back to Plan'),
            style: ElevatedButton.styleFrom(
              backgroundColor: const Color(0xFF38BDF8),
              foregroundColor: Colors.white,
              minimumSize: const Size.fromHeight(48),
              shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(12)),
            ),
          ),
        ],
      ),
    );
  }
}

// ── Sub-widgets ────────────────────────────────────────────────────────────────

class _SectionHeader extends StatelessWidget {
  final String text;
  const _SectionHeader(this.text);

  @override
  Widget build(BuildContext context) => Text(
        text,
        style: const TextStyle(
          color: Colors.white70,
          fontSize: 14,
          fontWeight: FontWeight.w600,
          letterSpacing: 0.5,
        ),
      );
}

class _ScoreCard extends StatelessWidget {
  final FormSessionSummary summary;
  const _ScoreCard({required this.summary});

  Color get _scoreColor {
    if (summary.averageQuality >= 0.7) return Colors.green;
    if (summary.averageQuality >= 0.4) return Colors.amber;
    return Colors.red;
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: const Color(0xFF1A1A2E),
        borderRadius: BorderRadius.circular(16),
      ),
      child: Row(
        children: [
          // Circular score gauge
          SizedBox(
            width: 90,
            height: 90,
            child: Stack(
              alignment: Alignment.center,
              children: [
                CircularProgressIndicator(
                  value: summary.averageQuality,
                  strokeWidth: 8,
                  backgroundColor: Colors.white12,
                  valueColor: AlwaysStoppedAnimation<Color>(_scoreColor),
                ),
                Text(
                  summary.qualityPercent,
                  style: TextStyle(
                    color: _scoreColor,
                    fontWeight: FontWeight.bold,
                    fontSize: 18,
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(width: 20),
          // Stats column
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  summary.exerciseDisplay,
                  style: const TextStyle(
                      color: Colors.white,
                      fontSize: 17,
                      fontWeight: FontWeight.bold),
                ),
                const SizedBox(height: 6),
                _statRow(Icons.repeat,         'Reps: ${summary.totalReps}'),
                _statRow(Icons.timer,           'Duration: ${summary.durationSeconds}s'),
                _statRow(Icons.trending_up,     'Trend: ${summary.qualityTrend}'),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _statRow(IconData icon, String text) => Padding(
        padding: const EdgeInsets.only(top: 4),
        child: Row(
          children: [
            Icon(icon, color: Colors.white54, size: 14),
            const SizedBox(width: 5),
            Text(text, style: const TextStyle(color: Colors.white70, fontSize: 13)),
          ],
        ),
      );
}

class _RepBar extends StatelessWidget {
  final int repIndex;
  final double score;
  const _RepBar({required this.repIndex, required this.score});

  Color get _color {
    if (score >= 0.7) return Colors.green;
    if (score >= 0.4) return Colors.amber;
    return Colors.red;
  }

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: Row(
        children: [
          SizedBox(
            width: 50,
            child: Text('Rep $repIndex',
                style: const TextStyle(color: Colors.white70, fontSize: 12)),
          ),
          Expanded(
            child: ClipRRect(
              borderRadius: BorderRadius.circular(4),
              child: LinearProgressIndicator(
                value: score,
                backgroundColor: Colors.white12,
                valueColor: AlwaysStoppedAnimation<Color>(_color),
                minHeight: 10,
              ),
            ),
          ),
          const SizedBox(width: 8),
          SizedBox(
            width: 38,
            child: Text(
              '${(score * 100).round()}%',
              style: TextStyle(color: _color, fontSize: 12, fontWeight: FontWeight.bold),
              textAlign: TextAlign.right,
            ),
          ),
        ],
      ),
    );
  }
}

class _RepDetailCard extends StatelessWidget {
  final RepResult rep;
  const _RepDetailCard({required this.rep});

  Color get _color {
    if (rep.quality >= 0.7) return Colors.green;
    if (rep.quality >= 0.4) return Colors.amber;
    return Colors.red;
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.only(bottom: 10),
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
      decoration: BoxDecoration(
        color: const Color(0xFF1A1A2E),
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: _color.withValues(alpha: 0.25)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Top row — rep number + quality bar + percentage
          Row(
            children: [
              SizedBox(
                width: 50,
                child: Text(
                  'Rep ${rep.repIdx}',
                  style: const TextStyle(
                    color: Colors.white,
                    fontSize: 13,
                    fontWeight: FontWeight.w600,
                  ),
                ),
              ),
              Expanded(
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(4),
                  child: LinearProgressIndicator(
                    value: rep.quality,
                    backgroundColor: Colors.white12,
                    valueColor: AlwaysStoppedAnimation<Color>(_color),
                    minHeight: 8,
                  ),
                ),
              ),
              const SizedBox(width: 8),
              SizedBox(
                width: 38,
                child: Text(
                  rep.qualityPercent,
                  style: TextStyle(
                    color: _color,
                    fontSize: 12,
                    fontWeight: FontWeight.bold,
                  ),
                  textAlign: TextAlign.right,
                ),
              ),
            ],
          ),
          // Top errors row
          if (rep.topErrors.isNotEmpty) ...[
            const SizedBox(height: 6),
            Wrap(
              spacing: 6,
              runSpacing: 4,
              children: rep.topErrors.map((e) {
                return Container(
                  padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                  decoration: BoxDecoration(
                    color: Colors.red.withValues(alpha: 0.12),
                    borderRadius: BorderRadius.circular(6),
                    border: Border.all(color: Colors.red.withValues(alpha: 0.3)),
                  ),
                  child: Text(
                    '${e.name} ${(e.value * 100).round()}%',
                    style: const TextStyle(
                      color: Colors.white70,
                      fontSize: 11,
                    ),
                  ),
                );
              }).toList(),
            ),
          ],
        ],
      ),
    );
  }
}


class _ErrorChip extends StatelessWidget {
  final String joint;
  final int repCount;
  const _ErrorChip({required this.joint, required this.repCount});

  @override
  Widget build(BuildContext context) {
    final color = repCount >= 3 ? Colors.red : Colors.amber;
    return Container(
      margin: const EdgeInsets.only(bottom: 8),
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
      decoration: BoxDecoration(
        color: color.withValues(alpha:0.12),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: color.withValues(alpha:0.4)),
      ),
      child: Row(
        children: [
          Icon(Icons.warning_amber_rounded, color: color, size: 16),
          const SizedBox(width: 8),
          Expanded(
            child: Text(joint,
                style: const TextStyle(color: Colors.white, fontSize: 13)),
          ),
          Text(
            '$repCount rep${repCount > 1 ? "s" : ""}',
            style: TextStyle(color: color, fontSize: 12, fontWeight: FontWeight.bold),
          ),
        ],
      ),
    );
  }
}

class _FeedbackCard extends StatelessWidget {
  final String feedback;
  const _FeedbackCard({required this.feedback});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFF1A1A2E),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: const Color(0xFF38BDF8).withValues(alpha:0.4)),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Icon(Icons.fitness_center,
              color: Color(0xFF38BDF8), size: 20),
          const SizedBox(width: 10),
          Expanded(
            child: Text(
              feedback.isNotEmpty ? feedback : 'Great work! Keep it up.',
              style: const TextStyle(
                  color: Colors.white70, fontSize: 14, height: 1.5),
            ),
          ),
        ],
      ),
    );
  }
}
