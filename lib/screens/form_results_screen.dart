import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../models/form_models.dart';
import '../providers/form_session_provider.dart';

/// Renders a completed [FormReport] (D-05): per-rep KIE/KFE detections + a
/// plain-language session feedback line. No percentages, no quality gauge — the
/// backend produces binary detections + a confidence-derived severity word only.
class FormResultsScreen extends StatelessWidget {
  const FormResultsScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final provider = context.watch<FormSessionProvider>();
    final report = provider.report;

    if (report == null) {
      return const Scaffold(
        body: Center(child: CircularProgressIndicator()),
      );
    }

    return Scaffold(
      backgroundColor: const Color(0xFF0F0F1E),
      appBar: AppBar(
        backgroundColor: const Color(0xFF1A1A2E),
        foregroundColor: Colors.white,
        title: Text('${report.exerciseDisplay} — Results'),
        automaticallyImplyLeading: false,
      ),
      body: ListView(
        padding: const EdgeInsets.all(20),
        children: [
          _SummaryCard(report: report),
          const SizedBox(height: 20),

          _SectionHeader("Coach's Feedback"),
          const SizedBox(height: 10),
          _FeedbackCard(feedback: report.sessionFeedback),
          const SizedBox(height: 20),

          if (report.reps.isNotEmpty) ...[
            _SectionHeader('Per-Rep Breakdown'),
            const SizedBox(height: 10),
            ...report.reps.asMap().entries.map(
                  (entry) => _RepCard(index: entry.key + 1, rep: entry.value),
                ),
            const SizedBox(height: 20),
          ],

          ElevatedButton.icon(
            onPressed: () {
              provider.reset();
              Navigator.of(context).popUntil(
                (r) => r.isFirst || r.settings.name == '/plan',
              );
            },
            icon: const Icon(Icons.arrow_back),
            label: const Text('Done'),
            style: ElevatedButton.styleFrom(
              backgroundColor: const Color(0xFF38BDF8),
              foregroundColor: Colors.white,
              minimumSize: const Size.fromHeight(48),
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(12),
              ),
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

class _SummaryCard extends StatelessWidget {
  final FormReport report;
  const _SummaryCard({required this.report});

  @override
  Widget build(BuildContext context) {
    final kie = report.detectedCount('KIE');
    final kfe = report.detectedCount('KFE');
    final clean = kie == 0 && kfe == 0;

    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: const Color(0xFF1A1A2E),
        borderRadius: BorderRadius.circular(16),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(
                clean ? Icons.check_circle : Icons.warning_amber_rounded,
                color: clean ? Colors.green : Colors.amber,
                size: 28,
              ),
              const SizedBox(width: 10),
              Expanded(
                child: Text(
                  report.exerciseDisplay,
                  style: const TextStyle(
                    color: Colors.white,
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          _statRow(Icons.fact_check, '${report.totalReps} analyzed'),
          _statRow(
            Icons.compare_arrows,
            'Knees caving inward: ${kie > 0 ? "$kie flagged" : "none"}',
            color: kie > 0 ? Colors.redAccent : Colors.white70,
          ),
          _statRow(
            Icons.swap_vert,
            'Knees too far forward: ${kfe > 0 ? "$kfe flagged" : "none"}',
            color: kfe > 0 ? Colors.redAccent : Colors.white70,
          ),
        ],
      ),
    );
  }

  Widget _statRow(IconData icon, String text, {Color color = Colors.white70}) =>
      Padding(
        padding: const EdgeInsets.only(top: 6),
        child: Row(
          children: [
            Icon(icon, color: Colors.white54, size: 16),
            const SizedBox(width: 8),
            Expanded(
              child: Text(text, style: TextStyle(color: color, fontSize: 13)),
            ),
          ],
        ),
      );
}

class _RepCard extends StatelessWidget {
  final int index;
  final FormRep rep;
  const _RepCard({required this.index, required this.rep});

  @override
  Widget build(BuildContext context) {
    final detected = rep.detectedErrors;
    final clean = detected.isEmpty;
    final borderColor =
        clean ? Colors.green.withValues(alpha: 0.3) : Colors.red.withValues(alpha: 0.3);

    return Container(
      margin: const EdgeInsets.only(bottom: 10),
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
      decoration: BoxDecoration(
        color: const Color(0xFF1A1A2E),
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: borderColor),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Text(
                'Rep ${rep.repNumber ?? index}',
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 13,
                  fontWeight: FontWeight.w600,
                ),
              ),
              const Spacer(),
              if (clean)
                const Text(
                  'Clean',
                  style: TextStyle(
                    color: Colors.greenAccent,
                    fontSize: 12,
                    fontWeight: FontWeight.bold,
                  ),
                ),
            ],
          ),
          if (!clean) ...[
            const SizedBox(height: 8),
            Wrap(
              spacing: 6,
              runSpacing: 4,
              children: detected.map((e) => _ErrorChip(error: e)).toList(),
            ),
          ],
        ],
      ),
    );
  }
}

class _ErrorChip extends StatelessWidget {
  final FormError error;
  const _ErrorChip({required this.error});

  @override
  Widget build(BuildContext context) {
    final color = switch (error.severityWord) {
      'strong' => Colors.redAccent.shade400,
      'moderate' => Colors.orangeAccent.shade400,
      _ => Colors.amberAccent.shade400,
    };
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.15),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: color.withValues(alpha: 0.5)),
      ),
      child: Text(
        '${error.label} (${error.severityWord})',
        style: const TextStyle(color: Colors.white, fontSize: 11),
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
        border: Border.all(color: const Color(0xFF38BDF8).withValues(alpha: 0.4)),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Icon(Icons.fitness_center, color: Color(0xFF38BDF8), size: 20),
          const SizedBox(width: 10),
          Expanded(
            child: Text(
              feedback.isNotEmpty ? feedback : 'Great work! Keep it up.',
              style: const TextStyle(
                color: Colors.white70,
                fontSize: 14,
                height: 1.5,
              ),
            ),
          ),
        ],
      ),
    );
  }
}
