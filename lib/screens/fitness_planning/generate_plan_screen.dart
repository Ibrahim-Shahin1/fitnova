import 'dart:async';

import 'package:flutter/material.dart';

import '../../models/active_plan.dart';
import '../../services/plan_service.dart';
import '../../services/plan_stream_service.dart';
import '../../theme/app_spacing.dart';
import '../../widgets/plan/plan_schedule.dart';
import '../../widgets/ui/app_button.dart';
import '../../widgets/ui/app_card.dart';

/// Full-screen live view of the 4-agent crew building the plan. Streams agent
/// progress (Profiler → Generator → Critic → Optimizer, with per-round critic
/// scores), then shows the validated final plan. Pops `true` on success.
class GeneratePlanScreen extends StatefulWidget {
  const GeneratePlanScreen({super.key, required this.params});

  /// Override params captured by the coach (may be empty → use the profile).
  final Map<String, dynamic> params;

  @override
  State<GeneratePlanScreen> createState() => _GeneratePlanScreenState();
}

class _GeneratePlanScreenState extends State<GeneratePlanScreen> {
  static const _agents = ['Profiler', 'Generator', 'Critic', 'Optimizer'];
  static const _icons = {
    'Profiler': Icons.person_search,
    'Generator': Icons.auto_awesome,
    'Critic': Icons.fact_check,
    'Optimizer': Icons.tune,
  };
  static const _blurb = {
    'Profiler': 'Reading your goal, frequency, injuries & equipment',
    'Generator': 'Drafting the split from the recommended program',
    'Critic': 'Checking each exercise lands on the right day',
    'Optimizer': 'Fixing flaws and polishing the schedule',
  };

  final Map<String, String> _status = {for (final a in _agents) a: 'pending'};
  final List<String> _log = [];
  StreamSubscription<Map<String, dynamic>>? _sub;
  bool _done = false;
  bool _error = false;
  String? _errMsg;
  int? _score;
  ActivePlan? _finalPlan;

  @override
  void initState() {
    super.initState();
    _sub = PlanStreamService.generate(widget.params).listen(
      _onEvent,
      onError: (e) =>
          mounted ? setState(() => _fail('$e')) : null,
    );
  }

  @override
  void dispose() {
    _sub?.cancel();
    super.dispose();
  }

  void _fail(String? msg) {
    _error = true;
    _errMsg = msg ?? 'Something went wrong building your plan.';
  }

  Future<void> _onEvent(Map<String, dynamic> ev) async {
    final type = ev['event'];
    if (type == 'agent') {
      final name = ev['name'] as String?;
      final status = ev['status'] as String?;
      if (name != null && _status.containsKey(name)) {
        setState(() => _status[name] = status == 'done' ? 'done' : 'running');
      }
      if (name == 'Profiler' && status == 'done') {
        final split = (ev['detail'] as Map?)?['split'];
        _push(split is List
            ? 'Split chosen: ${split.join(' / ')}'
            : 'Training split chosen');
      } else if (name == 'Generator' && status == 'done') {
        _push('Generator drafted the split from the program');
      } else if (name == 'Critic' && status == 'done') {
        final r = ev['round'];
        final sc = ev['score'];
        final hv = (ev['hard_violations'] as int?) ?? 0;
        final issues = (ev['issues'] as List?)?.cast<String>() ?? const [];
        if (sc is int) setState(() => _score = sc);
        final when = r == 0 ? 'reviewed the draft' : 'reviewed round $r';
        _push(hv > 0
            ? 'Critic $when — $sc/10 · $hv constraint issue${hv == 1 ? '' : 's'} to fix:'
            : 'Critic $when — $sc/10 · all constraints clean ✓');
        for (final iss in issues) {
          _push('• $iss');
        }
      } else if (name == 'Optimizer' && status == 'done') {
        _push('Optimizer round ${ev['round']} — moved/swapped exercises to fix them');
      }
    } else if (type == 'done') {
      final q = ev['quality_report'] as Map?;
      if (q?['score'] is int) setState(() => _score = q!['score'] as int);
      _push('Final validation — all constraints satisfied'
          '${_score != null ? ' ($_score/10)' : ''}');
      await _loadFinal();
    } else if (type == 'error') {
      setState(() => _fail(ev['error']?.toString()));
    }
  }

  void _push(String line) {
    if (mounted) setState(() => _log.add(line));
  }

  Future<void> _loadFinal() async {
    try {
      final plan = await PlanService.fetchActive();
      if (!mounted) return;
      setState(() {
        _finalPlan = plan;
        _done = true;
        for (final a in _agents) {
          _status[a] = 'done';
        }
      });
    } catch (_) {
      if (mounted) setState(() => _done = true);
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Scaffold(
      appBar: AppBar(
        title: Text(_done ? 'Your plan is ready' : 'Building your plan'),
        automaticallyImplyLeading: _done || _error,
      ),
      body: _error
          ? _ErrorView(message: _errMsg, onClose: () => Navigator.of(context).pop(false))
          : _done
              ? _SuccessView(plan: _finalPlan, score: _score)
              : _ProgressView(
                  agents: _agents,
                  icons: _icons,
                  blurb: _blurb,
                  status: _status,
                  log: _log,
                  score: _score,
                  theme: theme,
                ),
    );
  }
}

class _ProgressView extends StatelessWidget {
  const _ProgressView({
    required this.agents,
    required this.icons,
    required this.blurb,
    required this.status,
    required this.log,
    required this.score,
    required this.theme,
  });

  final List<String> agents;
  final Map<String, IconData> icons;
  final Map<String, String> blurb;
  final Map<String, String> status;
  final List<String> log;
  final int? score;
  final ThemeData theme;

  @override
  Widget build(BuildContext context) {
    final cs = theme.colorScheme;
    return ListView(
      padding: const EdgeInsets.all(AppSpacing.lg),
      children: [
        Text('The coaching crew is building your plan',
            style: theme.textTheme.titleMedium
                ?.copyWith(fontWeight: FontWeight.w700)),
        const SizedBox(height: AppSpacing.xs),
        Text('Four specialists collaborate — and the Critic keeps scoring it '
            'until it passes.',
            style: theme.textTheme.bodySmall
                ?.copyWith(color: cs.onSurfaceVariant)),
        const SizedBox(height: AppSpacing.lg),
        for (final a in agents)
          _AgentRow(
            name: a,
            icon: icons[a]!,
            blurb: blurb[a]!,
            state: status[a]!,
          ),
        const SizedBox(height: AppSpacing.md),
        if (log.isNotEmpty)
          AppCard(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Icon(Icons.timeline, size: 18, color: cs.primary),
                    const SizedBox(width: AppSpacing.sm),
                    Text('Live progress',
                        style: theme.textTheme.labelLarge
                            ?.copyWith(fontWeight: FontWeight.w700)),
                    const Spacer(),
                    if (score != null)
                      Text('latest $score/10',
                          style: theme.textTheme.labelMedium?.copyWith(
                              color: cs.primary, fontWeight: FontWeight.w800)),
                  ],
                ),
                const SizedBox(height: AppSpacing.sm),
                for (final line in log)
                  Builder(builder: (_) {
                    final isIssue = line.startsWith('• ');
                    return Padding(
                      padding: EdgeInsets.only(
                          bottom: 4, left: isIssue ? AppSpacing.lg : 0),
                      child: Row(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Icon(isIssue ? Icons.close : Icons.check,
                              size: 14,
                              color: isIssue ? cs.error : Colors.green),
                          const SizedBox(width: AppSpacing.sm),
                          Expanded(
                            child: Text(isIssue ? line.substring(2) : line,
                                style: theme.textTheme.bodySmall?.copyWith(
                                    color: isIssue
                                        ? cs.error
                                        : cs.onSurfaceVariant)),
                          ),
                        ],
                      ),
                    );
                  }),
              ],
            ),
          ),
        const SizedBox(height: AppSpacing.lg),
        Center(
          child: Text('This usually takes about a minute…',
              style: theme.textTheme.bodySmall
                  ?.copyWith(color: cs.onSurfaceVariant)),
        ),
      ],
    );
  }
}

class _AgentRow extends StatelessWidget {
  const _AgentRow({
    required this.name,
    required this.icon,
    required this.blurb,
    required this.state,
  });

  final String name;
  final IconData icon;
  final String blurb;
  final String state; // pending | running | done

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final cs = theme.colorScheme;
    final active = state == 'running';
    final done = state == 'done';
    final accent = done
        ? Colors.green
        : active
            ? cs.primary
            : cs.onSurfaceVariant;

    Widget trailing;
    if (done) {
      trailing = const Icon(Icons.check_circle, color: Colors.green, size: 22);
    } else if (active) {
      trailing = SizedBox(
        width: 18,
        height: 18,
        child: CircularProgressIndicator(strokeWidth: 2, color: cs.primary),
      );
    } else {
      trailing =
          Icon(Icons.radio_button_unchecked, color: cs.outlineVariant, size: 20);
    }

    return Opacity(
      opacity: state == 'pending' ? 0.55 : 1,
      child: Padding(
        padding: const EdgeInsets.symmetric(vertical: AppSpacing.xs),
        child: Row(
          children: [
            CircleAvatar(
              radius: 18,
              backgroundColor: accent.withValues(alpha: 0.14),
              child: Icon(icon, size: 18, color: accent),
            ),
            const SizedBox(width: AppSpacing.md),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(name,
                      style: theme.textTheme.titleSmall?.copyWith(
                          fontWeight: FontWeight.w700,
                          color: active ? cs.primary : null)),
                  Text(blurb,
                      style: theme.textTheme.bodySmall
                          ?.copyWith(color: cs.onSurfaceVariant)),
                ],
              ),
            ),
            const SizedBox(width: AppSpacing.sm),
            trailing,
          ],
        ),
      ),
    );
  }
}

class _SuccessView extends StatelessWidget {
  const _SuccessView({required this.plan, required this.score});

  final ActivePlan? plan;
  final int? score;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final cs = theme.colorScheme;
    return Column(
      children: [
        Expanded(
          child: ListView(
            padding: const EdgeInsets.all(AppSpacing.lg),
            children: [
              Row(
                children: [
                  const Icon(Icons.verified, color: Colors.green, size: 28),
                  const SizedBox(width: AppSpacing.sm),
                  Expanded(
                    child: Text(
                      'Plan validated by the crew'
                      '${score != null ? ' — $score/10' : ''}',
                      style: theme.textTheme.titleMedium
                          ?.copyWith(fontWeight: FontWeight.w800),
                    ),
                  ),
                ],
              ),
              const SizedBox(height: AppSpacing.lg),
              if (plan != null)
                PlanScheduleView(plan: plan!)
              else
                Text('Your plan is saved — open the Planning tab to see it.',
                    style: theme.textTheme.bodyMedium
                        ?.copyWith(color: cs.onSurfaceVariant)),
              const SizedBox(height: AppSpacing.lg),
            ],
          ),
        ),
        SafeArea(
          top: false,
          child: Padding(
            padding: const EdgeInsets.fromLTRB(
                AppSpacing.lg, 0, AppSpacing.lg, AppSpacing.md),
            child: AppButton(
              label: 'Done',
              icon: Icons.check,
              expand: true,
              onPressed: () => Navigator.of(context).pop(true),
            ),
          ),
        ),
      ],
    );
  }
}

class _ErrorView extends StatelessWidget {
  const _ErrorView({required this.message, required this.onClose});

  final String? message;
  final VoidCallback onClose;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final cs = theme.colorScheme;
    return Padding(
      padding: const EdgeInsets.all(AppSpacing.lg),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(Icons.error_outline, size: 56, color: cs.error),
          const SizedBox(height: AppSpacing.md),
          Text("Couldn't build the plan",
              style: theme.textTheme.titleMedium
                  ?.copyWith(fontWeight: FontWeight.w700)),
          const SizedBox(height: AppSpacing.sm),
          Text(message ?? 'Please try again.',
              textAlign: TextAlign.center,
              style: theme.textTheme.bodyMedium
                  ?.copyWith(color: cs.onSurfaceVariant)),
          const SizedBox(height: AppSpacing.lg),
          AppButton(label: 'Close', expand: true, onPressed: onClose),
        ],
      ),
    );
  }
}
