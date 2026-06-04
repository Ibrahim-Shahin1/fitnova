import 'dart:async';

import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../../models/meal_plan.dart';
import '../../providers/nutrition_provider.dart';
import '../../services/nutrition_service.dart';
import '../../theme/app_spacing.dart';
import '../../widgets/ui/app_button.dart';
import '../../widgets/ui/app_card.dart';
import '../../widgets/nutrition/meal_plan_view.dart';

/// Live view of the 4-agent nutrition crew building the meal plan. Streams agent
/// progress (Dietitian → Composer → Critic → Coach) via SSE, then reveals the
/// validated plan. Mirrors the fitness GeneratePlanScreen. Pops `true` on success.
class GenerateMealPlanScreen extends StatefulWidget {
  const GenerateMealPlanScreen({super.key, required this.params});

  final Map<String, dynamic> params;

  @override
  State<GenerateMealPlanScreen> createState() => _GenerateMealPlanScreenState();
}

class _GenerateMealPlanScreenState extends State<GenerateMealPlanScreen> {
  static const _agents = ['Dietitian', 'Composer', 'Critic', 'Coach'];
  static const _icons = {
    'Dietitian': Icons.medical_information,
    'Composer': Icons.restaurant_menu,
    'Critic': Icons.fact_check,
    'Coach': Icons.tips_and_updates,
  };
  static const _blurb = {
    'Dietitian': 'Explaining your calorie & macro targets',
    'Composer': 'Naming the plan & each day from its recipes',
    'Critic': 'Checking every day hits your targets',
    'Coach': 'Writing prep tips & swap suggestions',
  };

  final Map<String, String> _status = {for (final a in _agents) a: 'pending'};
  final List<String> _log = [];
  StreamSubscription<Map<String, dynamic>>? _sub;
  bool _finished = false;
  bool _proceeded = false;
  bool _error = false;
  String? _errMsg;
  MealPlan? _finalPlan;

  @override
  void initState() {
    super.initState();
    _sub = NutritionService.generateStream(widget.params).listen(
      _onEvent,
      onError: (e) => mounted ? setState(() => _fail('$e')) : null,
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
    if (type == 'started') {
      _push('Computing your calorie & macro targets…');
    } else if (type == 'agent') {
      final name = ev['name'] as String?;
      final status = ev['status'] as String?;
      if (name != null && _status.containsKey(name)) {
        setState(() => _status[name] = status == 'done' ? 'done' : 'running');
        if (status == 'done') _push('$name finished');
      }
    } else if (type == 'plan') {
      _finalPlan = MealPlan.fromJson(ev);
      for (final a in _agents) {
        _status[a] = 'done';
      }
      await context.read<NutritionProvider>().setPlan(_finalPlan!, ev);
      _push('Plan ready — ${_finalPlan!.avgCalorieMatchPct.toStringAsFixed(0)}% '
          'avg calorie match');
      if (mounted) setState(() => _finished = true);
    } else if (type == 'error') {
      setState(() => _fail(ev['error']?.toString()));
    }
  }

  void _push(String line) {
    if (mounted) setState(() => _log.add(line));
  }

  @override
  Widget build(BuildContext context) {
    final title = _proceeded
        ? 'Your meal plan'
        : _finished
            ? 'The crew is done'
            : 'Building your meal plan';
    return Scaffold(
      appBar: AppBar(title: Text(title), automaticallyImplyLeading: _proceeded || _error),
      body: _error
          ? _ErrorView(message: _errMsg, onClose: () => Navigator.of(context).pop(false))
          : _proceeded && _finalPlan != null
              ? _SuccessView(
                  plan: _finalPlan!,
                  onDone: () => Navigator.of(context).pop(true),
                )
              : _ProgressView(
                  agents: _agents,
                  icons: _icons,
                  blurb: _blurb,
                  status: _status,
                  log: _log,
                  finished: _finished,
                  onProceed: () => setState(() => _proceeded = true),
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
    required this.finished,
    required this.onProceed,
  });

  final List<String> agents;
  final Map<String, IconData> icons;
  final Map<String, String> blurb;
  final Map<String, String> status;
  final List<String> log;
  final bool finished;
  final VoidCallback onProceed;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final cs = theme.colorScheme;
    return ListView(
      padding: const EdgeInsets.all(AppSpacing.lg),
      children: [
        Text('A nutrition crew is building your plan',
            style: theme.textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w700)),
        const SizedBox(height: AppSpacing.xs),
        Text('Recipes come from a cleaned 167k Food.com dataset and are chosen to hit '
            'your calorie & macro targets. The crew then names, checks, and coaches it.',
            style: theme.textTheme.bodySmall?.copyWith(color: cs.onSurfaceVariant)),
        const SizedBox(height: AppSpacing.lg),
        for (final a in agents)
          _AgentRow(name: a, icon: icons[a]!, blurb: blurb[a]!, state: status[a]!),
        const SizedBox(height: AppSpacing.md),
        if (log.isNotEmpty)
          AppCard(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(children: [
                  Icon(Icons.timeline, size: 18, color: cs.primary),
                  const SizedBox(width: AppSpacing.sm),
                  Text('Live progress',
                      style: theme.textTheme.labelLarge?.copyWith(fontWeight: FontWeight.w700)),
                ]),
                const SizedBox(height: AppSpacing.sm),
                for (final line in log)
                  Padding(
                    padding: const EdgeInsets.only(bottom: 4),
                    child: Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
                      const Icon(Icons.check, size: 14, color: Colors.green),
                      const SizedBox(width: AppSpacing.sm),
                      Expanded(child: Text(line,
                          style: theme.textTheme.bodySmall?.copyWith(color: cs.onSurfaceVariant))),
                    ]),
                  ),
              ],
            ),
          ),
        const SizedBox(height: AppSpacing.lg),
        if (!finished)
          Center(
            child: Text('This usually takes about a minute…',
                style: theme.textTheme.bodySmall?.copyWith(color: cs.onSurfaceVariant)),
          )
        else ...[
          Row(children: [
            const Icon(Icons.verified, color: Colors.green, size: 22),
            const SizedBox(width: AppSpacing.sm),
            Expanded(child: Text('The crew finished — review their work above.',
                style: theme.textTheme.bodyMedium?.copyWith(fontWeight: FontWeight.w600))),
          ]),
          const SizedBox(height: AppSpacing.md),
          AppButton(
            label: 'See my meal plan',
            icon: Icons.arrow_forward,
            size: AppButtonSize.lg,
            expand: true,
            onPressed: onProceed,
          ),
          const SizedBox(height: AppSpacing.lg),
        ],
      ],
    );
  }
}

class _AgentRow extends StatelessWidget {
  const _AgentRow({required this.name, required this.icon, required this.blurb, required this.state});
  final String name;
  final IconData icon;
  final String blurb;
  final String state;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final cs = theme.colorScheme;
    final active = state == 'running';
    final done = state == 'done';
    final accent = done ? Colors.green : (active ? cs.primary : cs.onSurfaceVariant);

    Widget trailing;
    if (done) {
      trailing = const Icon(Icons.check_circle, color: Colors.green, size: 22);
    } else if (active) {
      trailing = SizedBox(
          width: 18, height: 18, child: CircularProgressIndicator(strokeWidth: 2, color: cs.primary));
    } else {
      trailing = Icon(Icons.radio_button_unchecked, color: cs.outlineVariant, size: 20);
    }

    return Opacity(
      opacity: state == 'pending' ? 0.55 : 1,
      child: Padding(
        padding: const EdgeInsets.symmetric(vertical: AppSpacing.xs),
        child: Row(children: [
          CircleAvatar(
            radius: 18,
            backgroundColor: accent.withValues(alpha: 0.14),
            child: Icon(icon, size: 18, color: accent),
          ),
          const SizedBox(width: AppSpacing.md),
          Expanded(
            child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
              Text(name,
                  style: theme.textTheme.titleSmall
                      ?.copyWith(fontWeight: FontWeight.w700, color: active ? cs.primary : null)),
              Text(blurb, style: theme.textTheme.bodySmall?.copyWith(color: cs.onSurfaceVariant)),
            ]),
          ),
          const SizedBox(width: AppSpacing.sm),
          trailing,
        ]),
      ),
    );
  }
}

class _SuccessView extends StatelessWidget {
  const _SuccessView({required this.plan, required this.onDone});
  final MealPlan plan;
  final VoidCallback onDone;

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Expanded(child: MealPlanView(plan: plan)),
        SafeArea(
          top: false,
          child: Padding(
            padding: const EdgeInsets.fromLTRB(AppSpacing.lg, 0, AppSpacing.lg, AppSpacing.md),
            child: AppButton(label: 'Done', icon: Icons.check, expand: true, onPressed: onDone),
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
      child: Column(mainAxisAlignment: MainAxisAlignment.center, children: [
        Icon(Icons.error_outline, size: 56, color: cs.error),
        const SizedBox(height: AppSpacing.md),
        Text("Couldn't build the plan",
            style: theme.textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w700)),
        const SizedBox(height: AppSpacing.sm),
        Text(message ?? 'Please try again.',
            textAlign: TextAlign.center,
            style: theme.textTheme.bodyMedium?.copyWith(color: cs.onSurfaceVariant)),
        const SizedBox(height: AppSpacing.lg),
        AppButton(label: 'Close', expand: true, onPressed: onClose),
      ]),
    );
  }
}
