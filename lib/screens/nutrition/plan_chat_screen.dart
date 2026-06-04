import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../../providers/nutrition_provider.dart';
import '../../services/nutrition_service.dart';
import '../../theme/app_spacing.dart';

/// Chat to ADJUST the existing meal plan — "swap day 2 lunch", "no fish",
/// "more protein at breakfast". The backend maps each message to a deterministic
/// action (swap_meal / add_exclusion / answer) and returns a constraint-verified
/// plan, which we apply to the provider so the Nutrition tab reflects it live.
class PlanChatScreen extends StatefulWidget {
  const PlanChatScreen({super.key});

  @override
  State<PlanChatScreen> createState() => _PlanChatScreenState();
}

class _Turn {
  _Turn(this.isUser, this.text, {this.changed = false});
  final bool isUser;
  final String text;
  final bool changed; // assistant turn that modified the plan
}

class _PlanChatScreenState extends State<PlanChatScreen> {
  final _input = TextEditingController();
  final _scroll = ScrollController();
  final List<_Turn> _turns = [];
  bool _sending = false;

  @override
  void dispose() {
    _input.dispose();
    _scroll.dispose();
    super.dispose();
  }

  Future<void> _send(String text) async {
    text = text.trim();
    if (text.isEmpty || _sending) return;
    _input.clear();
    final prov = context.read<NutritionProvider>();
    final raw = prov.planRaw;
    if (raw == null) return;
    setState(() {
      _turns.add(_Turn(true, text));
      _sending = true;
    });
    _scrollDown();
    try {
      final res = await NutritionService.planChat(plan: raw, message: text);
      if (res.plan != null) {
        await prov.applyPlanJson(res.plan!);
      }
      if (mounted) {
        setState(() => _turns.add(_Turn(false, res.reply, changed: res.plan != null)));
      }
    } catch (e) {
      if (mounted) setState(() => _turns.add(_Turn(false, "Sorry — $e")));
    } finally {
      if (mounted) setState(() => _sending = false);
      _scrollDown();
    }
  }

  void _scrollDown() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scroll.hasClients) {
        _scroll.animateTo(_scroll.position.maxScrollExtent,
            duration: const Duration(milliseconds: 250), curve: Curves.easeOut);
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final cs = theme.colorScheme;
    return Scaffold(
      appBar: AppBar(title: const Text('Adjust your plan')),
      body: Column(
        children: [
          Expanded(
            child: _turns.isEmpty
                ? const _Intro()
                : ListView.builder(
                    controller: _scroll,
                    padding: const EdgeInsets.all(AppSpacing.md),
                    itemCount: _turns.length + (_sending ? 1 : 0),
                    itemBuilder: (_, i) {
                      if (i >= _turns.length) return const _Typing();
                      return _Bubble(turn: _turns[i]);
                    },
                  ),
          ),
          SafeArea(
            top: false,
            child: Padding(
              padding: const EdgeInsets.fromLTRB(
                  AppSpacing.md, AppSpacing.sm, AppSpacing.md, AppSpacing.md),
              child: Row(children: [
                Expanded(
                  child: TextField(
                    controller: _input,
                    enabled: !_sending,
                    minLines: 1,
                    maxLines: 4,
                    textInputAction: TextInputAction.send,
                    onSubmitted: _send,
                    decoration: InputDecoration(
                      hintText: _sending ? 'Updating…' : 'e.g. "no fish", "swap day 2 dinner"',
                      filled: true,
                      fillColor: cs.surfaceContainerHighest,
                      contentPadding: const EdgeInsets.symmetric(
                          horizontal: AppSpacing.md, vertical: AppSpacing.sm + 2),
                      border: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(AppRadius.rl),
                        borderSide: BorderSide.none,
                      ),
                    ),
                  ),
                ),
                const SizedBox(width: AppSpacing.sm),
                IconButton.filled(
                  onPressed: _sending ? null : () => _send(_input.text),
                  icon: const Icon(Icons.arrow_upward),
                ),
              ]),
            ),
          ),
        ],
      ),
    );
  }
}

class _Intro extends StatelessWidget {
  const _Intro();

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final cs = theme.colorScheme;
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(AppSpacing.xl),
        child: Column(mainAxisSize: MainAxisSize.min, children: [
          Icon(Icons.tune, size: 48, color: cs.primary),
          const SizedBox(height: AppSpacing.md),
          Text('Adjust your plan by chat',
              style: theme.textTheme.titleLarge?.copyWith(fontWeight: FontWeight.w700)),
          const SizedBox(height: AppSpacing.sm),
          Text(
            'Swap a meal or add a restriction — your plan updates instantly and is '
            're-checked against every rule.',
            textAlign: TextAlign.center,
            style: theme.textTheme.bodyMedium?.copyWith(color: cs.onSurfaceVariant),
          ),
          const SizedBox(height: AppSpacing.lg),
          // A natural example, not tappable buttons.
          Text(
            '“I don’t want any fish — and swap day 2 dinner for something '
            'lighter.”',
            textAlign: TextAlign.center,
            style: theme.textTheme.bodyMedium?.copyWith(
                color: cs.primary, fontStyle: FontStyle.italic, height: 1.4),
          ),
        ]),
      ),
    );
  }
}

class _Bubble extends StatelessWidget {
  const _Bubble({required this.turn});
  final _Turn turn;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final cs = theme.colorScheme;
    return Align(
      alignment: turn.isUser ? Alignment.centerRight : Alignment.centerLeft,
      child: Container(
        margin: const EdgeInsets.symmetric(vertical: AppSpacing.xs),
        padding: const EdgeInsets.symmetric(
            horizontal: AppSpacing.md, vertical: AppSpacing.sm),
        constraints: BoxConstraints(maxWidth: MediaQuery.of(context).size.width * 0.8),
        decoration: BoxDecoration(
          color: turn.isUser ? cs.primary : cs.surfaceContainerHighest,
          borderRadius: BorderRadius.circular(AppRadius.rl),
        ),
        child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Text(turn.text,
              style: theme.textTheme.bodyMedium?.copyWith(
                  color: turn.isUser ? cs.onPrimary : cs.onSurface, height: 1.4)),
          if (turn.changed)
            Padding(
              padding: const EdgeInsets.only(top: 4),
              child: Row(mainAxisSize: MainAxisSize.min, children: [
                Icon(Icons.check_circle, size: 13, color: cs.secondary),
                const SizedBox(width: 4),
                Text('Plan updated & re-verified',
                    style: theme.textTheme.labelSmall?.copyWith(
                        color: cs.secondary, fontWeight: FontWeight.w700)),
              ]),
            ),
        ]),
      ),
    );
  }
}

class _Typing extends StatelessWidget {
  const _Typing();

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    return Align(
      alignment: Alignment.centerLeft,
      child: Container(
        margin: const EdgeInsets.symmetric(vertical: AppSpacing.xs),
        padding: const EdgeInsets.all(AppSpacing.md),
        decoration: BoxDecoration(
          color: cs.surfaceContainerHighest,
          borderRadius: BorderRadius.circular(AppRadius.rl),
        ),
        child: SizedBox(
          width: 16, height: 16,
          child: CircularProgressIndicator(strokeWidth: 2, color: cs.primary),
        ),
      ),
    );
  }
}
