import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../../models/active_plan.dart';
import '../../providers/conversation_provider.dart';
import '../../theme/app_spacing.dart';
import '../../widgets/coach/coach_message_bubble.dart';
import '../../widgets/plan/plan_schedule.dart';
import '../../widgets/ui/app_button.dart';
import 'generate_plan_screen.dart';

/// The persistent AI coach chat. Builds + manages the user's plan conversationally.
class CoachChatScreen extends StatelessWidget {
  const CoachChatScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return ChangeNotifierProvider(
      create: (_) => ConversationProvider()..loadHistory(),
      child: const _CoachChatView(),
    );
  }
}

class _CoachChatView extends StatefulWidget {
  const _CoachChatView();

  @override
  State<_CoachChatView> createState() => _CoachChatViewState();
}

class _CoachChatViewState extends State<_CoachChatView> {
  final _input = TextEditingController();
  final _scroll = ScrollController();

  @override
  void dispose() {
    _input.dispose();
    _scroll.dispose();
    super.dispose();
  }

  void _scrollToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scroll.hasClients) {
        _scroll.animateTo(
          _scroll.position.maxScrollExtent,
          duration: const Duration(milliseconds: 250),
          curve: Curves.easeOut,
        );
      }
    });
  }

  Future<void> _send() async {
    final text = _input.text.trim();
    if (text.isEmpty) return;
    _input.clear();
    final prov = context.read<ConversationProvider>();
    _scrollToBottom();
    await prov.send(text);
    _scrollToBottom();
  }

  Future<void> _openGenerate() async {
    final prov = context.read<ConversationProvider>();
    final params = Map<String, dynamic>.from(prov.pendingPlanParams ?? const {});
    final ok = await Navigator.of(context).push<bool>(
      MaterialPageRoute(builder: (_) => GeneratePlanScreen(params: params)),
    );
    if (ok == true) {
      await prov.onPlanGenerated();
      _scrollToBottom();
    }
  }

  @override
  Widget build(BuildContext context) {
    final prov = context.watch<ConversationProvider>();
    final showList = prov.messages.isNotEmpty || prov.sending;

    return Scaffold(
      appBar: AppBar(title: const Text('AI Coach')),
      body: Column(
        children: [
          Expanded(
            child: prov.loading
                ? const Center(child: CircularProgressIndicator())
                : showList
                    ? ListView.builder(
                        controller: _scroll,
                        padding: const EdgeInsets.all(AppSpacing.md),
                        itemCount: prov.messages.length + (prov.sending ? 1 : 0),
                        itemBuilder: (context, i) {
                          if (i >= prov.messages.length) {
                            return const _TypingBubble();
                          }
                          final m = prov.messages[i];
                          if (m.isPlanReveal) {
                            return _PlanRevealCard(plan: m.plan!);
                          }
                          return CoachMessageBubble(
                            isUser: m.isUser,
                            text: m.content ?? '',
                          );
                        },
                      )
                    : const _Intro(),
          ),
          if (prov.readyToGenerate)
            Padding(
              padding: const EdgeInsets.fromLTRB(
                  AppSpacing.md, AppSpacing.sm, AppSpacing.md, 0),
              child: AppButton(
                label: 'Generate Plan',
                icon: Icons.auto_awesome,
                size: AppButtonSize.lg,
                expand: true,
                onPressed: _openGenerate,
              ),
            ),
          _InputBar(
            controller: _input,
            enabled: !prov.sending,
            onSend: _send,
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
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(Icons.auto_awesome, size: 48, color: cs.primary),
            const SizedBox(height: AppSpacing.md),
            Text(
              "I'm your FitNova coach",
              style: theme.textTheme.titleLarge
                  ?.copyWith(fontWeight: FontWeight.w700),
            ),
            const SizedBox(height: AppSpacing.sm),
            Text(
              'Tell me your goal and I\'ll build your plan — and adjust it '
              'whenever you ask.\n\ne.g. "Build me a 4-day powerbuilding plan, '
              'I train at a full gym."',
              textAlign: TextAlign.center,
              style: theme.textTheme.bodyMedium
                  ?.copyWith(color: cs.onSurfaceVariant, height: 1.4),
            ),
          ],
        ),
      ),
    );
  }
}

class _TypingBubble extends StatelessWidget {
  const _TypingBubble();

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final cs = theme.colorScheme;
    return Align(
      alignment: Alignment.centerLeft,
      child: Container(
        margin: const EdgeInsets.symmetric(vertical: AppSpacing.xs),
        padding: const EdgeInsets.symmetric(
          horizontal: AppSpacing.md,
          vertical: AppSpacing.sm + 2,
        ),
        decoration: BoxDecoration(
          color: cs.surfaceContainerHighest,
          borderRadius: BorderRadius.circular(AppRadius.rl),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            SizedBox(
              width: 16,
              height: 16,
              child: CircularProgressIndicator(strokeWidth: 2, color: cs.primary),
            ),
            const SizedBox(width: AppSpacing.sm),
            Text('Coach is thinking…',
                style: theme.textTheme.bodySmall
                    ?.copyWith(color: cs.onSurfaceVariant)),
          ],
        ),
      ),
    );
  }
}

class _InputBar extends StatelessWidget {
  const _InputBar({
    required this.controller,
    required this.enabled,
    required this.onSend,
  });

  final TextEditingController controller;
  final bool enabled;
  final VoidCallback onSend;

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    return SafeArea(
      top: false,
      child: Padding(
        padding: const EdgeInsets.fromLTRB(
          AppSpacing.md, AppSpacing.sm, AppSpacing.md, AppSpacing.md,
        ),
        child: Row(
          children: [
            Expanded(
              child: TextField(
                controller: controller,
                enabled: enabled,
                minLines: 1,
                maxLines: 4,
                textInputAction: TextInputAction.send,
                onSubmitted: (_) => enabled ? onSend() : null,
                decoration: InputDecoration(
                  hintText: enabled ? 'Message your coach…' : 'Coach is replying…',
                  filled: true,
                  fillColor: cs.surfaceContainerHighest,
                  contentPadding: const EdgeInsets.symmetric(
                    horizontal: AppSpacing.md, vertical: AppSpacing.sm + 2,
                  ),
                  border: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(AppRadius.rl),
                    borderSide: BorderSide.none,
                  ),
                ),
              ),
            ),
            const SizedBox(width: AppSpacing.sm),
            IconButton.filled(
              onPressed: enabled ? onSend : null,
              icon: const Icon(Icons.arrow_upward),
            ),
          ],
        ),
      ),
    );
  }
}

/// One-time, in-chat reveal of a freshly generated plan — the same clean
/// schedule design as the Planning tab, animated in, with a pointer that the
/// canonical view lives in the Planning tab.
class _PlanRevealCard extends StatelessWidget {
  const _PlanRevealCard({required this.plan});

  final ActivePlan plan;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final cs = theme.colorScheme;
    return TweenAnimationBuilder<double>(
      tween: Tween<double>(begin: 0, end: 1),
      duration: const Duration(milliseconds: 420),
      curve: Curves.easeOutCubic,
      builder: (context, t, child) => Opacity(
        opacity: t.clamp(0.0, 1.0),
        child: Transform.translate(offset: Offset(0, (1 - t) * 14), child: child),
      ),
      child: Container(
        margin: const EdgeInsets.symmetric(vertical: AppSpacing.sm),
        padding: const EdgeInsets.all(AppSpacing.md),
        decoration: BoxDecoration(
          color: cs.surface,
          borderRadius: BorderRadius.circular(AppRadius.rl),
          border: Border.all(color: cs.outlineVariant),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Row(
              children: [
                Icon(Icons.event_available, size: 18, color: cs.primary),
                const SizedBox(width: AppSpacing.sm),
                Text('Your plan is ready',
                    style: theme.textTheme.labelLarge?.copyWith(
                      color: cs.primary,
                      fontWeight: FontWeight.w800,
                    )),
              ],
            ),
            const SizedBox(height: AppSpacing.md),
            PlanScheduleView(plan: plan),
            const SizedBox(height: AppSpacing.xs),
            Text('View it anytime in the Planning tab.',
                style: theme.textTheme.bodySmall
                    ?.copyWith(color: cs.onSurfaceVariant)),
          ],
        ),
      ),
    );
  }
}
