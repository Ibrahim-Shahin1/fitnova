import 'package:flutter/material.dart';

import '../../models/meal_plan.dart';
import '../../services/nutrition_service.dart';
import '../../theme/app_spacing.dart';
import '../../widgets/ui/app_button.dart';
import '../../widgets/ui/app_card.dart';

/// Full recipe screen. Shows 100%-dataset provenance (Food.com id, ingredients,
/// original steps), then two clearly-separated AI capabilities:
///   1. "Generate with our model" — the fine-tuned DistilGPT-2 (academic showcase)
///   2. A chat thread — GPT-4o-mini, grounded in this recipe, for substitutions/
///      scaling/technique follow-ups.
class RecipeDetailScreen extends StatefulWidget {
  const RecipeDetailScreen({super.key, required this.meal});

  final Meal meal;

  @override
  State<RecipeDetailScreen> createState() => _RecipeDetailScreenState();
}

class _ChatTurn {
  _ChatTurn(this.isUser, this.text);
  final bool isUser;
  final String text;
}

class _RecipeDetailScreenState extends State<RecipeDetailScreen> {
  final _input = TextEditingController();
  final _scroll = ScrollController();

  String? _aiInstructions;
  bool _generating = false;
  String? _genError;

  final List<_ChatTurn> _chat = [];
  bool _chatSending = false;

  @override
  void dispose() {
    _input.dispose();
    _scroll.dispose();
    super.dispose();
  }

  Map<String, dynamic> get _recipeContext => {
        'name': widget.meal.name,
        'ingredients': widget.meal.ingredients,
        'steps': widget.meal.steps,
        if (_aiInstructions != null) 'ai_instructions': _aiInstructions,
      };

  Future<void> _generate() async {
    setState(() {
      _generating = true;
      _genError = null;
    });
    try {
      final text = await NutritionService.recipeInstructions(
        name: widget.meal.name,
        ingredients: widget.meal.ingredients,
        calorieLevel: widget.meal.calorieLevelWord,
      );
      if (mounted) setState(() => _aiInstructions = text);
    } catch (e) {
      if (mounted) setState(() => _genError = '$e');
    } finally {
      if (mounted) setState(() => _generating = false);
    }
  }

  Future<void> _sendChat() async {
    final text = _input.text.trim();
    if (text.isEmpty || _chatSending) return;
    _input.clear();
    setState(() {
      _chat.add(_ChatTurn(true, text));
      _chatSending = true;
    });
    _scrollDown();
    try {
      final history = _chat
          .map((t) => {'role': t.isUser ? 'user' : 'assistant', 'content': t.text})
          .toList();
      final reply = await NutritionService.recipeChat(
        recipe: _recipeContext,
        messages: history,
      );
      if (mounted) setState(() => _chat.add(_ChatTurn(false, reply)));
    } catch (e) {
      if (mounted) {
        setState(() => _chat.add(_ChatTurn(false, "Sorry — I couldn't answer that. $e")));
      }
    } finally {
      if (mounted) setState(() => _chatSending = false);
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
    final m = widget.meal;

    return Scaffold(
      appBar: AppBar(title: Text(m.slot[0].toUpperCase() + m.slot.substring(1))),
      body: Column(
        children: [
          Expanded(
            child: ListView(
              controller: _scroll,
              padding: const EdgeInsets.all(AppSpacing.lg),
              children: [
                Text(m.name,
                    style: theme.textTheme.titleLarge?.copyWith(fontWeight: FontWeight.w800)),
                const SizedBox(height: AppSpacing.sm),
                Row(children: [
                  _Badge(icon: Icons.verified, text: 'Food.com #${m.recipeId}', color: cs.secondary),
                  const SizedBox(width: AppSpacing.sm),
                  Flexible(
                    child: Text(
                      '${m.calories.round()} kcal · P${m.proteinG.round()} '
                      'C${m.carbsG.round()} F${m.fatG.round()}'
                      '${m.minutes > 0 ? ' · ${m.minutes}min' : ''}',
                      style: theme.textTheme.labelMedium?.copyWith(color: cs.onSurfaceVariant),
                    ),
                  ),
                ]),
                const SizedBox(height: AppSpacing.lg),

                if (m.ingredients.isNotEmpty) ...[
                  _SectionHeader(icon: Icons.shopping_basket, title: 'Ingredients', tag: 'dataset'),
                  const SizedBox(height: AppSpacing.sm),
                  Wrap(
                    spacing: AppSpacing.sm,
                    runSpacing: AppSpacing.xs,
                    children: [
                      for (final i in m.ingredients)
                        Chip(label: Text(i), visualDensity: VisualDensity.compact),
                    ],
                  ),
                  const SizedBox(height: AppSpacing.lg),
                ],

                if (m.steps.isNotEmpty) ...[
                  _SectionHeader(
                      icon: Icons.menu_book, title: 'Original recipe', tag: 'Food.com dataset'),
                  const SizedBox(height: AppSpacing.sm),
                  for (var i = 0; i < m.steps.length; i++)
                    Padding(
                      padding: const EdgeInsets.only(bottom: AppSpacing.xs),
                      child: Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
                        Text('${i + 1}. ',
                            style: theme.textTheme.bodyMedium
                                ?.copyWith(fontWeight: FontWeight.w700, color: cs.secondary)),
                        Expanded(
                            child: Text(m.steps[i],
                                style: theme.textTheme.bodyMedium?.copyWith(height: 1.4))),
                      ]),
                    ),
                  const SizedBox(height: AppSpacing.lg),
                ],

                const Divider(),
                const SizedBox(height: AppSpacing.sm),

                _SectionHeader(
                    icon: Icons.auto_awesome, title: 'Fine-tuned DistilGPT-2', tag: 'generated'),
                const SizedBox(height: AppSpacing.sm),
                if (_aiInstructions == null && _genError == null)
                  AppButton(
                    label: _generating ? 'Writing recipe…' : 'Generate instructions',
                    icon: Icons.auto_awesome,
                    variant: AppButtonVariant.secondary,
                    expand: true,
                    isLoading: _generating,
                    onPressed: _generating ? null : _generate,
                  ),
                if (_genError != null) ...[
                  Text(_genError!, style: theme.textTheme.bodySmall?.copyWith(color: cs.error)),
                  const SizedBox(height: AppSpacing.sm),
                  AppButton(label: 'Retry', expand: true, onPressed: _generate),
                ],
                if (_aiInstructions != null)
                  AppCard(
                    elevated: true,
                    child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                      Text(_aiInstructions!,
                          style: theme.textTheme.bodyMedium?.copyWith(height: 1.5)),
                      const SizedBox(height: AppSpacing.xs),
                      Text('Generated by DistilGPT-2, fine-tuned on Food.com '
                          '(Majumder et al. recipe-generation task).',
                          style: theme.textTheme.labelSmall?.copyWith(
                              color: cs.onSurfaceVariant, fontStyle: FontStyle.italic)),
                    ]),
                  ),

                const SizedBox(height: AppSpacing.lg),

                _SectionHeader(
                    icon: Icons.forum, title: 'Ask about this recipe', tag: 'AI assistant'),
                const SizedBox(height: AppSpacing.xs),
                Text(
                  'Substitutions, scaling, technique — e.g. "make it vegan", '
                  '"no oven, what else?", "double the servings".',
                  style: theme.textTheme.bodySmall?.copyWith(color: cs.onSurfaceVariant),
                ),
                const SizedBox(height: AppSpacing.sm),
                for (final turn in _chat) _Bubble(turn: turn),
                if (_chatSending) const _TypingBubble(),
                const SizedBox(height: AppSpacing.md),
              ],
            ),
          ),
          _ChatInput(controller: _input, enabled: !_chatSending, onSend: _sendChat),
        ],
      ),
    );
  }
}

class _SectionHeader extends StatelessWidget {
  const _SectionHeader({required this.icon, required this.title, required this.tag});
  final IconData icon;
  final String title;
  final String tag;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final cs = theme.colorScheme;
    final isAi = tag == 'generated' || tag == 'AI assistant';
    final color = isAi ? cs.primary : cs.secondary;
    return Row(children: [
      Icon(icon, size: 18, color: color),
      const SizedBox(width: AppSpacing.sm),
      Text(title, style: theme.textTheme.titleSmall?.copyWith(fontWeight: FontWeight.w700)),
      const SizedBox(width: AppSpacing.sm),
      Container(
        padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
        decoration: BoxDecoration(
          color: color.withValues(alpha: 0.14),
          borderRadius: BorderRadius.circular(AppRadius.pill),
        ),
        child: Text(tag,
            style: theme.textTheme.labelSmall?.copyWith(color: color, fontWeight: FontWeight.w700)),
      ),
    ]);
  }
}

class _Badge extends StatelessWidget {
  const _Badge({required this.icon, required this.text, required this.color});
  final IconData icon;
  final String text;
  final Color color;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: AppSpacing.sm, vertical: 3),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.14),
        borderRadius: BorderRadius.circular(AppRadius.pill),
      ),
      child: Row(mainAxisSize: MainAxisSize.min, children: [
        Icon(icon, size: 13, color: color),
        const SizedBox(width: 4),
        Text(text,
            style: Theme.of(context).textTheme.labelSmall
                ?.copyWith(color: color, fontWeight: FontWeight.w700)),
      ]),
    );
  }
}

class _Bubble extends StatelessWidget {
  const _Bubble({required this.turn});
  final _ChatTurn turn;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final cs = theme.colorScheme;
    return Align(
      alignment: turn.isUser ? Alignment.centerRight : Alignment.centerLeft,
      child: Container(
        margin: const EdgeInsets.symmetric(vertical: AppSpacing.xs),
        padding: const EdgeInsets.symmetric(horizontal: AppSpacing.md, vertical: AppSpacing.sm),
        constraints: BoxConstraints(maxWidth: MediaQuery.of(context).size.width * 0.78),
        decoration: BoxDecoration(
          color: turn.isUser ? cs.primary : cs.surfaceContainerHighest,
          borderRadius: BorderRadius.circular(AppRadius.rl),
        ),
        child: Text(turn.text,
            style: theme.textTheme.bodyMedium?.copyWith(
                color: turn.isUser ? cs.onPrimary : cs.onSurface, height: 1.4)),
      ),
    );
  }
}

class _TypingBubble extends StatelessWidget {
  const _TypingBubble();

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

class _ChatInput extends StatelessWidget {
  const _ChatInput({required this.controller, required this.enabled, required this.onSend});
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
            AppSpacing.md, AppSpacing.sm, AppSpacing.md, AppSpacing.md),
        child: Row(children: [
          Expanded(
            child: TextField(
              controller: controller,
              enabled: enabled,
              minLines: 1,
              maxLines: 4,
              textInputAction: TextInputAction.send,
              onSubmitted: (_) => enabled ? onSend() : null,
              decoration: InputDecoration(
                hintText: enabled ? 'Ask about this recipe…' : 'Thinking…',
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
            onPressed: enabled ? onSend : null,
            icon: const Icon(Icons.arrow_upward),
          ),
        ]),
      ),
    );
  }
}
