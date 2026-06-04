import 'package:flutter/material.dart';

import '../../theme/app_spacing.dart';

/// A single chat bubble — right-aligned brand bubble for the user, left-aligned
/// surface bubble for the coach.
class CoachMessageBubble extends StatelessWidget {
  const CoachMessageBubble({super.key, required this.isUser, required this.text});

  final bool isUser;
  final String text;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final cs = theme.colorScheme;
    return Align(
      alignment: isUser ? Alignment.centerRight : Alignment.centerLeft,
      child: Container(
        constraints: BoxConstraints(
          maxWidth: MediaQuery.of(context).size.width * 0.82,
        ),
        margin: const EdgeInsets.symmetric(vertical: AppSpacing.xs),
        padding: const EdgeInsets.symmetric(
          horizontal: AppSpacing.md,
          vertical: AppSpacing.sm + 2,
        ),
        decoration: BoxDecoration(
          color: isUser ? cs.primary : cs.surfaceContainerHighest,
          borderRadius: BorderRadius.only(
            topLeft: const Radius.circular(AppRadius.rl),
            topRight: const Radius.circular(AppRadius.rl),
            bottomLeft: Radius.circular(isUser ? AppRadius.rl : AppRadius.rs),
            bottomRight: Radius.circular(isUser ? AppRadius.rs : AppRadius.rl),
          ),
        ),
        child: Text(
          text,
          style: theme.textTheme.bodyMedium?.copyWith(
            color: isUser ? cs.onPrimary : cs.onSurface,
            height: 1.35,
          ),
        ),
      ),
    );
  }
}
