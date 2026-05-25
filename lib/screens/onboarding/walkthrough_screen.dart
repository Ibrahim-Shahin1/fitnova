import 'package:flutter/material.dart';

import '../../theme/app_spacing.dart';
import '../../widgets/ui/app_button.dart';

const _pages = [
  (
    icon: Icons.calendar_today_outlined,
    title: 'Plan your training',
    body: 'Chat with your AI coach to get a personalized 7-day plan — then log '
        'every set, rep, and weight as you go.',
  ),
  (
    icon: Icons.videocam_outlined,
    title: 'Fix your form',
    body: 'Record or stream a lift and get plain-language feedback on your '
        'form, grounded in real movement analysis.',
  ),
  (
    icon: Icons.insights_outlined,
    title: 'Track your progress',
    body: 'Watch your numbers climb over time, and ask the coach anything '
        'about your plan or history — anytime.',
  ),
];

/// Step 2 of onboarding: a short three-card intro to the app's features.
class WalkthroughScreen extends StatefulWidget {
  const WalkthroughScreen({super.key, required this.onFinish});

  final VoidCallback onFinish;

  @override
  State<WalkthroughScreen> createState() => _WalkthroughScreenState();
}

class _WalkthroughScreenState extends State<WalkthroughScreen> {
  final _controller = PageController();
  int _page = 0;

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  bool get _isLast => _page == _pages.length - 1;

  void _next() {
    if (_isLast) {
      widget.onFinish();
    } else {
      _controller.nextPage(
        duration: const Duration(milliseconds: 300),
        curve: Curves.easeOutCubic,
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Scaffold(
      body: SafeArea(
        child: Column(
          children: [
            Expanded(
              child: PageView.builder(
                controller: _controller,
                itemCount: _pages.length,
                onPageChanged: (i) => setState(() => _page = i),
                itemBuilder: (context, i) {
                  final p = _pages[i];
                  return Padding(
                    padding: const EdgeInsets.symmetric(
                        horizontal: AppSpacing.xl, vertical: AppSpacing.lg),
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        Icon(p.icon, size: 88, color: theme.colorScheme.primary),
                        const SizedBox(height: AppSpacing.xl),
                        Text(p.title,
                            textAlign: TextAlign.center,
                            style: theme.textTheme.headlineSmall
                                ?.copyWith(fontWeight: FontWeight.w700)),
                        const SizedBox(height: AppSpacing.md),
                        Text(p.body,
                            textAlign: TextAlign.center,
                            style: theme.textTheme.bodyLarge?.copyWith(
                                color: theme.colorScheme.onSurfaceVariant,
                                height: 1.5)),
                      ],
                    ),
                  );
                },
              ),
            ),
            Padding(
              padding: const EdgeInsets.all(AppSpacing.lg),
              child: Column(
                children: [
                  Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: List.generate(_pages.length, (i) {
                      final active = i == _page;
                      return AnimatedContainer(
                        duration: AppDuration.fast,
                        margin:
                            const EdgeInsets.symmetric(horizontal: AppSpacing.xs),
                        width: active ? 22 : 8,
                        height: 8,
                        decoration: BoxDecoration(
                          color: active
                              ? theme.colorScheme.primary
                              : theme.colorScheme.outlineVariant,
                          borderRadius: BorderRadius.circular(AppRadius.pill),
                        ),
                      );
                    }),
                  ),
                  const SizedBox(height: AppSpacing.lg),
                  AppButton(
                    label: _isLast ? 'Get started' : 'Next',
                    onPressed: _next,
                    expand: true,
                    size: AppButtonSize.lg,
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}
