import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../../providers/auth_provider.dart';
import '../../providers/nutrition_provider.dart';
import '../../theme/app_spacing.dart';
import '../../widgets/nutrition/meal_plan_view.dart';
import '../../widgets/ui/app_button.dart';
import 'nutrition_intake_screen.dart';
import 'plan_chat_screen.dart';

/// Nutrition tab — shows the saved meal plan, or a CTA to build one. Mirrors the
/// Planning tab: own provider (scoped to the auth user), reloads on return from
/// the intake/generate flow.
class NutritionTab extends StatelessWidget {
  const NutritionTab({super.key});

  @override
  Widget build(BuildContext context) {
    // Scope persistence to the signed-in user so a new account never inherits a
    // previous session's plan from this device.
    final userId = context.read<AuthProvider>().user?.id;
    return ChangeNotifierProvider(
      create: (_) => NutritionProvider(userId: userId)..load(),
      child: const _NutritionView(),
    );
  }
}

class _NutritionView extends StatelessWidget {
  const _NutritionView();

  Future<void> _openIntake(BuildContext context) async {
    // Pass the tab's provider across the Navigator boundary so the pushed intake
    // (and the generate screen it pushes) share this SAME instance — a new route
    // is not a descendant of this tab's provider scope.
    final prov = context.read<NutritionProvider>();
    await Navigator.of(context).push(
      MaterialPageRoute(
        builder: (_) => ChangeNotifierProvider<NutritionProvider>.value(
          value: prov,
          child: const NutritionIntakeScreen(),
        ),
      ),
    );
  }

  Future<void> _openPlanChat(BuildContext context) async {
    final prov = context.read<NutritionProvider>();
    await Navigator.of(context).push(
      MaterialPageRoute(
        builder: (_) => ChangeNotifierProvider<NutritionProvider>.value(
          value: prov,
          child: const PlanChatScreen(),
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final prov = context.watch<NutritionProvider>();
    return Scaffold(
      appBar: AppBar(
        title: const Text('Nutrition'),
        actions: [
          if (prov.hasPlan) ...[
            IconButton(
              tooltip: 'Adjust plan',
              icon: const Icon(Icons.chat_bubble_outline),
              onPressed: () => _openPlanChat(context),
            ),
            IconButton(
              tooltip: 'Rebuild plan',
              icon: const Icon(Icons.refresh),
              onPressed: () => _openIntake(context),
            ),
          ],
        ],
      ),
      floatingActionButton: prov.hasPlan
          ? FloatingActionButton.extended(
              onPressed: () => _openPlanChat(context),
              icon: const Icon(Icons.tune),
              label: const Text('Adjust plan'),
            )
          : null,
      body: prov.loading
          ? const Center(child: CircularProgressIndicator())
          : prov.hasPlan
              ? MealPlanView(plan: prov.plan!)
              : _NutritionCta(onOpen: () => _openIntake(context)),
    );
  }
}

class _NutritionCta extends StatelessWidget {
  const _NutritionCta({required this.onOpen});
  final VoidCallback onOpen;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final cs = theme.colorScheme;
    return SafeArea(
      child: Padding(
        padding: const EdgeInsets.all(AppSpacing.lg),
        child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
          const SizedBox(height: AppSpacing.xl),
          Icon(Icons.restaurant, size: 64, color: cs.primary),
          const SizedBox(height: AppSpacing.lg),
          Text('Build your meal plan',
              textAlign: TextAlign.center,
              style: theme.textTheme.headlineSmall?.copyWith(fontWeight: FontWeight.w700)),
          const SizedBox(height: AppSpacing.sm),
          Text(
            'Enter your goal and preferences — we use your profile stats to compute '
            'your calorie & macro targets, then a nutrition crew builds a weekly '
            'plan from real recipes that hit them.',
            textAlign: TextAlign.center,
            style: theme.textTheme.bodyMedium?.copyWith(color: cs.onSurfaceVariant),
          ),
          const Spacer(),
          AppButton(
            label: 'Get started',
            icon: Icons.auto_awesome,
            size: AppButtonSize.lg,
            expand: true,
            onPressed: onOpen,
          ),
          const SizedBox(height: AppSpacing.lg),
        ]),
      ),
    );
  }
}
