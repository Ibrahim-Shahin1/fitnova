import 'package:flutter/material.dart';

import '../theme/app_spacing.dart';

class SplashScreen extends StatefulWidget {
  const SplashScreen({super.key});

  @override
  State<SplashScreen> createState() => _SplashScreenState();
}

class _SplashScreenState extends State<SplashScreen> {
  @override
  void initState() {
    super.initState();
    Future.delayed(const Duration(milliseconds: 1500), () {
      if (mounted) {
        Navigator.pushReplacementNamed(context, '/register');
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final cs = theme.colorScheme;

    return Scaffold(
      body: Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            SizedBox(
              width: 260,
              child: Image.asset(
                'assets/logo/wordmark.png',
                semanticLabel: 'FITNOVA',
              ),
            ),
            const SizedBox(height: AppSpacing.md),
            Text(
              'AI-Powered Fitness Planning',
              style: theme.textTheme.bodyLarge?.copyWith(
                color: cs.onSurfaceVariant,
                letterSpacing: 1,
              ),
            ),
          ],
        ),
      ),
    );
  }
}
