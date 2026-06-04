import 'package:flutter/widgets.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:provider/provider.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'package:fitnova_application/main.dart';
import 'package:fitnova_application/providers/user_provider.dart';
import 'package:fitnova_application/providers/form_session_provider.dart';
import 'package:fitnova_application/theme/theme_controller.dart';

void main() {
  testWidgets('App launches and shows splash screen', (WidgetTester tester) async {
    SharedPreferences.setMockInitialValues(<String, Object>{});
    final themeController = ThemeController();
    await themeController.init();

    await tester.pumpWidget(
      MultiProvider(
        providers: [
          ChangeNotifierProvider.value(value: themeController),
          ChangeNotifierProvider(create: (_) => UserProvider()),
          ChangeNotifierProvider(create: (_) => FormSessionProvider()),
        ],
        child: const FitNovaApp(),
      ),
    );

    expect(find.image(const AssetImage('assets/logo/wordmark.png')), findsOneWidget);
    expect(find.text('AI-Powered Fitness Planning'), findsOneWidget);

    // Splash schedules a 1500ms delayed navigation; let it fire so the
    // test framework doesn't flag a pending Timer on disposal.
    await tester.pumpAndSettle(const Duration(seconds: 2));
  });
}
