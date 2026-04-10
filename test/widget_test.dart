import 'package:flutter_test/flutter_test.dart';
import 'package:provider/provider.dart';

import 'package:fitnova_application/main.dart';
import 'package:fitnova_application/providers/user_provider.dart';

void main() {
  testWidgets('App launches and shows splash screen', (WidgetTester tester) async {
    await tester.pumpWidget(
      ChangeNotifierProvider(
        create: (_) => UserProvider(),
        child: const FitNovaApp(),
      ),
    );

    expect(find.text('FitNova'), findsOneWidget);
    expect(find.text('AI-Powered Fitness Planning'), findsOneWidget);
  });
}
