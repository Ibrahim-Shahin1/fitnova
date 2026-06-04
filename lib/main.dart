import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'providers/user_provider.dart';
import 'providers/form_session_provider.dart';
import 'providers/benchmark_provider.dart';
import 'theme/app_theme.dart';
import 'theme/theme_controller.dart';
import 'screens/splash_screen.dart';
import 'screens/registration_screen.dart';
import 'screens/home_screen.dart';
import 'screens/chat_screen.dart';
import 'screens/plan_screen.dart';
import 'screens/mode_select_screen.dart';
import 'screens/benchmark_browse_screen.dart';
import 'screens/benchmark_results_screen.dart';
import 'screens/benchmark_filmstrip_screen.dart';
import 'providers/auth_provider.dart';
import 'providers/profile_provider.dart';
import 'services/supabase_service.dart';
import 'screens/auth/auth_gate.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final themeController = ThemeController();
  await themeController.init();

  await SupabaseService.initialize();

  runApp(
    MultiProvider(
      providers: [
        ChangeNotifierProvider.value(value: themeController),
        ChangeNotifierProvider(create: (_) => AuthProvider()),
        ChangeNotifierProxyProvider<AuthProvider, ProfileProvider>(
          create: (_) => ProfileProvider(),
          update: (_, auth, profile) =>
              profile!..syncWithAuth(auth.user?.id),
        ),
        ChangeNotifierProvider(create: (_) => UserProvider()),
        ChangeNotifierProvider(create: (_) => FormSessionProvider()),
        ChangeNotifierProvider(create: (_) => BenchmarkProvider()),
      ],
      child: const FitNovaApp(),
    ),
  );
}

class FitNovaApp extends StatelessWidget {
  const FitNovaApp({super.key});

  @override
  Widget build(BuildContext context) {
    return Consumer<ThemeController>(
      builder: (context, themeController, _) {
        return MaterialApp(
          title: 'FitNova',
          debugShowCheckedModeBanner: false,
          theme: AppTheme.light,
          darkTheme: AppTheme.dark,
          themeMode: themeController.mode,
          home: const AuthGate(),
          routes: {
            '/splash':               (_) => const SplashScreen(),
            '/register':             (_) => const RegistrationScreen(),
            '/mode-select':          (_) => const ModeSelectScreen(),
            '/home':                 (_) => const HomeScreen(),
            '/chat':                 (_) => const ChatScreen(),
            '/plan':                 (_) => const PlanScreen(),
            '/benchmark/browse':     (_) => const BenchmarkBrowseScreen(),
            '/benchmark/results':    (_) => const BenchmarkResultsScreen(),
            '/benchmark/filmstrip':  (_) => const BenchmarkFilmstripScreen(),
          },
        );
      },
    );
  }
}
