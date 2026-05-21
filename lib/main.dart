import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'providers/user_provider.dart';
import 'providers/form_session_provider.dart';
import 'theme/app_theme.dart';
import 'theme/theme_controller.dart';
import 'screens/splash_screen.dart';
import 'screens/registration_screen.dart';
import 'screens/home_screen.dart';
import 'screens/chat_screen.dart';
import 'screens/plan_screen.dart';
import 'screens/form_check_screen.dart';
import 'screens/form_results_screen.dart';
import 'models/exercise_meta.dart';
import 'screens/exercise_selection_screen.dart';
import 'screens/guidelines_screen.dart';
import 'screens/video_upload_screen.dart';
import 'screens/mode_select_screen.dart';
import 'screens/form_replay_screen.dart';
import 'models/form_models.dart';
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
        ChangeNotifierProvider(create: (_) => ProfileProvider()),
        ChangeNotifierProvider(create: (_) => UserProvider()),
        ChangeNotifierProvider(create: (_) => FormSessionProvider()),
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
            '/splash':            (_) => const SplashScreen(),
            '/register':          (_) => const RegistrationScreen(),
            '/mode-select':       (_) => const ModeSelectScreen(),
            '/home':              (_) => const HomeScreen(),
            '/chat':              (_) => const ChatScreen(),
            '/plan':              (_) => const PlanScreen(),
            '/exercise-select':   (_) => const ExerciseSelectionScreen(),
            '/form-results':      (_) => const FormResultsScreen(),
          },
          onGenerateRoute: (settings) {
            if (settings.name == '/form-check') {
              final hint = settings.arguments as String?;
              return MaterialPageRoute(
                builder: (_) => FormCheckScreen(exerciseHint: hint),
                settings: settings,
              );
            }
            if (settings.name == '/guidelines') {
              final meta = settings.arguments as ExerciseMeta;
              return MaterialPageRoute(
                builder: (_) => GuidelinesScreen(meta: meta),
                settings: settings,
              );
            }
            if (settings.name == '/video-upload') {
              final meta = settings.arguments as ExerciseMeta;
              return MaterialPageRoute(
                builder: (_) => VideoUploadScreen(meta: meta),
                settings: settings,
              );
            }
            if (settings.name == '/form-replay') {
              final args = settings.arguments as Map<String, dynamic>;
              return MaterialPageRoute(
                builder: (_) => FormReplayScreen(
                  videoPath: args['videoPath'] as String,
                  summary:   args['summary']  as FormSessionSummary,
                ),
                settings: settings,
              );
            }
            return null;
          },
        );
      },
    );
  }
}
