import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import '../../features/shell/app_shell.dart';
import '../../features/live_detection/live_detection_screen.dart';
import '../../features/training/training_screen.dart';
import '../../features/config/config_screen.dart';
import '../../features/database/database_screen.dart';
import '../../features/settings/settings_screen.dart';

/// Navigation index provider for NavigationRail
final navigationIndexProvider = StateProvider<int>((ref) => 0);

/// Router provider
final routerProvider = Provider<GoRouter>((ref) {
  return GoRouter(
    initialLocation: '/live',
    routes: [
      ShellRoute(
        builder: (context, state, child) => AppShell(child: child),
        routes: [
          GoRoute(
            path: '/live',
            name: 'live',
            pageBuilder: (context, state) => const NoTransitionPage(
              child: LiveDetectionScreen(),
            ),
          ),
          GoRoute(
            path: '/training',
            name: 'training',
            pageBuilder: (context, state) => const NoTransitionPage(
              child: TrainingScreen(),
            ),
          ),
          GoRoute(
            path: '/config',
            name: 'config',
            pageBuilder: (context, state) => const NoTransitionPage(
              child: ConfigScreen(),
            ),
          ),
          GoRoute(
            path: '/database',
            name: 'database',
            pageBuilder: (context, state) => const NoTransitionPage(
              child: DatabaseScreen(),
            ),
          ),
          GoRoute(
            path: '/settings',
            name: 'settings',
            pageBuilder: (context, state) => const NoTransitionPage(
              child: SettingsScreen(),
            ),
          ),
        ],
      ),
    ],
  );
});

/// Route paths
class AppRoutes {
  static const String live = '/live';
  static const String training = '/training';
  static const String config = '/config';
  static const String database = '/database';
  static const String settings = '/settings';
}
