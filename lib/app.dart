import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'core/config/theme.dart';
import 'core/config/router.dart';
import 'core/services/backend_launcher.dart';

class G20App extends ConsumerStatefulWidget {
  const G20App({super.key});

  @override
  ConsumerState<G20App> createState() => _G20AppState();
}

class _G20AppState extends ConsumerState<G20App> with WidgetsBindingObserver {
  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    // Auto-start the Python backend server on app launch
    _initializeBackend();
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    // Kill the backend when app closes
    ref.read(backendLauncherProvider.notifier).stopBackend();
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    // Kill backend on app detach/pause (window close)
    if (state == AppLifecycleState.detached) {
      ref.read(backendLauncherProvider.notifier).stopBackend();
    }
  }

  Future<void> _initializeBackend() async {
    // Small delay to let the widget tree settle
    await Future.delayed(const Duration(milliseconds: 100));
    
    // Start the backend server
    final launcher = ref.read(backendLauncherProvider.notifier);
    await launcher.startBackend();
  }

  @override
  Widget build(BuildContext context) {
    final router = ref.watch(routerProvider);
    final backendState = ref.watch(backendLauncherProvider);

    return MaterialApp.router(
      title: 'G20 RF Platform',
      debugShowCheckedModeBanner: false,
      theme: G20Theme.light,
      darkTheme: G20Theme.dark,
      themeMode: ThemeMode.dark, // Default to dark theme for RF visualization
      routerConfig: router,
      builder: (context, child) {
        // Show backend status banner if not running
        if (backendState.state == BackendState.error) {
          return Column(
            children: [
              MaterialBanner(
                backgroundColor: Colors.red.shade900,
                content: Text(
                  'Backend Error: ${backendState.errorMessage ?? "Unknown error"}',
                  style: const TextStyle(color: Colors.white),
                ),
                actions: [
                  TextButton(
                    onPressed: () {
                      ref.read(backendLauncherProvider.notifier).restartBackend();
                    },
                    child: const Text('RETRY', style: TextStyle(color: Colors.white)),
                  ),
                ],
              ),
              Expanded(child: child ?? const SizedBox()),
            ],
          );
        }
        return child ?? const SizedBox();
      },
    );
  }
}
