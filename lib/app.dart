import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'core/config/theme.dart';
import 'core/config/router.dart';
import 'core/services/backend_launcher.dart';
import 'features/settings/settings_screen.dart';
import 'features/live_detection/providers/video_stream_provider.dart';

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
    // Load saved user preferences
    _loadSavedSettings();
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

  /// Load saved user settings from SharedPreferences
  Future<void> _loadSavedSettings() async {
    try {
      final prefs = await SharedPreferences.getInstance();

      // Load FFT size preference (default 65536 if not set)
      final savedFftSize = prefs.getInt('waterfall_fft_size');
      if (savedFftSize != null) {
        // Validate it's a valid FFT size
        const validSizes = [8192, 16384, 32768, 65536];
        if (validSizes.contains(savedFftSize)) {
          ref.read(waterfallFftSizeProvider.notifier).state = savedFftSize;
          debugPrint('[App] Loaded saved FFT size: $savedFftSize');
        }
      }

      // Load score threshold preference (default 0.5 if not set)
      final savedThreshold = prefs.getDouble('score_threshold');
      if (savedThreshold != null) {
        ref.read(scoreThresholdProvider.notifier).state = savedThreshold;
        debugPrint('[App] Loaded saved score threshold: $savedThreshold');
      }

    } catch (e) {
      debugPrint('[App] Failed to load saved settings: $e');
    }
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
        Widget content = child ?? const SizedBox();

        // Show backend status banner if not running
        if (backendState.state == BackendState.error) {
          content = Column(
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
              Expanded(child: content),
            ],
          );
        }

        // Enforce 16:9 aspect ratio with black letterboxing
        return Container(
          color: Colors.black,  // Letterbox color
          child: Center(
            child: AspectRatio(
              aspectRatio: 16 / 9,
              child: content,
            ),
          ),
        );
      },
    );
  }
}
