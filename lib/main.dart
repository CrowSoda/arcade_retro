import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:media_kit/media_kit.dart';
import 'package:window_manager/window_manager.dart';
import 'app.dart';
import 'core/services/backend_launcher.dart';

/// ============================================================
/// CONSOLE SPAM CONTROL
/// ============================================================
/// Set to true to enable ALL debug logging
/// Set to false for clean console (only Flutter framework errors)
///
/// To enable temporarily: run with --dart-define=ENABLE_DEBUG_LOGS=true
const bool _enableDebugLogs = bool.fromEnvironment(
  'ENABLE_DEBUG_LOGS',
  defaultValue: false,  // CLEAN CONSOLE BY DEFAULT
);

/// Original debugPrint function (saved for optional restoration)
final _originalDebugPrint = debugPrint;

/// Silent debugPrint - does nothing
void _silentDebugPrint(String? message, {int? wrapWidth}) {
  // Intentionally empty - silences ALL debugPrint calls
}

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  // ============================================================
  // SILENCE ALL DEBUGPRINT CALLS BY DEFAULT
  // ============================================================
  // This globally silences ALL debugPrint() calls throughout the app
  // To enable debug logging, run with: --dart-define=ENABLE_DEBUG_LOGS=true
  if (!_enableDebugLogs) {
    debugPrint = _silentDebugPrint;
  }

  // Initialize MediaKit for video playback (H.264 hardware decoding)
  MediaKit.ensureInitialized();

  // Silence "Cancelled" exceptions from tile loading during zoom
  // These are expected when tiles are cancelled due to zoom changes
  FlutterError.onError = (FlutterErrorDetails details) {
    final exception = details.exception;
    if (exception.toString().contains('Cancelled')) {
      // Silently ignore cancelled tile loads
      return;
    }
    // Show actual errors
    FlutterError.presentError(details);
  };

  // Handle async errors (like tile loading cancellations)
  PlatformDispatcher.instance.onError = (error, stack) {
    if (error.toString().contains('Cancelled')) {
      // Silently ignore cancelled operations
      return true;
    }
    // Let other errors propagate
    return false;
  };

  // ============================================================
  // WINDOW MANAGER - Reliable backend cleanup on window close
  // ============================================================
  // Initialize window_manager for desktop platforms
  if (Platform.isWindows || Platform.isLinux || Platform.isMacOS) {
    await windowManager.ensureInitialized();

    // Prevent default close to allow cleanup
    await windowManager.setPreventClose(true);
  }

  // Create ProviderScope container with ref access
  final container = ProviderContainer();

  // Add window close listener for reliable backend cleanup
  if (Platform.isWindows || Platform.isLinux || Platform.isMacOS) {
    windowManager.addListener(_BackendCleanupListener(container));
  }

  runApp(
    UncontrolledProviderScope(
      container: container,
      child: const G20App(),
    ),
  );
}

/// Window listener that stops the backend on window close
class _BackendCleanupListener extends WindowListener {
  final ProviderContainer container;

  _BackendCleanupListener(this.container);

  @override
  void onWindowClose() async {
    // Stop the backend before closing
    debugPrint('[WindowManager] Window closing, stopping backend...');
    await container.read(backendLauncherProvider.notifier).stopBackend();

    // Allow the window to close
    await windowManager.destroy();
  }
}
