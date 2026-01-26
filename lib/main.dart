import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:media_kit/media_kit.dart';
import 'app.dart';

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
  
  runApp(
    const ProviderScope(
      child: G20App(),
    ),
  );
}
