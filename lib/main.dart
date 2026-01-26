import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:media_kit/media_kit.dart';
import 'app.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  
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
    // Log other errors normally
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
