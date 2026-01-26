/// Settings Providers - Centralized state management for all app settings
/// 
/// This file contains all settings-related StateProviders, separated from
/// UI widgets for better SRP compliance.
library;

import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../../live_detection/providers/video_stream_provider.dart';
import '../../../core/services/backend_launcher.dart';

// ============================================================================
// AUTO-TUNE SETTINGS
// ============================================================================

/// Auto-tune delay setting - DEFAULT 2 seconds
/// Automatically tunes to frequency after user stops typing
/// null = disabled, otherwise seconds delay
final autoTuneDelayProvider = StateProvider<int?>((ref) => 2);

// ============================================================================
// INFERENCE SETTINGS  
// ============================================================================

/// Score threshold for detection filtering (0.0 - 1.0)
/// Default 0.9 (90%) - higher confidence for production use
final scoreThresholdProvider = StateProvider<double>((ref) => 0.9);

// ============================================================================
// WATERFALL DISPLAY SETTINGS
// ============================================================================

/// Skip first waterfall frame on connection (avoids garbage/initialization data)
final skipFirstWaterfallFrameProvider = StateProvider<bool>((ref) => false);

/// Waterfall min dB setting - noise floor display
final waterfallMinDbProvider = StateProvider<double>((ref) => -100.0);

/// Waterfall max dB setting - peak display
final waterfallMaxDbProvider = StateProvider<double>((ref) => -20.0);

/// Waterfall FFT size setting
/// Controls frequency resolution vs performance tradeoff
/// Default 65536 (maximum resolution, ~10ms GPU)
/// Valid sizes: 8192, 16384, 32768, 65536
final waterfallFftSizeProvider = StateProvider<int>((ref) => 65536);

/// Waterfall display time span in seconds
/// FIXED at 2.5s for optimal performance - larger buffers cause FPS drops
final waterfallTimeSpanProvider = StateProvider<double>((ref) => 2.5);

/// Waterfall FPS setting (frames per second)
/// Controls how fast the waterfall streams
/// Default 30fps - full speed
final waterfallFpsProvider = StateProvider<int>((ref) => 30);

/// Colormap setting for waterfall display
/// 0=Viridis, 1=Plasma, 2=Inferno, 3=Magma, 4=Turbo
final waterfallColormapProvider = StateProvider<int>((ref) => 0);

// ============================================================================
// CONNECTION/SYSTEM INFO
// ============================================================================

/// Backend version - derived from connection state
final backendVersionProvider = Provider<String>((ref) {
  final backendState = ref.watch(backendLauncherProvider);
  if (backendState.wsPort != null) {
    return backendState.version ?? '1.0.0 (connected)';
  }
  return 'Not Connected';
});

// ============================================================================
// CONSTANTS
// ============================================================================

/// FFT size options with descriptions
const Map<int, ({String label, String sublabel, String resolution})> fftSizeOptions = {
  8192: (label: '8K', sublabel: 'Fastest', resolution: '2441 Hz/bin'),
  16384: (label: '16K', sublabel: 'Fast', resolution: '1221 Hz/bin'),
  32768: (label: '32K', sublabel: 'Balanced', resolution: '610 Hz/bin'),
  65536: (label: '64K', sublabel: 'Detailed', resolution: '305 Hz/bin'),
};

/// Colormap names for display
const List<String> colormapNames = ['Viridis', 'Plasma', 'Inferno', 'Magma', 'Turbo'];
