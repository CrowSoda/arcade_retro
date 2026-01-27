/// Settings Providers - Centralized state management for all app settings
///
/// This file contains all settings-related StateProviders with PERSISTENCE
/// via SharedPreferences. All settings are saved on change and loaded on startup.
library;

import 'package:flutter/foundation.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:shared_preferences/shared_preferences.dart';
import '../../live_detection/providers/video_stream_provider.dart';
import '../../../core/services/backend_launcher.dart';

// ============================================================================
// SETTINGS PERSISTENCE HELPER
// ============================================================================

/// Async SharedPreferences instance - initialized once
Future<SharedPreferences>? _prefsInstance;
Future<SharedPreferences> _getPrefs() {
  _prefsInstance ??= SharedPreferences.getInstance();
  return _prefsInstance!;
}

/// Helper to load a setting synchronously on provider creation
/// Falls back to default if not found
T _loadSetting<T>(String key, T defaultValue) {
  // Use a sync approach - prefs are cached after first load
  // This returns the default initially, then updates when prefs are ready
  return defaultValue;
}

/// Helper notifier that persists its value to SharedPreferences
class PersistentSettingNotifier<T> extends StateNotifier<T> {
  final String _key;
  final T _defaultValue;

  PersistentSettingNotifier(this._key, this._defaultValue) : super(_defaultValue) {
    _loadFromDisk();
  }

  Future<void> _loadFromDisk() async {
    try {
      final prefs = await _getPrefs();
      final T? loaded;

      if (_defaultValue is int) {
        loaded = prefs.getInt(_key) as T?;
      } else if (_defaultValue is double) {
        loaded = prefs.getDouble(_key) as T?;
      } else if (_defaultValue is bool) {
        loaded = prefs.getBool(_key) as T?;
      } else if (_defaultValue is String) {
        loaded = prefs.getString(_key) as T?;
      } else {
        loaded = null;
      }

      if (loaded != null) {
        state = loaded;
        debugPrint('[Settings] Loaded $_key = $loaded');
      }
    } catch (e) {
      debugPrint('[Settings] Error loading $_key: $e');
    }
  }

  Future<void> _saveToDisk(T value) async {
    try {
      final prefs = await _getPrefs();

      if (value is int) {
        await prefs.setInt(_key, value);
      } else if (value is double) {
        await prefs.setDouble(_key, value);
      } else if (value is bool) {
        await prefs.setBool(_key, value);
      } else if (value is String) {
        await prefs.setString(_key, value);
      }

      debugPrint('[Settings] Saved $_key = $value');
    } catch (e) {
      debugPrint('[Settings] Error saving $_key: $e');
    }
  }

  void setValue(T value) {
    state = value;
    _saveToDisk(value);
  }
}

/// Nullable int notifier (for optional settings like auto-tune delay)
class PersistentNullableIntNotifier extends StateNotifier<int?> {
  final String _key;
  final int? _defaultValue;

  PersistentNullableIntNotifier(this._key, this._defaultValue) : super(_defaultValue) {
    _loadFromDisk();
  }

  Future<void> _loadFromDisk() async {
    try {
      final prefs = await _getPrefs();
      if (prefs.containsKey(_key)) {
        final value = prefs.getInt(_key);
        state = value == -1 ? null : value;  // -1 stored as null
        debugPrint('[Settings] Loaded $_key = $state');
      }
    } catch (e) {
      debugPrint('[Settings] Error loading $_key: $e');
    }
  }

  void setValue(int? value) {
    state = value;
    _saveToDisk(value);
  }

  Future<void> _saveToDisk(int? value) async {
    try {
      final prefs = await _getPrefs();
      await prefs.setInt(_key, value ?? -1);  // Store null as -1
      debugPrint('[Settings] Saved $_key = $value');
    } catch (e) {
      debugPrint('[Settings] Error saving $_key: $e');
    }
  }
}

// ============================================================================
// AUTO-TUNE SETTINGS
// ============================================================================

/// Auto-tune delay setting - DEFAULT 2 seconds
/// Automatically tunes to frequency after user stops typing
/// null = disabled, otherwise seconds delay
final autoTuneDelayProvider = StateNotifierProvider<PersistentNullableIntNotifier, int?>((ref) {
  return PersistentNullableIntNotifier('g20_autotune_delay', 2);
});

// ============================================================================
// INFERENCE SETTINGS
// ============================================================================

/// Score threshold for detection filtering (0.0 - 1.0)
/// Default 0.9 (90%) - higher confidence for production use
final scoreThresholdProvider = StateNotifierProvider<PersistentSettingNotifier<double>, double>((ref) {
  return PersistentSettingNotifier<double>('g20_score_threshold', 0.9);
});

// ============================================================================
// WATERFALL DISPLAY SETTINGS
// ============================================================================

/// Skip first waterfall frame on connection (avoids garbage/initialization data)
final skipFirstWaterfallFrameProvider = StateNotifierProvider<PersistentSettingNotifier<bool>, bool>((ref) {
  return PersistentSettingNotifier<bool>('g20_skip_first_frame', false);
});

/// Show stats overlay on waterfall (FPS, resolution, row count)
/// Default false - clean display for demos/production
final showStatsOverlayProvider = StateNotifierProvider<PersistentSettingNotifier<bool>, bool>((ref) {
  return PersistentSettingNotifier<bool>('g20_show_stats', false);
});

/// Waterfall min dB setting - noise floor display
final waterfallMinDbProvider = StateNotifierProvider<PersistentSettingNotifier<double>, double>((ref) {
  return PersistentSettingNotifier<double>('g20_waterfall_min_db', -100.0);
});

/// Waterfall max dB setting - peak display
final waterfallMaxDbProvider = StateNotifierProvider<PersistentSettingNotifier<double>, double>((ref) {
  return PersistentSettingNotifier<double>('g20_waterfall_max_db', -20.0);
});

/// Waterfall FFT size setting
/// Controls frequency resolution vs performance tradeoff
/// Default 65536 (maximum resolution, ~10ms GPU)
/// Valid sizes: 8192, 16384, 32768, 65536
final waterfallFftSizeProvider = StateNotifierProvider<PersistentSettingNotifier<int>, int>((ref) {
  return PersistentSettingNotifier<int>('g20_waterfall_fft_size', 65536);
});

/// Waterfall display time span in seconds
/// FIXED at 2.5s for optimal performance - larger buffers cause FPS drops
final waterfallTimeSpanProvider = StateNotifierProvider<PersistentSettingNotifier<double>, double>((ref) {
  return PersistentSettingNotifier<double>('g20_waterfall_time_span', 2.5);
});

/// Waterfall FPS setting (frames per second)
/// Controls how fast the waterfall streams
/// Default 30fps - full speed
final waterfallFpsProvider = StateNotifierProvider<PersistentSettingNotifier<int>, int>((ref) {
  return PersistentSettingNotifier<int>('g20_waterfall_fps', 30);
});

/// Colormap setting for waterfall display
/// 0=Viridis, 1=Plasma, 2=Inferno, 3=Magma, 4=Turbo
final waterfallColormapProvider = StateNotifierProvider<PersistentSettingNotifier<int>, int>((ref) {
  return PersistentSettingNotifier<int>('g20_waterfall_colormap', 0);
});

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
