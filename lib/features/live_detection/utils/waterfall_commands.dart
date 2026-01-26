/// Waterfall Commands - WebSocket command helpers for waterfall control
/// 
/// Provides type-safe command builders for backend communication
library;

import 'dart:convert';

/// Command types for waterfall backend control
enum WaterfallCommand {
  setFps,
  setFftSize,
  setColormap,
  setDbRange,
  setTimeSpan,
  setScoreThreshold,
}

/// Build a JSON command message for the backend
/// 
/// Returns encoded JSON string ready to send over WebSocket
class WaterfallCommandBuilder {
  /// Set waterfall FPS (frames per second)
  /// Valid range: 1-30
  static String setFps(int fps) {
    return json.encode({
      'command': 'set_fps',
      'fps': fps.clamp(1, 60),
    });
  }

  /// Set FFT size for waterfall processing
  /// Valid sizes: 8192, 16384, 32768, 65536
  static String setFftSize(int size) {
    const validSizes = [8192, 16384, 32768, 65536];
    final validatedSize = validSizes.contains(size) ? size : 65536;
    return json.encode({
      'command': 'set_fft_size',
      'size': validatedSize,
    });
  }

  /// Set colormap for waterfall display
  /// 0=Viridis, 1=Plasma, 2=Inferno, 3=Magma, 4=Turbo
  static String setColormap(int colormapIndex) {
    return json.encode({
      'command': 'set_colormap',
      'colormap': colormapIndex.clamp(0, 4),
    });
  }

  /// Set dB range for waterfall display normalization
  /// [minDb] - noise floor (e.g., -100)
  /// [maxDb] - peak level (e.g., -20)
  static String setDbRange(double minDb, double maxDb) {
    return json.encode({
      'command': 'set_db_range',
      'min_db': minDb,
      'max_db': maxDb,
    });
  }

  /// Set time span (how many seconds of data to show)
  static String setTimeSpan(double seconds) {
    return json.encode({
      'command': 'set_time_span',
      'seconds': seconds.clamp(0.1, 60.0),
    });
  }

  /// Set detection score threshold (0.0 - 1.0)
  static String setScoreThreshold(double threshold) {
    return json.encode({
      'command': 'set_score_threshold',
      'threshold': threshold.clamp(0.0, 1.0),
    });
  }
}

/// Constants for FFT size options
class FftSizeConstants {
  static const int fft8k = 8192;
  static const int fft16k = 16384;
  static const int fft32k = 32768;
  static const int fft64k = 65536;

  static const List<int> validSizes = [fft8k, fft16k, fft32k, fft64k];

  /// Get estimated processing time for FFT size
  static String estimatedTime(int size) {
    switch (size) {
      case fft8k: return '~2ms';
      case fft16k: return '~4ms';
      case fft32k: return '~6ms';
      case fft64k: return '~10ms';
      default: return '?';
    }
  }

  /// Get frequency resolution for FFT size at 20 MHz sample rate
  static String resolution(int size) {
    const sampleRate = 20000000;  // 20 MHz
    final hzPerBin = sampleRate / size;
    if (hzPerBin >= 1000) {
      return '${(hzPerBin / 1000).toStringAsFixed(1)} kHz/bin';
    }
    return '${hzPerBin.toInt()} Hz/bin';
  }

  /// Check if size is valid
  static bool isValid(int size) => validSizes.contains(size);
}

/// Constants for colormap options
class ColormapConstants {
  static const int viridis = 0;
  static const int plasma = 1;
  static const int inferno = 2;
  static const int magma = 3;
  static const int turbo = 4;

  static const List<String> names = ['Viridis', 'Plasma', 'Inferno', 'Magma', 'Turbo'];

  static String nameFor(int index) => names[index.clamp(0, names.length - 1)];
}
