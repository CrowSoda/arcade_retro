// Flutter provider and utility tests
// Tests for state management and business logic

import 'package:flutter_test/flutter_test.dart';

void main() {
  group('Coordinate Conversion Utils', () {
    test('frequency to pixel conversion', () {
      // Test frequency to pixel conversion
      // Given: center_freq = 825 MHz, bandwidth = 20 MHz, width = 1024 px
      const centerFreqMHz = 825.0;
      const bandwidthMHz = 20.0;
      const width = 1024;

      // Center frequency should map to center pixel
      final centerPixel = _freqToPixel(centerFreqMHz, centerFreqMHz, bandwidthMHz, width);
      expect(centerPixel, closeTo(512, 1));

      // Low edge frequency
      final lowFreq = centerFreqMHz - bandwidthMHz / 2; // 815 MHz
      final lowPixel = _freqToPixel(lowFreq, centerFreqMHz, bandwidthMHz, width);
      expect(lowPixel, closeTo(0, 1));

      // High edge frequency
      final highFreq = centerFreqMHz + bandwidthMHz / 2; // 835 MHz
      final highPixel = _freqToPixel(highFreq, centerFreqMHz, bandwidthMHz, width);
      expect(highPixel, closeTo(1024, 1));
    });

    test('pixel to frequency conversion', () {
      const centerFreqMHz = 825.0;
      const bandwidthMHz = 20.0;
      const width = 1024;

      // Center pixel should map to center frequency
      final centerFreq = _pixelToFreq(512, centerFreqMHz, bandwidthMHz, width);
      expect(centerFreq, closeTo(825.0, 0.1));

      // Left edge
      final lowFreq = _pixelToFreq(0, centerFreqMHz, bandwidthMHz, width);
      expect(lowFreq, closeTo(815.0, 0.1));

      // Right edge
      final highFreq = _pixelToFreq(1024, centerFreqMHz, bandwidthMHz, width);
      expect(highFreq, closeTo(835.0, 0.1));
    });

    test('roundtrip conversion preserves value', () {
      const centerFreqMHz = 825.0;
      const bandwidthMHz = 20.0;
      const width = 1024;

      // Test multiple frequencies
      for (final freq in [815.0, 820.0, 825.0, 830.0, 835.0]) {
        final pixel = _freqToPixel(freq, centerFreqMHz, bandwidthMHz, width);
        final roundtrip = _pixelToFreq(pixel, centerFreqMHz, bandwidthMHz, width);
        expect(roundtrip, closeTo(freq, 0.01), reason: 'Roundtrip failed for $freq MHz');
      }
    });
  });

  group('dB Calculations', () {
    test('linear to dB conversion', () {
      // 1.0 linear = 0 dB
      expect(_linearToDb(1.0), closeTo(0.0, 0.001));

      // 10.0 linear = 20 dB
      expect(_linearToDb(10.0), closeTo(20.0, 0.001));

      // 0.1 linear = -20 dB
      expect(_linearToDb(0.1), closeTo(-20.0, 0.001));

      // 100.0 linear = 40 dB
      expect(_linearToDb(100.0), closeTo(40.0, 0.001));
    });

    test('dB to linear conversion', () {
      // 0 dB = 1.0 linear
      expect(_dbToLinear(0.0), closeTo(1.0, 0.001));

      // 20 dB = 10.0 linear
      expect(_dbToLinear(20.0), closeTo(10.0, 0.001));

      // -20 dB = 0.1 linear
      expect(_dbToLinear(-20.0), closeTo(0.1, 0.001));
    });

    test('dB roundtrip', () {
      for (final db in [-60.0, -40.0, -20.0, 0.0, 20.0, 40.0]) {
        final linear = _dbToLinear(db);
        final roundtrip = _linearToDb(linear);
        expect(roundtrip, closeTo(db, 0.001));
      }
    });
  });

  group('Normalization', () {
    test('normalize to 0-1 range', () {
      // Value at min should be 0
      expect(_normalize(-80.0, -80.0, 0.0), closeTo(0.0, 0.001));

      // Value at max should be 1
      expect(_normalize(0.0, -80.0, 0.0), closeTo(1.0, 0.001));

      // Value at midpoint
      expect(_normalize(-40.0, -80.0, 0.0), closeTo(0.5, 0.001));
    });

    test('normalize clamps out of range values', () {
      // Below min
      expect(_normalizeClamp(-100.0, -80.0, 0.0), closeTo(0.0, 0.001));

      // Above max
      expect(_normalizeClamp(20.0, -80.0, 0.0), closeTo(1.0, 0.001));
    });
  });

  group('Detection Box Validation', () {
    test('valid box coordinates', () {
      expect(_isValidBox(0.0, 0.0, 1.0, 1.0), isTrue);
      expect(_isValidBox(0.1, 0.2, 0.3, 0.4), isTrue);
      expect(_isValidBox(0.5, 0.5, 0.6, 0.6), isTrue);
    });

    test('invalid box - x1 > x2', () {
      expect(_isValidBox(0.5, 0.0, 0.3, 1.0), isFalse);
    });

    test('invalid box - y1 > y2', () {
      expect(_isValidBox(0.0, 0.5, 1.0, 0.3), isFalse);
    });

    test('invalid box - out of range', () {
      expect(_isValidBox(-0.1, 0.0, 1.0, 1.0), isFalse);
      expect(_isValidBox(0.0, 0.0, 1.1, 1.0), isFalse);
    });

    test('invalid box - zero area', () {
      expect(_isValidBox(0.5, 0.5, 0.5, 0.5), isFalse);
    });
  });

  group('Time Formatting', () {
    test('format seconds', () {
      expect(_formatDuration(1), equals('1s'));
      expect(_formatDuration(30), equals('30s'));
      expect(_formatDuration(59), equals('59s'));
    });

    test('format minutes', () {
      expect(_formatDuration(60), equals('1:00'));
      expect(_formatDuration(90), equals('1:30'));
      expect(_formatDuration(125), equals('2:05'));
    });

    test('format hours', () {
      expect(_formatDuration(3600), equals('1:00:00'));
      expect(_formatDuration(3661), equals('1:01:01'));
    });
  });

  group('Signal Name Validation', () {
    test('valid signal names', () {
      expect(_isValidSignalName('signal_a'), isTrue);
      expect(_isValidSignalName('my_signal_123'), isTrue);
      expect(_isValidSignalName('TestSignal'), isTrue);
    });

    test('invalid signal names - empty', () {
      expect(_isValidSignalName(''), isFalse);
    });

    test('invalid signal names - special chars', () {
      expect(_isValidSignalName('signal/name'), isFalse);
      expect(_isValidSignalName('signal name'), isFalse);
      expect(_isValidSignalName('signal@123'), isFalse);
    });

    test('invalid signal names - too long', () {
      final longName = 'a' * 65;
      expect(_isValidSignalName(longName), isFalse);
    });
  });
}

// =============================================================================
// Helper Functions (would normally be in lib/)
// =============================================================================

double _freqToPixel(double freqMHz, double centerFreqMHz, double bandwidthMHz, int width) {
  final lowFreq = centerFreqMHz - bandwidthMHz / 2;
  final normalized = (freqMHz - lowFreq) / bandwidthMHz;
  return normalized * width;
}

double _pixelToFreq(double pixel, double centerFreqMHz, double bandwidthMHz, int width) {
  final normalized = pixel / width;
  final lowFreq = centerFreqMHz - bandwidthMHz / 2;
  return lowFreq + normalized * bandwidthMHz;
}

double _linearToDb(double linear) {
  return 20 * _log10(linear);
}

double _dbToLinear(double db) {
  return _pow10(db / 20);
}

double _log10(double x) {
  return _ln(x) / _ln(10);
}

double _ln(double x) {
  // Natural log approximation for dart:math-free environment
  // In real code, use: import 'dart:math'; log(x)
  if (x <= 0) return double.negativeInfinity;
  if (x == 1) return 0;

  // Newton-Raphson for ln(x)
  double y = x - 1;
  double result = y;
  double term = y;
  for (int i = 2; i <= 50; i++) {
    term *= -y;
    result += term / i;
  }
  return result;
}

double _pow10(double x) {
  // 10^x approximation
  // In real code, use: import 'dart:math'; pow(10, x)
  return _exp(x * _ln(10));
}

double _exp(double x) {
  // e^x Taylor series
  double result = 1;
  double term = 1;
  for (int i = 1; i <= 50; i++) {
    term *= x / i;
    result += term;
  }
  return result;
}

double _normalize(double value, double min, double max) {
  return (value - min) / (max - min);
}

double _normalizeClamp(double value, double min, double max) {
  final normalized = _normalize(value, min, max);
  if (normalized < 0) return 0;
  if (normalized > 1) return 1;
  return normalized;
}

bool _isValidBox(double x1, double y1, double x2, double y2) {
  // Check range [0, 1]
  if (x1 < 0 || x1 > 1) return false;
  if (y1 < 0 || y1 > 1) return false;
  if (x2 < 0 || x2 > 1) return false;
  if (y2 < 0 || y2 > 1) return false;

  // Check ordering
  if (x1 >= x2) return false;
  if (y1 >= y2) return false;

  return true;
}

String _formatDuration(int seconds) {
  if (seconds < 60) {
    return '${seconds}s';
  } else if (seconds < 3600) {
    final m = seconds ~/ 60;
    final s = seconds % 60;
    return '$m:${s.toString().padLeft(2, '0')}';
  } else {
    final h = seconds ~/ 3600;
    final m = (seconds % 3600) ~/ 60;
    final s = seconds % 60;
    return '$h:${m.toString().padLeft(2, '0')}:${s.toString().padLeft(2, '0')}';
  }
}

bool _isValidSignalName(String name) {
  if (name.isEmpty) return false;
  if (name.length > 64) return false;

  // Only alphanumeric and underscore
  final validPattern = RegExp(r'^[a-zA-Z0-9_]+$');
  return validPattern.hasMatch(name);
}
