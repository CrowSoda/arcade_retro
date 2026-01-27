import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

/// RX Channel mode
enum RxMode {
  scanning,   // Auto scanning (inference pipeline)
  manual,     // Manual tune (user commanded)
  idle,       // Not active
  error,      // Error state
}

/// Single RX channel state
class RxChannelState {
  final int rxNumber;       // 1-4
  final RxMode mode;
  final double centerFreqMHz;
  final double bandwidthMHz;
  final bool isConnected;
  final int? countdownSeconds;  // Countdown for manual mode timeout
  final String? errorMessage;

  const RxChannelState({
    required this.rxNumber,
    this.mode = RxMode.idle,
    this.centerFreqMHz = 825.0,
    this.bandwidthMHz = 20.0,
    this.isConnected = false,
    this.countdownSeconds,
    this.errorMessage,
  });

  RxChannelState copyWith({
    int? rxNumber,
    RxMode? mode,
    double? centerFreqMHz,
    double? bandwidthMHz,
    bool? isConnected,
    int? countdownSeconds,
    String? errorMessage,
  }) {
    return RxChannelState(
      rxNumber: rxNumber ?? this.rxNumber,
      mode: mode ?? this.mode,
      centerFreqMHz: centerFreqMHz ?? this.centerFreqMHz,
      bandwidthMHz: bandwidthMHz ?? this.bandwidthMHz,
      isConnected: isConnected ?? this.isConnected,
      countdownSeconds: countdownSeconds,  // Explicitly allow null
      errorMessage: errorMessage,
    );
  }

  /// Display string for mode
  String get modeDisplayString {
    switch (mode) {
      case RxMode.scanning:
        return '🤖 SCAN';
      case RxMode.manual:
        if (countdownSeconds != null) {
          return '🔴 ${countdownSeconds}s';  // Recording with countdown
        }
        return '🔴 REC';  // Manual recording (no countdown = permanent)
      case RxMode.idle:
        return '⏸️ IDLE';
      case RxMode.error:
        return '❌ ERR';
    }
  }

  /// Color for mode display
  Color get modeColor {
    switch (mode) {
      case RxMode.scanning:
        return const Color(0xFF4CAF50);  // Green
      case RxMode.manual:
        return const Color(0xFFFF9800);  // Orange/Warning
      case RxMode.idle:
        return const Color(0xFF9E9E9E);  // Grey
      case RxMode.error:
        return const Color(0xFFF44336);  // Red
    }
  }
}

/// Multi-RX state - supports up to 4 RX channels
class MultiRxState {
  final List<RxChannelState> channels;

  MultiRxState({List<RxChannelState>? channels})
      : channels = channels ?? const [];

  /// Get RX by number (1-indexed)
  RxChannelState? getRx(int rxNumber) {
    try {
      return channels.firstWhere((ch) => ch.rxNumber == rxNumber);
    } catch (_) {
      return null;
    }
  }

  /// Get all connected RX channels
  List<RxChannelState> get connectedChannels =>
      channels.where((ch) => ch.isConnected).toList();

  /// Get count of connected channels
  int get connectedCount => connectedChannels.length;

  MultiRxState copyWith({List<RxChannelState>? channels}) {
    return MultiRxState(channels: channels ?? this.channels);
  }
}

/// Multi-RX state notifier with state preservation for RX2 timeout restore
class MultiRxNotifier extends StateNotifier<MultiRxState> {
  /// Store RX2's previous state before manual mode (for timeout restore)
  RxChannelState? _rx2PreviousState;

  MultiRxNotifier() : super(MultiRxState()) {
    // Initialize with simulated 2 RX channels (RX1 scanning, RX2 idle)
    _initializeSimulated();
  }

  void _initializeSimulated() {
    state = MultiRxState(channels: [
      // RX1: Always scanning - inference pipeline (NEVER interrupted)
      const RxChannelState(
        rxNumber: 1,
        mode: RxMode.scanning,
        centerFreqMHz: 825.0,
        bandwidthMHz: 20.0,
        isConnected: true,
      ),
      // RX2: Initially idle - for manual tuning/collection
      const RxChannelState(
        rxNumber: 2,
        mode: RxMode.idle,
        centerFreqMHz: 825.0,
        bandwidthMHz: 20.0,
        isConnected: true,
      ),
    ]);
  }

  /// Update a specific RX channel
  void updateRx(int rxNumber, RxChannelState Function(RxChannelState) updater) {
    final updatedChannels = state.channels.map((ch) {
      if (ch.rxNumber == rxNumber) {
        return updater(ch);
      }
      return ch;
    }).toList();
    state = state.copyWith(channels: updatedChannels);
  }

  /// Set RX to manual mode
  void setRxManual(int rxNumber, double centerMHz, double bwMHz, int? timeoutSeconds) {
    updateRx(rxNumber, (ch) => ch.copyWith(
      mode: RxMode.manual,
      centerFreqMHz: centerMHz,
      bandwidthMHz: bwMHz,
      countdownSeconds: timeoutSeconds,
    ));

    // STUB: Simulate hardware tune
    _simulateHardwareTune(rxNumber, centerMHz, bwMHz);
  }

  /// Set RX to scanning mode
  void setRxScanning(int rxNumber, double centerMHz, double bwMHz) {
    updateRx(rxNumber, (ch) => ch.copyWith(
      mode: RxMode.scanning,
      centerFreqMHz: centerMHz,
      bandwidthMHz: bwMHz,
      countdownSeconds: null,
    ));

    // STUB: Simulate hardware tune
    _simulateHardwareTune(rxNumber, centerMHz, bwMHz);
  }

  /// Set RX to idle
  void setRxIdle(int rxNumber) {
    updateRx(rxNumber, (ch) => ch.copyWith(
      mode: RxMode.idle,
      countdownSeconds: null,
    ));
  }

  /// Update countdown for RX
  void updateCountdown(int rxNumber, int seconds) {
    updateRx(rxNumber, (ch) => ch.copyWith(countdownSeconds: seconds));
  }

  /// Tune RX2 (manual tuning channel) - SAVES previous state for timeout restore
  void tuneRx2(double centerMHz, double bwMHz, int? timeoutSeconds) {
    // SAVE previous state before switching to manual (for timeout restore)
    final currentRx2 = state.getRx(2);
    if (currentRx2 != null && currentRx2.mode != RxMode.manual) {
      _rx2PreviousState = currentRx2.copyWith();
      debugPrint('📻 Saved RX2 previous state: ${_rx2PreviousState?.centerFreqMHz} MHz, mode: ${_rx2PreviousState?.mode}');
    }

    setRxManual(2, centerMHz, bwMHz, timeoutSeconds);
    debugPrint('📻 RX2 tuned to manual: $centerMHz MHz, BW: $bwMHz MHz, timeout: ${timeoutSeconds ?? "∞"}s');
  }

  /// Resume RX2 to SAVED state after manual mode timeout
  /// If no saved state, falls back to idle
  void rx2ResumeToSaved() {
    if (_rx2PreviousState != null) {
      final prev = _rx2PreviousState!;
      debugPrint('📻 Restoring RX2 to saved state: ${prev.centerFreqMHz} MHz, mode: ${prev.mode}');

      if (prev.mode == RxMode.scanning) {
        setRxScanning(2, prev.centerFreqMHz, prev.bandwidthMHz);
      } else if (prev.mode == RxMode.idle) {
        setRxIdle(2);
      } else {
        // For other modes, just go to idle
        setRxIdle(2);
      }

      // STUB: Simulate hardware retune
      _simulateHardwareTune(2, prev.centerFreqMHz, prev.bandwidthMHz);

      _rx2PreviousState = null;
    } else {
      debugPrint('📻 No saved state for RX2, setting to idle');
      setRxIdle(2);
    }
  }

  /// Resume RX2 to idle after manual mode (legacy - use rx2ResumeToSaved for timeout)
  void rx2ResumeIdle() {
    rx2ResumeToSaved();
  }

  // =========================================================================
  // HARDWARE STUBS - Replace with libsidekiq calls in production
  // =========================================================================

  /// STUB: Simulate hardware tune command
  /// In production, this calls libsidekiq API to tune the SDR
  Future<void> _simulateHardwareTune(int rxNumber, double centerMHz, double bwMHz) async {
    debugPrint('📻 [STUB] Hardware tune RX$rxNumber -> $centerMHz MHz, BW: $bwMHz MHz');

    // Simulate tune time (real hardware takes 10-50ms)
    await Future.delayed(const Duration(milliseconds: 50));

    // TODO: Production implementation with libsidekiq:
    // try {
    //   await _sidekiqApi.tuneRx(
    //     rxNumber: rxNumber,
    //     centerFreqHz: (centerMHz * 1e6).toInt(),
    //     bandwidthHz: (bwMHz * 1e6).toInt(),
    //   );
    //   debugPrint('📻 Hardware tune RX$rxNumber complete');
    // } catch (e) {
    //   debugPrint('📻 Hardware tune RX$rxNumber FAILED: $e');
    //   updateRx(rxNumber, (ch) => ch.copyWith(
    //     mode: RxMode.error,
    //     errorMessage: 'Tune failed: $e',
    //   ));
    // }

    debugPrint('📻 [STUB] Hardware tune complete');
  }

  /// STUB: Get hardware status
  /// In production, this queries libsidekiq for actual RX status
  Future<Map<String, dynamic>> getHardwareStatus(int rxNumber) async {
    debugPrint('📻 [STUB] Getting hardware status for RX$rxNumber');

    // TODO: Production implementation:
    // return await _sidekiqApi.getRxStatus(rxNumber);

    // Return simulated status
    final rx = state.getRx(rxNumber);
    return {
      'rxNumber': rxNumber,
      'connected': rx?.isConnected ?? false,
      'centerFreqHz': ((rx?.centerFreqMHz ?? 0) * 1e6).toInt(),
      'bandwidthHz': ((rx?.bandwidthMHz ?? 0) * 1e6).toInt(),
      'temperature': 42.5,
      'rssi': -45.0,
      'stub': true,
    };
  }

  /// Add a new RX channel (for hardware that supports more)
  void addRx(int rxNumber) {
    if (state.getRx(rxNumber) != null) return;  // Already exists

    state = state.copyWith(
      channels: [
        ...state.channels,
        RxChannelState(rxNumber: rxNumber, isConnected: true),
      ],
    );
  }

  /// Remove an RX channel
  void removeRx(int rxNumber) {
    state = state.copyWith(
      channels: state.channels.where((ch) => ch.rxNumber != rxNumber).toList(),
    );
  }
}

/// Provider for multi-RX state
final multiRxProvider = StateNotifierProvider<MultiRxNotifier, MultiRxState>((ref) {
  return MultiRxNotifier();
});

/// Provider for specific RX channel (by number)
final rxChannelProvider = Provider.family<RxChannelState?, int>((ref, rxNumber) {
  return ref.watch(multiRxProvider).getRx(rxNumber);
});

/// Provider for connected RX count
final connectedRxCountProvider = Provider<int>((ref) {
  return ref.watch(multiRxProvider).connectedCount;
});
