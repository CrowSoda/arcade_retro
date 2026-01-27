import 'dart:async';

import 'package:flutter/foundation.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../../core/services/g20_api_service.dart';
import '../../live_detection/providers/sdr_config_provider.dart';
import '../../live_detection/providers/rx_state_provider.dart';

/// Tuning operation mode
enum TuningMode {
  /// Automatic scanning - system controls tuning
  auto,

  /// Manual control - user controls tuning, RX2 assigned
  manual,
}

/// Tuning state model
class TuningState {
  /// Current operating mode
  final TuningMode mode;

  /// Selected timeout in seconds (null = permanent manual)
  final int? timeoutSeconds;

  /// Remaining seconds until auto-resume (0 when in auto mode)
  final int remainingSeconds;

  /// Whether RX2 is assigned to manual viewing
  final bool rx2InUse;

  /// Whether a tuning operation is in progress
  final bool isTuning;

  /// Last error message (null if no error)
  final String? errorMessage;

  /// Timestamp of last state change
  final DateTime lastUpdated;

  const TuningState({
    this.mode = TuningMode.auto,
    this.timeoutSeconds,
    this.remainingSeconds = 0,
    this.rx2InUse = false,
    this.isTuning = false,
    this.errorMessage,
    required this.lastUpdated,
  });

  TuningState copyWith({
    TuningMode? mode,
    int? timeoutSeconds,
    bool clearTimeout = false,
    int? remainingSeconds,
    bool? rx2InUse,
    bool? isTuning,
    String? errorMessage,
    bool clearError = false,
    DateTime? lastUpdated,
  }) {
    return TuningState(
      mode: mode ?? this.mode,
      timeoutSeconds: clearTimeout ? null : (timeoutSeconds ?? this.timeoutSeconds),
      remainingSeconds: remainingSeconds ?? this.remainingSeconds,
      rx2InUse: rx2InUse ?? this.rx2InUse,
      isTuning: isTuning ?? this.isTuning,
      errorMessage: clearError ? null : (errorMessage ?? this.errorMessage),
      lastUpdated: lastUpdated ?? DateTime.now(),
    );
  }

  /// Check if in manual mode with active countdown
  bool get hasActiveCountdown => mode == TuningMode.manual && remainingSeconds > 0;

  /// Check if in permanent manual mode (no timeout)
  bool get isPermanentManual => mode == TuningMode.manual && timeoutSeconds == null;

  /// Format remaining time as MM:SS
  String get remainingTimeFormatted {
    if (remainingSeconds <= 0) return '00:00';
    final min = remainingSeconds ~/ 60;
    final sec = remainingSeconds % 60;
    return '${min.toString().padLeft(2, '0')}:${sec.toString().padLeft(2, '0')}';
  }

  /// Get mode display string
  String get modeDisplayString {
    if (mode == TuningMode.auto) {
      return '🤖 AUTO';
    } else if (isPermanentManual) {
      return '✋ MANUAL';
    } else {
      return '✋ ${remainingSeconds}s';
    }
  }
}

/// Tuning state notifier - manages auto/manual mode transitions
class TuningStateNotifier extends StateNotifier<TuningState> {
  final G20ApiService _apiService;
  final Ref _ref;
  Timer? _countdownTimer;

  TuningStateNotifier(this._apiService, this._ref)
      : super(TuningState(lastUpdated: DateTime.now()));

  @override
  void dispose() {
    _countdownTimer?.cancel();
    super.dispose();
  }

  /// Set manual mode with optional timeout
  ///
  /// [timeoutSeconds] - Seconds until auto-resume (null = permanent manual)
  /// Typical values: 60, 120, 300, or null
  Future<bool> setManualMode(int? timeoutSeconds) async {
    debugPrint('[TuningState] Setting manual mode (timeout: ${timeoutSeconds ?? "permanent"})');

    // Cancel any existing countdown
    _countdownTimer?.cancel();

    // Update state to show we're transitioning
    state = state.copyWith(
      isTuning: true,
      clearError: true,
    );

    // Assign RX2 to manual mode
    final result = await _apiService.assignRx2ToManual();

    if (!result.success) {
      state = state.copyWith(
        isTuning: false,
        errorMessage: result.errorMessage ?? 'Failed to assign RX2',
      );
      return false;
    }

    // Update state to manual mode
    state = state.copyWith(
      mode: TuningMode.manual,
      timeoutSeconds: timeoutSeconds,
      remainingSeconds: timeoutSeconds ?? 0,
      rx2InUse: true,
      isTuning: false,
      lastUpdated: DateTime.now(),
    );

    // Start countdown timer if we have a timeout
    if (timeoutSeconds != null && timeoutSeconds > 0) {
      _startCountdown();
    }

    debugPrint('[TuningState] Manual mode active, RX2 assigned');
    return true;
  }

  /// Resume auto mode - restores RX2 to its saved state before manual mode
  Future<bool> resumeAuto() async {
    debugPrint('[TuningState] Resuming auto mode...');

    // Cancel countdown
    _countdownTimer?.cancel();

    // Update state to show we're transitioning
    state = state.copyWith(
      isTuning: true,
      clearError: true,
    );

    // Release RX2 via API
    final result = await _apiService.releaseRx2();

    if (!result.success) {
      state = state.copyWith(
        isTuning: false,
        errorMessage: result.errorMessage ?? 'Failed to release RX2',
      );
      return false;
    }

    // RESTORE RX2 to saved state (before manual mode)
    // This handles the stub simulation and will call libsidekiq in production
    _ref.read(multiRxProvider.notifier).rx2ResumeToSaved();

    // Update state to auto mode
    state = state.copyWith(
      mode: TuningMode.auto,
      clearTimeout: true,
      remainingSeconds: 0,
      rx2InUse: false,
      isTuning: false,
      lastUpdated: DateTime.now(),
    );

    debugPrint('[TuningState] Auto mode active, RX2 restored to saved state');
    return true;
  }

  /// Update tuning parameters (center freq, bandwidth)
  ///
  /// Also updates the SDR config provider for display consistency
  Future<bool> updateTuning({
    required double centerMHz,
    required double bwMHz,
  }) async {
    debugPrint('[TuningState] Updating tuning: $centerMHz MHz, BW: $bwMHz MHz');

    state = state.copyWith(isTuning: true, clearError: true);

    final result = await _apiService.setTuning(
      centerMHz: centerMHz,
      bwMHz: bwMHz,
    );

    if (!result.success) {
      state = state.copyWith(
        isTuning: false,
        errorMessage: result.errorMessage ?? 'Tuning failed',
      );
      return false;
    }

    // Update SDR config provider to keep display in sync
    _ref.read(sdrConfigProvider.notifier).setFrequency(centerMHz);
    _ref.read(sdrConfigProvider.notifier).setBandwidth(bwMHz);

    state = state.copyWith(
      isTuning: false,
      lastUpdated: DateTime.now(),
    );

    debugPrint('[TuningState] Tuning updated successfully');
    return true;
  }

  /// Clear error message
  void clearError() {
    state = state.copyWith(clearError: true);
  }

  /// Start countdown timer
  void _startCountdown() {
    _countdownTimer?.cancel();
    _countdownTimer = Timer.periodic(const Duration(seconds: 1), (timer) {
      if (state.remainingSeconds <= 1) {
        timer.cancel();
        // Auto-resume when countdown reaches zero
        debugPrint('[TuningState] Countdown complete, auto-resuming...');
        resumeAuto();
      } else {
        state = state.copyWith(
          remainingSeconds: state.remainingSeconds - 1,
        );
      }
    });
  }

  /// Extend timeout by additional seconds
  void extendTimeout(int additionalSeconds) {
    if (state.mode != TuningMode.manual) return;

    state = state.copyWith(
      remainingSeconds: state.remainingSeconds + additionalSeconds,
    );

    // Restart timer if it wasn't running (was in permanent mode)
    if (_countdownTimer == null || !_countdownTimer!.isActive) {
      _startCountdown();
    }
  }

  /// Switch to permanent manual (cancel timeout)
  void setPermanentManual() {
    if (state.mode != TuningMode.manual) return;

    _countdownTimer?.cancel();
    state = state.copyWith(
      clearTimeout: true,
      remainingSeconds: 0,
    );
    debugPrint('[TuningState] Switched to permanent manual mode');
  }
}

// ============================================================
// RIVERPOD PROVIDERS
// ============================================================

/// Tuning state provider
final tuningStateProvider = StateNotifierProvider<TuningStateNotifier, TuningState>((ref) {
  final apiService = ref.watch(g20ApiServiceProvider);
  return TuningStateNotifier(apiService, ref);
});

/// Current tuning mode (convenience provider)
final tuningModeProvider = Provider<TuningMode>((ref) {
  return ref.watch(tuningStateProvider).mode;
});

/// Whether system is in manual mode (convenience provider)
final isManualModeProvider = Provider<bool>((ref) {
  return ref.watch(tuningStateProvider).mode == TuningMode.manual;
});

/// Remaining countdown seconds (convenience provider)
final tuningCountdownProvider = Provider<int>((ref) {
  return ref.watch(tuningStateProvider).remainingSeconds;
});

/// Mode display string (convenience provider)
final tuningModeDisplayProvider = Provider<String>((ref) {
  return ref.watch(tuningStateProvider).modeDisplayString;
});

// ============================================================
// MANUAL CAPTURE PROVIDER (for RecordingIndicator)
// ============================================================

/// Capture phase enum
enum CapturePhase {
  /// Not capturing
  idle,

  /// Preparing to capture
  preparing,

  /// Actively capturing
  capturing,

  /// Finalizing capture
  finalizing,
}

/// Manual capture state model
class ManualCaptureState {
  /// Current capture phase
  final CapturePhase phase;

  /// Signal name being captured
  final String? signalName;

  /// Capture progress (0.0 to 1.0)
  final double captureProgress;

  /// Total capture duration in minutes
  final int captureDurationMinutes;

  /// Number of captures in queue
  final int queueLength;

  /// Error message if any
  final String? errorMessage;

  const ManualCaptureState({
    this.phase = CapturePhase.idle,
    this.signalName,
    this.captureProgress = 0.0,
    this.captureDurationMinutes = 1,
    this.queueLength = 0,
    this.errorMessage,
  });

  ManualCaptureState copyWith({
    CapturePhase? phase,
    String? signalName,
    double? captureProgress,
    int? captureDurationMinutes,
    int? queueLength,
    String? errorMessage,
    bool clearSignal = false,
    bool clearError = false,
  }) {
    return ManualCaptureState(
      phase: phase ?? this.phase,
      signalName: clearSignal ? null : (signalName ?? this.signalName),
      captureProgress: captureProgress ?? this.captureProgress,
      captureDurationMinutes: captureDurationMinutes ?? this.captureDurationMinutes,
      queueLength: queueLength ?? this.queueLength,
      errorMessage: clearError ? null : (errorMessage ?? this.errorMessage),
    );
  }
}

/// Manual capture state notifier
class ManualCaptureNotifier extends StateNotifier<ManualCaptureState> {
  ManualCaptureNotifier() : super(const ManualCaptureState());

  /// Start a capture
  void startCapture({
    required String signalName,
    int durationMinutes = 1,
  }) {
    state = state.copyWith(
      phase: CapturePhase.capturing,
      signalName: signalName,
      captureProgress: 0.0,
      captureDurationMinutes: durationMinutes,
      clearError: true,
    );
  }

  /// Update capture progress
  void updateProgress(double progress) {
    state = state.copyWith(captureProgress: progress.clamp(0.0, 1.0));
  }

  /// Cancel the current capture
  void cancel() {
    debugPrint('[ManualCapture] Cancelling capture');
    state = state.copyWith(
      phase: CapturePhase.idle,
      captureProgress: 0.0,
      clearSignal: true,
    );
  }

  /// Complete the capture
  void complete() {
    state = state.copyWith(
      phase: CapturePhase.idle,
      captureProgress: 0.0,
      clearSignal: true,
    );
  }

  /// Update queue length
  void setQueueLength(int length) {
    state = state.copyWith(queueLength: length);
  }

  /// Set error
  void setError(String message) {
    state = state.copyWith(
      phase: CapturePhase.idle,
      errorMessage: message,
    );
  }
}

/// Manual capture provider
final manualCaptureProvider =
    StateNotifierProvider<ManualCaptureNotifier, ManualCaptureState>((ref) {
  return ManualCaptureNotifier();
});
