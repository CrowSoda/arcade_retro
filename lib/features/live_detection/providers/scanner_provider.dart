// lib/features/live_detection/providers/scanner_provider.dart
/// Scanner provider - Steps through frequency ranges with dwell time
/// 
/// Example: Range 2400-2480 MHz at 20 MHz BW, 5s dwell
/// Step 1: 2410 MHz (2400-2420), wait 5s
/// Step 2: 2430 MHz (2420-2440), wait 5s
/// Step 3: 2450 MHz (2440-2460), wait 5s
/// Step 4: 2470 MHz (2460-2480), wait 5s
/// Loop back to step 1

import 'dart:async';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../../config/config_screen.dart' show Mission, FreqRange;
import 'sdr_config_provider.dart';
import 'waterfall_provider.dart';
import 'rx_state_provider.dart';

/// A single scan step (center frequency)
class ScanStep {
  final double centerMHz;
  final double bandwidthMHz;
  final int rangeIndex;  // Which frequency range this belongs to
  final int stepInRange; // Step number within the range

  const ScanStep({
    required this.centerMHz,
    required this.bandwidthMHz,
    required this.rangeIndex,
    required this.stepInRange,
  });
  
  @override
  String toString() => 'Step ${rangeIndex + 1}.${stepInRange + 1}: ${centerMHz.toInt()} MHz';
}

/// Scanner state
class ScannerState {
  final bool isScanning;
  final List<ScanStep> steps;
  final int currentStepIndex;
  final double dwellTimeSec;
  final double timeRemaining;  // Seconds until next step
  final String? missionName;

  const ScannerState({
    this.isScanning = false,
    this.steps = const [],
    this.currentStepIndex = 0,
    this.dwellTimeSec = 5.0,
    this.timeRemaining = 0,
    this.missionName,
  });

  ScanStep? get currentStep => 
      steps.isNotEmpty && currentStepIndex < steps.length 
          ? steps[currentStepIndex] 
          : null;

  int get totalSteps => steps.length;
  
  String get statusText {
    if (!isScanning) return 'IDLE';
    if (steps.isEmpty) return 'NO STEPS';
    return 'Step ${currentStepIndex + 1}/${steps.length}';
  }

  ScannerState copyWith({
    bool? isScanning,
    List<ScanStep>? steps,
    int? currentStepIndex,
    double? dwellTimeSec,
    double? timeRemaining,
    String? missionName,
  }) => ScannerState(
    isScanning: isScanning ?? this.isScanning,
    steps: steps ?? this.steps,
    currentStepIndex: currentStepIndex ?? this.currentStepIndex,
    dwellTimeSec: dwellTimeSec ?? this.dwellTimeSec,
    timeRemaining: timeRemaining ?? this.timeRemaining,
    missionName: missionName ?? this.missionName,
  );
}

/// Scanner notifier - manages stepped frequency scanning
class ScannerNotifier extends StateNotifier<ScannerState> {
  final Ref _ref;
  Timer? _dwellTimer;
  Timer? _countdownTimer;

  ScannerNotifier(this._ref) : super(const ScannerState());

  /// Load a mission and calculate all scan steps
  void loadMission(Mission mission) {
    stopScanning();
    
    final steps = <ScanStep>[];
    
    for (int rangeIdx = 0; rangeIdx < mission.freqRanges.length; rangeIdx++) {
      final range = mission.freqRanges[rangeIdx];
      final stepsForRange = _calculateStepsForRange(
        range, 
        mission.bandwidthMhz, 
        rangeIdx,
      );
      steps.addAll(stepsForRange);
    }
    
    state = state.copyWith(
      steps: steps,
      currentStepIndex: 0,
      dwellTimeSec: mission.dwellTimeSec,
      missionName: mission.name,
    );
    
    debugPrint('[Scanner] Loaded mission "${mission.name}" with ${steps.length} steps');
    for (final step in steps) {
      debugPrint('[Scanner]   $step');
    }
  }

  /// Calculate steps to cover a frequency range
  List<ScanStep> _calculateStepsForRange(FreqRange range, double bandwidthMHz, int rangeIndex) {
    final steps = <ScanStep>[];
    final rangeWidth = range.endMhz - range.startMhz;
    
    // Number of steps needed to cover the range
    // Each step covers `bandwidthMHz` of spectrum
    // Center of first step is at startMhz + bandwidthMHz/2
    final numSteps = (rangeWidth / bandwidthMHz).ceil();
    
    for (int i = 0; i < numSteps; i++) {
      // Center frequency for this step
      // First step: startMhz + bw/2
      // Second step: startMhz + bw + bw/2
      // etc.
      final centerMHz = range.startMhz + (bandwidthMHz / 2) + (i * bandwidthMHz);
      
      // Don't exceed end of range
      if (centerMHz - bandwidthMHz / 2 >= range.endMhz) break;
      
      steps.add(ScanStep(
        centerMHz: centerMHz,
        bandwidthMHz: bandwidthMHz,
        rangeIndex: rangeIndex,
        stepInRange: i,
      ));
    }
    
    return steps;
  }

  /// Start scanning through steps
  void startScanning() {
    if (state.steps.isEmpty) {
      debugPrint('[Scanner] Cannot start - no steps loaded');
      return;
    }
    
    state = state.copyWith(
      isScanning: true,
      currentStepIndex: 0,
      timeRemaining: state.dwellTimeSec,
    );
    
    // Tune to first step immediately
    _tuneToCurrentStep();
    
    // SINGLE FREQUENCY OPTIMIZATION: Don't start dwell timer if only one step
    // This avoids unnecessary API calls when there's nothing to scan through
    if (state.steps.length > 1) {
      _startDwellTimer();
      debugPrint('[Scanner] Started scanning with ${state.dwellTimeSec}s dwell (${state.steps.length} steps)');
    } else {
      debugPrint('[Scanner] Single frequency mode - no dwell timer needed');
    }
  }

  /// Stop scanning
  void stopScanning() {
    _dwellTimer?.cancel();
    _countdownTimer?.cancel();
    _dwellTimer = null;
    _countdownTimer = null;
    
    state = state.copyWith(
      isScanning: false,
      timeRemaining: 0,
    );
    
    debugPrint('[Scanner] Stopped scanning');
  }

  /// Move to next step manually (or called by timer)
  void nextStep() {
    if (state.steps.isEmpty) return;
    
    final nextIndex = (state.currentStepIndex + 1) % state.steps.length;
    
    state = state.copyWith(
      currentStepIndex: nextIndex,
      timeRemaining: state.dwellTimeSec,
    );
    
    _tuneToCurrentStep();
  }

  /// Move to previous step manually
  void previousStep() {
    if (state.steps.isEmpty) return;
    
    final prevIndex = state.currentStepIndex == 0 
        ? state.steps.length - 1 
        : state.currentStepIndex - 1;
    
    state = state.copyWith(
      currentStepIndex: prevIndex,
      timeRemaining: state.dwellTimeSec,
    );
    
    _tuneToCurrentStep();
  }

  /// Jump to a specific step
  void goToStep(int stepIndex) {
    if (stepIndex < 0 || stepIndex >= state.steps.length) return;
    
    state = state.copyWith(
      currentStepIndex: stepIndex,
      timeRemaining: state.dwellTimeSec,
    );
    
    _tuneToCurrentStep();
  }

  void _startDwellTimer() {
    _dwellTimer?.cancel();
    _countdownTimer?.cancel();
    
    // Main dwell timer - fires when it's time to move to next step
    _dwellTimer = Timer.periodic(
      Duration(milliseconds: (state.dwellTimeSec * 1000).round()),
      (_) => nextStep(),
    );
    
    // Countdown timer - updates every 100ms for smooth countdown display
    _countdownTimer = Timer.periodic(
      const Duration(milliseconds: 100),
      (_) {
        final remaining = state.timeRemaining - 0.1;
        if (remaining > 0) {
          state = state.copyWith(timeRemaining: remaining);
        }
      },
    );
  }

  void _tuneToCurrentStep() {
    final step = state.currentStep;
    if (step == null) return;
    
    // Update SDR config
    _ref.read(sdrConfigProvider.notifier).setFrequency(step.centerMHz);
    _ref.read(sdrConfigProvider.notifier).setBandwidth(step.bandwidthMHz);
    
    // Update waterfall
    _ref.read(waterfallProvider.notifier).setCenterFrequency(step.centerMHz);
    _ref.read(waterfallProvider.notifier).setBandwidth(step.bandwidthMHz);
    
    // Update RX1 status
    _ref.read(multiRxProvider.notifier).setRxScanning(1, step.centerMHz, step.bandwidthMHz);
    
    debugPrint('[Scanner] Tuned to $step');
  }

  @override
  void dispose() {
    _dwellTimer?.cancel();
    _countdownTimer?.cancel();
    super.dispose();
  }
}

/// Scanner provider
final scannerProvider = StateNotifierProvider<ScannerNotifier, ScannerState>((ref) {
  return ScannerNotifier(ref);
});

/// Convenience provider for current step
final currentScanStepProvider = Provider<ScanStep?>((ref) {
  return ref.watch(scannerProvider).currentStep;
});

/// Convenience provider for scanning status
final isScanningProvider = Provider<bool>((ref) {
  return ref.watch(scannerProvider).isScanning;
});
