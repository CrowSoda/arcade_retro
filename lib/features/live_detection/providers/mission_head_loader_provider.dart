// lib/features/live_detection/providers/mission_head_loader_provider.dart
/// Watches mission changes and loads/unloads detection heads via Hydra
/// This provider bridges mission config to video stream head management

import 'package:flutter/foundation.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../../config/providers/mission_provider.dart';
import 'video_stream_provider.dart';

/// State for head loading
class MissionHeadState {
  final List<String> loadedHeads;
  final List<String> requestedHeads;
  final bool isLoading;
  final String? error;

  const MissionHeadState({
    this.loadedHeads = const [],
    this.requestedHeads = const [],
    this.isLoading = false,
    this.error,
  });

  MissionHeadState copyWith({
    List<String>? loadedHeads,
    List<String>? requestedHeads,
    bool? isLoading,
    String? error,
  }) {
    return MissionHeadState(
      loadedHeads: loadedHeads ?? this.loadedHeads,
      requestedHeads: requestedHeads ?? this.requestedHeads,
      isLoading: isLoading ?? this.isLoading,
      error: error,
    );
  }
}

/// Notifier that manages head loading based on mission changes
class MissionHeadLoaderNotifier extends StateNotifier<MissionHeadState> {
  final Ref _ref;
  String? _lastMissionPath;

  MissionHeadLoaderNotifier(this._ref) : super(const MissionHeadState()) {
    // Listen for mission changes
    _ref.listen<MissionState>(missionProvider, (previous, next) {
      _onMissionChanged(previous, next);
    });
    
    // Check initial state
    final currentMission = _ref.read(missionProvider).activeMission;
    if (currentMission != null) {
      _loadHeadsForMission(currentMission.effectiveSignals);
    }
  }

  void _onMissionChanged(MissionState? previous, MissionState next) {
    final mission = next.activeMission;
    
    // Mission cleared - unload all heads
    if (mission == null && previous?.activeMission != null) {
      debugPrint('[MissionHeadLoader] Mission cleared - unloading all heads');
      _unloadAllHeads();
      _lastMissionPath = null;
      return;
    }
    
    if (mission == null) return;
    
    // Only apply if mission actually changed (different file or modified)
    if (_lastMissionPath == mission.filePath) {
      // Same mission file - check if modified timestamp changed
      if (previous?.activeMission?.modified == mission.modified) {
        return;  // No change
      }
    }
    
    _lastMissionPath = mission.filePath;
    
    final signals = mission.effectiveSignals;
    debugPrint('[MissionHeadLoader] Mission changed: ${mission.name}');
    debugPrint('[MissionHeadLoader] Loading heads: $signals');
    
    _loadHeadsForMission(signals);
  }

  void _loadHeadsForMission(List<String> signals) {
    if (signals.isEmpty) {
      debugPrint('[MissionHeadLoader] No signals to load');
      _unloadAllHeads();
      return;
    }
    
    // Find heads to unload (in current but not in new)
    final currentHeads = Set<String>.from(state.loadedHeads);
    final newHeads = Set<String>.from(signals);
    final toUnload = currentHeads.difference(newHeads).toList();
    
    if (toUnload.isNotEmpty) {
      debugPrint('[MissionHeadLoader] Unloading old heads: $toUnload');
      _ref.read(videoStreamProvider.notifier).unloadHeads(toUnload);
    }
    
    state = state.copyWith(
      requestedHeads: signals,
      isLoading: true,
    );
    
    // Call video stream to load new heads
    debugPrint('[MissionHeadLoader] Loading heads: $signals');
    _ref.read(videoStreamProvider.notifier).loadHeads(signals);
    
    // Update state (actual loaded heads confirmed by backend response)
    state = state.copyWith(
      loadedHeads: signals,  // Optimistic update
      isLoading: false,
    );
  }

  void _unloadAllHeads() {
    state = state.copyWith(
      requestedHeads: [],
      isLoading: true,
    );
    
    _ref.read(videoStreamProvider.notifier).unloadHeads();
    
    state = state.copyWith(
      loadedHeads: [],
      isLoading: false,
    );
  }

  /// Manually refresh heads for current mission
  void refreshHeads() {
    final mission = _ref.read(missionProvider).activeMission;
    if (mission != null) {
      _loadHeadsForMission(mission.effectiveSignals);
    }
  }

  /// Manually load specific heads (override mission)
  void loadHeads(List<String> signals) {
    _loadHeadsForMission(signals);
  }

  /// Manually unload all heads
  void unloadAllHeads() {
    _unloadAllHeads();
  }
}

/// Provider for mission head loader
/// Watch this provider in your main widget tree to enable auto head loading
final missionHeadLoaderProvider = StateNotifierProvider<MissionHeadLoaderNotifier, MissionHeadState>(
  (ref) => MissionHeadLoaderNotifier(ref),
);

/// Convenience provider for currently loaded heads
final loadedHeadsProvider = Provider<List<String>>((ref) {
  return ref.watch(missionHeadLoaderProvider).loadedHeads;
});

/// Convenience provider for whether heads are loading
final headsLoadingProvider = Provider<bool>((ref) {
  return ref.watch(missionHeadLoaderProvider).isLoading;
});
