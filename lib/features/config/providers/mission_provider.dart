/// lib/features/config/providers/mission_provider.dart
/// Mission configuration state management
/// Handles loading, saving, and switching between mission files

import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:path/path.dart' as path;
import '../models/mission_config.dart';

/// State for mission management
class MissionState {
  /// Currently active mission configuration
  final MissionConfig? activeMission;
  
  /// List of available mission files (paths)
  final List<String> availableMissions;
  
  /// Whether we're currently loading
  final bool isLoading;
  
  /// Error message if any
  final String? error;
  
  /// Path to the missions directory
  final String missionsDir;

  const MissionState({
    this.activeMission,
    this.availableMissions = const [],
    this.isLoading = false,
    this.error,
    this.missionsDir = 'config/missions',
  });

  MissionState copyWith({
    MissionConfig? activeMission,
    List<String>? availableMissions,
    bool? isLoading,
    String? error,
    String? missionsDir,
  }) {
    return MissionState(
      activeMission: activeMission ?? this.activeMission,
      availableMissions: availableMissions ?? this.availableMissions,
      isLoading: isLoading ?? this.isLoading,
      error: error,
      missionsDir: missionsDir ?? this.missionsDir,
    );
  }

  /// Get just the filename from a full path
  static String getFileName(String filePath) {
    return path.basename(filePath);
  }
}

/// Provider for mission configuration management
class MissionNotifier extends StateNotifier<MissionState> {
  MissionNotifier() : super(const MissionState()) {
    // Scan for available missions on creation (but don't auto-load any)
    _initialize();
  }

  Future<void> _initialize() async {
    await scanMissions();
    // No auto-load - user must explicitly select a mission
  }

  /// Get the absolute path to the missions directory
  String get _missionsPath {
    // Get the executable directory or current directory
    final execDir = Platform.resolvedExecutable.isNotEmpty 
        ? path.dirname(Platform.resolvedExecutable)
        : Directory.current.path;
    
    // For development, check if we're in the project directory
    final projectMissionsDir = path.join(Directory.current.path, state.missionsDir);
    if (Directory(projectMissionsDir).existsSync()) {
      return projectMissionsDir;
    }
    
    // Fall back to relative to executable
    return path.join(execDir, state.missionsDir);
  }

  /// Scan the missions directory for available mission files
  Future<void> scanMissions() async {
    state = state.copyWith(isLoading: true, error: null);
    
    try {
      final dir = Directory(_missionsPath);
      
      if (!await dir.exists()) {
        await dir.create(recursive: true);
        debugPrint('[MissionProvider] Created missions directory: $_missionsPath');
      }
      
      final files = await dir.list()
          .where((entity) => entity is File && entity.path.endsWith('.mission.yaml'))
          .map((entity) => entity.path)
          .toList();
      
      // Sort alphabetically
      files.sort((a, b) => path.basename(a).compareTo(path.basename(b)));
      
      debugPrint('[MissionProvider] Found ${files.length} mission files in $_missionsPath');
      
      state = state.copyWith(
        availableMissions: files,
        isLoading: false,
      );
    } catch (e) {
      debugPrint('[MissionProvider] Error scanning missions: $e');
      state = state.copyWith(
        isLoading: false,
        error: 'Failed to scan missions: $e',
      );
    }
  }

  /// Load a mission from file
  Future<bool> loadMission(String filePath) async {
    state = state.copyWith(isLoading: true, error: null);
    
    try {
      final mission = await MissionConfig.loadFromFile(filePath);
      
      debugPrint('[MissionProvider] Loaded mission: ${mission.name} from $filePath');
      
      state = state.copyWith(
        activeMission: mission,
        isLoading: false,
      );
      
      return true;
    } catch (e) {
      debugPrint('[MissionProvider] Error loading mission: $e');
      state = state.copyWith(
        isLoading: false,
        error: 'Failed to load mission: $e',
      );
      return false;
    }
  }

  /// Save the current mission to its file (or a new file)
  Future<bool> saveMission({String? newPath}) async {
    final mission = state.activeMission;
    if (mission == null) {
      state = state.copyWith(error: 'No active mission to save');
      return false;
    }
    
    state = state.copyWith(isLoading: true, error: null);
    
    try {
      final savePath = newPath ?? mission.filePath;
      if (savePath == null) {
        state = state.copyWith(
          isLoading: false,
          error: 'No file path specified',
        );
        return false;
      }
      
      // Update modified timestamp
      final updatedMission = mission.copyWith(
        modified: DateTime.now(),
        filePath: savePath,
      );
      
      await updatedMission.saveToFile(savePath);
      
      debugPrint('[MissionProvider] Saved mission to $savePath');
      
      state = state.copyWith(
        activeMission: updatedMission,
        isLoading: false,
      );
      
      // Rescan to pick up any new files
      await scanMissions();
      
      return true;
    } catch (e) {
      debugPrint('[MissionProvider] Error saving mission: $e');
      state = state.copyWith(
        isLoading: false,
        error: 'Failed to save mission: $e',
      );
      return false;
    }
  }

  /// Create a new mission with default values
  Future<void> createNewMission({
    required String name,
    String description = '',
  }) async {
    final now = DateTime.now();
    
    // Generate filename from name
    final fileName = '${name.toLowerCase().replaceAll(RegExp(r'[^a-z0-9]'), '_')}.mission.yaml';
    final filePath = path.join(_missionsPath, fileName);
    
    // Check if file already exists
    if (File(filePath).existsSync()) {
      state = state.copyWith(error: 'Mission file already exists: $fileName');
      return;
    }
    
    final newMission = MissionConfig.defaultConfig().copyWith(
      name: name,
      description: description,
      created: now,
      modified: now,
      filePath: filePath,
    );
    
    state = state.copyWith(
      activeMission: newMission,
    );
    
    // Save immediately
    await saveMission();
  }

  /// Duplicate the current mission with a new name
  Future<void> duplicateMission({required String newName}) async {
    final current = state.activeMission;
    if (current == null) {
      state = state.copyWith(error: 'No active mission to duplicate');
      return;
    }
    
    final now = DateTime.now();
    
    // Generate filename from name
    final fileName = '${newName.toLowerCase().replaceAll(RegExp(r'[^a-z0-9]'), '_')}.mission.yaml';
    final filePath = path.join(_missionsPath, fileName);
    
    // Check if file already exists
    if (File(filePath).existsSync()) {
      state = state.copyWith(error: 'Mission file already exists: $fileName');
      return;
    }
    
    final duplicated = current.copyWith(
      name: newName,
      created: now,
      modified: now,
      filePath: filePath,
    );
    
    state = state.copyWith(
      activeMission: duplicated,
    );
    
    // Save immediately
    await saveMission();
  }

  /// Delete a mission file
  Future<bool> deleteMission(String filePath) async {
    try {
      final file = File(filePath);
      if (await file.exists()) {
        await file.delete();
        debugPrint('[MissionProvider] Deleted mission: $filePath');
        
        // If we deleted the active mission, clear it
        if (state.activeMission?.filePath == filePath) {
          state = state.copyWith(activeMission: null);
        }
        
        // Rescan
        await scanMissions();
        return true;
      }
      return false;
    } catch (e) {
      debugPrint('[MissionProvider] Error deleting mission: $e');
      state = state.copyWith(error: 'Failed to delete mission: $e');
      return false;
    }
  }

  /// Update the active mission with new values
  void updateActiveMission(MissionConfig Function(MissionConfig) updater) {
    final current = state.activeMission;
    if (current == null) return;
    
    state = state.copyWith(
      activeMission: updater(current),
    );
  }

  /// Update a specific field in the active mission
  void updateField<T>({
    required T value,
    required MissionConfig Function(MissionConfig, T) updater,
  }) {
    final current = state.activeMission;
    if (current == null) return;
    
    state = state.copyWith(
      activeMission: updater(current, value),
    );
  }
}

/// Provider instance
final missionProvider = StateNotifierProvider<MissionNotifier, MissionState>(
  (ref) => MissionNotifier(),
);

/// Convenience provider for active mission
final activeMissionProvider = Provider<MissionConfig?>((ref) {
  return ref.watch(missionProvider).activeMission;
});

/// Convenience provider for available missions list
final availableMissionsProvider = Provider<List<String>>((ref) {
  return ref.watch(missionProvider).availableMissions;
});

/// Convenience provider for loading state
final missionLoadingProvider = Provider<bool>((ref) {
  return ref.watch(missionProvider).isLoading;
});

/// Provider that applies mission settings to the video stream when mission changes
/// Import this provider in your main app to enable auto-apply
final missionApplierProvider = Provider<void>((ref) {
  // Watch for mission changes
  ref.listen<MissionState>(missionProvider, (previous, next) {
    final mission = next.activeMission;
    if (mission == null) return;
    
    // Only apply if mission actually changed
    if (previous?.activeMission?.filePath == mission.filePath &&
        previous?.activeMission?.modified == mission.modified) {
      return;
    }
    
    debugPrint('[MissionApplier] Applying mission settings: ${mission.name}');
    
    // Import video stream provider dynamically to avoid circular deps
    // Settings will be applied via WebSocket commands
    // This is handled by mission_screen.dart via direct provider access
  });
});
