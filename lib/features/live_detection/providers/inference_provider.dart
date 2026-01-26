// lib/features/live_detection/providers/inference_provider.dart
/// Unified inference provider - coordinates waterfall + detections
/// Connects to Python unified pipeline, routes data to waterfall + detections

import 'dart:async';
import 'package:flutter/foundation.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../../../core/grpc/inference_client.dart';
import '../../../core/services/backend_launcher.dart';
import '../../../core/database/signal_database.dart';
import '../../settings/settings_screen.dart';
import 'detection_provider.dart' as det;
import 'waterfall_provider.dart';

/// WebSocket port provider
final wsPortProvider = Provider<int?>((ref) {
  final state = ref.watch(backendLauncherProvider);
  return state.wsPort;
});

/// Unified pipeline manager provider - singleton
final unifiedPipelineManagerProvider = Provider<UnifiedPipelineManager?>((ref) {
  ref.keepAlive();  // Don't auto-dispose!
  
  final backendState = ref.read(backendLauncherProvider);
  
  if (backendState.wsPort == null) {
    debugPrint('[InferenceProvider] WebSocket port not yet discovered');
    return null;
  }
  
  debugPrint('[InferenceProvider] Creating UnifiedPipelineManager with wsPort: ${backendState.wsPort}');
  final manager = UnifiedPipelineManager(
    host: 'localhost',
    port: backendState.wsPort!,
  );
  
  ref.onDispose(() {
    debugPrint('[InferenceProvider] Disposing UnifiedPipelineManager');
    manager.dispose();
  });
  
  return manager;
});

/// Live inference state
enum LiveInferenceState {
  stopped,
  starting,
  running,
  error,
}

/// Unified inference notifier - manages pipeline + routes data
class UnifiedInferenceNotifier extends StateNotifier<LiveInferenceState> {
  UnifiedInferenceNotifier(this._ref) : super(LiveInferenceState.stopped);

  final Ref _ref;
  StreamSubscription<DetectionFrame>? _detectionSub;
  int _frameCount = 0;

  /// Start unified pipeline
  Future<bool> start() async {
    if (state == LiveInferenceState.running) return true;
    if (state == LiveInferenceState.starting) return false;

    state = LiveInferenceState.starting;

    final manager = _ref.read(unifiedPipelineManagerProvider);
    if (manager == null) {
      debugPrint('[UnifiedInference] No pipeline manager available');
      state = LiveInferenceState.error;
      return false;
    }

    try {
      // Connect to WebSocket
      final success = await manager.connect();
      if (!success) {
        state = LiveInferenceState.error;
        return false;
      }

      // Connect waterfall provider to pipeline
      final waterfallNotifier = _ref.read(waterfallProvider.notifier);
      waterfallNotifier.connectToPipeline(manager);
      
      // Subscribe to detection frames
      _detectionSub = manager.detections.listen(_handleDetectionFrame);
      
      state = LiveInferenceState.running;
      debugPrint('[UnifiedInference] Pipeline started');
      return true;
    } catch (e) {
      debugPrint('[UnifiedInference] Error starting: $e');
      state = LiveInferenceState.error;
      return false;
    }
  }

  void _handleDetectionFrame(DetectionFrame frame) {
    _frameCount++;
    final detectionPts = frame.pts;
    
    // Get current score threshold from settings
    final scoreThreshold = _ref.read(scoreThresholdProvider);
    
    // Convert detections with PTS for waterfall scrolling
    // Filter by confidence threshold
    final detections = frame.detections.map((d) {
      // Skip boxes with zero or negative size
      if (d.x2 <= d.x1 || d.y2 <= d.y1) {
        return null;
      }
      
      // Filter by confidence threshold
      if (d.confidence < scoreThreshold) {
        return null;
      }
      
      return det.Detection(
        id: 'det_${frame.frameId}_${d.detectionId}',
        classId: d.classId,
        className: d.className.isEmpty ? 'unknown' : d.className,
        confidence: d.confidence,
        x1: d.x1,
        y1: d.y1,
        x2: d.x2,
        y2: d.y2,
        freqMHz: d.freqCenterMhz,
        bandwidthMHz: d.freqBandwidthMhz,
        timestamp: DateTime.now(),
        latitude: 39.7275,
        longitude: -104.7303,
        mgrsLocation: '13SDE1234567890',
        pts: detectionPts,  // PTS for waterfall scrolling
      );
    }).whereType<det.Detection>().toList();

    final notifier = _ref.read(det.detectionProvider.notifier);
    
    // Get current PTS from waterfall
    final waterfallState = _ref.read(waterfallProvider);
    final currentPts = waterfallState.currentPts;
    
    // Get time span from settings and prune old detections
    final timeSpan = _ref.read(waterfallTimeSpanProvider);
    notifier.pruneByPts(currentPts, displayTimeSpan: timeSpan);
    
    // Add new detections (don't clear - they persist and scroll!)
    if (detections.isNotEmpty) {
      notifier.addDetections(detections);
      debugPrint('[Inference] +${detections.length} @ PTS ${detectionPts.toStringAsFixed(1)}s');
      
      // Log detailed detection records to database for high-confidence detections
      final dbNotifier = _ref.read(signalDatabaseProvider.notifier);
      for (final detection in detections) {
        if (detection.confidence >= 0.9) {
          // Log full detection details - score, frequency, bandwidth, location
          dbNotifier.addDetectionRecord(
            signalName: detection.className,
            score: detection.confidence,
            freqMHz: detection.freqMHz,
            bandwidthMHz: detection.bandwidthMHz,
            mgrsLocation: detection.mgrsLocation,
            trackId: detection.trackId,
          );
        }
      }
    }
  }

  /// Stop pipeline
  Future<void> stop() async {
    await _detectionSub?.cancel();
    _detectionSub = null;

    final manager = _ref.read(unifiedPipelineManagerProvider);
    await manager?.stop();

    state = LiveInferenceState.stopped;
    _frameCount = 0;
    debugPrint('[UnifiedInference] Stopped');
  }

  @override
  void dispose() {
    stop();
    super.dispose();
  }
}

/// Provider for unified inference state
final unifiedInferenceProvider =
    StateNotifierProvider<UnifiedInferenceNotifier, LiveInferenceState>((ref) {
  return UnifiedInferenceNotifier(ref);
});

/// Auto-start unified pipeline when backend is ready
/// Uses ref.listen() to safely respond to state changes without stale closures
final autoStartUnifiedProvider = Provider<void>((ref) {
  // Track if we've already started (prevents multiple starts)
  var hasStarted = false;
  
  // Listen for backend state changes
  ref.listen<BackendLauncherState>(backendLauncherProvider, (previous, next) {
    // Only auto-start once when backend becomes ready
    if (!hasStarted && 
        next.state == BackendState.running && 
        next.wsPort != null) {
      hasStarted = true;
      debugPrint('[AutoStart] Backend ready with wsPort: ${next.wsPort}');
      
      // Use Timer instead of Future.delayed to avoid closure issues
      Timer(const Duration(milliseconds: 500), () {
        // Safe to read here because we're in a timer callback, not provider build
        try {
          final inference = ref.read(unifiedInferenceProvider.notifier);
          if (inference.state == LiveInferenceState.stopped) {
            debugPrint('[AutoStart] Starting unified pipeline...');
            inference.start();
          }
        } catch (e) {
          debugPrint('[AutoStart] Failed to start: $e');
        }
      });
    }
  }, fireImmediately: true);
  
  return;
});

// ============ Legacy providers for backward compatibility ============

final inferenceManagerProvider = Provider<InferenceManager?>((ref) {
  ref.keepAlive();
  final backendState = ref.read(backendLauncherProvider);
  
  if (backendState.wsPort == null) return null;
  
  final manager = InferenceManager(
    host: 'localhost',
    port: backendState.wsPort!,
  );
  
  ref.onDispose(() => manager.dispose());
  return manager;
});

class LiveInferenceNotifier extends StateNotifier<LiveInferenceState> {
  LiveInferenceNotifier(this._ref) : super(LiveInferenceState.stopped);
  final Ref _ref;
  
  Future<bool> start({double scoreThreshold = 0.9}) async {
    // Delegate to unified provider
    return _ref.read(unifiedInferenceProvider.notifier).start();
  }
  
  Future<void> stop() async {
    return _ref.read(unifiedInferenceProvider.notifier).stop();
  }
}

final liveInferenceProvider =
    StateNotifierProvider<LiveInferenceNotifier, LiveInferenceState>((ref) {
  return LiveInferenceNotifier(ref);
});

final autoStartInferenceProvider = Provider<void>((ref) {
  // Delegate to unified auto-start
  ref.watch(autoStartUnifiedProvider);
  return;
});
