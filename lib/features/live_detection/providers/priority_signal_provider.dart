/// Priority Signal Handler
/// 
/// Manages priority-based signal detection and RX2 tasking.
/// 
/// KEY DESIGN PRINCIPLES:
/// - RX1 ALWAYS SCANS: RX1 runs inference continuously, never interrupted
/// - RX2 IS TASKABLE: RX2 can be assigned to manual capture or priority collection
/// - PRIORITY QUEUE: Higher priority signals get RX2 access first
/// 
/// ARCHITECTURE:
/// ```
/// RX1 (Scanning) ──► Detections ──► Priority Queue ──► RX2 Tasking
///     │                                │
///     └── NEVER INTERRUPTED           └── Can queue multiple signals
/// ```
/// 
/// When same signal detected on multiple frequencies (e.g., 700MHz and 800MHz),
/// the queue handles deconfliction based on priority and timing.

import 'package:flutter/foundation.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'rx_state_provider.dart';
import '../../config/providers/tuning_state_provider.dart';

/// Signal priority levels (1 = highest)
enum SignalPriority {
  priority1,  // Critical - immediate attention
  priority2,  // High - queue for next availability
  priority3,  // Normal - queue after higher priorities
  priority4,  // Low - collect if RX2 idle
}

/// A detected priority signal awaiting RX2 collection
class PrioritySignal {
  final String id;
  final String signalType;      // SOI class name (e.g., "FH_SIGNAL")
  final SignalPriority priority;
  final double frequencyMHz;    // Center frequency where detected
  final double bandwidthMHz;    // Estimated signal bandwidth
  final double confidence;      // Detection confidence 0-1
  final DateTime detectedAt;
  final bool isCollected;       // Whether RX2 has collected this
  
  const PrioritySignal({
    required this.id,
    required this.signalType,
    required this.priority,
    required this.frequencyMHz,
    required this.bandwidthMHz,
    required this.confidence,
    required this.detectedAt,
    this.isCollected = false,
  });
  
  PrioritySignal copyWith({
    bool? isCollected,
  }) {
    return PrioritySignal(
      id: id,
      signalType: signalType,
      priority: priority,
      frequencyMHz: frequencyMHz,
      bandwidthMHz: bandwidthMHz,
      confidence: confidence,
      detectedAt: detectedAt,
      isCollected: isCollected ?? this.isCollected,
    );
  }
  
  /// Signals are same if same type and within 10 MHz
  bool isSameSignal(PrioritySignal other) {
    return signalType == other.signalType &&
           (frequencyMHz - other.frequencyMHz).abs() < 10.0;
  }
}

/// Priority signal queue state
class PrioritySignalState {
  /// Queue of signals awaiting collection (ordered by priority then time)
  final List<PrioritySignal> queue;
  
  /// Currently collecting signal (null if RX2 idle)
  final PrioritySignal? currentlyCollecting;
  
  /// Recently collected signals (for deduplication)
  final List<PrioritySignal> recentlyCollected;
  
  /// Whether auto-collection is enabled
  final bool autoCollectEnabled;
  
  /// Collection duration in seconds
  final int collectionDurationSec;
  
  const PrioritySignalState({
    this.queue = const [],
    this.currentlyCollecting,
    this.recentlyCollected = const [],
    this.autoCollectEnabled = true,
    this.collectionDurationSec = 60,
  });
  
  PrioritySignalState copyWith({
    List<PrioritySignal>? queue,
    PrioritySignal? currentlyCollecting,
    bool clearCurrentlyCollecting = false,
    List<PrioritySignal>? recentlyCollected,
    bool? autoCollectEnabled,
    int? collectionDurationSec,
  }) {
    return PrioritySignalState(
      queue: queue ?? this.queue,
      currentlyCollecting: clearCurrentlyCollecting ? null : (currentlyCollecting ?? this.currentlyCollecting),
      recentlyCollected: recentlyCollected ?? this.recentlyCollected,
      autoCollectEnabled: autoCollectEnabled ?? this.autoCollectEnabled,
      collectionDurationSec: collectionDurationSec ?? this.collectionDurationSec,
    );
  }
  
  /// Get next signal to collect (highest priority, oldest)
  PrioritySignal? get nextInQueue {
    if (queue.isEmpty) return null;
    return queue.first;  // Already sorted by priority then time
  }
  
  /// Check if a signal is already in queue or recently collected
  bool isDuplicate(PrioritySignal signal) {
    // Check queue
    for (final queued in queue) {
      if (queued.isSameSignal(signal)) return true;
    }
    // Check currently collecting
    if (currentlyCollecting != null && currentlyCollecting!.isSameSignal(signal)) return true;
    // Check recently collected (within 5 minutes)
    final fiveMinAgo = DateTime.now().subtract(const Duration(minutes: 5));
    for (final recent in recentlyCollected) {
      if (recent.detectedAt.isAfter(fiveMinAgo) && recent.isSameSignal(signal)) return true;
    }
    return false;
  }
}

/// Priority signal queue manager
class PrioritySignalNotifier extends StateNotifier<PrioritySignalState> {
  final Ref _ref;
  
  PrioritySignalNotifier(this._ref) : super(const PrioritySignalState());
  
  /// Add a detected signal to the queue
  /// Returns false if duplicate, true if added
  bool addDetection(PrioritySignal signal) {
    // Check for duplicates
    if (state.isDuplicate(signal)) {
      debugPrint('[PrioritySignal] Duplicate signal ignored: ${signal.signalType} @ ${signal.frequencyMHz} MHz');
      return false;
    }
    
    // Add to queue, sorted by priority (enum index) then time
    final newQueue = [...state.queue, signal];
    newQueue.sort((a, b) {
      final priorityCompare = a.priority.index.compareTo(b.priority.index);
      if (priorityCompare != 0) return priorityCompare;
      return a.detectedAt.compareTo(b.detectedAt);
    });
    
    state = state.copyWith(queue: newQueue);
    debugPrint('[PrioritySignal] Queued: ${signal.signalType} @ ${signal.frequencyMHz} MHz (priority ${signal.priority.index + 1}, queue size: ${newQueue.length})');
    
    // Auto-start collection if enabled and RX2 is free
    if (state.autoCollectEnabled && state.currentlyCollecting == null) {
      _tryStartNextCollection();
    }
    
    return true;
  }
  
  /// Simulate detection from inference result
  /// Called when RX1 detects a priority signal
  void onInferenceDetection({
    required String signalType,
    required SignalPriority priority,
    required double frequencyMHz,
    required double bandwidthMHz,
    required double confidence,
  }) {
    final signal = PrioritySignal(
      id: '${signalType}_${DateTime.now().millisecondsSinceEpoch}',
      signalType: signalType,
      priority: priority,
      frequencyMHz: frequencyMHz,
      bandwidthMHz: bandwidthMHz,
      confidence: confidence,
      detectedAt: DateTime.now(),
    );
    addDetection(signal);
  }
  
  /// Start collecting next signal in queue
  void _tryStartNextCollection() {
    final next = state.nextInQueue;
    if (next == null) return;
    
    // Check if RX2 is available (not in manual mode)
    final tuningState = _ref.read(tuningStateProvider);
    if (tuningState.mode == TuningMode.manual) {
      debugPrint('[PrioritySignal] RX2 in manual mode, waiting...');
      return;
    }
    
    // Remove from queue, set as current
    final newQueue = state.queue.where((s) => s.id != next.id).toList();
    state = state.copyWith(
      queue: newQueue,
      currentlyCollecting: next,
    );
    
    // TASK RX2 - tune to signal frequency
    _ref.read(multiRxProvider.notifier).tuneRx2(
      next.frequencyMHz,
      next.bandwidthMHz,
      state.collectionDurationSec,  // Auto-return after collection
    );
    
    debugPrint('[PrioritySignal] RX2 tasked to ${next.signalType} @ ${next.frequencyMHz} MHz');
  }
  
  /// Mark current collection as complete
  void completeCurrentCollection() {
    if (state.currentlyCollecting == null) return;
    
    final completed = state.currentlyCollecting!.copyWith(isCollected: true);
    
    // Add to recently collected (trim to last 20)
    final recent = [...state.recentlyCollected, completed];
    if (recent.length > 20) {
      recent.removeRange(0, recent.length - 20);
    }
    
    state = state.copyWith(
      clearCurrentlyCollecting: true,
      recentlyCollected: recent,
    );
    
    debugPrint('[PrioritySignal] Collection complete: ${completed.signalType}');
    
    // Try next in queue
    if (state.autoCollectEnabled) {
      _tryStartNextCollection();
    }
  }
  
  /// Cancel current collection
  void cancelCurrentCollection() {
    if (state.currentlyCollecting == null) return;
    
    // Put back in queue at front
    final current = state.currentlyCollecting!;
    state = state.copyWith(
      queue: [current, ...state.queue],
      clearCurrentlyCollecting: true,
    );
    
    debugPrint('[PrioritySignal] Collection cancelled, returned to queue');
  }
  
  /// Clear entire queue
  void clearQueue() {
    state = state.copyWith(queue: []);
    debugPrint('[PrioritySignal] Queue cleared');
  }
  
  /// Toggle auto-collection
  void setAutoCollect(bool enabled) {
    state = state.copyWith(autoCollectEnabled: enabled);
    debugPrint('[PrioritySignal] Auto-collect: $enabled');
    
    if (enabled && state.currentlyCollecting == null) {
      _tryStartNextCollection();
    }
  }
  
  /// Set collection duration
  void setCollectionDuration(int seconds) {
    state = state.copyWith(collectionDurationSec: seconds);
  }
}

// ============================================================
// PROVIDERS
// ============================================================

/// Priority signal state provider
final prioritySignalProvider = StateNotifierProvider<PrioritySignalNotifier, PrioritySignalState>((ref) {
  return PrioritySignalNotifier(ref);
});

/// Queue length (convenience)
final priorityQueueLengthProvider = Provider<int>((ref) {
  return ref.watch(prioritySignalProvider).queue.length;
});

/// Is RX2 collecting (convenience)
final isRx2CollectingProvider = Provider<bool>((ref) {
  return ref.watch(prioritySignalProvider).currentlyCollecting != null;
});
