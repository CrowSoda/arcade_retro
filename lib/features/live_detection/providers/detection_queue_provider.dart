// lib/features/live_detection/providers/detection_queue_provider.dart
/// Priority-based detection queue for dual-RX operation
///
/// ARCHITECTURE:
/// - RX1: ALWAYS scanning (constant detections, never compromised)
/// - RX2: Collection channel (tunes to detected signals for recording)
///
/// When RX1 detects a priority signal:
/// 1. Detection is added to queue with priority
/// 2. If RX2 is free, it tunes to the signal and collects
/// 3. If RX2 is busy, detection stays queued
/// 4. When collection completes, next priority item starts

import 'package:flutter/foundation.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'rx_state_provider.dart';

/// Signal priority levels - determines queue ordering
enum SignalPriority {
  critical,  // 0 - Highest: Must collect immediately
  high,      // 1 - High priority threats
  medium,    // 2 - Standard priority
  low,       // 3 - Collect if time permits
}

extension SignalPriorityExtension on SignalPriority {
  String get label {
    switch (this) {
      case SignalPriority.critical: return 'CRITICAL';
      case SignalPriority.high: return 'HIGH';
      case SignalPriority.medium: return 'MED';
      case SignalPriority.low: return 'LOW';
    }
  }

  /// Get priority from class name (configurable mapping)
  static SignalPriority fromClassName(String className) {
    // TODO: Make this configurable via mission config
    switch (className.toLowerCase()) {
      case 'jamming':
      case 'threat':
        return SignalPriority.critical;
      case 'radar':
      case 'datalink':
        return SignalPriority.high;
      case 'creamy_chicken':
      case 'comm':
        return SignalPriority.medium;
      default:
        return SignalPriority.low;
    }
  }
}

/// Entry in the detection queue
class DetectionQueueEntry {
  final String detectionId;
  final double freqMHz;
  final double bwMHz;
  final String className;
  final SignalPriority priority;
  final double confidence;
  final DateTime detectedAt;
  final int? collectionDurationSec;  // How long to collect (null = default)

  DetectionQueueEntry({
    required this.detectionId,
    required this.freqMHz,
    required this.bwMHz,
    required this.className,
    required this.priority,
    this.confidence = 0.0,
    this.collectionDurationSec,
    DateTime? detectedAt,
  }) : detectedAt = detectedAt ?? DateTime.now();

  @override
  String toString() => 'Queue[$priority]: $className @ $freqMHz MHz';
}

/// Current collection status
class CollectionStatus {
  final DetectionQueueEntry entry;
  final DateTime startedAt;
  final int durationSec;
  final double progress;  // 0.0 - 1.0

  const CollectionStatus({
    required this.entry,
    required this.startedAt,
    required this.durationSec,
    this.progress = 0.0,
  });

  int get elapsedSec => DateTime.now().difference(startedAt).inSeconds;
  int get remainingSec => (durationSec - elapsedSec).clamp(0, durationSec);
  bool get isComplete => elapsedSec >= durationSec;
}

/// Detection queue state
class DetectionQueueState {
  final List<DetectionQueueEntry> queue;
  final CollectionStatus? currentCollection;
  final bool rx2Available;
  final int totalCollected;  // Count of completed collections
  final int totalDropped;    // Count of detections that aged out

  const DetectionQueueState({
    this.queue = const [],
    this.currentCollection,
    this.rx2Available = true,
    this.totalCollected = 0,
    this.totalDropped = 0,
  });

  DetectionQueueState copyWith({
    List<DetectionQueueEntry>? queue,
    CollectionStatus? currentCollection,
    bool clearCurrentCollection = false,
    bool? rx2Available,
    int? totalCollected,
    int? totalDropped,
  }) {
    return DetectionQueueState(
      queue: queue ?? this.queue,
      currentCollection: clearCurrentCollection ? null : (currentCollection ?? this.currentCollection),
      rx2Available: rx2Available ?? this.rx2Available,
      totalCollected: totalCollected ?? this.totalCollected,
      totalDropped: totalDropped ?? this.totalDropped,
    );
  }

  int get queueLength => queue.length;
  bool get hasQueue => queue.isNotEmpty;
  bool get isCollecting => currentCollection != null;
}

/// Detection queue notifier - manages priority queue and RX2 handoff
class DetectionQueueNotifier extends StateNotifier<DetectionQueueState> {
  final Ref _ref;
  static const int _defaultCollectionDurationSec = 30;
  static const int _maxQueueSize = 20;
  static const int _maxAgeMinutes = 5;  // Drop detections older than this

  DetectionQueueNotifier(this._ref) : super(const DetectionQueueState());

  /// Called when RX1 detects a signal worth collecting
  /// Priority is determined by class name mapping
  void onDetection({
    required String detectionId,
    required double freqMHz,
    required double bwMHz,
    required String className,
    double confidence = 0.0,
    int? collectionDurationSec,
  }) {
    final priority = SignalPriorityExtension.fromClassName(className);

    final entry = DetectionQueueEntry(
      detectionId: detectionId,
      freqMHz: freqMHz,
      bwMHz: bwMHz,
      className: className,
      priority: priority,
      confidence: confidence,
      collectionDurationSec: collectionDurationSec,
    );

    // Check for duplicate (same freq within 1 MHz)
    final isDuplicate = state.queue.any((e) =>
      (e.freqMHz - freqMHz).abs() < 1.0 &&
      e.className == className
    );

    if (isDuplicate) {
      debugPrint('🎯 Skipping duplicate detection: $className @ $freqMHz MHz');
      return;
    }

    // Add to queue
    var newQueue = [...state.queue, entry];

    // Sort by priority, then by time (oldest first within same priority)
    newQueue.sort((a, b) {
      final priorityCmp = a.priority.index.compareTo(b.priority.index);
      if (priorityCmp != 0) return priorityCmp;
      return a.detectedAt.compareTo(b.detectedAt);
    });

    // Trim queue if too large (drop lowest priority items)
    if (newQueue.length > _maxQueueSize) {
      final dropped = newQueue.length - _maxQueueSize;
      newQueue = newQueue.sublist(0, _maxQueueSize);
      state = state.copyWith(
        queue: newQueue,
        totalDropped: state.totalDropped + dropped,
      );
      debugPrint('🎯 Queue full, dropped $dropped low-priority items');
    } else {
      state = state.copyWith(queue: newQueue);
    }

    debugPrint('🎯 Queued detection: $entry (queue: ${state.queueLength})');

    // Try to start collection if RX2 is free
    _tryStartNextCollection();
  }

  /// Manually queue a detection with explicit priority
  void queueManual({
    required double freqMHz,
    required double bwMHz,
    required String label,
    SignalPriority priority = SignalPriority.medium,
    int collectionDurationSec = 60,
  }) {
    onDetection(
      detectionId: 'manual_${DateTime.now().millisecondsSinceEpoch}',
      freqMHz: freqMHz,
      bwMHz: bwMHz,
      className: label,
      collectionDurationSec: collectionDurationSec,
    );
  }

  /// Try to start the next collection (if RX2 is available)
  void _tryStartNextCollection() {
    if (!state.rx2Available || state.queue.isEmpty || state.isCollecting) {
      return;
    }

    // Age out old detections
    _pruneOldDetections();

    if (state.queue.isEmpty) return;

    final next = state.queue.first;
    final remaining = state.queue.sublist(1);

    // Tune RX2 to collection frequency
    _ref.read(multiRxProvider.notifier).tuneRx2(next.freqMHz, next.bwMHz, null);

    // Start collection
    final collection = CollectionStatus(
      entry: next,
      startedAt: DateTime.now(),
      durationSec: next.collectionDurationSec ?? _defaultCollectionDurationSec,
    );

    state = state.copyWith(
      queue: remaining,
      currentCollection: collection,
      rx2Available: false,
    );

    debugPrint('🎯 RX2 collecting: ${next.className} @ ${next.freqMHz} MHz for ${collection.durationSec}s');

    // Start collection timer (simulated for now)
    _simulateCollection(collection);
  }

  /// Simulate collection completion (replace with real IQ capture in production)
  Future<void> _simulateCollection(CollectionStatus collection) async {
    final durationSec = collection.durationSec;

    // Update progress periodically
    for (int i = 0; i < durationSec; i++) {
      await Future.delayed(const Duration(seconds: 1));

      if (state.currentCollection == null) {
        debugPrint('🎯 Collection cancelled');
        return;
      }

      final progress = (i + 1) / durationSec;
      state = state.copyWith(
        currentCollection: CollectionStatus(
          entry: collection.entry,
          startedAt: collection.startedAt,
          durationSec: collection.durationSec,
          progress: progress,
        ),
      );
    }

    // Collection complete
    onCollectionComplete();
  }

  /// Called when RX2 collection completes
  void onCollectionComplete() {
    if (state.currentCollection != null) {
      debugPrint('🎯 Collection complete: ${state.currentCollection!.entry.className}');
    }

    state = state.copyWith(
      clearCurrentCollection: true,
      rx2Available: true,
      totalCollected: state.totalCollected + 1,
    );

    // Return RX2 to saved state
    _ref.read(multiRxProvider.notifier).rx2ResumeToSaved();

    // Start next in queue
    _tryStartNextCollection();
  }

  /// Cancel current collection
  void cancelCurrentCollection() {
    if (state.currentCollection == null) return;

    debugPrint('🎯 Cancelling collection: ${state.currentCollection!.entry.className}');

    state = state.copyWith(
      clearCurrentCollection: true,
      rx2Available: true,
    );

    _ref.read(multiRxProvider.notifier).rx2ResumeToSaved();
    _tryStartNextCollection();
  }

  /// Remove old detections from queue
  void _pruneOldDetections() {
    final cutoff = DateTime.now().subtract(Duration(minutes: _maxAgeMinutes));
    final oldCount = state.queue.where((e) => e.detectedAt.isBefore(cutoff)).length;

    if (oldCount > 0) {
      final newQueue = state.queue.where((e) => e.detectedAt.isAfter(cutoff)).toList();
      state = state.copyWith(
        queue: newQueue,
        totalDropped: state.totalDropped + oldCount,
      );
      debugPrint('🎯 Pruned $oldCount aged-out detections');
    }
  }

  /// Clear all queued detections
  void clearQueue() {
    state = state.copyWith(
      queue: [],
      totalDropped: state.totalDropped + state.queueLength,
    );
    debugPrint('🎯 Queue cleared');
  }

  /// Skip to next in queue (cancel current, start next)
  void skipToNext() {
    cancelCurrentCollection();
  }
}

/// Provider for detection queue
final detectionQueueProvider = StateNotifierProvider<DetectionQueueNotifier, DetectionQueueState>((ref) {
  return DetectionQueueNotifier(ref);
});

/// Convenience provider for current collection
final currentCollectionProvider = Provider<CollectionStatus?>((ref) {
  return ref.watch(detectionQueueProvider).currentCollection;
});

/// Convenience provider for queue length
final queueLengthProvider = Provider<int>((ref) {
  return ref.watch(detectionQueueProvider).queueLength;
});
