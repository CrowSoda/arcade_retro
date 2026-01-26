// lib/core/database/signal_database.dart
/// Signal Database - Persistent storage for signal metadata
/// 
/// Each signal entry includes:
/// - Name (signal class name)
/// - Modulation type and rate
/// - Data labels count (how many samples trained on)
/// - F1 score from last training
/// - Training history
/// 
/// Updated by training tab when models are trained

import 'dart:convert';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

/// Modulation types for dropdown
const List<String> kModTypes = [
  '--',
  'BPSK',
  'QPSK',
  'OQPSK',
  '8PSK',
  '16QAM',
  '64QAM',
  'OFDM',
  'FSK',
  'GFSK',
  'MSK',
  'GMSK',
  'FHSS',
  'DSSS',
  'AM',
  'FM',
  'Chirp',
  'Burst',
  'Unknown',
];

/// Individual detection record with score, timestamp, and context
class DetectionRecord {
  final DateTime timestamp;
  final double score;         // Confidence score 0-1
  final double freqMHz;       // Detection frequency
  final double? bandwidthMHz; // Signal bandwidth
  final String? mgrsLocation; // Device location
  final int? trackId;         // Track ID if assigned

  const DetectionRecord({
    required this.timestamp,
    required this.score,
    required this.freqMHz,
    this.bandwidthMHz,
    this.mgrsLocation,
    this.trackId,
  });

  Map<String, dynamic> toJson() => {
    'timestamp': timestamp.toIso8601String(),
    'score': score,
    'freqMHz': freqMHz,
    'bandwidthMHz': bandwidthMHz,
    'mgrsLocation': mgrsLocation,
    'trackId': trackId,
  };

  factory DetectionRecord.fromJson(Map<String, dynamic> json) => DetectionRecord(
    timestamp: DateTime.parse(json['timestamp']),
    score: (json['score'] as num).toDouble(),
    freqMHz: (json['freqMHz'] as num).toDouble(),
    bandwidthMHz: (json['bandwidthMHz'] as num?)?.toDouble(),
    mgrsLocation: json['mgrsLocation'],
    trackId: json['trackId'],
  );
}

/// Training result from a single training run
class TrainingResult {
  final DateTime timestamp;
  final int dataLabels;      // Number of labeled samples used
  final double f1Score;      // F1 score achieved
  final double precision;    // Precision metric
  final double recall;       // Recall metric
  final int epochs;          // Training epochs
  final double? loss;        // Final loss value
  final String? modelPath;   // Path to saved model file

  const TrainingResult({
    required this.timestamp,
    required this.dataLabels,
    required this.f1Score,
    this.precision = 0.0,
    this.recall = 0.0,
    this.epochs = 0,
    this.loss,
    this.modelPath,
  });

  Map<String, dynamic> toJson() => {
    'timestamp': timestamp.toIso8601String(),
    'dataLabels': dataLabels,
    'f1Score': f1Score,
    'precision': precision,
    'recall': recall,
    'epochs': epochs,
    'loss': loss,
    'modelPath': modelPath,
  };

  factory TrainingResult.fromJson(Map<String, dynamic> json) => TrainingResult(
    timestamp: DateTime.parse(json['timestamp']),
    dataLabels: json['dataLabels'] ?? 0,
    f1Score: (json['f1Score'] as num?)?.toDouble() ?? 0.0,
    precision: (json['precision'] as num?)?.toDouble() ?? 0.0,
    recall: (json['recall'] as num?)?.toDouble() ?? 0.0,
    epochs: json['epochs'] ?? 0,
    loss: (json['loss'] as num?)?.toDouble(),
    modelPath: json['modelPath'],
  );
}

/// Signal entry for the database
class SignalEntry {
  final String id;
  String name;
  String modType;
  double? modRate;        // Symbol rate in sps
  double? bandwidth;      // Signal bandwidth in kHz
  String? notes;
  
  // Training stats
  int totalDataLabels;    // Total labeled samples across all training
  double? f1Score;        // Best/latest F1 score
  double? precision;      // Best/latest precision
  double? recall;         // Best/latest recall
  int timesAbove90;       // Detection count with >90% confidence
  
  // History
  List<TrainingResult> trainingHistory;
  List<DetectionRecord> detectionHistory;  // Recent high-confidence detections
  
  // Timestamps
  DateTime created;
  DateTime modified;

  SignalEntry({
    required this.id,
    required this.name,
    this.modType = '--',
    this.modRate,
    this.bandwidth,
    this.notes,
    this.totalDataLabels = 0,
    this.f1Score,
    this.precision,
    this.recall,
    this.timesAbove90 = 0,
    List<TrainingResult>? trainingHistory,
    List<DetectionRecord>? detectionHistory,
    DateTime? created,
    DateTime? modified,
  }) : trainingHistory = trainingHistory ?? [],
       detectionHistory = detectionHistory ?? [],
       created = created ?? DateTime.now(),
       modified = modified ?? DateTime.now();

  /// Backwards-compatible getter for totalDataLabels
  int get sampleCount => totalDataLabels;

  /// Get the latest training result
  TrainingResult? get latestTraining => 
      trainingHistory.isNotEmpty ? trainingHistory.last : null;

  /// Get best F1 score from history
  double? get bestF1Score {
    if (trainingHistory.isEmpty) return f1Score;
    return trainingHistory.map((t) => t.f1Score).reduce((a, b) => a > b ? a : b);
  }

  /// Add a new training result
  void addTrainingResult(TrainingResult result) {
    trainingHistory.add(result);
    totalDataLabels += result.dataLabels;
    f1Score = result.f1Score;
    precision = result.precision;
    recall = result.recall;
    modified = DateTime.now();
  }

  /// Increment detection count (for >90% confidence detections)
  void incrementDetectionCount() {
    timesAbove90++;
    modified = DateTime.now();
  }

  /// Add a detection record (for >90% confidence detections)
  /// Keeps last 100 detections per signal
  void addDetectionRecord(DetectionRecord record) {
    detectionHistory.add(record);
    timesAbove90++;
    // Keep only last 100 detections per signal to prevent unbounded growth
    if (detectionHistory.length > 100) {
      detectionHistory.removeAt(0);
    }
    modified = DateTime.now();
  }

  /// Get recent detections (last N)
  List<DetectionRecord> getRecentDetections([int count = 10]) {
    final start = detectionHistory.length > count ? detectionHistory.length - count : 0;
    return detectionHistory.sublist(start);
  }

  /// Get average score from recent detections
  double? get averageRecentScore {
    if (detectionHistory.isEmpty) return null;
    final recent = getRecentDetections(20);
    return recent.map((d) => d.score).reduce((a, b) => a + b) / recent.length;
  }

  SignalEntry copyWith({
    String? name,
    String? modType,
    double? modRate,
    double? bandwidth,
    String? notes,
    int? totalDataLabels,
    double? f1Score,
    double? precision,
    double? recall,
    int? timesAbove90,
    List<TrainingResult>? trainingHistory,
    List<DetectionRecord>? detectionHistory,
  }) {
    return SignalEntry(
      id: id,
      name: name ?? this.name,
      modType: modType ?? this.modType,
      modRate: modRate ?? this.modRate,
      bandwidth: bandwidth ?? this.bandwidth,
      notes: notes ?? this.notes,
      totalDataLabels: totalDataLabels ?? this.totalDataLabels,
      f1Score: f1Score ?? this.f1Score,
      precision: precision ?? this.precision,
      recall: recall ?? this.recall,
      timesAbove90: timesAbove90 ?? this.timesAbove90,
      trainingHistory: trainingHistory ?? this.trainingHistory,
      detectionHistory: detectionHistory ?? this.detectionHistory,
      created: created,
      modified: DateTime.now(),
    );
  }

  Map<String, dynamic> toJson() => {
    'id': id,
    'name': name,
    'modType': modType,
    'modRate': modRate,
    'bandwidth': bandwidth,
    'notes': notes,
    'totalDataLabels': totalDataLabels,
    'f1Score': f1Score,
    'precision': precision,
    'recall': recall,
    'timesAbove90': timesAbove90,
    'trainingHistory': trainingHistory.map((t) => t.toJson()).toList(),
    'detectionHistory': detectionHistory.map((d) => d.toJson()).toList(),
    'created': created.toIso8601String(),
    'modified': modified.toIso8601String(),
  };

  factory SignalEntry.fromJson(Map<String, dynamic> json) => SignalEntry(
    id: json['id'],
    name: json['name'],
    modType: json['modType'] ?? '--',
    modRate: (json['modRate'] as num?)?.toDouble(),
    bandwidth: (json['bandwidth'] as num?)?.toDouble(),
    notes: json['notes'],
    totalDataLabels: json['totalDataLabels'] ?? json['sampleCount'] ?? 0,
    f1Score: (json['f1Score'] as num?)?.toDouble(),
    precision: (json['precision'] as num?)?.toDouble(),
    recall: (json['recall'] as num?)?.toDouble(),
    timesAbove90: json['timesAbove90'] ?? 0,
    trainingHistory: (json['trainingHistory'] as List<dynamic>?)
        ?.map((t) => TrainingResult.fromJson(t))
        .toList() ?? [],
    detectionHistory: (json['detectionHistory'] as List<dynamic>?)
        ?.map((d) => DetectionRecord.fromJson(d))
        .toList() ?? [],
    created: json['created'] != null ? DateTime.parse(json['created']) : DateTime.now(),
    modified: json['modified'] != null ? DateTime.parse(json['modified']) : DateTime.now(),
  );
}

/// Signal Database Notifier - persists to config/signals.json
class SignalDatabaseNotifier extends StateNotifier<List<SignalEntry>> {
  static const _filePath = 'config/signals.json';
  
  SignalDatabaseNotifier() : super(_loadFromDiskSync());

  /// Synchronously load from disk at startup
  static List<SignalEntry> _loadFromDiskSync() {
    try {
      final file = File(_filePath);
      if (file.existsSync()) {
        final jsonStr = file.readAsStringSync();
        final List<dynamic> jsonList = json.decode(jsonStr);
        final entries = jsonList.map((j) => SignalEntry.fromJson(j)).toList();
        debugPrint('[SignalDB] Loaded ${entries.length} signals from disk');
        return entries;
      }
    } catch (e) {
      debugPrint('[SignalDB] Error loading from disk: $e');
    }
    
    // Return default entries if no file exists
    return [
      SignalEntry(
        id: '1', 
        name: 'creamy_chicken', 
        modType: '--', 
        totalDataLabels: 127, 
        f1Score: 0.91, 
        timesAbove90: 47,
      ),
      SignalEntry(
        id: '2', 
        name: 'lte_uplink', 
        modType: 'OFDM', 
        modRate: 15000, 
        bandwidth: 10000, 
        totalDataLabels: 89, 
        f1Score: 0.87, 
        timesAbove90: 23,
      ),
      SignalEntry(
        id: '3', 
        name: 'wifi_24', 
        modType: 'OFDM', 
        modRate: 20000, 
        bandwidth: 20000, 
        totalDataLabels: 156, 
        f1Score: 0.82, 
        timesAbove90: 56,
      ),
      SignalEntry(
        id: '4', 
        name: 'bluetooth', 
        modType: 'GFSK', 
        modRate: 1000000, 
        bandwidth: 1000, 
        totalDataLabels: 78, 
        f1Score: 0.79, 
        timesAbove90: 18,
      ),
      SignalEntry(
        id: '5', 
        name: 'unk_220001ZJAN26_825', 
        modType: '--', 
        totalDataLabels: 34, 
        timesAbove90: 8,
      ),
    ];
  }

  Future<void> _saveToDisk() async {
    try {
      final dir = Directory('config');
      if (!await dir.exists()) {
        await dir.create(recursive: true);
      }
      final file = File(_filePath);
      final jsonList = state.map((e) => e.toJson()).toList();
      await file.writeAsString(const JsonEncoder.withIndent('  ').convert(jsonList));
      debugPrint('[SignalDB] Saved ${state.length} signals to disk');
    } catch (e) {
      debugPrint('[SignalDB] Error saving to disk: $e');
    }
  }

  /// Get signal by name (case-insensitive)
  SignalEntry? getByName(String name) {
    final lower = name.toLowerCase();
    try {
      return state.firstWhere((e) => e.name.toLowerCase() == lower);
    } catch (_) {
      return null;
    }
  }

  /// Get signal by ID
  SignalEntry? getById(String id) {
    try {
      return state.firstWhere((e) => e.id == id);
    } catch (_) {
      return null;
    }
  }

  /// Add a new signal entry
  void addSignal(SignalEntry entry) {
    state = [...state, entry];
    _saveToDisk();
  }

  /// Update an existing signal
  void updateSignal(String id, SignalEntry updated) {
    state = [
      for (final entry in state)
        entry.id == id ? updated : entry,
    ];
    _saveToDisk();
  }

  /// Delete a signal
  void deleteSignal(String id) {
    state = state.where((e) => e.id != id).toList();
    _saveToDisk();
  }

  /// Add training result to a signal (by name or create new)
  void addTrainingResult(String signalName, TrainingResult result) {
    final existing = getByName(signalName);
    
    if (existing != null) {
      existing.addTrainingResult(result);
      state = [...state]; // Trigger rebuild
      _saveToDisk();
    } else {
      // Create new signal entry
      final newEntry = SignalEntry(
        id: 'sig_${DateTime.now().millisecondsSinceEpoch}',
        name: signalName,
        totalDataLabels: result.dataLabels,
        f1Score: result.f1Score,
        precision: result.precision,
        recall: result.recall,
        trainingHistory: [result],
      );
      addSignal(newEntry);
    }
    
    debugPrint('[SignalDB] Training result added for "$signalName": F1=${result.f1Score.toStringAsFixed(2)}, labels=${result.dataLabels}');
  }

  /// Increment detection count for a signal (legacy - use addDetectionRecord instead)
  void incrementDetectionCount(String signalName) {
    final existing = getByName(signalName);
    if (existing != null) {
      existing.incrementDetectionCount();
      state = [...state];
      _saveToDisk();
    }
  }

  /// Add a detailed detection record to a signal (for >90% confidence detections)
  /// Creates the signal entry if it doesn't exist
  void addDetectionRecord({
    required String signalName,
    required double score,
    required double freqMHz,
    double? bandwidthMHz,
    String? mgrsLocation,
    int? trackId,
  }) {
    final record = DetectionRecord(
      timestamp: DateTime.now(),
      score: score,
      freqMHz: freqMHz,
      bandwidthMHz: bandwidthMHz,
      mgrsLocation: mgrsLocation,
      trackId: trackId,
    );

    var entry = getByName(signalName);
    if (entry == null) {
      // Create new entry for unknown signal
      entry = SignalEntry(
        id: 'sig_${DateTime.now().millisecondsSinceEpoch}',
        name: signalName.toLowerCase(),
      );
      state = [...state, entry];
    }

    entry.addDetectionRecord(record);
    state = [...state]; // Trigger rebuild
    _saveToDisk();
    
    debugPrint('[SignalDB] Detection logged: $signalName @ ${freqMHz.toStringAsFixed(2)} MHz, score=${(score * 100).toStringAsFixed(0)}%');
  }

  /// Get or create signal entry (for detection/training)
  SignalEntry getOrCreate(String signalName) {
    final existing = getByName(signalName);
    if (existing != null) return existing;
    
    // Create new entry
    final newEntry = SignalEntry(
      id: 'sig_${DateTime.now().millisecondsSinceEpoch}',
      name: signalName.toLowerCase(),
    );
    addSignal(newEntry);
    return newEntry;
  }
}

/// Provider for signal database
final signalDatabaseProvider = StateNotifierProvider<SignalDatabaseNotifier, List<SignalEntry>>((ref) {
  return SignalDatabaseNotifier();
});

/// Provider to get a single signal by name
final signalByNameProvider = Provider.family<SignalEntry?, String>((ref, name) {
  final db = ref.watch(signalDatabaseProvider);
  final lower = name.toLowerCase();
  try {
    return db.firstWhere((e) => e.name.toLowerCase() == lower);
  } catch (_) {
    return null;
  }
});
