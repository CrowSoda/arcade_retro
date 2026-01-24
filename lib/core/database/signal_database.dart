import 'dart:convert';
import 'package:flutter_riverpod/flutter_riverpod.dart';

/// Signal profile stored in database
class SignalProfile {
  final String id;
  final String name;           // e.g., "creamy_chicken"
  final String modType;        // FSK, PSK, AM, FM, etc.
  final double? modRate;       // Modulation rate in Hz
  final double freqMin;        // Min frequency in MHz
  final double freqMax;        // Max frequency in MHz
  final double? pri;           // Pulse Repetition Interval in ms
  final double? pulseWidth;    // Pulse width in us
  final String? modelPath;     // Path to trained model
  final int detectionCount;    // Number of times detected
  final double avgConfidence;  // Average detection confidence
  final DateTime createdAt;
  final DateTime updatedAt;

  SignalProfile({
    required this.id,
    required this.name,
    required this.modType,
    this.modRate,
    required this.freqMin,
    required this.freqMax,
    this.pri,
    this.pulseWidth,
    this.modelPath,
    this.detectionCount = 0,
    this.avgConfidence = 0.0,
    DateTime? createdAt,
    DateTime? updatedAt,
  })  : createdAt = createdAt ?? DateTime.now(),
        updatedAt = updatedAt ?? DateTime.now();

  SignalProfile copyWith({
    String? name,
    String? modType,
    double? modRate,
    double? freqMin,
    double? freqMax,
    double? pri,
    double? pulseWidth,
    String? modelPath,
    int? detectionCount,
    double? avgConfidence,
  }) {
    return SignalProfile(
      id: id,
      name: name ?? this.name,
      modType: modType ?? this.modType,
      modRate: modRate ?? this.modRate,
      freqMin: freqMin ?? this.freqMin,
      freqMax: freqMax ?? this.freqMax,
      pri: pri ?? this.pri,
      pulseWidth: pulseWidth ?? this.pulseWidth,
      modelPath: modelPath ?? this.modelPath,
      detectionCount: detectionCount ?? this.detectionCount,
      avgConfidence: avgConfidence ?? this.avgConfidence,
      createdAt: createdAt,
      updatedAt: DateTime.now(),
    );
  }

  Map<String, dynamic> toJson() => {
    'id': id,
    'name': name,
    'modType': modType,
    'modRate': modRate,
    'freqMin': freqMin,
    'freqMax': freqMax,
    'pri': pri,
    'pulseWidth': pulseWidth,
    'modelPath': modelPath,
    'detectionCount': detectionCount,
    'avgConfidence': avgConfidence,
    'createdAt': createdAt.toIso8601String(),
    'updatedAt': updatedAt.toIso8601String(),
  };

  factory SignalProfile.fromJson(Map<String, dynamic> json) => SignalProfile(
    id: json['id'],
    name: json['name'],
    modType: json['modType'],
    modRate: json['modRate']?.toDouble(),
    freqMin: json['freqMin'].toDouble(),
    freqMax: json['freqMax'].toDouble(),
    pri: json['pri']?.toDouble(),
    pulseWidth: json['pulseWidth']?.toDouble(),
    modelPath: json['modelPath'],
    detectionCount: json['detectionCount'] ?? 0,
    avgConfidence: json['avgConfidence']?.toDouble() ?? 0.0,
    createdAt: DateTime.parse(json['createdAt']),
    updatedAt: DateTime.parse(json['updatedAt']),
  );
}

/// Detection record with MGRS location
class DetectionRecord {
  final String id;
  final String signalProfileId;
  final DateTime timestamp;
  final String mgrsLocation;     // MGRS grid reference (e.g., "17TLJ8834916401")
  final double latitude;
  final double longitude;
  final double freqCenter;       // Detected center frequency MHz
  final double bandwidth;        // Detected bandwidth MHz
  final double confidence;
  final double snr;              // Signal-to-noise ratio dB
  final String? iqFilePath;      // Path to captured IQ data
  final bool addedToTraining;    // Whether used for model training

  DetectionRecord({
    required this.id,
    required this.signalProfileId,
    required this.timestamp,
    required this.mgrsLocation,
    required this.latitude,
    required this.longitude,
    required this.freqCenter,
    required this.bandwidth,
    required this.confidence,
    this.snr = 0.0,
    this.iqFilePath,
    this.addedToTraining = false,
  });

  Map<String, dynamic> toJson() => {
    'id': id,
    'signalProfileId': signalProfileId,
    'timestamp': timestamp.toIso8601String(),
    'mgrsLocation': mgrsLocation,
    'latitude': latitude,
    'longitude': longitude,
    'freqCenter': freqCenter,
    'bandwidth': bandwidth,
    'confidence': confidence,
    'snr': snr,
    'iqFilePath': iqFilePath,
    'addedToTraining': addedToTraining,
  };

  factory DetectionRecord.fromJson(Map<String, dynamic> json) => DetectionRecord(
    id: json['id'],
    signalProfileId: json['signalProfileId'],
    timestamp: DateTime.parse(json['timestamp']),
    mgrsLocation: json['mgrsLocation'],
    latitude: json['latitude'].toDouble(),
    longitude: json['longitude'].toDouble(),
    freqCenter: json['freqCenter'].toDouble(),
    bandwidth: json['bandwidth'].toDouble(),
    confidence: json['confidence'].toDouble(),
    snr: json['snr']?.toDouble() ?? 0.0,
    iqFilePath: json['iqFilePath'],
    addedToTraining: json['addedToTraining'] ?? false,
  );
}

/// Mission config - snapshot of signals to detect
class MissionConfig {
  final String id;
  final String name;
  final List<String> signalProfileIds;
  final double minConfidenceThreshold;
  final DateTime createdAt;

  MissionConfig({
    required this.id,
    required this.name,
    required this.signalProfileIds,
    this.minConfidenceThreshold = 0.7,
    DateTime? createdAt,
  }) : createdAt = createdAt ?? DateTime.now();

  Map<String, dynamic> toJson() => {
    'id': id,
    'name': name,
    'signalProfileIds': signalProfileIds,
    'minConfidenceThreshold': minConfidenceThreshold,
    'createdAt': createdAt.toIso8601String(),
  };

  factory MissionConfig.fromJson(Map<String, dynamic> json) => MissionConfig(
    id: json['id'],
    name: json['name'],
    signalProfileIds: List<String>.from(json['signalProfileIds']),
    minConfidenceThreshold: json['minConfidenceThreshold']?.toDouble() ?? 0.7,
    createdAt: DateTime.parse(json['createdAt']),
  );

  String toConfigFile() => const JsonEncoder.withIndent('  ').convert(toJson());
}

/// Signal Database - manages all signal profiles and detections
class SignalDatabaseNotifier extends StateNotifier<SignalDatabaseState> {
  SignalDatabaseNotifier() : super(SignalDatabaseState.initial()) {
    _loadMockData();
  }

  void _loadMockData() {
    // Add some mock signal profiles
    final mockProfiles = [
      SignalProfile(
        id: 'sig_001',
        name: 'Alpha-1',
        modType: 'FSK',
        modRate: 9600,
        freqMin: 905.0,
        freqMax: 910.0,
        detectionCount: 47,
        avgConfidence: 0.92,
        modelPath: '/models/alpha1.trt',
      ),
      SignalProfile(
        id: 'sig_002',
        name: 'Bravo-2',
        modType: 'PSK',
        modRate: 19200,
        freqMin: 912.0,
        freqMax: 918.0,
        pri: 5.0,
        detectionCount: 23,
        avgConfidence: 0.87,
        modelPath: '/models/bravo2.trt',
      ),
      SignalProfile(
        id: 'sig_003',
        name: 'Charlie-3',
        modType: 'AM',
        freqMin: 920.0,
        freqMax: 925.0,
        detectionCount: 12,
        avgConfidence: 0.78,
        modelPath: '/models/charlie3.trt',
      ),
    ];

    // Mock MGRS locations (simulated positions)
    final mockDetections = [
      DetectionRecord(
        id: 'det_001',
        signalProfileId: 'sig_001',
        timestamp: DateTime.now().subtract(const Duration(minutes: 5)),
        mgrsLocation: '17TLJ8834916401',
        latitude: 38.8977,
        longitude: -77.0365,
        freqCenter: 907.5,
        bandwidth: 2.0,
        confidence: 0.94,
        snr: 18.5,
      ),
      DetectionRecord(
        id: 'det_002',
        signalProfileId: 'sig_002',
        timestamp: DateTime.now().subtract(const Duration(minutes: 3)),
        mgrsLocation: '17TLJ8835016410',
        latitude: 38.8980,
        longitude: -77.0360,
        freqCenter: 915.0,
        bandwidth: 4.0,
        confidence: 0.89,
        snr: 15.2,
      ),
    ];

    state = state.copyWith(
      profiles: Map.fromEntries(mockProfiles.map((p) => MapEntry(p.id, p))),
      detections: mockDetections,
    );
  }

  /// Add or update a signal profile
  void upsertProfile(SignalProfile profile) {
    final updated = {...state.profiles};
    updated[profile.id] = profile;
    state = state.copyWith(profiles: updated);
  }

  /// Record a new detection
  void addDetection(DetectionRecord detection) {
    final detections = [...state.detections, detection];
    
    // Update profile stats if high confidence
    if (detection.confidence >= 0.7 && state.profiles.containsKey(detection.signalProfileId)) {
      final profile = state.profiles[detection.signalProfileId]!;
      final newCount = profile.detectionCount + 1;
      final newAvg = ((profile.avgConfidence * profile.detectionCount) + detection.confidence) / newCount;
      
      upsertProfile(profile.copyWith(
        detectionCount: newCount,
        avgConfidence: newAvg,
      ));
    }
    
    state = state.copyWith(detections: detections);
  }

  /// Create a mission config from selected profiles
  MissionConfig createMissionConfig(String name, List<String> profileIds, {double threshold = 0.7}) {
    final config = MissionConfig(
      id: 'mission_${DateTime.now().millisecondsSinceEpoch}',
      name: name,
      signalProfileIds: profileIds,
      minConfidenceThreshold: threshold,
    );
    
    final configs = [...state.missionConfigs, config];
    state = state.copyWith(missionConfigs: configs);
    
    return config;
  }

  /// Load a mission config
  void loadMissionConfig(MissionConfig config) {
    state = state.copyWith(activeMissionConfig: config);
  }

  /// Get profiles for active mission
  List<SignalProfile> getActiveMissionProfiles() {
    if (state.activeMissionConfig == null) return [];
    return state.activeMissionConfig!.signalProfileIds
        .where((id) => state.profiles.containsKey(id))
        .map((id) => state.profiles[id]!)
        .toList();
  }
}

class SignalDatabaseState {
  final Map<String, SignalProfile> profiles;
  final List<DetectionRecord> detections;
  final List<MissionConfig> missionConfigs;
  final MissionConfig? activeMissionConfig;

  SignalDatabaseState({
    required this.profiles,
    required this.detections,
    required this.missionConfigs,
    this.activeMissionConfig,
  });

  factory SignalDatabaseState.initial() => SignalDatabaseState(
    profiles: {},
    detections: [],
    missionConfigs: [],
  );

  /// Copy with support for nullable activeMissionConfig
  /// Use clearActiveMission: true to explicitly set activeMissionConfig to null
  SignalDatabaseState copyWith({
    Map<String, SignalProfile>? profiles,
    List<DetectionRecord>? detections,
    List<MissionConfig>? missionConfigs,
    MissionConfig? activeMissionConfig,
    bool clearActiveMission = false,
  }) {
    return SignalDatabaseState(
      profiles: profiles ?? this.profiles,
      detections: detections ?? this.detections,
      missionConfigs: missionConfigs ?? this.missionConfigs,
      activeMissionConfig: clearActiveMission ? null : (activeMissionConfig ?? this.activeMissionConfig),
    );
  }
}

/// Provider for the signal database
final signalDatabaseProvider = StateNotifierProvider<SignalDatabaseNotifier, SignalDatabaseState>((ref) {
  return SignalDatabaseNotifier();
});

/// Mock MGRS generator (in production, use real GPS + conversion)
String generateMockMgrs() {
  final random = DateTime.now().millisecondsSinceEpoch;
  final zone = '17T';
  final grid = 'LJ';
  final easting = 88340 + (random % 100);
  final northing = 16400 + ((random ~/ 100) % 100);
  return '$zone$grid$easting$northing';
}
