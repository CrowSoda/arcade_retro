import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../../../core/utils/dtg_formatter.dart';
import 'video_stream_provider.dart' as video_stream;

/// Detection model - represents a detected signal
class Detection {
  final String id;
  final int classId;
  final String className;
  final double confidence;
  final double x1; // Normalized 0-1 (freq position start)
  final double y1; // Normalized 0-1 (time position in detection frame)
  final double x2; // Normalized 0-1 (freq position end)
  final double y2; // Normalized 0-1 (time position in detection frame)
  final double freqMHz;      // Center frequency in MHz
  final double bandwidthMHz; // Bandwidth in MHz
  final String mgrsLocation; // MGRS grid reference (device location)
  final double latitude;     // Actual lat for map
  final double longitude;    // Actual lng for map
  final DateTime timestamp;
  final bool isSelected;
  final bool isActive;
  final int trackId;
  final double pts;          // Presentation timestamp - used to scroll with waterfall
  final int absoluteRow;     // Absolute row index when detection was made (for row-based pruning)

  const Detection({
    required this.id,
    required this.classId,
    required this.className,
    required this.confidence,
    required this.x1,
    required this.y1,
    required this.x2,
    required this.y2,
    this.freqMHz = 915.0,
    this.bandwidthMHz = 5.0,
    this.mgrsLocation = '13SDE1234567890',
    this.latitude = 39.7275,
    this.longitude = -104.7303,
    required this.timestamp,
    this.isSelected = false,
    this.isActive = true,
    this.trackId = 0,
    this.pts = 0.0,
    this.absoluteRow = 0,
  });

  Detection copyWith({
    String? id,
    int? classId,
    String? className,
    double? confidence,
    double? x1,
    double? y1,
    double? x2,
    double? y2,
    double? freqMHz,
    double? bandwidthMHz,
    String? mgrsLocation,
    double? latitude,
    double? longitude,
    DateTime? timestamp,
    bool? isSelected,
    bool? isActive,
    int? trackId,
    double? pts,
    int? absoluteRow,
  }) {
    return Detection(
      id: id ?? this.id,
      classId: classId ?? this.classId,
      className: className ?? this.className,
      confidence: confidence ?? this.confidence,
      x1: x1 ?? this.x1,
      y1: y1 ?? this.y1,
      x2: x2 ?? this.x2,
      y2: y2 ?? this.y2,
      freqMHz: freqMHz ?? this.freqMHz,
      bandwidthMHz: bandwidthMHz ?? this.bandwidthMHz,
      mgrsLocation: mgrsLocation ?? this.mgrsLocation,
      latitude: latitude ?? this.latitude,
      longitude: longitude ?? this.longitude,
      timestamp: timestamp ?? this.timestamp,
      isSelected: isSelected ?? this.isSelected,
      isActive: isActive ?? this.isActive,
      trackId: trackId ?? this.trackId,
      pts: pts ?? this.pts,
      absoluteRow: absoluteRow ?? this.absoluteRow,
    );
  }
}

/// Detection notifier - manages detection state
/// Detections come from gRPC inference stream or manual adds
class DetectionNotifier extends StateNotifier<List<Detection>> {
  DetectionNotifier() : super([]);

  // Call this to load mock data for testing without backend
  void loadMockData() {
    _addMockDetections();
  }

  /// Add mock detections for demo/testing - around Aurora, CO (39.7275, -104.7303)
  /// All detections are at FIXED positions - NO MOVEMENT
  void _addMockDetections() {
    final mockDetections = [
      Detection(
        id: 'det_001',
        classId: 1,
        className: 'LTE_UPLINK',
        confidence: 0.95,
        x1: 0.2, y1: 0.1, x2: 0.25, y2: 0.3,
        freqMHz: 820.5,
        bandwidthMHz: 5.0,
        mgrsLocation: '13SDE1234567890',
        latitude: 39.7275,
        longitude: -104.7303,
        timestamp: DateTime.now().subtract(const Duration(minutes: 5)),
        trackId: 1,
      ),
      Detection(
        id: 'det_002',
        classId: 2,
        className: 'WIFI_24',
        confidence: 0.88,
        x1: 0.45, y1: 0.2, x2: 0.55, y2: 0.5,
        freqMHz: 823.5,
        bandwidthMHz: 8.0,
        mgrsLocation: '13SDE1244567900',
        latitude: 39.735,
        longitude: -104.725,
        timestamp: DateTime.now().subtract(const Duration(minutes: 3)),
        trackId: 2,
      ),
      _createUnkDetection(
        id: 'det_003',
        confidence: 0.72,
        x1: 0.7, y1: 0.3, x2: 0.78, y2: 0.6,
        freqMHz: 830.2,
        bandwidthMHz: 4.0,
        mgrsLocation: '13SDE1254567910',
        latitude: 39.740,
        longitude: -104.745,
        timestamp: DateTime.now().subtract(const Duration(minutes: 1)),
        trackId: 3,
      ),
      Detection(
        id: 'det_004',
        classId: 3,
        className: 'creamy_chicken',
        confidence: 0.91,
        x1: 0.3, y1: 0.5, x2: 0.4, y2: 0.8,
        freqMHz: 822.0,
        bandwidthMHz: 6.0,
        mgrsLocation: '13SDE1264567920',
        latitude: 39.722,
        longitude: -104.715,
        timestamp: DateTime.now(),
        trackId: 4,
      ),
      Detection(
        id: 'det_005',
        classId: 4,
        className: 'BLUETOOTH',
        confidence: 0.85,
        x1: 0.6, y1: 0.4, x2: 0.65, y2: 0.7,
        freqMHz: 828.0,
        bandwidthMHz: 2.0,
        mgrsLocation: '13SDE1274567930',
        latitude: 39.718,
        longitude: -104.720,
        timestamp: DateTime.now().subtract(const Duration(seconds: 30)),
        trackId: 5,
      ),
      // Extra detections for more map coverage
      Detection(
        id: 'det_006',
        classId: 1,
        className: 'LTE_UPLINK',
        confidence: 0.89,
        x1: 0.15, y1: 0.2, x2: 0.2, y2: 0.4,
        freqMHz: 819.5,
        bandwidthMHz: 5.0,
        mgrsLocation: '13SDE1284567940',
        latitude: 39.750,
        longitude: -104.705,
        timestamp: DateTime.now().subtract(const Duration(seconds: 45)),
        trackId: 6,
      ),
      _createUnkDetection(
        id: 'det_007',
        confidence: 0.65,
        x1: 0.8, y1: 0.1, x2: 0.85, y2: 0.35,
        freqMHz: 832.0,
        bandwidthMHz: 3.0,
        mgrsLocation: '13SDE1294567950',
        latitude: 39.742,
        longitude: -104.740,
        timestamp: DateTime.now().subtract(const Duration(seconds: 20)),
        trackId: 7,
      ),
      Detection(
        id: 'det_008',
        classId: 3,
        className: 'creamy_chicken',
        confidence: 0.78,
        x1: 0.35, y1: 0.6, x2: 0.42, y2: 0.85,
        freqMHz: 821.5,
        bandwidthMHz: 7.0,
        mgrsLocation: '13SDE1304567960',
        latitude: 39.715,
        longitude: -104.755,
        timestamp: DateTime.now().subtract(const Duration(seconds: 10)),
        trackId: 8,
      ),
    ];

    state = mockDetections;
  }


  /// Add a detection from external source (model inference, etc.)
  void addDetection(Detection detection) {
    state = [...state, detection];
  }

  /// Add multiple detections
  void addDetections(List<Detection> detections) {
    state = [...state, ...detections];
  }

  /// Update an existing detection
  void updateDetection(String id, Detection updated) {
    state = [
      for (final det in state)
        det.id == id ? updated : det,
    ];
  }

  /// Remove a detection
  void removeDetection(String id) {
    state = state.where((det) => det.id != id).toList();
  }

  /// Clear all detections
  void clearAll() {
    state = [];
  }

  /// Select a detection by ID
  void selectDetection(String id) {
    state = [
      for (final det in state)
        det.copyWith(isSelected: det.id == id),
    ];
  }

  /// Clear all selections
  void clearSelection() {
    state = [
      for (final det in state) det.copyWith(isSelected: false),
    ];
  }

  /// Remove inactive/old detections by timestamp
  void pruneOld(Duration maxAge) {
    final cutoff = DateTime.now().subtract(maxAge);
    state = state.where((det) => det.timestamp.isAfter(cutoff)).toList();
  }

  /// Remove detections that have scrolled off screen based on PTS
  /// displayTimeSpan is ~8.5 seconds for 256 rows at 30fps
  void pruneByPts(double currentPts, {double displayTimeSpan = 8.5}) {
    final minPts = currentPts - displayTimeSpan;
    state = state.where((det) => det.pts >= minPts).toList();
  }

  /// Remove detections that have scrolled off based on absolute row position
  /// This is the preferred method for PSD box lifecycle management
  ///
  /// [currentRow] - The current total rows received (bottom of waterfall)
  /// [bufferHeight] - The visible buffer height in rows
  void pruneByAbsoluteRow(int currentRow, int bufferHeight) {
    final cutoffRow = currentRow - bufferHeight;
    final before = state.length;
    state = state.where((det) => det.absoluteRow >= cutoffRow).toList();
    final after = state.length;
    if (before != after) {
      // Debug logging - can be removed in production
      // debugPrint('[Detection] Pruned ${before - after} boxes, $after remaining');
    }
  }
}

/// Provider for detections
final detectionProvider =
    StateNotifierProvider<DetectionNotifier, List<Detection>>((ref) {
  return DetectionNotifier();
});

/// Provider for currently selected detection
final selectedDetectionProvider = Provider<Detection?>((ref) {
  final detections = ref.watch(detectionProvider);
  try {
    return detections.firstWhere((det) => det.isSelected);
  } catch (_) {
    return null;
  }
});

/// Provider for detection count
final detectionCountProvider = Provider<int>((ref) {
  return ref.watch(detectionProvider).length;
});

/// Helper to create UNK detections with proper naming: unk_[DTG]_[freq]MHz
/// Example: unk_220001ZJAN26_830.2MHz
Detection _createUnkDetection({
  required String id,
  required double confidence,
  required double x1,
  required double y1,
  required double x2,
  required double y2,
  required double freqMHz,
  required double bandwidthMHz,
  required String mgrsLocation,
  required double latitude,
  required double longitude,
  required DateTime timestamp,
  required int trackId,
}) {
  // Generate className as unk_DTG_freqMHz
  final className = generateUnkFilename(timestamp, freqMHz.toStringAsFixed(1));

  return Detection(
    id: id,
    classId: 0,
    className: className,
    confidence: confidence,
    x1: x1,
    y1: y1,
    x2: x2,
    y2: y2,
    freqMHz: freqMHz,
    bandwidthMHz: bandwidthMHz,
    mgrsLocation: mgrsLocation,
    latitude: latitude,
    longitude: longitude,
    timestamp: timestamp,
    trackId: trackId,
  );
}

/// Convert VideoStream Detection to DetectionProvider Detection
/// Assumes center freq = 825 MHz, bandwidth = 20 MHz
Detection convertVideoDetection(video_stream.VideoDetection vd, double pts) {
  const centerFreqMHz = 825.0;
  const bandwidthMHz = 20.0;

  // Calculate frequency from normalized y coordinates (frequency axis)
  final freqStart = centerFreqMHz - (bandwidthMHz / 2);
  final freqMHz = freqStart + ((vd.y1 + vd.y2) / 2) * bandwidthMHz;
  final detBandwidthMHz = (vd.y2 - vd.y1) * bandwidthMHz;

  return Detection(
    id: 'vid_${vd.detectionId}_${pts.toStringAsFixed(2)}',
    classId: vd.classId,
    className: vd.className,
    confidence: vd.confidence,
    x1: vd.x1,
    y1: vd.y1,
    x2: vd.x2,
    y2: vd.y2,
    freqMHz: freqMHz,
    bandwidthMHz: detBandwidthMHz,
    mgrsLocation: '13SDE1234567890', // TODO: Get from GPS
    latitude: 39.7275,  // TODO: Get from GPS
    longitude: -104.7303,  // TODO: Get from GPS
    timestamp: DateTime.now(),
    trackId: vd.detectionId,
    pts: pts,
    absoluteRow: vd.absoluteRow,  // Pass through for row-based pruning
  );
}
