/// lib/features/live_detection/models/detection.dart
/// Detection model for signal detections

/// A detected signal with location and classification info
class Detection {
  /// Unique detection ID
  final String id;

  /// Class ID from the model (0 = unknown/background)
  final int classId;

  /// Human-readable class name
  final String className;

  /// Detection confidence (0.0 - 1.0)
  final double confidence;

  /// Bounding box coordinates (normalized 0.0 - 1.0)
  final double x1;
  final double y1;
  final double x2;
  final double y2;

  /// Frequency in MHz
  final double freqMHz;

  /// Bandwidth in MHz
  final double bandwidthMHz;

  /// MGRS grid location
  final String mgrsLocation;

  /// GPS latitude
  final double latitude;

  /// GPS longitude
  final double longitude;

  /// Detection timestamp
  final DateTime timestamp;

  /// Track ID for persistent tracking
  final int trackId;

  /// Presentation timestamp (video frame time)
  final double pts;

  /// Absolute row in waterfall (for pruning)
  final int absoluteRow;

  const Detection({
    required this.id,
    required this.classId,
    required this.className,
    required this.confidence,
    required this.x1,
    required this.y1,
    required this.x2,
    required this.y2,
    required this.freqMHz,
    required this.bandwidthMHz,
    required this.mgrsLocation,
    required this.latitude,
    required this.longitude,
    required this.timestamp,
    this.trackId = 0,
    this.pts = 0.0,
    this.absoluteRow = 0,
  });

  /// Width of the bounding box
  double get width => x2 - x1;

  /// Height of the bounding box
  double get height => y2 - y1;

  /// Center X coordinate
  double get centerX => (x1 + x2) / 2;

  /// Center Y coordinate
  double get centerY => (y1 + y2) / 2;

  /// Is this an unknown/UNK detection?
  bool get isUnknown => classId == 0 || className.toLowerCase().startsWith('unk');

  /// Create a copy with updated fields
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
      trackId: trackId ?? this.trackId,
      pts: pts ?? this.pts,
      absoluteRow: absoluteRow ?? this.absoluteRow,
    );
  }

  @override
  String toString() =>
      'Detection($className @ ${freqMHz.toStringAsFixed(1)} MHz, conf=${(confidence * 100).toStringAsFixed(1)}%)';

  @override
  bool operator ==(Object other) {
    if (identical(this, other)) return true;
    return other is Detection && other.id == id;
  }

  @override
  int get hashCode => id.hashCode;
}
