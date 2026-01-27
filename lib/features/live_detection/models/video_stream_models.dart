/// Video Stream Models - Data models for WebSocket video streaming
///
/// Contains VideoDetection, StreamMetadata, VideoStreamState, and WaterfallSource
library;

import 'dart:typed_data';

/// Message types from backend (must match unified_pipeline.py)
class MessageType {
  static const int strip = 0x01;      // Row strip data
  static const int detection = 0x02;  // Detection JSON
  static const int metadata = 0x03;   // Stream metadata

  // Alias for compatibility
  static const int videoFrame = strip;
}

/// Waterfall source indicator - which RX stream feeds the display
/// Values must match backend WaterfallSource enum in unified_pipeline.py
enum WaterfallSource {
  rx1Scanning,   // 0 - RX1 scanning, no detection/recording
  rx1Recording,  // 1 - RX1 detected something and is recording
  rx2Recording,  // 2 - RX2 is collecting (handoff from RX1)
  manual,        // 3 - Manual collection on any RX
}

/// Extension to get display properties for WaterfallSource
extension WaterfallSourceExtension on WaterfallSource {
  /// Human-readable label for the source
  String get label {
    switch (this) {
      case WaterfallSource.rx1Scanning:
        return 'SCANNING';
      case WaterfallSource.rx1Recording:
        return 'RX1 REC';
      case WaterfallSource.rx2Recording:
        return 'RX2 REC';
      case WaterfallSource.manual:
        return 'MANUAL';
    }
  }

  /// Whether this source indicates active recording
  bool get isRecording => this != WaterfallSource.rx1Scanning;
}

/// Metadata about the stream
class StreamMetadata {
  final String mode;           // 'row_strip' or 'video'
  final int stripWidth;        // Width of each strip (2048)
  final int rowsPerStrip;      // Rows per strip (~38)
  final int videoFps;          // Target FPS (30)
  final int suggestedBufferHeight;  // How many rows client should buffer
  final double timeSpanSeconds;     // Current time span
  final String encoder;        // 'rgba_raw' or 'image/jpeg'

  StreamMetadata({
    required this.mode,
    required this.stripWidth,
    required this.rowsPerStrip,
    required this.videoFps,
    required this.suggestedBufferHeight,
    required this.timeSpanSeconds,
    required this.encoder,
  });

  factory StreamMetadata.fromJson(Map<String, dynamic> json) {
    return StreamMetadata(
      mode: json['mode'] ?? 'video',
      stripWidth: json['strip_width'] ?? json['video_width'] ?? 2048,
      rowsPerStrip: json['rows_per_strip'] ?? 38,
      videoFps: json['video_fps'] ?? 30,
      suggestedBufferHeight: json['suggested_buffer_height'] ?? 5700,
      timeSpanSeconds: (json['time_span_seconds'] ?? 5.0).toDouble(),
      encoder: json['encoder'] ?? 'rgba_raw',
    );
  }

  bool get isRowStripMode => mode == 'row_strip';
  bool get isJpeg => encoder == 'image/jpeg';

  @override
  String toString() => 'StreamMetadata(mode=$mode, ${stripWidth}×$rowsPerStrip strips, buffer=$suggestedBufferHeight)';
}

/// Detection from inference with ROW-BASED positioning
class VideoDetection {
  final int detectionId;
  final double x1, y1, x2, y2;
  final double confidence;
  final int classId;
  final String className;
  final double pts;
  final bool isSelected;
  final int absoluteRow;  // Absolute row index when detection was made
  final int rowSpan;      // Number of rows detection spans

  VideoDetection({
    required this.detectionId,
    required this.x1,
    required this.y1,
    required this.x2,
    required this.y2,
    required this.confidence,
    required this.classId,
    required this.className,
    required this.pts,
    this.isSelected = false,
    this.absoluteRow = 0,
    this.rowSpan = 1,
  });

  factory VideoDetection.fromJson(Map<String, dynamic> json, double pts, {int baseRow = 0, int rowsInFrame = 38}) {
    final rowOffset = json['row_offset'] as int? ?? 0;
    final rowSpan = json['row_span'] as int? ?? 1;

    return VideoDetection(
      detectionId: json['detection_id'] ?? 0,
      x1: (json['x1'] ?? 0).toDouble(),
      y1: (json['y1'] ?? 0).toDouble(),
      x2: (json['x2'] ?? 0).toDouble(),
      y2: (json['y2'] ?? 0).toDouble(),
      confidence: (json['confidence'] ?? 0).toDouble(),
      classId: json['class_id'] ?? 0,
      className: json['class_name'] ?? 'unknown',
      pts: pts,
      absoluteRow: baseRow + rowOffset,
      rowSpan: rowSpan > 0 ? rowSpan : 1,
    );
  }

  VideoDetection copyWith({bool? isSelected}) {
    return VideoDetection(
      detectionId: detectionId,
      x1: x1, y1: y1, x2: x2, y2: y2,
      confidence: confidence,
      classId: classId,
      className: className,
      pts: pts,
      isSelected: isSelected ?? this.isSelected,
      absoluteRow: absoluteRow,
      rowSpan: rowSpan,
    );
  }

  @override
  String toString() => 'VideoDetection(id: $detectionId, class: $className, conf: ${(confidence * 100).toStringAsFixed(0)}%, row: $absoluteRow)';
}

/// State for the row-strip stream
class VideoStreamState {
  final bool isConnected;
  final bool isConnecting;
  final StreamMetadata? metadata;

  // ROW-STRIP MODE: Local pixel buffer maintained by Flutter
  final Uint8List? pixelBuffer;  // RGBA pixel buffer (width × bufferHeight × 4)
  final int bufferWidth;         // Width of pixel buffer (2048)
  final int bufferHeight;        // Height of pixel buffer (suggested by backend)

  final List<VideoDetection> detections;
  final double currentPts;
  final int frameCount;
  final String? error;
  final double fps;

  // ROW SYNC: Row-based tracking
  final int totalRowsReceived;  // Monotonic counter
  final int rowsPerStrip;       // Rows per strip (~38)

  // PSD data for the line graph at bottom of screen
  final Float32List? psd;       // Raw dB values for PSD chart

  // Waterfall source indicator - which RX stream is feeding the display
  final WaterfallSource waterfallSource;

  const VideoStreamState({
    this.isConnected = false,
    this.isConnecting = false,
    this.metadata,
    this.pixelBuffer,
    this.bufferWidth = 2048,
    this.bufferHeight = 2850,  // ~2.5s at 30fps, reduced for performance
    this.detections = const [],
    this.currentPts = 0.0,
    this.frameCount = 0,
    this.error,
    this.fps = 0.0,
    this.totalRowsReceived = 0,
    this.rowsPerStrip = 38,
    this.psd,
    this.waterfallSource = WaterfallSource.rx1Scanning,
  });

  /// Whether the waterfall is showing a recording stream (not just scanning)
  bool get isRecording => waterfallSource.isRecording;

  VideoStreamState copyWith({
    bool? isConnected,
    bool? isConnecting,
    StreamMetadata? metadata,
    Uint8List? pixelBuffer,
    int? bufferWidth,
    int? bufferHeight,
    List<VideoDetection>? detections,
    double? currentPts,
    int? frameCount,
    String? error,
    double? fps,
    int? totalRowsReceived,
    int? rowsPerStrip,
    Float32List? psd,
    WaterfallSource? waterfallSource,
  }) {
    return VideoStreamState(
      isConnected: isConnected ?? this.isConnected,
      isConnecting: isConnecting ?? this.isConnecting,
      metadata: metadata ?? this.metadata,
      pixelBuffer: pixelBuffer ?? this.pixelBuffer,
      bufferWidth: bufferWidth ?? this.bufferWidth,
      bufferHeight: bufferHeight ?? this.bufferHeight,
      detections: detections ?? this.detections,
      currentPts: currentPts ?? this.currentPts,
      frameCount: frameCount ?? this.frameCount,
      error: error,
      fps: fps ?? this.fps,
      totalRowsReceived: totalRowsReceived ?? this.totalRowsReceived,
      rowsPerStrip: rowsPerStrip ?? this.rowsPerStrip,
      psd: psd ?? this.psd,
      waterfallSource: waterfallSource ?? this.waterfallSource,
    );
  }

  static VideoStreamState initial() => const VideoStreamState();
}

/// Callback type for forwarding detections
typedef DetectionCallback = void Function(List<VideoDetection> detections, double pts);
