// lib/features/live_detection/providers/video_stream_provider.dart
/// Row-strip streaming provider for waterfall display
/// Receives RGBA row strips from backend and stitches into local pixel buffer
/// Handles detection overlays with row-index based positioning

import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter/foundation.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:web_socket_channel/web_socket_channel.dart';

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

/// Provider for row-strip video stream
class VideoStreamNotifier extends StateNotifier<VideoStreamState> {
  WebSocketChannel? _channel;
  StreamSubscription? _subscription;
  DetectionCallback? _onDetections;
  
  // Local pixel buffer for stitching strips
  Uint8List? _pixelBuffer;
  int _bufferWidth = 2048;
  int _bufferHeight = 2850;  // Default: 2.5s × 30fps × 38 rows
  static const int _bytesPerPixel = 4;  // RGBA
  
  // FPS calculation
  DateTime? _lastFrameTime;
  int _fpsFrameCount = 0;
  double _measuredFps = 0.0;
  
  // Detection buffer
  List<VideoDetection> _detectionBuffer = [];
  double _currentTimeSpan = 5.0;
  
  VideoStreamNotifier() : super(VideoStreamState.initial());
  
  void setDetectionCallback(DetectionCallback callback) {
    _onDetections = callback;
  }

  /// Initialize or resize pixel buffer
  void _initPixelBuffer(int width, int height) {
    if (_pixelBuffer != null && _bufferWidth == width && _bufferHeight == height) {
      return;  // Already correct size
    }
    
    debugPrint('[VideoStream] Initializing pixel buffer: ${width}×$height (${(width * height * _bytesPerPixel / 1024 / 1024).toStringAsFixed(1)} MB)');
    
    _bufferWidth = width;
    _bufferHeight = height;
    _pixelBuffer = Uint8List(width * height * _bytesPerPixel);
    
    // Initialize with viridis dark purple (68, 1, 84, 255)
    _fillWithViridisBackground();
    
    state = state.copyWith(
      pixelBuffer: _pixelBuffer,
      bufferWidth: width,
      bufferHeight: height,
    );
  }
  
  /// Clear pixel buffer to viridis dark purple background
  /// Called when waterfall source changes to prevent mixing old/new data
  void _clearPixelBuffer() {
    if (_pixelBuffer == null) return;
    _fillWithViridisBackground();
  }
  
  /// Fill buffer with viridis dark purple (68, 1, 84, 255)
  void _fillWithViridisBackground() {
    if (_pixelBuffer == null) return;
    for (int i = 0; i < _pixelBuffer!.length; i += _bytesPerPixel) {
      _pixelBuffer![i] = 68;      // R
      _pixelBuffer![i + 1] = 1;   // G
      _pixelBuffer![i + 2] = 84;  // B
      _pixelBuffer![i + 3] = 255; // A
    }
  }

  Future<void> connect(String host, int port) async {
    if (state.isConnected || state.isConnecting) {
      debugPrint('[VideoStream] Already connected or connecting');
      return;
    }
    
    state = state.copyWith(isConnecting: true, error: null);
    
    try {
      final url = 'ws://$host:$port/ws/video';
      debugPrint('[VideoStream] Connecting to $url');

      _channel = WebSocketChannel.connect(Uri.parse(url));

      _subscription = _channel!.stream.listen(
        _onMessage,
        onError: _onError,
        onDone: _onDone,
      );

      // Reset first frame flag on connect
      _firstFrameReceived = false;

      state = state.copyWith(isConnected: true, isConnecting: false);
      debugPrint('[VideoStream] Connected!');
    } catch (e) {
      debugPrint('[VideoStream] Connection error: $e');
      state = state.copyWith(
        isConnected: false,
        isConnecting: false,
        error: e.toString(),
      );
    }
  }

  void disconnect() {
    debugPrint('[VideoStream] Disconnecting...');
    _subscription?.cancel();
    _subscription = null;
    _channel?.sink.close();
    _channel = null;
    _pixelBuffer = null;
    state = VideoStreamState.initial();
  }

  void _onMessage(dynamic message) {
    if (message is! List<int>) {
      debugPrint('[VideoStream] Unexpected message type: ${message.runtimeType}');
      return;
    }

    final bytes = message is Uint8List ? message : Uint8List.fromList(message);
    if (bytes.isEmpty) return;

    final messageType = bytes[0];
    final payload = bytes.sublist(1);

    switch (messageType) {
      case MessageType.strip:
        _handleStrip(payload);
        break;
      case MessageType.detection:
        _handleDetection(payload);
        break;
      case MessageType.metadata:
        _handleMetadata(payload);
        break;
      default:
        debugPrint('[VideoStream] Unknown message type: $messageType');
    }
  }

  /// Handle row strip - scroll buffer and paste new rows at bottom
  void _handleStrip(Uint8List data) {
    // PERF TIMING: Measure strip handling time
    final stopwatch = Stopwatch()..start();
    
    // Skip first frame if setting is enabled (avoids garbage/initialization data)
    if (_skipFirstFrame && !_firstFrameReceived) {
      _firstFrameReceived = true;
      debugPrint('[VideoStream] Skipping first frame per skip setting');
      return;
    }
    _firstFrameReceived = true;  // Mark as received either way
    
    // Parse binary header (17 bytes):
    // - frame_id: uint32 (4 bytes)
    // - total_rows: uint32 (4 bytes)
    // - rows_in_strip: uint16 (2 bytes)
    // - strip_width: uint16 (2 bytes)
    // - pts: float32 (4 bytes)
    // - source_id: uint8 (1 byte) - 0=SCAN, 1=RX1_REC, 2=RX2_REC, 3=MANUAL
    
    if (data.length < 17) {
      debugPrint('[VideoStream] Strip too short: ${data.length} bytes');
      return;
    }
    
    final header = ByteData.sublistView(data, 0, 17);
    // final frameId = header.getUint32(0, Endian.little);
    final totalRows = header.getUint32(4, Endian.little);
    final rowsInStrip = header.getUint16(8, Endian.little);
    final stripWidth = header.getUint16(10, Endian.little);
    final pts = header.getFloat32(12, Endian.little);
    final sourceId = header.getUint8(16);  // Waterfall source indicator
    
    // Parse source ID to enum (clamp to valid range)
    final newSource = WaterfallSource.values[sourceId.clamp(0, WaterfallSource.values.length - 1)];
    
    // Check if source changed - clear buffer to prevent mixing old/new data
    if (newSource != state.waterfallSource && _pixelBuffer != null) {
      debugPrint('[VideoStream] Source changed: ${state.waterfallSource.label} -> ${newSource.label}, clearing buffer');
      _clearPixelBuffer();
      _detectionBuffer.clear();
    }
    
    // Extract RGBA pixel data
    final expectedRgbaBytes = rowsInStrip * stripWidth * _bytesPerPixel;
    
    if (data.length < 17 + expectedRgbaBytes) {
      debugPrint('[VideoStream] Strip pixel data too short: ${data.length} < ${17 + expectedRgbaBytes}');
      return;
    }
    
    final pixelData = data.sublist(17, 17 + expectedRgbaBytes);
    
    // Extract PSD dB values (Float32, after RGBA data)
    // NOTE: Must copy to aligned buffer since 17-byte header breaks 4-byte alignment
    final psdStart = 17 + expectedRgbaBytes;
    final expectedPsdBytes = stripWidth * 4;  // Float32 = 4 bytes per element
    Float32List? psdData;
    if (data.length >= psdStart + expectedPsdBytes) {
      // Copy to aligned buffer (can't use view due to 17-byte header alignment issue)
      final psdBytes = data.sublist(psdStart, psdStart + expectedPsdBytes);
      final alignedBuffer = Uint8List(expectedPsdBytes);
      alignedBuffer.setAll(0, psdBytes);
      psdData = Float32List.view(alignedBuffer.buffer);
    }
    
    // Initialize buffer if needed
    if (_pixelBuffer == null) {
      _initPixelBuffer(stripWidth, state.metadata?.suggestedBufferHeight ?? 5700);
    }
    
    // Calculate FPS with rolling window
    final now = DateTime.now();
    _fpsFrameCount++;
    if (_lastFrameTime != null) {
      final elapsed = now.difference(_lastFrameTime!).inMilliseconds;
      if (elapsed >= 1000) {
        _measuredFps = _fpsFrameCount * 1000.0 / elapsed;
        _fpsFrameCount = 0;
        _lastFrameTime = now;  // Reset timer after measurement
      }
    } else {
      _lastFrameTime = now;  // Initialize on first frame
    }
    
    // SCROLL: Move buffer up by rowsInStrip rows
    final bytesPerRow = _bufferWidth * _bytesPerPixel;
    final shiftBytes = rowsInStrip * bytesPerRow;
    
    if (shiftBytes < _pixelBuffer!.length) {
      // Shift existing data up (oldest at top, newest at bottom)
      _pixelBuffer!.setRange(0, _pixelBuffer!.length - shiftBytes, 
          _pixelBuffer!.sublist(shiftBytes));
      
      // PASTE: Copy new strip to bottom of buffer
      final bottomStart = _pixelBuffer!.length - shiftBytes;
      _pixelBuffer!.setRange(bottomStart, bottomStart + expectedRgbaBytes, pixelData);
    }
    
    // Cull old detections
    final cutoffRow = totalRows - _bufferHeight;
    _detectionBuffer = _detectionBuffer.where((d) => d.absoluteRow >= cutoffRow).toList();
    
    // Update state with PSD data and waterfall source
    state = state.copyWith(
      pixelBuffer: _pixelBuffer,
      frameCount: state.frameCount + 1,
      fps: _measuredFps,
      totalRowsReceived: totalRows + rowsInStrip,
      rowsPerStrip: rowsInStrip,
      currentPts: pts.toDouble(),
      detections: List.from(_detectionBuffer),
      psd: psdData,
      waterfallSource: newSource,  // Update source indicator
    );
    
    // PERF TIMING: Print every 30 frames
    stopwatch.stop();
    if (state.frameCount % 30 == 0) {
      debugPrint('[PERF] Strip handle: ${stopwatch.elapsedMilliseconds}ms (shift: ${shiftBytes ~/ 1024}KB)');
    }
  }

  void _handleDetection(Uint8List jsonData) {
    try {
      final jsonStr = utf8.decode(jsonData);
      final data = json.decode(jsonStr) as Map<String, dynamic>;

      final pts = (data['pts'] ?? 0).toDouble();
      final detList = (data['detections'] as List<dynamic>?) ?? [];
      
      final baseRow = data['base_row'] as int? ?? state.totalRowsReceived;
      final rowsInFrame = data['rows_in_frame'] as int? ?? 38;

      final newDetections = detList
          .map((d) => VideoDetection.fromJson(
                d as Map<String, dynamic>,
                pts,
                baseRow: baseRow,
                rowsInFrame: rowsInFrame,
              ))
          .toList();

      _detectionBuffer.addAll(newDetections);
      
      state = state.copyWith(
        detections: List.from(_detectionBuffer),
        currentPts: pts,
      );
      
      if (_onDetections != null && newDetections.isNotEmpty) {
        _onDetections!(newDetections, pts);
      }
    } catch (e) {
      debugPrint('[VideoStream] Detection parse error: $e');
    }
  }

  void _handleMetadata(Uint8List jsonData) {
    try {
      final jsonStr = utf8.decode(jsonData);
      final data = json.decode(jsonStr) as Map<String, dynamic>;

      final metadata = StreamMetadata.fromJson(data);
      debugPrint('[VideoStream] Metadata: $metadata');
      
      // ROW-STRIP MODE: Initialize pixel buffer with suggested size
      if (metadata.isRowStripMode) {
        _initPixelBuffer(metadata.stripWidth, metadata.suggestedBufferHeight);
      }
      
      state = state.copyWith(
        metadata: metadata,
        rowsPerStrip: metadata.rowsPerStrip,
      );
    } catch (e) {
      debugPrint('[VideoStream] Metadata parse error: $e');
    }
  }

  void _onError(dynamic error) {
    debugPrint('[VideoStream] Error: $error');
    state = state.copyWith(isConnected: false, error: error.toString());
  }

  void _onDone() {
    debugPrint('[VideoStream] Connection closed');
    state = state.copyWith(isConnected: false);
  }
  
  void selectDetection(int detectionId) {
    final newDetections = state.detections.map((d) {
      if (d.detectionId == detectionId) {
        return d.copyWith(isSelected: !d.isSelected);
      }
      return d.copyWith(isSelected: false);
    }).toList();
    
    state = state.copyWith(detections: newDetections);
  }

  void setTimeSpan(double seconds) {
    debugPrint('[VideoStream] setTimeSpan: $seconds');
    
    _currentTimeSpan = seconds;
    
    if (_channel == null) {
      debugPrint('[VideoStream] Cannot set time span - not connected');
      return;
    }
    
    final msg = json.encode({
      'command': 'set_time_span',
      'seconds': seconds,
    });
    
    try {
      _channel!.sink.add(msg);
      debugPrint('[VideoStream] Sent time span command: ${seconds}s');
    } catch (e) {
      debugPrint('[VideoStream] Send FAILED: $e');
    }
  }

  void setFps(int fps) {
    debugPrint('[VideoStream] setFps: $fps');
    
    if (_channel == null) {
      debugPrint('[VideoStream] Cannot set FPS - not connected');
      return;
    }
    
    final msg = json.encode({
      'command': 'set_fps',
      'fps': fps,
    });
    
    try {
      _channel!.sink.add(msg);
      debugPrint('[VideoStream] Sent FPS command: ${fps}fps');
    } catch (e) {
      debugPrint('[VideoStream] Send FAILED: $e');
    }
  }

  void setScoreThreshold(double threshold) {
    debugPrint('[VideoStream] setScoreThreshold: $threshold');
    
    if (_channel == null) {
      debugPrint('[VideoStream] Cannot set score threshold - not connected');
      return;
    }
    
    final msg = json.encode({
      'command': 'set_score_threshold',
      'threshold': threshold,
    });
    
    try {
      _channel!.sink.add(msg);
      debugPrint('[VideoStream] Sent score threshold command: ${(threshold * 100).toInt()}%');
    } catch (e) {
      debugPrint('[VideoStream] Send FAILED: $e');
    }
  }

  /// Set FFT size for waterfall processing
  /// Valid sizes: 8192, 16384, 32768, 65536
  /// NOTE: Backend will warmup cuFFT kernels (100-500ms delay)
  void setFftSize(int size) {
    debugPrint('[VideoStream] setFftSize: $size');
    
    if (_channel == null) {
      debugPrint('[VideoStream] Cannot set FFT size - not connected');
      return;
    }
    
    // Validate size
    const validSizes = [8192, 16384, 32768, 65536];
    if (!validSizes.contains(size)) {
      debugPrint('[VideoStream] Invalid FFT size: $size. Valid: $validSizes');
      return;
    }
    
    final msg = json.encode({
      'command': 'set_fft_size',
      'size': size,
    });
    
    try {
      _channel!.sink.add(msg);
      debugPrint('[VideoStream] Sent FFT size command: $size (warmup may take 100-500ms)');
    } catch (e) {
      debugPrint('[VideoStream] Send FAILED: $e');
    }
  }

  /// Set dB range for waterfall display normalization
  /// minDb: noise floor (e.g., -100)
  /// maxDb: peak level (e.g., -20)
  void setDbRange(double minDb, double maxDb) {
    debugPrint('[VideoStream] setDbRange: $minDb to $maxDb dB');
    
    if (_channel == null) {
      debugPrint('[VideoStream] Cannot set dB range - not connected');
      return;
    }
    
    final msg = json.encode({
      'command': 'set_db_range',
      'min_db': minDb,
      'max_db': maxDb,
    });
    
    try {
      _channel!.sink.add(msg);
      debugPrint('[VideoStream] Sent dB range command: $minDb to $maxDb dB');
    } catch (e) {
      debugPrint('[VideoStream] Send FAILED: $e');
    }
  }

  // Track if first frame should be skipped
  bool _skipFirstFrame = false;
  bool _firstFrameReceived = false;

  /// Set whether to skip the first waterfall frame on connection
  /// Useful to avoid garbage/initialization data
  void setSkipFirstFrame(bool skip) {
    debugPrint('[VideoStream] setSkipFirstFrame: $skip');
    _skipFirstFrame = skip;
    _firstFrameReceived = false;  // Reset on setting change
  }

  /// Set colormap for waterfall display
  /// 0=Viridis, 1=Plasma, 2=Inferno, 3=Magma, 4=Turbo
  void setColormap(int colormapIndex) {
    debugPrint('[VideoStream] setColormap: $colormapIndex');
    
    if (_channel == null) {
      debugPrint('[VideoStream] Cannot set colormap - not connected');
      return;
    }
    
    // Clamp to valid range
    final index = colormapIndex.clamp(0, 4);
    
    final msg = json.encode({
      'command': 'set_colormap',
      'colormap': index,
    });
    
    try {
      _channel!.sink.add(msg);
      debugPrint('[VideoStream] Sent colormap command: $index');
    } catch (e) {
      debugPrint('[VideoStream] Send FAILED: $e');
    }
  }

  @override
  void dispose() {
    disconnect();
    super.dispose();
  }
}

/// Provider instance
final videoStreamProvider =
    StateNotifierProvider<VideoStreamNotifier, VideoStreamState>(
  (ref) => VideoStreamNotifier(),
);

/// Provider for connection URL configuration
final videoStreamUrlProvider = StateProvider<({String host, int port})>(
  (ref) => (host: 'localhost', port: 8765),
);
