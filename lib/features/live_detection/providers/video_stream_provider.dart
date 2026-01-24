// lib/features/live_detection/providers/video_stream_provider.dart
/// Video stream provider for H.264/JPEG waterfall streaming
/// Handles WebSocket connection and message parsing

import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter/foundation.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:web_socket_channel/web_socket_channel.dart';

/// Message types from backend (must match unified_pipeline.py)
class MessageType {
  static const int videoFrame = 0x01;
  static const int detection = 0x02;
  static const int metadata = 0x03;
}

/// Metadata about the video stream
class StreamMetadata {
  final int videoWidth;
  final int videoHeight;
  final int videoFps;
  final String encoder; // 'video/h264' or 'image/jpeg'

  StreamMetadata({
    required this.videoWidth,
    required this.videoHeight,
    required this.videoFps,
    required this.encoder,
  });

  factory StreamMetadata.fromJson(Map<String, dynamic> json) {
    return StreamMetadata(
      videoWidth: json['video_width'] ?? 2048,
      videoHeight: json['video_height'] ?? 1024,
      videoFps: json['video_fps'] ?? 30,
      encoder: json['encoder'] ?? 'image/jpeg',
    );
  }

  bool get isH264 => encoder == 'video/h264';
  bool get isJpeg => encoder == 'image/jpeg';
  
  @override
  String toString() => 'StreamMetadata($videoWidth×$videoHeight @$videoFps fps, $encoder)';
}

/// Detection from inference
class VideoDetection {
  final int detectionId;
  final double x1, y1, x2, y2;
  final double confidence;
  final int classId;
  final String className;
  final double pts;
  final bool isSelected;

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
  });

  factory VideoDetection.fromJson(Map<String, dynamic> json, double pts) {
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
    );
  }
}

/// State for the video stream
class VideoStreamState {
  final bool isConnected;
  final bool isConnecting;
  final StreamMetadata? metadata;
  final Uint8List? currentFrame; // JPEG bytes (for JPEG mode)
  final List<VideoDetection> detections;
  final double currentPts;
  final int frameCount;
  final String? error;
  final double fps; // Measured FPS

  const VideoStreamState({
    this.isConnected = false,
    this.isConnecting = false,
    this.metadata,
    this.currentFrame,
    this.detections = const [],
    this.currentPts = 0.0,
    this.frameCount = 0,
    this.error,
    this.fps = 0.0,
  });

  VideoStreamState copyWith({
    bool? isConnected,
    bool? isConnecting,
    StreamMetadata? metadata,
    Uint8List? currentFrame,
    List<VideoDetection>? detections,
    double? currentPts,
    int? frameCount,
    String? error,
    double? fps,
  }) {
    return VideoStreamState(
      isConnected: isConnected ?? this.isConnected,
      isConnecting: isConnecting ?? this.isConnecting,
      metadata: metadata ?? this.metadata,
      currentFrame: currentFrame ?? this.currentFrame,
      detections: detections ?? this.detections,
      currentPts: currentPts ?? this.currentPts,
      frameCount: frameCount ?? this.frameCount,
      error: error,
      fps: fps ?? this.fps,
    );
  }
  
  static VideoStreamState initial() => const VideoStreamState();
}

/// Callback type for forwarding detections
typedef DetectionCallback = void Function(List<VideoDetection> detections, double pts);

/// Provider for video stream state
class VideoStreamNotifier extends StateNotifier<VideoStreamState> {
  WebSocketChannel? _channel;
  StreamSubscription? _subscription;
  DetectionCallback? _onDetections;
  
  // FPS calculation
  DateTime? _lastFrameTime;
  int _fpsFrameCount = 0;
  double _measuredFps = 0.0;
  
  // Detection accumulator - keeps detections for timeSpan seconds
  List<VideoDetection> _detectionBuffer = [];
  double _currentTimeSpan = 5.0;  // Default 5 seconds
  
  VideoStreamNotifier() : super(VideoStreamState.initial());
  
  /// Set callback for when detections are received
  void setDetectionCallback(DetectionCallback callback) {
    _onDetections = callback;
  }

  /// Connect to the backend WebSocket (video endpoint)
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

      state = state.copyWith(isConnected: true, isConnecting: false);
      debugPrint('[VideoStream] Connected!');
      
      // Note: Initial time span will be sent by the listener in video_waterfall_display.dart
      // when the widget rebuilds after connection state changes
    } catch (e) {
      debugPrint('[VideoStream] Connection error: $e');
      state = state.copyWith(
        isConnected: false,
        isConnecting: false,
        error: e.toString(),
      );
    }
  }

  /// Disconnect from the backend
  void disconnect() {
    debugPrint('[VideoStream] Disconnecting...');
    _subscription?.cancel();
    _subscription = null;
    _channel?.sink.close();
    _channel = null;
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
      case MessageType.videoFrame:
        _handleVideoFrame(payload);
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

  void _handleVideoFrame(Uint8List frameData) {
    // Calculate FPS
    final now = DateTime.now();
    if (_lastFrameTime != null) {
      _fpsFrameCount++;
      final elapsed = now.difference(_lastFrameTime!).inMilliseconds;
      if (elapsed >= 1000) {
        _measuredFps = _fpsFrameCount * 1000.0 / elapsed;
        _fpsFrameCount = 0;
        _lastFrameTime = now;
      }
    } else {
      _lastFrameTime = now;
    }
    
    // For JPEG mode, update the frame directly
    // For H.264 mode, this would feed to a video player
    if (state.metadata?.isJpeg == true) {
      state = state.copyWith(
        currentFrame: frameData,
        frameCount: state.frameCount + 1,
        fps: _measuredFps,
      );
    } else {
      // H.264 mode - would need to feed to media_kit player
      // For now, just track frame count
      state = state.copyWith(
        frameCount: state.frameCount + 1,
        fps: _measuredFps,
      );
      
      if (state.frameCount % 30 == 0) {
        debugPrint('[VideoStream] H.264 frame ${state.frameCount}: ${frameData.length} bytes');
      }
    }
  }

  void _handleDetection(Uint8List jsonData) {
    try {
      final jsonStr = utf8.decode(jsonData);
      final data = json.decode(jsonStr) as Map<String, dynamic>;

      final pts = (data['pts'] ?? 0).toDouble();
      final detList = (data['detections'] as List<dynamic>?) ?? [];
      

      // Parse new detections from this message
      final newDetections = detList
          .map((d) => VideoDetection.fromJson(d as Map<String, dynamic>, pts))
          .toList();

      // Add new detections to buffer
      _detectionBuffer.addAll(newDetections);
      
      // Prune old detections (older than timeSpan from current PTS)
      final cutoffPts = pts - _currentTimeSpan;
      _detectionBuffer = _detectionBuffer.where((d) => d.pts >= cutoffPts).toList();

      state = state.copyWith(
        detections: List.from(_detectionBuffer),  // Expose full buffer
        currentPts: pts,
      );
      
      // Forward to callback (for detection_provider)
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
      state = state.copyWith(metadata: metadata);

      debugPrint('[VideoStream] Metadata: $metadata');
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
  
  /// Select a detection
  void selectDetection(int detectionId) {
    final newDetections = state.detections.map((d) {
      if (d.detectionId == detectionId) {
        return d.copyWith(isSelected: !d.isSelected);
      }
      return d.copyWith(isSelected: false);
    }).toList();
    
    state = state.copyWith(detections: newDetections);
  }

  /// Send time span change to backend and update local buffer
  /// This resizes the waterfall buffer on the backend to match Flutter's display setting
  void setTimeSpan(double seconds) {
    debugPrint('[VideoStream] setTimeSpan CALLED with $seconds');
    
    _currentTimeSpan = seconds;
    
    // Prune detection buffer immediately to new time span
    if (state.currentPts > 0) {
      final cutoffPts = state.currentPts - _currentTimeSpan;
      _detectionBuffer = _detectionBuffer.where((d) => d.pts >= cutoffPts).toList();
      state = state.copyWith(detections: List.from(_detectionBuffer));
    }
    
    if (_channel == null) {
      debugPrint('[VideoStream] Cannot set time span - _channel is NULL!');
      return;
    }
    
    debugPrint('[VideoStream] _channel is NOT null, proceeding to send');
    
    final msg = json.encode({
      'command': 'set_time_span',
      'seconds': seconds,
    });
    
    debugPrint('[VideoStream] About to send: $msg');
    
    try {
      _channel!.sink.add(msg);
      debugPrint('[VideoStream] Sent time span command: ${seconds}s');
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
  (ref) => (host: 'localhost', port: 8765), // Same port as existing backend
);
