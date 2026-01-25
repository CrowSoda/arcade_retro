// lib/core/grpc/inference_client.dart
/// Unified WebSocket client - receives BOTH waterfall and detections
/// Connects to Python unified pipeline at /ws/unified
/// 
/// OPTIMIZED: Binary protocol for waterfall (pre-rendered RGBA from Python)

import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter/foundation.dart';
import 'package:web_socket_channel/web_socket_channel.dart';

/// Detection from inference
class InferenceDetection {
  final int detectionId;
  final String className;
  final int classId;
  final double confidence;
  final double x1, y1, x2, y2;
  final double freqCenterMhz;
  final double freqBandwidthMhz;

  InferenceDetection({
    required this.detectionId,
    required this.className,
    required this.classId,
    required this.confidence,
    required this.x1,
    required this.y1,
    required this.x2,
    required this.y2,
    this.freqCenterMhz = 0,
    this.freqBandwidthMhz = 0,
  });

  factory InferenceDetection.fromJson(Map<String, dynamic> json) {
    return InferenceDetection(
      detectionId: json['detection_id'] ?? 0,
      className: json['class_name'] ?? 'unknown',
      classId: json['class_id'] ?? 0,
      confidence: (json['confidence'] ?? 0).toDouble(),
      x1: (json['x1'] ?? 0).toDouble(),
      y1: (json['y1'] ?? 0).toDouble(),
      x2: (json['x2'] ?? 0).toDouble(),
      y2: (json['y2'] ?? 0).toDouble(),
      freqCenterMhz: (json['freq_center_mhz'] ?? 0).toDouble(),
      freqBandwidthMhz: (json['freq_bandwidth_mhz'] ?? 0).toDouble(),
    );
  }
}

/// Detection frame with PTS for waterfall positioning
class DetectionFrame {
  final int frameId;
  final int timestampMs;
  final double pts;  // Presentation timestamp (seconds)
  final double inferenceMs;
  final List<InferenceDetection> detections;

  DetectionFrame({
    required this.frameId,
    required this.timestampMs,
    required this.pts,
    required this.inferenceMs,
    required this.detections,
  });

  factory DetectionFrame.fromJson(Map<String, dynamic> json) {
    final detectionsJson = json['detections'] as List<dynamic>? ?? [];
    return DetectionFrame(
      frameId: json['frame_id'] ?? 0,
      timestampMs: json['timestamp_ms'] ?? DateTime.now().millisecondsSinceEpoch,
      pts: (json['pts'] ?? 0).toDouble(),
      inferenceMs: (json['inference_ms'] ?? 0).toDouble(),
      detections: detectionsJson
          .map((d) => InferenceDetection.fromJson(d as Map<String, dynamic>))
          .toList(),
    );
  }
}

/// OPTIMIZED: Waterfall row with pre-rendered RGBA pixels + raw dB for PSD
class WaterfallRow {
  final int sequenceId;
  final double pts;  // Presentation timestamp (seconds)
  final int width;
  final Uint8List rgbaPixels;  // Pre-rendered RGBA from Python
  final Float32List psdData;   // Raw dB values for PSD chart

  WaterfallRow({
    required this.sequenceId,
    required this.pts,
    required this.width,
    required this.rgbaPixels,
    required this.psdData,
  });

  /// Parse binary waterfall message
  /// Format (20-byte header, padded for 4-byte alignment):
  /// - Byte 0: Message type (0x01 = waterfall)
  /// - Bytes 1-3: Padding (3 bytes)
  /// - Bytes 4-7: Sequence ID (uint32, little-endian)
  /// - Bytes 8-15: PTS (float64, little-endian)
  /// - Bytes 16-19: Width (uint32, little-endian)
  /// - Bytes 20 - (20 + width*4): RGBA pixel data (width * 4 bytes)
  /// - Remaining: Float32 dB values for PSD (width * 4 bytes)
  factory WaterfallRow.fromBinary(Uint8List data) {
    final byteData = ByteData.sublistView(data);
    
    // Skip message type (byte 0) and padding (bytes 1-3)
    final sequenceId = byteData.getUint32(4, Endian.little);
    final pts = byteData.getFloat64(8, Endian.little);
    final width = byteData.getUint32(16, Endian.little);
    
    // Extract RGBA pixels (already pre-rendered by Python!)
    const pixelStart = 20;  // Header is 20 bytes
    final pixelLength = width * 4;
    final rgbaPixels = Uint8List.sublistView(data, pixelStart, pixelStart + pixelLength);
    
    // Extract dB values for PSD chart (starts at 4-byte aligned offset)
    final psdStart = pixelStart + pixelLength;
    final psdLength = width;  // Number of Float32 elements (not bytes)
    Float32List psdData;
    if (psdStart + psdLength * 4 <= data.length) {
      // Create a view of the Float32 data (psdStart is 4-byte aligned: 20+4096=4116)
      psdData = data.buffer.asFloat32List(data.offsetInBytes + psdStart, psdLength);
    } else {
      psdData = Float32List(width);  // Empty fallback
    }
    
    return WaterfallRow(
      sequenceId: sequenceId,
      pts: pts,
      width: width,
      rgbaPixels: rgbaPixels,
      psdData: psdData,
    );
  }

  /// Legacy JSON factory (for backward compatibility)
  factory WaterfallRow.fromJson(Map<String, dynamic> json) {
    final rowData = json['row'] as List<dynamic>? ?? [];
    return WaterfallRow(
      sequenceId: json['sequence_id'] ?? 0,
      pts: (json['pts'] ?? 0).toDouble(),
      width: rowData.length,
      rgbaPixels: Uint8List(0),
      psdData: Float32List(0),
    );
  }
}

/// Unified WebSocket manager - handles waterfall + detections
/// OPTIMIZED: Binary protocol for waterfall rows
class UnifiedPipelineManager {
  final String host;
  final int port;
  
  WebSocketChannel? _channel;
  StreamSubscription? _subscription;
  bool _isConnected = false;
  
  // Streams for waterfall and detections
  final _waterfallController = StreamController<WaterfallRow>.broadcast();
  final _detectionController = StreamController<DetectionFrame>.broadcast();
  
  /// Stream of waterfall rows (pre-rendered RGBA)
  Stream<WaterfallRow> get waterfallRows => _waterfallController.stream;
  
  /// Stream of detection frames
  Stream<DetectionFrame> get detections => _detectionController.stream;
  
  /// Whether connected
  bool get isConnected => _isConnected;

  UnifiedPipelineManager({required this.host, required this.port});

  /// Connect to unified pipeline
  Future<bool> connect() async {
    if (_isConnected) return true;

    try {
      final wsUrl = 'ws://$host:$port/ws/unified';
      debugPrint('[UnifiedPipeline] Connecting to $wsUrl (BINARY MODE)');
      
      _channel = WebSocketChannel.connect(Uri.parse(wsUrl));
      
      // Listen for messages - handle BOTH binary and text
      _subscription = _channel!.stream.listen(
        _handleMessage,
        onError: (e) {
          debugPrint('[UnifiedPipeline] WebSocket error: $e');
          _isConnected = false;
        },
        onDone: () {
          debugPrint('[UnifiedPipeline] WebSocket closed');
          _isConnected = false;
        },
      );
      
      _isConnected = true;
      debugPrint('[UnifiedPipeline] Connected!');
      return true;
    } catch (e) {
      debugPrint('[UnifiedPipeline] Connection failed: $e');
      return false;
    }
  }

  void _handleMessage(dynamic message) {
    try {
      // BINARY message - ignore, this client uses /ws/unified which is deprecated
      // The new video stream uses /ws/video with a different binary format
      if (message is List<int>) {
        // Silently ignore binary messages - they're for the video stream
        return;
      }
      
      // TEXT message = JSON (detections, status, errors)
      if (message is String) {
        final json = jsonDecode(message) as Map<String, dynamic>;
        final msgType = json['type'] as String?;
        
        if (msgType == 'waterfall') {
          // Legacy JSON waterfall (shouldn't happen with new backend)
          debugPrint('[UnifiedPipeline] WARNING: Received legacy JSON waterfall');
          // Skip - we expect binary now
        } else if (msgType == 'detection_frame') {
          // Detection frame from inference
          final frame = DetectionFrame.fromJson(json);
          debugPrint('[UnifiedPipeline] Detections: ${frame.detections.length} @ PTS ${frame.pts.toStringAsFixed(3)}s');
          _detectionController.add(frame);
        } else if (msgType == 'error') {
          debugPrint('[UnifiedPipeline] Error: ${json['message']}');
        } else if (msgType == 'status') {
          debugPrint('[UnifiedPipeline] Status: ${json['is_running']}');
        }
      }
    } catch (e) {
      debugPrint('[UnifiedPipeline] Parse error: $e');
    }
  }

  /// Send stop command
  Future<void> stop() async {
    if (_channel != null) {
      _channel!.sink.add(jsonEncode({'command': 'stop'}));
      await Future.delayed(const Duration(milliseconds: 100));
    }
  }

  /// Disconnect
  Future<void> disconnect() async {
    await stop();
    await _subscription?.cancel();
    _subscription = null;
    await _channel?.sink.close();
    _channel = null;
    _isConnected = false;
    debugPrint('[UnifiedPipeline] Disconnected');
  }

  /// Dispose resources
  Future<void> dispose() async {
    await disconnect();
    await _waterfallController.close();
    await _detectionController.close();
  }
}

// Legacy InferenceManager for backward compatibility
// Can be removed once everything uses UnifiedPipelineManager
class InferenceManager {
  final String host;
  final int port;
  
  WebSocketChannel? _channel;
  StreamSubscription? _subscription;
  String? _currentSessionId;
  
  final _detectionController = StreamController<DetectionFrame>.broadcast();
  
  Stream<DetectionFrame> get detections => _detectionController.stream;
  String? get sessionId => _currentSessionId;
  bool get isRunning => _currentSessionId != null;

  InferenceManager({required this.host, required this.port});

  Future<bool> startInference({
    String? modelPath,
    double scoreThreshold = 0.5,
    int nfft = 4096,
    int noverlap = 2048,
    int chunkMs = 200,
    double dynRangeDb = 80.0,
  }) async {
    if (isRunning) return true;

    try {
      final wsUrl = 'ws://$host:$port/ws/inference';
      debugPrint('[InferenceManager] Connecting to $wsUrl');
      
      _channel = WebSocketChannel.connect(Uri.parse(wsUrl));
      
      final startCmd = jsonEncode({
        'command': 'start',
        'model_path': modelPath ?? '',
        'score_threshold': scoreThreshold,
        'nfft': nfft,
        'noverlap': noverlap,
        'chunk_ms': chunkMs,
        'dyn_range_db': dynRangeDb,
      });
      
      _channel!.sink.add(startCmd);
      
      _subscription = _channel!.stream.listen(
        (message) {
          try {
            final json = jsonDecode(message as String) as Map<String, dynamic>;
            final msgType = json['type'] as String?;
            
            if (msgType == 'session_started') {
              _currentSessionId = json['session_id'] as String?;
              debugPrint('[InferenceManager] Session started: $_currentSessionId');
            } else if (msgType == 'detection_frame') {
              final frame = DetectionFrame.fromJson(json);
              _detectionController.add(frame);
            } else if (msgType == 'error') {
              debugPrint('[InferenceManager] Error: ${json['message']}');
            }
          } catch (e) {
            debugPrint('[InferenceManager] Parse error: $e');
          }
        },
        onError: (e) {
          debugPrint('[InferenceManager] WebSocket error: $e');
          _currentSessionId = null;
        },
        onDone: () {
          debugPrint('[InferenceManager] WebSocket closed');
          _currentSessionId = null;
        },
      );
      
      await Future.delayed(const Duration(milliseconds: 500));
      _currentSessionId = 'pending';
      
      final runCmd = jsonEncode({
        'command': 'run',
        'center_freq_mhz': 825.0,
        'bandwidth_mhz': 20.0,
        'sample_rate': 20e6,
        'chunk_ms': chunkMs,
      });
      
      _channel!.sink.add(runCmd);
      return true;
    } catch (e) {
      debugPrint('[InferenceManager] Error starting: $e');
      return false;
    }
  }

  Future<void> stopInference() async {
    if (_channel != null) {
      _channel!.sink.add(jsonEncode({'command': 'stop'}));
      await Future.delayed(const Duration(milliseconds: 100));
    }
    await _subscription?.cancel();
    _subscription = null;
    await _channel?.sink.close();
    _channel = null;
    _currentSessionId = null;
  }

  Future<void> dispose() async {
    await stopInference();
    await _detectionController.close();
  }
}
