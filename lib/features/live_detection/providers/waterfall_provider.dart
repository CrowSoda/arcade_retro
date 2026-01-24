import 'dart:async';
import 'dart:typed_data';
import 'package:flutter/foundation.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../../../core/grpc/inference_client.dart';
import '../../settings/settings_screen.dart';
import 'video_stream_provider.dart';

/// OPTIMIZED: Waterfall state with pre-rendered pixel buffer
/// Python sends RGBA pixels, we just scroll the buffer and insert new rows
class WaterfallState {
  final Uint8List pixelBuffer;  // RGBA pixels ready for display
  final Float32List psd;        // Raw dB values for PSD chart (smoothed)
  final int width;
  final int height;
  final double minDb;           // For PSD Y-axis
  final double maxDb;           // For PSD Y-axis
  final double centerFreqMHz;
  final double bandwidthMHz;
  final double sampleRateMHz;
  final bool isLoading;
  final String? error;
  final double currentPts;
  final int frameCount;  // Increment on each update to trigger rebuild

  const WaterfallState({
    required this.pixelBuffer,
    required this.psd,
    required this.width,
    required this.height,
    this.minDb = -80,
    this.maxDb = -20,
    this.centerFreqMHz = 825.0,
    this.bandwidthMHz = 20.0,
    this.sampleRateMHz = 20.0,
    this.isLoading = false,
    this.error,
    this.currentPts = 0.0,
    this.frameCount = 0,
  });

  WaterfallState copyWith({
    Uint8List? pixelBuffer,
    Float32List? psd,
    int? width,
    int? height,
    double? minDb,
    double? maxDb,
    double? centerFreqMHz,
    double? bandwidthMHz,
    double? sampleRateMHz,
    bool? isLoading,
    String? error,
    double? currentPts,
    int? frameCount,
  }) {
    return WaterfallState(
      pixelBuffer: pixelBuffer ?? this.pixelBuffer,
      psd: psd ?? this.psd,
      width: width ?? this.width,
      height: height ?? this.height,
      minDb: minDb ?? this.minDb,
      maxDb: maxDb ?? this.maxDb,
      centerFreqMHz: centerFreqMHz ?? this.centerFreqMHz,
      bandwidthMHz: bandwidthMHz ?? this.bandwidthMHz,
      sampleRateMHz: sampleRateMHz ?? this.sampleRateMHz,
      isLoading: isLoading ?? this.isLoading,
      error: error,
      currentPts: currentPts ?? this.currentPts,
      frameCount: frameCount ?? this.frameCount,
    );
  }

  static WaterfallState empty() => WaterfallState(
        pixelBuffer: Uint8List(0),
        psd: Float32List(0),
        width: 0,
        height: 0,
        isLoading: true,
      );
}

/// OPTIMIZED Waterfall provider
/// - Receives pre-rendered RGBA rows from Python
/// - Scrolls pixel buffer in-place (memmove, no allocation)
/// - DYNAMIC buffer size based on time span setting
class WaterfallNotifier extends StateNotifier<WaterfallState> {
  WaterfallNotifier(this._ref) : super(WaterfallState.empty()) {
    _init();
  }

  final Ref _ref;

  // Display parameters - WIDTH is fixed, HEIGHT is dynamic
  static const int _displayWidth = 2048;  // High resolution
  static const int _bytesPerPixel = 4;  // RGBA
  static const int _rowBytes = _displayWidth * _bytesPerPixel;
  static const int _rowsPerSecond = 30;  // 30 rows/sec = 33ms per row
  
  // DYNAMIC buffer size based on time span
  int _displayHeight = 150;  // Default: 5s × 30 = 150 rows
  int _totalBytes = 2048 * 150 * 4;  // 2048 width × 150 rows × 4 bytes
  
  // Pixel buffer - reallocated when time span changes
  late Uint8List _pixelBuffer;
  
  // WebSocket subscription
  StreamSubscription<WaterfallRow>? _waterfallSub;
  UnifiedPipelineManager? _pipeline;
  
  // Skip rendering when map is shown
  bool _skipRendering = false;
  
  // Frame timing - update state every N rows
  int _rowsReceived = 0;
  int _frameCount = 0;
  static const int _rowsPerStateUpdate = 2;  // Update every 2 rows (~15fps state updates)
  
  // PTS tracking
  double _currentPts = 0.0;
  
  // PSD smoothing (exponential moving average)
  Float64List? _smoothedPSD;
  static const double _psdAlpha = 0.05;  // Smoothing factor (~20 frames to converge)
  Float32List _latestPSD = Float32List(0);
  
  // Track current time span setting
  double _currentTimeSpan = 5.0;

  void _init() {
    // Get initial time span from provider
    _currentTimeSpan = _ref.read(waterfallTimeSpanProvider);
    _resizeBuffer(_currentTimeSpan);
    
    // Listen for time span changes
    _ref.listen<double>(waterfallTimeSpanProvider, (previous, next) {
      if ((previous ?? 5.0) != next) {
        debugPrint('[Waterfall] Time span changed: ${previous ?? 5.0}s → ${next}s');
        _currentTimeSpan = next;
        _resizeBuffer(next);
        
        // ALSO send to video stream backend to resize its buffer
        _ref.read(videoStreamProvider.notifier).setTimeSpan(next);
      }
    });
    
    debugPrint('[Waterfall] OPTIMIZED: Initialized with ${_currentTimeSpan}s time span (${_displayHeight} rows)');
  }

  /// Resize buffer based on time span (timeSpan × 30 rows per second)
  void _resizeBuffer(double timeSpan) {
    final newHeight = (timeSpan * _rowsPerSecond).round().clamp(30, 900);  // Min 1s, max 30s
    final newTotalBytes = _displayWidth * newHeight * _bytesPerPixel;
    
    debugPrint('[Waterfall] Resizing buffer: $_displayHeight → $newHeight rows (${newTotalBytes ~/ 1024}KB)');
    
    _displayHeight = newHeight;
    _totalBytes = newTotalBytes;
    
    // Allocate new buffer - filled with dark purple (viridis 0)
    _pixelBuffer = Uint8List(_totalBytes);
    
    // Initialize with background color (viridis dark purple: RGB 68, 1, 84)
    for (int i = 0; i < _totalBytes; i += _bytesPerPixel) {
      _pixelBuffer[i] = 68;      // R
      _pixelBuffer[i + 1] = 1;   // G
      _pixelBuffer[i + 2] = 84;  // B
      _pixelBuffer[i + 3] = 255; // A
    }
    
    // Reset frame counter to force immediate update
    _frameCount++;
    
    state = state.copyWith(
      pixelBuffer: _pixelBuffer,
      isLoading: false,
      width: _displayWidth,
      height: _displayHeight,
      centerFreqMHz: state.centerFreqMHz,
      bandwidthMHz: state.bandwidthMHz,
      sampleRateMHz: state.sampleRateMHz,
      frameCount: _frameCount,
    );
  }

  /// Connect to the unified pipeline manager
  void connectToPipeline(UnifiedPipelineManager pipeline) {
    if (_pipeline == pipeline && _waterfallSub != null) {
      debugPrint('[Waterfall] Already connected to pipeline');
      return;
    }
    
    // Cancel existing subscription
    _waterfallSub?.cancel();
    
    _pipeline = pipeline;
    _waterfallSub = pipeline.waterfallRows.listen(_onWaterfallRow);
    
    debugPrint('[Waterfall] Connected to unified pipeline (BINARY MODE)');
  }

  void _onWaterfallRow(WaterfallRow row) {
    if (_skipRendering) return;
    
    // Validate incoming row
    if (row.rgbaPixels.isEmpty || row.width != _displayWidth) {
      return;
    }
    
    // SCROLL: Move all rows UP by 1 row
    _pixelBuffer.setRange(0, _totalBytes - _rowBytes, 
        _pixelBuffer.sublist(_rowBytes));
    
    // INSERT: Copy new row to bottom
    final bottomRowStart = _totalBytes - _rowBytes;
    _pixelBuffer.setRange(bottomRowStart, _totalBytes, row.rgbaPixels);
    
    // PSD smoothing - exponential moving average
    if (row.psdData.isNotEmpty) {
      if (_smoothedPSD == null || _smoothedPSD!.length != row.psdData.length) {
        _smoothedPSD = Float64List(row.psdData.length);
        for (int i = 0; i < row.psdData.length; i++) {
          _smoothedPSD![i] = row.psdData[i].toDouble();
        }
      } else {
        for (int i = 0; i < row.psdData.length && i < _smoothedPSD!.length; i++) {
          _smoothedPSD![i] = _psdAlpha * row.psdData[i] + (1 - _psdAlpha) * _smoothedPSD![i];
        }
      }
      
      _latestPSD = Float32List(row.psdData.length);
      for (int i = 0; i < _smoothedPSD!.length; i++) {
        _latestPSD[i] = _smoothedPSD![i].toDouble();
      }
    }
    
    // Track PTS
    _currentPts = row.pts;
    _rowsReceived++;
    
    // Update state every N rows
    if (_rowsReceived >= _rowsPerStateUpdate) {
      _rowsReceived = 0;
      _frameCount++;
      state = state.copyWith(
        currentPts: _currentPts,
        frameCount: _frameCount,
        psd: _latestPSD,
      );
    }
  }

  /// Get pixel buffer for rendering - DIRECT ACCESS, no copy
  Uint8List get pixelBuffer => _pixelBuffer;

  void updateParams({double? centerFreqMHz, double? bandwidthMHz}) {
    state = state.copyWith(
      centerFreqMHz: centerFreqMHz, 
      bandwidthMHz: bandwidthMHz, 
    );
  }

  void setCenterFrequency(double freqMHz) {
    state = state.copyWith(centerFreqMHz: freqMHz);
    debugPrint('Center frequency set to: $freqMHz MHz');
  }

  void setBandwidth(double bwMHz) {
    state = state.copyWith(bandwidthMHz: bwMHz);
    debugPrint('Bandwidth set to: $bwMHz MHz');
  }

  void skipRendering() {
    _skipRendering = true;
    debugPrint('Waterfall: skipping renders (map visible)');
  }

  void resumeRendering() {
    _skipRendering = false;
    // Force immediate update
    _frameCount++;
    state = state.copyWith(
      currentPts: _currentPts,
      frameCount: _frameCount,
    );
    debugPrint('Waterfall: resumed rendering');
  }

  bool get isRenderingSkipped => _skipRendering;

  @override
  void dispose() {
    _waterfallSub?.cancel();
    super.dispose();
  }
}

final waterfallProvider = StateNotifierProvider<WaterfallNotifier, WaterfallState>((ref) {
  return WaterfallNotifier(ref);
});

/// Provider for the unified pipeline manager
final unifiedPipelineProvider = Provider<UnifiedPipelineManager?>((ref) {
  // This will be set when backend is ready
  return null;
});
