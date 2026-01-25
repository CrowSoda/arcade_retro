// lib/features/live_detection/widgets/video_waterfall_display.dart
/// Row-strip waterfall display - renders from local RGBA pixel buffer
/// Receives strips from backend, stitches locally, renders with RawImage
/// Detection boxes tracked by row index for perfect sync

import 'dart:async';
import 'dart:typed_data';
import 'dart:ui' as ui;
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../../../core/config/theme.dart';
import '../../../core/services/backend_launcher.dart';
import '../providers/video_stream_provider.dart';
import '../providers/map_provider.dart' show getSOIColor, soiVisibilityProvider;
import '../../settings/settings_screen.dart' show waterfallTimeSpanProvider;

/// Waterfall display using row-strip streaming
class VideoWaterfallDisplay extends ConsumerStatefulWidget {
  final String host;
  
  const VideoWaterfallDisplay({
    super.key,
    this.host = 'localhost',
  });

  @override
  ConsumerState<VideoWaterfallDisplay> createState() => _VideoWaterfallDisplayState();
}

class _VideoWaterfallDisplayState extends ConsumerState<VideoWaterfallDisplay> {
  bool _hasAttemptedConnect = false;
  ui.Image? _waterfallImage;
  ui.Image? _previousImage;  // Double-buffer: keeps old image alive for one more frame
  int _lastFrameCount = -1;

  @override
  void initState() {
    super.initState();
  }

  @override
  void dispose() {
    _waterfallImage?.dispose();
    _previousImage?.dispose();
    super.dispose();
  }

  void _tryConnect(int port) {
    if (_hasAttemptedConnect) return;
    _hasAttemptedConnect = true;
    
    final notifier = ref.read(videoStreamProvider.notifier);
    debugPrint('[VideoWaterfallDisplay] Auto-connecting to ws://${widget.host}:$port/ws/video');
    notifier.connect(widget.host, port);
  }

  /// Convert RGBA pixel buffer to ui.Image for rendering
  Future<ui.Image?> _createImageFromPixels(Uint8List pixels, int width, int height) async {
    if (pixels.isEmpty || width <= 0 || height <= 0) return null;
    
    final completer = Completer<ui.Image>();
    
    ui.decodeImageFromPixels(
      pixels,
      width,
      height,
      ui.PixelFormat.rgba8888,
      (image) => completer.complete(image),
    );
    
    return completer.future;
  }

  @override
  Widget build(BuildContext context) {
    final streamState = ref.watch(videoStreamProvider);
    final backendState = ref.watch(backendLauncherProvider);
    
    // Listen for time span changes and send to backend
    ref.listen<double>(waterfallTimeSpanProvider, (previous, next) {
      final currentState = ref.read(videoStreamProvider);
      debugPrint('[Waterfall] Time span listener fired: $previous → $next, connected: ${currentState.isConnected}');
      if (previous != next && currentState.isConnected) {
        ref.read(videoStreamProvider.notifier).setTimeSpan(next);
      }
    });
    
    // Send initial time span when connection state changes to connected
    ref.listen<VideoStreamState>(videoStreamProvider, (previous, next) {
      if (previous?.isConnected != true && next.isConnected) {
        final currentTimeSpan = ref.read(waterfallTimeSpanProvider);
        if ((currentTimeSpan - 5.0).abs() > 0.1) {
          ref.read(videoStreamProvider.notifier).setTimeSpan(currentTimeSpan);
        }
      }
    });
    
    // Auto-connect when backend port is discovered
    if (backendState.wsPort != null && 
        !streamState.isConnected && 
        !streamState.isConnecting &&
        !_hasAttemptedConnect) {
      WidgetsBinding.instance.addPostFrameCallback((_) {
        _tryConnect(backendState.wsPort!);
      });
    }

    return LayoutBuilder(
      builder: (context, constraints) {
        final size = constraints.biggest;
        const leftMargin = 50.0;
        const bottomMargin = 25.0;
        const topMargin = 8.0;
        const rightMargin = 8.0;

        final plotRect = Rect.fromLTRB(
          leftMargin,
          topMargin,
          size.width - rightMargin,
          size.height - bottomMargin,
        );

        return RepaintBoundary(
          child: Stack(
            children: [
              // Background
              Container(color: G20Colors.surfaceDark),
              
              // Waterfall image (rendered from pixel buffer)
              Positioned(
                left: plotRect.left,
                top: plotRect.top,
                width: plotRect.width,
                height: plotRect.height,
                child: _buildWaterfall(streamState),
              ),
              
              // Detection overlay (row-index based positioning)
              Positioned(
                left: plotRect.left,
                top: plotRect.top,
                width: plotRect.width,
                height: plotRect.height,
                child: _DetectionOverlayLayer(
                  totalRowsReceived: streamState.totalRowsReceived,
                  bufferHeight: streamState.bufferHeight,
                  detections: streamState.detections,
                ),
              ),

              // Time axis
              Positioned(
                left: 0,
                top: topMargin,
                bottom: bottomMargin,
                width: leftMargin - 4,
                child: const _VideoTimeAxis(),
              ),

              // Frequency axis
              Positioned(
                left: leftMargin,
                right: rightMargin,
                bottom: 0,
                height: bottomMargin,
                child: _VideoFrequencyAxis(metadata: streamState.metadata),
              ),

              // Connection status overlay
              if (!streamState.isConnected) _buildConnectionOverlay(streamState),
              
              // Stats overlay (top-right corner)
              Positioned(
                top: topMargin + 4,
                right: rightMargin + 4,
                child: _buildStatsOverlay(streamState),
              ),
            ],
          ),
        );
      },
    );
  }

  Widget _buildWaterfall(VideoStreamState state) {
    final pixelBuffer = state.pixelBuffer;
    
    if (pixelBuffer == null || pixelBuffer.isEmpty) {
      return Container(
        color: const Color(0xFF1A0033),
        child: const Center(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              CircularProgressIndicator(color: G20Colors.primary),
              SizedBox(height: 16),
              Text(
                'Waiting for waterfall stream...',
                style: TextStyle(color: Colors.white54),
              ),
            ],
          ),
        ),
      );
    }

    // Only rebuild image when frame count changes
    if (state.frameCount != _lastFrameCount) {
      _lastFrameCount = state.frameCount;
      
      // DOUBLE-BUFFER: Create image async, don't block
      _createImageFromPixels(pixelBuffer, state.bufferWidth, state.bufferHeight)
          .then((newImage) {
        if (newImage != null && mounted) {
          setState(() {
            // Dispose the OLD previous image (not the current one!)
            _previousImage?.dispose();
            // Move current to previous
            _previousImage = _waterfallImage;
            // Set new as current
            _waterfallImage = newImage;
          });
        }
      });
    }
    
    // Always return current image (or placeholder)
    if (_waterfallImage != null) {
      return RawImage(
        image: _waterfallImage,
        fit: BoxFit.fill,
        filterQuality: FilterQuality.low,
      );
    }
    
    return Container(color: const Color(0xFF1A0033));
  }

  Widget _buildConnectionOverlay(VideoStreamState state) {
    final backendState = ref.watch(backendLauncherProvider);
    final port = backendState.wsPort;
    
    return Positioned.fill(
      child: Container(
        color: Colors.black54,
        child: Center(
          child: Card(
            color: G20Colors.cardDark,
            child: Padding(
              padding: const EdgeInsets.all(24),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  if (state.isConnecting) ...[
                    const CircularProgressIndicator(color: G20Colors.primary),
                    const SizedBox(height: 16),
                    const Text(
                      'Connecting to row-strip stream...',
                      style: TextStyle(color: Colors.white),
                    ),
                  ] else if (port == null) ...[
                    const Icon(Icons.hourglass_empty, size: 48, color: Colors.white54),
                    const SizedBox(height: 16),
                    const Text(
                      'Waiting for backend...',
                      style: TextStyle(color: Colors.white70),
                      textAlign: TextAlign.center,
                    ),
                  ] else ...[
                    const Icon(Icons.videocam_off, size: 48, color: Colors.white54),
                    const SizedBox(height: 16),
                    Text(
                      state.error ?? 'Not connected',
                      style: const TextStyle(color: Colors.white70),
                      textAlign: TextAlign.center,
                    ),
                    const SizedBox(height: 16),
                    ElevatedButton.icon(
                      onPressed: () {
                        _hasAttemptedConnect = false;
                        ref.read(videoStreamProvider.notifier).connect(widget.host, port);
                      },
                      icon: const Icon(Icons.refresh),
                      label: const Text('Reconnect'),
                      style: ElevatedButton.styleFrom(
                        backgroundColor: G20Colors.primary,
                      ),
                    ),
                  ],
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildStatsOverlay(VideoStreamState state) {
    if (!state.isConnected) return const SizedBox.shrink();
    
    final metadata = state.metadata;
    final mode = metadata?.mode ?? 'unknown';
    final bufferInfo = '${state.bufferWidth}×${state.bufferHeight}';
    
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
      decoration: BoxDecoration(
        color: Colors.black54,
        borderRadius: BorderRadius.circular(4),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.end,
        mainAxisSize: MainAxisSize.min,
        children: [
          Text(
            '${state.fps.toStringAsFixed(1)} fps',
            style: const TextStyle(
              fontSize: 10,
              color: G20Colors.primary,
              fontWeight: FontWeight.bold,
            ),
          ),
          Text(
            bufferInfo,
            style: const TextStyle(fontSize: 9, color: Colors.white70),
          ),
          Text(
            mode.toUpperCase(),
            style: TextStyle(
              fontSize: 9,
              color: mode == 'row_strip' ? Colors.greenAccent : Colors.orangeAccent,
            ),
          ),
          Text(
            'rows: ${state.totalRowsReceived}',
            style: const TextStyle(fontSize: 8, color: Colors.white54),
          ),
        ],
      ),
    );
  }
}

/// Time axis for video waterfall
class _VideoTimeAxis extends ConsumerWidget {
  const _VideoTimeAxis();

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final displayTimeSec = ref.watch(waterfallTimeSpanProvider);

    String formatTime(double seconds) {
      if (seconds >= 1.0) {
        return '-${seconds.toStringAsFixed(0)}s';
      } else {
        return '-${(seconds * 1000).toStringAsFixed(0)}ms';
      }
    }

    return Container(
      padding: const EdgeInsets.only(right: 4),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        crossAxisAlignment: CrossAxisAlignment.end,
        children: [
          Text(
            formatTime(displayTimeSec),
            style: const TextStyle(fontSize: 10, color: G20Colors.textSecondaryDark),
          ),
          Text(
            formatTime(displayTimeSec * 2 / 3),
            style: const TextStyle(fontSize: 10, color: G20Colors.textSecondaryDark),
          ),
          Text(
            formatTime(displayTimeSec / 3),
            style: const TextStyle(fontSize: 10, color: G20Colors.textSecondaryDark),
          ),
          const Text(
            'Now',
            style: TextStyle(
              fontSize: 10,
              color: G20Colors.primary,
              fontWeight: FontWeight.bold,
            ),
          ),
        ],
      ),
    );
  }
}

/// Frequency axis for video waterfall
class _VideoFrequencyAxis extends StatelessWidget {
  final StreamMetadata? metadata;

  const _VideoFrequencyAxis({this.metadata});

  @override
  Widget build(BuildContext context) {
    const centerFreqMHz = 825.0;
    const bandwidthMHz = 20.0;

    final low = centerFreqMHz - bandwidthMHz / 2;
    final high = centerFreqMHz + bandwidthMHz / 2;

    return Container(
      padding: const EdgeInsets.only(top: 4),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(
            '${low.toStringAsFixed(1)}',
            style: const TextStyle(fontSize: 10, color: G20Colors.textSecondaryDark),
          ),
          Text(
            '${centerFreqMHz.toStringAsFixed(1)} MHz',
            style: const TextStyle(fontSize: 10, color: G20Colors.textSecondaryDark),
          ),
          Text(
            '${high.toStringAsFixed(1)}',
            style: const TextStyle(fontSize: 10, color: G20Colors.textSecondaryDark),
          ),
        ],
      ),
    );
  }
}

/// Detection overlay layer with ROW-INDEX positioning
/// 
/// Detection.absoluteRow = which row the detection was made at
/// totalRowsReceived = current bottom row of the buffer
/// Position: rowsAgo = totalRowsReceived - detection.absoluteRow
class _DetectionOverlayLayer extends ConsumerWidget {
  final int totalRowsReceived;
  final int bufferHeight;
  final List<VideoDetection> detections;

  const _DetectionOverlayLayer({
    required this.totalRowsReceived,
    required this.bufferHeight,
    required this.detections,
  });

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    if (detections.isEmpty) {
      return const SizedBox.shrink();
    }

    return LayoutBuilder(
      builder: (context, constraints) {
        final plotWidth = constraints.maxWidth;
        final plotHeight = constraints.maxHeight;
        
        // Pixels per row
        final pixelsPerRow = bufferHeight > 0 ? plotHeight / bufferHeight : 1.0;
        
        final visibleBoxes = <Widget>[];
        
        for (final det in detections) {
          // Check visibility toggle
          final isVisible = ref.watch(soiVisibilityProvider(det.className));
          if (!isVisible) continue;
          
          // === FREQUENCY AXIS (X) - from y1/y2 ===
          final left = det.y1 * plotWidth;
          final right = det.y2 * plotWidth;
          final boxWidth = (right - left).abs();
          
          // === TIME AXIS (Y) - ROW-INDEX positioning ===
          // How many rows ago was this detection?
          final rowsAgo = totalRowsReceived - det.absoluteRow;
          
          // Skip if outside visible range
          if (rowsAgo < 0 || rowsAgo >= bufferHeight) continue;
          
          // Box height from rowSpan
          final boxHeight = (det.rowSpan * pixelsPerRow).clamp(8.0, plotHeight * 0.3);
          
          // Y position: rowsAgo=0 → bottom, rowsAgo=bufferHeight → top
          final boxBottom = plotHeight - (rowsAgo * pixelsPerRow);
          final boxTop = boxBottom - boxHeight;
          
          // Skip if outside visible area
          if (boxTop > plotHeight || boxBottom < 0) continue;
          
          // Skip if too small
          if (boxWidth < 4) continue;
          
          final color = getSOIColor(det.className);
          
          // Debug info
          final debugStr = 'row=${det.absoluteRow} ago=$rowsAgo';
          
          visibleBoxes.add(
            Positioned(
              left: left.clamp(0.0, plotWidth - boxWidth),
              top: boxTop.clamp(0.0, plotHeight - boxHeight),
              width: boxWidth.clamp(4.0, plotWidth),
              height: boxHeight.clamp(8.0, plotHeight),
              child: _DetectionBoxWidget(
                detection: det,
                color: color,
                debugInfo: debugStr,
              ),
            ),
          );
        }
        
        return Stack(
          clipBehavior: Clip.hardEdge,
          children: visibleBoxes,
        );
      },
    );
  }
}

/// Individual detection box widget
class _DetectionBoxWidget extends StatelessWidget {
  final VideoDetection detection;
  final Color color;
  final String? debugInfo;

  const _DetectionBoxWidget({
    required this.detection,
    required this.color,
    this.debugInfo,
  });

  @override
  Widget build(BuildContext context) {
    final showDebug = debugInfo != null && debugInfo!.isNotEmpty;
    
    return Container(
      decoration: BoxDecoration(
        border: Border.all(
          color: color,
          width: detection.isSelected ? 3 : 2,
        ),
        color: color.withOpacity(detection.isSelected ? 0.25 : 0.1),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 3, vertical: 1),
            color: color.withOpacity(0.85),
            child: Text(
              '${detection.className} ${(detection.confidence * 100).toStringAsFixed(0)}%',
              style: const TextStyle(
                fontSize: 8,
                color: Colors.white,
                fontWeight: FontWeight.bold,
              ),
            ),
          ),
          if (showDebug)
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 3, vertical: 1),
              color: Colors.black54,
              child: Text(
                debugInfo!,
                style: const TextStyle(
                  fontSize: 7,
                  color: Colors.yellow,
                  fontFamily: 'monospace',
                ),
              ),
            ),
        ],
      ),
    );
  }
}

