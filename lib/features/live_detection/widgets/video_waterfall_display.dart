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
import '../providers/sdr_config_provider.dart';
import '../../settings/settings_screen.dart' show waterfallTimeSpanProvider, waterfallFpsProvider, showStatsOverlayProvider;

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
      if (previous != next && currentState.isConnected) {
        ref.read(videoStreamProvider.notifier).setTimeSpan(next);
      }
    });
    
    // Listen for FPS changes and send to backend
    ref.listen<int>(waterfallFpsProvider, (previous, next) {
      final currentState = ref.read(videoStreamProvider);
      if (previous != next && currentState.isConnected) {
        ref.read(videoStreamProvider.notifier).setFps(next);
      }
    });
    
    // Send initial time span and FPS when connection state changes to connected
    ref.listen<VideoStreamState>(videoStreamProvider, (previous, next) {
      if (previous?.isConnected != true && next.isConnected) {
        final currentTimeSpan = ref.read(waterfallTimeSpanProvider);
        if ((currentTimeSpan - 5.0).abs() > 0.1) {
          ref.read(videoStreamProvider.notifier).setTimeSpan(currentTimeSpan);
        }
        final currentFps = ref.read(waterfallFpsProvider);
        if (currentFps != 30) {
          ref.read(videoStreamProvider.notifier).setFps(currentFps);
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
              
              // Waterfall image (rendered from pixel buffer) with long-press for manual capture
              Positioned(
                left: plotRect.left,
                top: plotRect.top,
                width: plotRect.width,
                height: plotRect.height,
                child: GestureDetector(
                  onLongPress: () {
                    final sdrConfig = ref.read(sdrConfigProvider);
                    // Start drawing mode immediately - duration selection comes AFTER drawing
                    ref.read(manualCaptureProvider.notifier).startDrawingMode(
                      sdrConfig.centerFreqMHz.toStringAsFixed(1),
                      durationMinutes: 1,  // Will be updated after drawing
                    );
                  },
                  child: _buildWaterfall(streamState),
                ),
              ),
              
              // Manual capture drawing overlay
              _VideoDrawingOverlay(plotRect: plotRect),
              
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
              
              // Stats overlay (top-right corner) - controlled by settings toggle
              if (ref.watch(showStatsOverlayProvider))
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
    // Wrap in RepaintBoundary to isolate repaints from rest of tree
    if (_waterfallImage != null) {
      return RepaintBoundary(
        child: RawImage(
          image: _waterfallImage,
          fit: BoxFit.fill,
          filterQuality: FilterQuality.low,
        ),
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
    
    final isRecording = state.isRecording;
    final sourceLabel = state.waterfallSource.label;
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
          // Recording indicator with red dot for recording states
          Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              if (isRecording)
                Container(
                  width: 8,
                  height: 8,
                  margin: const EdgeInsets.only(right: 4),
                  decoration: const BoxDecoration(
                    color: Colors.red,
                    shape: BoxShape.circle,
                  ),
                ),
              Text(
                sourceLabel,
                style: TextStyle(
                  fontSize: 10,
                  color: isRecording ? Colors.red : Colors.grey,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ],
          ),
          // FPS display
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
            'rows: ${state.totalRowsReceived}',
            style: const TextStyle(fontSize: 8, color: Colors.white54),
          ),
        ],
      ),
    );
  }
  
  /// Show duration selection dialog before entering drawing mode
  void _showCaptureDurationDialog(BuildContext context, WidgetRef ref, double centerFreqMHz) {
    showDialog(
      context: context,
      builder: (context) => _CaptureDurationDialog(
        centerFreqMHz: centerFreqMHz,
      ),
    );
  }
}

/// Duration selection dialog - shown on long-press before drawing mode
class _CaptureDurationDialog extends ConsumerStatefulWidget {
  final double centerFreqMHz;

  const _CaptureDurationDialog({required this.centerFreqMHz});

  @override
  ConsumerState<_CaptureDurationDialog> createState() => _CaptureDurationDialogState();
}

class _CaptureDurationDialogState extends ConsumerState<_CaptureDurationDialog> {
  int _durationMinutes = 1;  // Default 1 min
  
  static const _durations = [1, 2, 5, 10];  // minutes

  @override
  Widget build(BuildContext context) {
    return AlertDialog(
      backgroundColor: G20Colors.surfaceDark,
      title: Row(
        children: [
          Container(
            width: 12, height: 12,
            decoration: const BoxDecoration(color: G20Colors.primary, shape: BoxShape.circle),
          ),
          const SizedBox(width: 8),
          const Expanded(
            child: Text(
              'Manual Capture',
              style: TextStyle(fontSize: 14, color: G20Colors.textPrimaryDark),
              overflow: TextOverflow.ellipsis,
            ),
          ),
        ],
      ),
      content: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Frequency info
          Text(
            'Center Freq: ${widget.centerFreqMHz.toStringAsFixed(2)} MHz',
            style: const TextStyle(fontSize: 12, color: G20Colors.textSecondaryDark),
          ),
          const SizedBox(height: 16),
          
          // Duration selector
          const Text('Capture Duration:', style: TextStyle(fontSize: 12, color: G20Colors.textPrimaryDark)),
          const SizedBox(height: 8),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceEvenly,
            children: _durations.map((d) => _DurationChip(
              minutes: d,
              isSelected: _durationMinutes == d,
              onTap: () => setState(() => _durationMinutes = d),
            )).toList(),
          ),
          
          const SizedBox(height: 16),
          Container(
            padding: const EdgeInsets.all(8),
            decoration: BoxDecoration(
              color: G20Colors.primary.withOpacity(0.1),
              borderRadius: BorderRadius.circular(6),
              border: Border.all(color: G20Colors.primary.withOpacity(0.3)),
            ),
            child: const Text(
              'After clicking Continue, swipe on the waterfall to select frequency range',
              style: TextStyle(fontSize: 11, color: G20Colors.textSecondaryDark),
            ),
          ),
        ],
      ),
      actions: [
        TextButton(
          onPressed: () => Navigator.pop(context),
          child: const Text('Cancel', style: TextStyle(color: G20Colors.textSecondaryDark)),
        ),
        ElevatedButton(
          onPressed: () => _startDrawingMode(context),
          style: ElevatedButton.styleFrom(backgroundColor: G20Colors.primary),
          child: const Text('Continue'),
        ),
      ],
    );
  }

  void _startDrawingMode(BuildContext context) {
    Navigator.pop(context);
    
    // Start drawing mode with selected duration
    ref.read(manualCaptureProvider.notifier).startDrawingMode(
      widget.centerFreqMHz.toStringAsFixed(2),
      durationMinutes: _durationMinutes,
    );
  }
}

/// Duration chip button
class _DurationChip extends StatelessWidget {
  final int minutes;
  final bool isSelected;
  final VoidCallback onTap;

  const _DurationChip({
    required this.minutes,
    required this.isSelected,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
        decoration: BoxDecoration(
          color: isSelected ? G20Colors.primary : G20Colors.cardDark,
          borderRadius: BorderRadius.circular(16),
        ),
        child: Text(
          '$minutes min',
          style: TextStyle(
            fontSize: 12,
            color: isSelected ? Colors.white : G20Colors.textSecondaryDark,
            fontWeight: isSelected ? FontWeight.bold : FontWeight.normal,
          ),
        ),
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

// Debug counter for _DetectionOverlayLayer - outside class for Dart static semantics
int _waterfallDebugCounter = 0;

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
        final pixelsPerRow = bufferHeight > 0 ? plotHeight / bufferHeight : 1.0;
        
        final visibleBoxes = <Widget>[];
        
        _waterfallDebugCounter++;
        
        for (final det in detections) {
          // Check visibility toggle
          final isVisible = ref.watch(soiVisibilityProvider(det.className));
          if (!isVisible) continue;
          
          // === FREQUENCY AXIS (X) - from y1/y2 with 20% padding ===
          // NOTE: Frequency axis is FLIPPED (1.0 - y) to match waterfall display
          final detWidth = det.y2 - det.y1;
          final freqPadding = detWidth * 0.2;  // 20% padding each side
          final paddedY1 = (det.y1 - freqPadding).clamp(0.0, 1.0);
          final paddedY2 = (det.y2 + freqPadding).clamp(0.0, 1.0);
          // Flip: y1 becomes right, y2 becomes left
          final left = (1.0 - paddedY2) * plotWidth;
          final right = (1.0 - paddedY1) * plotWidth;
          final boxWidth = (right - left).abs().clamp(8.0, plotWidth);
          
          // === TIME AXIS (Y) - ROW-INDEX positioning ===
          // How many rows ago was this detection?
          final rowsAgo = totalRowsReceived - det.absoluteRow;
          
          // Skip if outside visible range
          if (rowsAgo < 0 || rowsAgo >= bufferHeight) continue;
          
          // Box height from rowSpan - use model's actual output directly
          final boxHeight = det.rowSpan * pixelsPerRow;
          
          // Y position: rowsAgo=0 → bottom, rowsAgo=bufferHeight → top
          final boxBottom = plotHeight - (rowsAgo * pixelsPerRow);
          final boxTop = boxBottom - boxHeight;
          
          // Skip if outside visible area
          if (boxTop > plotHeight || boxBottom < 0) continue;
          
          // Skip if too small
          if (boxWidth < 4) continue;
          
          final color = getSOIColor(det.className);
          
          visibleBoxes.add(
            Positioned(
              left: left.clamp(0.0, plotWidth - boxWidth),
              top: boxTop.clamp(0.0, plotHeight - boxHeight),
              width: boxWidth,
              height: boxHeight,
              child: _DetectionBoxWidget(
                detection: det,
                color: color,
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

/// Individual detection box widget - simple colored rectangle (no text labels)
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
    return Container(
      decoration: BoxDecoration(
        border: Border.all(
          color: color,
          width: 2,
        ),
        color: color.withOpacity(0.15),
      ),
    );
  }
}

/// Drawing overlay for manual capture - long press waterfall, then drag to draw bounding box
/// This captures a subband and saves it as .rfcap for labeling in Training tab
class _VideoDrawingOverlay extends ConsumerWidget {
  final Rect plotRect;

  const _VideoDrawingOverlay({required this.plotRect});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final captureState = ref.watch(manualCaptureProvider);
    
    if (!captureState.isDrawing) {
      return const SizedBox.shrink();
    }

    return Positioned(
      left: plotRect.left,
      top: plotRect.top,
      width: plotRect.width,
      height: plotRect.height,
      child: Stack(
        children: [
          // Semi-transparent overlay
          Container(color: Colors.black.withOpacity(0.3)),
          
          // Gesture detector for drawing
          GestureDetector(
            onPanStart: (details) {
              final x = details.localPosition.dx / plotRect.width;
              ref.read(manualCaptureProvider.notifier).startDrawing(x.clamp(0, 1), 0);
            },
            onPanUpdate: (details) {
              final x = details.localPosition.dx / plotRect.width;
              ref.read(manualCaptureProvider.notifier).updateDrawing(x.clamp(0, 1), 1);
            },
            onPanEnd: (_) {
              ref.read(manualCaptureProvider.notifier).finishDrawing();
            },
            child: Container(color: Colors.transparent),
          ),
          
          // The drawn box (if any)
          if (captureState.hasPendingBox)
            _VideoDrawnBox(
              x1: captureState.pendingBoxX1!,
              y1: captureState.pendingBoxY1!,
              x2: captureState.pendingBoxX2!,
              y2: captureState.pendingBoxY2!,
              plotRect: plotRect,
            ),
          
          // Instructions at top
          Positioned(
            top: 8,
            left: 0,
            right: 0,
            child: Center(
              child: Container(
                padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                decoration: BoxDecoration(
                  color: G20Colors.warning.withOpacity(0.9),
                  borderRadius: BorderRadius.circular(4),
                ),
                child: Text(
                  captureState.hasPendingBox 
                      ? (captureState.isCapturing ? 'Queue this capture?' : 'Start this capture?')
                      : 'Swipe left to right to select frequency range',
                  style: const TextStyle(fontSize: 12, color: Colors.black, fontWeight: FontWeight.bold),
                ),
              ),
            ),
          ),
          
          // Action buttons at bottom
          if (captureState.hasPendingBox)
            Positioned(
              bottom: 8,
              left: 0,
              right: 0,
              child: Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  ElevatedButton(
                    onPressed: () => ref.read(manualCaptureProvider.notifier).adjustBox(),
                    style: ElevatedButton.styleFrom(backgroundColor: G20Colors.cardDark),
                    child: const Text('Redraw'),
                  ),
                  const SizedBox(width: 12),
                  ElevatedButton(
                    onPressed: () => ref.read(manualCaptureProvider.notifier).cancelDrawing(),
                    style: ElevatedButton.styleFrom(backgroundColor: G20Colors.error),
                    child: const Text('Cancel'),
                  ),
                  const SizedBox(width: 12),
                  ElevatedButton(
                    onPressed: () {
                      // Show duration dialog AFTER drawing box
                      _showDurationDialogAfterDraw(context, ref);
                    },
                    style: ElevatedButton.styleFrom(backgroundColor: G20Colors.success),
                    child: const Text('Next'),
                  ),
                ],
              ),
            ),
        ],
      ),
    );
  }
}

/// Show duration dialog AFTER user draws the box
void _showDurationDialogAfterDraw(BuildContext context, WidgetRef ref) {
  final sdrConfig = ref.read(sdrConfigProvider);
  showDialog(
    context: context,
    builder: (dialogContext) => _PostDrawDurationDialog(
      centerFreqMHz: sdrConfig.centerFreqMHz,
    ),
  );
}

/// Duration dialog shown AFTER drawing - selecting duration starts the capture
class _PostDrawDurationDialog extends ConsumerStatefulWidget {
  final double centerFreqMHz;

  const _PostDrawDurationDialog({required this.centerFreqMHz});

  @override
  ConsumerState<_PostDrawDurationDialog> createState() => _PostDrawDurationDialogState();
}

class _PostDrawDurationDialogState extends ConsumerState<_PostDrawDurationDialog> {
  int _durationMinutes = 1;  // Default 1 min
  
  static const _durations = [1, 2, 5, 10];  // minutes

  /// Calculate center frequency from the drawn box, not the current view
  double get boxCenterFreqMHz {
    final captureState = ref.read(manualCaptureProvider);
    final sdrConfig = ref.read(sdrConfigProvider);
    
    if (!captureState.hasPendingBox) return widget.centerFreqMHz;
    
    // Box X positions are normalized 0-1 representing the frequency axis
    // x1 = left edge, x2 = right edge (may be swapped if drawn right-to-left)
    final x1 = captureState.pendingBoxX1!;
    final x2 = captureState.pendingBoxX2!;
    final boxCenterNorm = (x1 + x2) / 2;
    
    // Convert to frequency: left = centerFreq - BW/2, right = centerFreq + BW/2
    final lowFreq = sdrConfig.centerFreqMHz - sdrConfig.bandwidthMHz / 2;
    final boxCenterFreq = lowFreq + (boxCenterNorm * sdrConfig.bandwidthMHz);
    
    return boxCenterFreq;
  }

  @override
  Widget build(BuildContext context) {
    final captureState = ref.watch(manualCaptureProvider);
    
    return AlertDialog(
      backgroundColor: G20Colors.surfaceDark,
      title: Row(
        children: [
          Container(
            width: 12, height: 12,
            decoration: const BoxDecoration(color: G20Colors.warning, shape: BoxShape.circle),
          ),
          const SizedBox(width: 8),
          const Expanded(
            child: Text(
              'Set Capture Duration',
              style: TextStyle(fontSize: 14, color: G20Colors.textPrimaryDark),
              overflow: TextOverflow.ellipsis,
            ),
          ),
        ],
      ),
      content: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Frequency info - FIXED: Use calculated box center freq instead of view CF
          Text(
            'Center Freq: ${boxCenterFreqMHz.toStringAsFixed(2)} MHz',
            style: const TextStyle(fontSize: 12, color: G20Colors.textSecondaryDark),
          ),
          const SizedBox(height: 16),
          
          // Duration selector
          const Text('How long to capture?', style: TextStyle(fontSize: 12, color: G20Colors.textPrimaryDark)),
          const SizedBox(height: 8),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceEvenly,
            children: _durations.map((d) => _DurationChip(
              minutes: d,
              isSelected: _durationMinutes == d,
              onTap: () => setState(() => _durationMinutes = d),
            )).toList(),
          ),
          
          const SizedBox(height: 16),
          Container(
            padding: const EdgeInsets.all(8),
            decoration: BoxDecoration(
              color: G20Colors.success.withOpacity(0.1),
              borderRadius: BorderRadius.circular(6),
              border: Border.all(color: G20Colors.success.withOpacity(0.3)),
            ),
            child: Text(
              captureState.isCapturing 
                  ? 'This capture will be queued after the current one completes'
                  : 'Capture will start immediately and save to Training tab',
              style: const TextStyle(fontSize: 11, color: G20Colors.textSecondaryDark),
            ),
          ),
        ],
      ),
      actions: [
        TextButton(
          onPressed: () {
            Navigator.pop(context);
            // FIXED: Cancel drawing mode and clear pending box completely
            ref.read(manualCaptureProvider.notifier).cancelDrawing();
          },
          child: const Text('Cancel', style: TextStyle(color: G20Colors.textSecondaryDark)),
        ),
        ElevatedButton(
          onPressed: () => _startCapture(context),
          style: ElevatedButton.styleFrom(backgroundColor: G20Colors.success),
          child: Text(captureState.isCapturing ? 'Queue Capture' : 'Start Capture'),
        ),
      ],
    );
  }

  void _startCapture(BuildContext context) {
    Navigator.pop(context);
    
    final notifier = ref.read(manualCaptureProvider.notifier);
    
    // Update the pending duration with the selected value
    notifier.setPendingDuration(_durationMinutes);
    
    // Now start the capture with the correct duration
    notifier.confirmAndStart();
  }
}

/// The drawn bounding box for subband selection
class _VideoDrawnBox extends StatelessWidget {
  final double x1, y1, x2, y2;
  final Rect plotRect;

  const _VideoDrawnBox({
    required this.x1, required this.y1,
    required this.x2, required this.y2,
    required this.plotRect,
  });

  @override
  Widget build(BuildContext context) {
    final left = (x1 < x2 ? x1 : x2) * plotRect.width;
    final top = (y1 < y2 ? y1 : y2) * plotRect.height;
    final width = (x1 - x2).abs() * plotRect.width;
    final height = (y1 - y2).abs() * plotRect.height;

    return Positioned(
      left: left,
      top: top,
      width: width.clamp(4.0, plotRect.width),
      height: height.clamp(4.0, plotRect.height),
      child: Container(
        decoration: BoxDecoration(
          border: Border.all(color: G20Colors.warning, width: 3),
          color: G20Colors.warning.withOpacity(0.2),
        ),
      ),
    );
  }
}

