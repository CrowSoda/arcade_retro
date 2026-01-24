// lib/features/live_detection/widgets/video_waterfall_display.dart
/// Video-based waterfall display using H.264/JPEG streaming
/// Supports JPEG frames (immediate) and H.264 (future media_kit integration)
/// 
/// Detection boxes are rendered as Flutter overlays (not burned into video)
/// This allows for proper time-synchronized scrolling based on PTS.

import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../../../core/config/theme.dart';
import '../../../core/services/backend_launcher.dart';
import '../providers/video_stream_provider.dart';
import '../providers/map_provider.dart' show getSOIColor, soiVisibilityProvider;
import '../../settings/settings_screen.dart' show waterfallTimeSpanProvider;

/// Waterfall display using video streaming
/// Supports JPEG frames (immediate) and H.264 (future)
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

  @override
  void initState() {
    super.initState();
    // Don't auto-connect here - wait for backend port discovery
  }

  @override
  void dispose() {
    // Don't disconnect here - let the provider handle lifecycle
    super.dispose();
  }

  void _tryConnect(int port) {
    if (_hasAttemptedConnect) return;
    _hasAttemptedConnect = true;
    
    final notifier = ref.read(videoStreamProvider.notifier);
    debugPrint('[VideoWaterfallDisplay] Auto-connecting to ws://${widget.host}:$port/ws/video');
    notifier.connect(widget.host, port);
  }

  @override
  Widget build(BuildContext context) {
    final streamState = ref.watch(videoStreamProvider);
    final backendState = ref.watch(backendLauncherProvider);
    
    // Listen for time span changes and send to backend
    ref.listen<double>(waterfallTimeSpanProvider, (previous, next) {
      // Must read current state inside callback, not use stale streamState
      final currentState = ref.read(videoStreamProvider);
      debugPrint('[Waterfall] Time span listener fired: $previous → $next, connected: ${currentState.isConnected}');
      if (previous != next && currentState.isConnected) {
        ref.read(videoStreamProvider.notifier).setTimeSpan(next);
      }
    });
    
    // Send initial time span when connection state changes to connected
    ref.listen<VideoStreamState>(videoStreamProvider, (previous, next) {
      if (previous?.isConnected != true && next.isConnected) {
        // Just became connected - send current time span
        final currentTimeSpan = ref.read(waterfallTimeSpanProvider);
        // Only send if not default (5.0) since backend starts with 5.0
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
              
              // Waterfall image (clean, no boxes burned in)
              Positioned(
                left: plotRect.left,
                top: plotRect.top,
                width: plotRect.width,
                height: plotRect.height,
                child: _buildWaterfall(streamState),
              ),
              
              // Detection overlay (Flutter-rendered, scrolls with PTS)
              Positioned(
                left: plotRect.left,
                top: plotRect.top,
                width: plotRect.width,
                height: plotRect.height,
                child: _DetectionOverlayLayer(
                  currentPts: streamState.currentPts,
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
    final frame = state.currentFrame;

    if (frame == null || frame.isEmpty) {
      return Container(
        color: const Color(0xFF1A0033), // Dark viridis color
        child: const Center(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              CircularProgressIndicator(color: G20Colors.primary),
              SizedBox(height: 16),
              Text(
                'Waiting for video stream...',
                style: TextStyle(color: Colors.white54),
              ),
            ],
          ),
        ),
      );
    }

    // JPEG mode - display image directly
    return Image.memory(
      frame,
      fit: BoxFit.fill,
      filterQuality: FilterQuality.high,
      gaplessPlayback: true, // Prevent flicker between frames
      errorBuilder: (context, error, stack) {
        return Container(
          color: const Color(0xFF1A0033),
          child: Center(
            child: Text(
              'Frame decode error: $error',
              style: const TextStyle(color: Colors.red),
            ),
          ),
        );
      },
    );
  }

  List<Widget> _buildDetectionOverlay(VideoStreamState state, Rect plotRect) {
    if (state.detections.isEmpty) {
      return [];
    }

    return state.detections.map((det) {
      return _VideoDetectionBox(
        detection: det,
        plotRect: plotRect,
        onTap: () => ref.read(videoStreamProvider.notifier).selectDetection(det.detectionId),
      );
    }).toList();
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
                      'Connecting to video stream...',
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
                        _hasAttemptedConnect = false; // Allow reconnect
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
    final encoder = metadata?.encoder ?? 'unknown';
    final resolution = metadata != null 
        ? '${metadata.videoWidth}×${metadata.videoHeight}'
        : '?×?';
    
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
            resolution,
            style: const TextStyle(fontSize: 9, color: Colors.white70),
          ),
          Text(
            encoder.split('/').last.toUpperCase(),
            style: TextStyle(
              fontSize: 9,
              color: encoder.contains('h264') ? Colors.greenAccent : Colors.orangeAccent,
            ),
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
    // Default values if no metadata
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

/// Detection bounding box for video waterfall
class _VideoDetectionBox extends StatelessWidget {
  final VideoDetection detection;
  final Rect plotRect;
  final VoidCallback onTap;

  const _VideoDetectionBox({
    required this.detection,
    required this.plotRect,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    // Detection coordinates are normalized 0-1 from the inference
    // x1, x2 = time axis (maps to Y in waterfall, but for video it's time within frame)
    // y1, y2 = frequency axis (maps to X in waterfall)
    
    // For video frames, detections are overlaid on the current frame
    // The y coordinates map to horizontal position (frequency)
    // The x coordinates represent the vertical position (time within the accumulated frame)
    
    final left = plotRect.left + detection.y1 * plotRect.width;
    final right = plotRect.left + detection.y2 * plotRect.width;
    final top = plotRect.top + detection.x1 * plotRect.height;
    final bottom = plotRect.top + detection.x2 * plotRect.height;

    final boxWidth = (right - left).abs();
    final boxHeight = (bottom - top).abs();

    // Skip if box is too small or outside bounds
    if (boxWidth < 4 || boxHeight < 4) {
      return const SizedBox.shrink();
    }

    final color = _getDetectionColor(detection.className);

    return Positioned(
      left: left,
      top: top,
      width: boxWidth,
      height: boxHeight,
      child: GestureDetector(
        onTap: onTap,
        child: Container(
          decoration: BoxDecoration(
            border: Border.all(
              color: color,
              width: detection.isSelected ? 3 : 2,
            ),
            color: color.withOpacity(detection.isSelected ? 0.2 : 0.05),
          ),
          child: Align(
            alignment: Alignment.topLeft,
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 2),
              color: color.withOpacity(0.8),
              child: Text(
                '${detection.className} ${(detection.confidence * 100).toStringAsFixed(0)}%',
                style: const TextStyle(
                  fontSize: 9,
                  color: Colors.white,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }

  Color _getDetectionColor(String className) {
    switch (className.toLowerCase()) {
      case 'creamy_chicken':
        return Colors.orange;
      case 'bluetooth':
        return Colors.blue;
      case 'wifi':
        return Colors.green;
      case 'lte':
        return Colors.purple;
      default:
        return Colors.cyan;
    }
  }
}

/// Detection overlay layer with DIRECT COORDINATE MAPPING
/// 
/// Detection coordinates (x1/x2/y1/y2) are normalized 0-1 from the inference image.
/// - x1/x2 = vertical position (time axis in waterfall)
/// - y1/y2 = horizontal position (frequency axis in waterfall)
class _DetectionOverlayLayer extends ConsumerWidget {
  final double currentPts;
  final List<VideoDetection> detections;

  const _DetectionOverlayLayer({
    required this.currentPts,
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
        
        final visibleBoxes = <Widget>[];
        
        for (final det in detections) {
          // Check visibility toggle
          final isVisible = ref.watch(soiVisibilityProvider(det.className));
          if (!isVisible) continue;
          
          // DIRECT COORDINATE MAPPING (not PTS-based)
          // x1/x2 = vertical position on inference image → maps to Y (time axis)
          // y1/y2 = horizontal position on inference image → maps to X (frequency axis)
          final top = det.x1 * plotHeight;
          final bottom = det.x2 * plotHeight;
          final left = det.y1 * plotWidth;
          final right = det.y2 * plotWidth;
          
          final width = (right - left).abs();
          final height = (bottom - top).abs();
          
          // Skip if too small
          if (width < 4 || height < 4) continue;
          
          // Use getSOIColor for correct matching color
          final color = getSOIColor(det.className);
          
          visibleBoxes.add(
            Positioned(
              left: left.clamp(0.0, plotWidth - width),
              top: top.clamp(0.0, plotHeight - height),
              width: width.clamp(4.0, plotWidth),
              height: height.clamp(4.0, plotHeight),
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

/// Individual detection box widget
class _DetectionBoxWidget extends StatelessWidget {
  final VideoDetection detection;
  final Color color;

  const _DetectionBoxWidget({
    required this.detection,
    required this.color,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: BoxDecoration(
        border: Border.all(
          color: color,
          width: detection.isSelected ? 3 : 2,
        ),
        color: color.withOpacity(detection.isSelected ? 0.25 : 0.1),
      ),
      child: Align(
        alignment: Alignment.topLeft,
        child: Container(
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
      ),
    );
  }
}
