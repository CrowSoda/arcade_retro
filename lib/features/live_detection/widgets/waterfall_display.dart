import 'dart:typed_data';
import 'dart:ui' as ui;
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../../../core/config/theme.dart';
import '../providers/waterfall_provider.dart';
import '../providers/detection_provider.dart';
import '../providers/map_provider.dart';
import '../providers/sdr_config_provider.dart';
import '../../settings/settings_screen.dart';

/// OPTIMIZED Waterfall display
/// - Uses pre-rendered RGBA pixel buffer from Python (no colormap calculation)
/// - Single decodeImageFromPixels per state update (every 2-3 rows, not every row)
/// - Efficient detection box rendering with select() instead of watch()
class WaterfallDisplay extends ConsumerStatefulWidget {
  const WaterfallDisplay({super.key});

  @override
  ConsumerState<WaterfallDisplay> createState() => _WaterfallDisplayState();
}

class _WaterfallDisplayState extends ConsumerState<WaterfallDisplay> {
  ui.Image? _displayImage;
  bool _isDecoding = false;
  int _lastFrameCount = -1;

  @override
  void dispose() {
    _displayImage?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    // Watch only frameCount changes for efficient updates
    final waterfallState = ref.watch(waterfallProvider);
    
    // Only decode new image when frameCount changes (every 2 rows, not every row)
    if (waterfallState.frameCount != _lastFrameCount && 
        waterfallState.pixelBuffer.isNotEmpty &&
        !_isDecoding) {
      _lastFrameCount = waterfallState.frameCount;
      _decodePixelBuffer(waterfallState);
    }

    // Watch detections separately - doesn't trigger waterfall redecode
    final detections = ref.watch(detectionProvider);

    return LayoutBuilder(
      builder: (context, constraints) {
        final size = constraints.biggest;
        const leftMargin = 50.0;
        const bottomMargin = 25.0;
        const topMargin = 8.0;
        const rightMargin = 8.0;

        final plotRect = Rect.fromLTRB(
          leftMargin, topMargin,
          size.width - rightMargin, size.height - bottomMargin,
        );

        return RepaintBoundary(
          child: Stack(
            children: [
              Container(color: G20Colors.surfaceDark),
              if (_displayImage != null)
                Positioned(
                  left: plotRect.left,
                  top: plotRect.top,
                  width: plotRect.width,
                  height: plotRect.height,
                  child: GestureDetector(
                    onLongPress: () {
                      final sdrConfig = ref.read(sdrConfigProvider);
                      ref.read(manualCaptureProvider.notifier).startDrawingMode(
                        sdrConfig.centerFreqMHz.toStringAsFixed(1),
                        durationMinutes: 1,
                      );
                    },
                    child: RawImage(
                      image: _displayImage,
                      fit: BoxFit.fill,
                      filterQuality: FilterQuality.low,
                    ),
                  ),
                ),
              // Detection boxes - use RepaintBoundary to isolate repaints
              ...(() {
                final captureState = ref.watch(manualCaptureProvider);
                final isDrawing = captureState.phase == CapturePhase.drawing ||
                                  captureState.phase == CapturePhase.promptTune ||
                                  captureState.phase == CapturePhase.confirmBox;
                if (isDrawing) {
                  return detections
                      .where((det) => det.className == captureState.signalName)
                      .map<Widget>((det) => _DetectionBox(
                        detection: det, 
                        plotRect: plotRect,
                        currentPts: waterfallState.currentPts,
                        onTap: () => ref.read(detectionProvider.notifier).selectDetection(det.id),
                      ))
                      .toList();
                }
                return detections.map<Widget>((det) => _DetectionBox(
                  detection: det, 
                  plotRect: plotRect,
                  currentPts: waterfallState.currentPts,
                  onTap: () => ref.read(detectionProvider.notifier).selectDetection(det.id),
                )).toList();
              })(),
              // Manual capture drawing overlay
              _DrawingOverlay(plotRect: plotRect),
              Positioned(
                left: 0, top: topMargin, bottom: bottomMargin, width: leftMargin - 4,
                child: _TimeAxis(),
              ),
              Positioned(
                left: leftMargin, right: rightMargin, bottom: 0, height: bottomMargin,
                child: _FrequencyAxis(state: waterfallState),
              ),
            ],
          ),
        );
      },
    );
  }

  void _decodePixelBuffer(WaterfallState state) {
    if (state.pixelBuffer.isEmpty || state.width == 0 || state.height == 0) return;
    
    _isDecoding = true;
    
    // Decode DIRECTLY from provider's pixel buffer
    ui.decodeImageFromPixels(
      state.pixelBuffer,
      state.width,
      state.height,
      ui.PixelFormat.rgba8888,
      (image) {
        if (!mounted) {
          image.dispose();
          _isDecoding = false;
          return;
        }
        
        final oldImage = _displayImage;
        _displayImage = image;
        _isDecoding = false;
        
        setState(() {});
        oldImage?.dispose();
      },
    );
  }
}

class _TimeAxis extends ConsumerWidget {
  @override
  Widget build(BuildContext context, WidgetRef ref) {
    // Get time span from settings provider
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
          Text(formatTime(displayTimeSec), style: const TextStyle(fontSize: 10, color: G20Colors.textSecondaryDark)),
          Text(formatTime(displayTimeSec * 2 / 3), style: const TextStyle(fontSize: 10, color: G20Colors.textSecondaryDark)),
          Text(formatTime(displayTimeSec / 3), style: const TextStyle(fontSize: 10, color: G20Colors.textSecondaryDark)),
          const Text('Now', style: TextStyle(fontSize: 10, color: G20Colors.primary, fontWeight: FontWeight.bold)),
        ],
      ),
    );
  }
}

/// Drawing overlay for manual capture - tap & drag to draw bounding box
class _DrawingOverlay extends ConsumerWidget {
  final Rect plotRect;

  const _DrawingOverlay({required this.plotRect});

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
          Container(color: Colors.black.withOpacity(0.3)),
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
          if (captureState.hasPendingBox)
            _DrawnBox(
              x1: captureState.pendingBoxX1!,
              y1: captureState.pendingBoxY1!,
              x2: captureState.pendingBoxX2!,
              y2: captureState.pendingBoxY2!,
              plotRect: plotRect,
            ),
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
                    onPressed: () => ref.read(manualCaptureProvider.notifier).confirmAndStart(),
                    style: ElevatedButton.styleFrom(backgroundColor: G20Colors.success),
                    child: Text(captureState.isCapturing ? 'Queue' : 'Start'),
                  ),
                ],
              ),
            ),
        ],
      ),
    );
  }
}

class _DrawnBox extends StatelessWidget {
  final double x1, y1, x2, y2;
  final Rect plotRect;

  const _DrawnBox({
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

class _FrequencyAxis extends StatelessWidget {
  final WaterfallState state;
  const _FrequencyAxis({required this.state});

  @override
  Widget build(BuildContext context) {
    final low = state.centerFreqMHz - state.bandwidthMHz / 2;
    final high = state.centerFreqMHz + state.bandwidthMHz / 2;
    return Container(
      padding: const EdgeInsets.only(top: 4),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text('${low.toStringAsFixed(1)}', style: const TextStyle(fontSize: 10, color: G20Colors.textSecondaryDark)),
          Text('${state.centerFreqMHz.toStringAsFixed(1)} MHz', style: const TextStyle(fontSize: 10, color: G20Colors.textSecondaryDark)),
          Text('${high.toStringAsFixed(1)}', style: const TextStyle(fontSize: 10, color: G20Colors.textSecondaryDark)),
        ],
      ),
    );
  }
}

/// OPTIMIZED Detection bounding box
/// - Receives currentPts as parameter instead of watching provider
/// - Uses RepaintBoundary for isolation
class _DetectionBox extends ConsumerWidget {
  final Detection detection;
  final Rect plotRect;
  final double currentPts;
  final VoidCallback onTap;

  const _DetectionBox({
    required this.detection, 
    required this.plotRect, 
    required this.currentPts,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    // Check visibility
    final isVisible = ref.watch(soiVisibilityProvider(detection.className));
    if (!isVisible) {
      return const SizedBox.shrink();
    }

    // Get time span from settings - matches waterfall display
    final displayTimeSpan = ref.watch(waterfallTimeSpanProvider);

    // Calculate Y position based on PTS offset
    final ptsAge = currentPts - detection.pts;
    final scrollProgress = 1.0 - (ptsAge / displayTimeSpan);

    if (scrollProgress < 0 || scrollProgress > 1.0) {
      return const SizedBox.shrink();
    }

    // X position from detection
    final x1 = plotRect.left + detection.y1 * plotRect.width;
    final x2 = plotRect.left + detection.y2 * plotRect.width;

    // Y position
    final detHeight = (detection.x2 - detection.x1) * plotRect.height * 0.3;
    final y1 = plotRect.top + scrollProgress * (plotRect.height - detHeight);
    final y2 = y1 + detHeight;

    if (y2 < plotRect.top || y1 > plotRect.bottom) {
      return const SizedBox.shrink();
    }

    final color = getSOIColor(detection.className);
    final boxHeight = (y2 - y1).abs();
    final safeHeight = boxHeight < 4 ? 4.0 : boxHeight;

    return Positioned(
      left: x1, 
      top: y1, 
      width: (x2 - x1).abs(),
      height: safeHeight,
      child: RepaintBoundary(
        child: GestureDetector(
          onTap: onTap,
          child: Container(
            decoration: BoxDecoration(
              border: Border.all(color: color, width: detection.isSelected ? 3 : 2),
              color: color.withOpacity(detection.isSelected ? 0.2 : 0.05),
            ),
          ),
        ),
      ),
    );
  }
}
