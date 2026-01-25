import 'dart:typed_data';
import 'dart:ui' as ui;
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/scheduler.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../../../core/config/theme.dart';
import '../../../core/utils/colormap.dart';
import '../providers/detection_provider.dart';
import '../providers/video_stream_provider.dart';
import '../providers/map_provider.dart';

/// PSD chart - GPU rendered, compute isolate for pixel generation
/// NOW READS FROM videoStreamProvider INSTEAD OF waterfallProvider
class PsdChart extends ConsumerStatefulWidget {
  const PsdChart({super.key});

  @override
  ConsumerState<PsdChart> createState() => _PsdChartState();
}

class _PsdChartState extends ConsumerState<PsdChart>
    with SingleTickerProviderStateMixin {
  ui.Image? _displayImage;
  ui.Image? _pendingImage;
  bool _computing = false;
  Ticker? _ticker;
  int _frameCount = 0;
  int _chartHeight = 150;

  // Smoothed PSD for display (exponential moving average)
  Float32List? _smoothedPsd;
  static const double _psdAlpha = 0.15;  // Smoothing factor

  @override
  void initState() {
    super.initState();
    _ticker = createTicker((_) {
      _frameCount++;
      if (_frameCount % 2 == 0 && _pendingImage != null) {
        final old = _displayImage;
        _displayImage = _pendingImage;
        _pendingImage = null;
        old?.dispose();
        setState(() {});
      }
    });
    _ticker!.start();
  }

  @override
  void dispose() {
    _ticker?.dispose();
    _displayImage?.dispose();
    _pendingImage?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    // READ PSD FROM videoStreamProvider - NOT waterfallProvider
    final videoState = ref.watch(videoStreamProvider);
    final detections = ref.watch(detectionProvider);

    return LayoutBuilder(
      builder: (context, constraints) {        
        _chartHeight = constraints.maxHeight.toInt().clamp(50, 300);
        
        // Build image from video stream PSD data
        _buildImageAsync(videoState);      

        const leftMargin = 35.0;
        const rightMargin = 8.0;
        const topMargin = 4.0;
        const bottomMargin = 18.0;

        final plotRect = Rect.fromLTRB(        
          leftMargin, topMargin,
          constraints.maxWidth - rightMargin,  
          constraints.maxHeight - bottomMargin,
        );

        // Frequency axis values (hardcoded for now, could come from metadata)
        const centerFreqMHz = 825.0;
        const bandwidthMHz = 20.0;

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
                  child: RawImage(
                    image: _displayImage,      
                    fit: BoxFit.fill,
                    filterQuality: FilterQuality.low,
                  ),
                ),
              // Detection frequency overlays  
              Positioned(
                left: plotRect.left,
                top: plotRect.top,
                width: plotRect.width,
                height: plotRect.height,       
                child: _DetectionOverlays(     
                  detections: detections,      
                  centerFreqMHz: centerFreqMHz,
                  bandwidthMHz: bandwidthMHz,
                  plotWidth: plotRect.width,   
                  plotHeight: plotRect.height, 
                ),
              ),
              Positioned(
                left: 0, top: topMargin, width: leftMargin - 2, height: plotRect.height,      
                child: const _YAxisLabels(),
              ),
              Positioned(
                left: leftMargin, right: rightMargin, bottom: 0, height: bottomMargin,        
                child: const _XAxisLabels(centerFreqMHz: centerFreqMHz, bandwidthMHz: bandwidthMHz),
              ),
            ],
          ),
        );
      },
    );
  }

  Future<void> _buildImageAsync(VideoStreamState state) async {
    if (_computing) return;
    
    final psdData = state.psd;
    if (psdData == null || psdData.isEmpty) return;

    _computing = true;

    // Apply exponential smoothing to PSD
    if (_smoothedPsd == null || _smoothedPsd!.length != psdData.length) {
      _smoothedPsd = Float32List.fromList(psdData);
    } else {
      for (int i = 0; i < psdData.length && i < _smoothedPsd!.length; i++) {
        _smoothedPsd![i] = (_psdAlpha * psdData[i] + (1 - _psdAlpha) * _smoothedPsd![i]);
      }
    }

    // Compute pixels in isolate
    final params = _PsdParams(
      psdData: _smoothedPsd!,
      width: state.bufferWidth,
      height: _chartHeight,
      minDb: -80.0,
      maxDb: -20.0,
    );

    final pixels = await compute(_generatePsdPixels, params);

    // Decode to image
    ui.decodeImageFromPixels(
      pixels, params.width, params.height, ui.PixelFormat.rgba8888,
      (image) {
        _pendingImage?.dispose();
        _pendingImage = image;
        _computing = false;
      },
    );
  }
}

class _PsdParams {
  final Float32List psdData;
  final int width;
  final int height;
  final double minDb;
  final double maxDb;

  _PsdParams({
    required this.psdData,
    required this.width,
    required this.height,
    required this.minDb,
    required this.maxDb,
  });
}

/// Runs in isolate - generate pixel buffer for PSD with green gradient
Uint8List _generatePsdPixels(_PsdParams p) {   
  final pixels = Uint8List(p.width * p.height * 4);

  // Background color
  const bgR = 0x12, bgG = 0x17, bgB = 0x1C;    

  // Fill with background
  for (int i = 0; i < p.width * p.height; i++) {
    final idx = i * 4;
    pixels[idx] = bgR;
    pixels[idx + 1] = bgG;
    pixels[idx + 2] = bgB;
    pixels[idx + 3] = 255;
  }

  if (p.psdData.isEmpty) return pixels;        

  // Smooth the PSD: decimate bins to width and apply smoothing
  final smoothedPsd = List<double>.filled(p.width, 0);
  final binsPerPixel = p.psdData.length / p.width;

  for (int x = 0; x < p.width; x++) {
    final startBin = (x * binsPerPixel).floor();
    final endBin = ((x + 1) * binsPerPixel).ceil().clamp(0, p.psdData.length);

    // Average the bins for this pixel
    double sum = 0;
    int count = 0;
    for (int b = startBin; b < endBin; b++) {  
      final v = p.psdData[b];
      if (v.isFinite) {
        sum += v;
        count++;
      }
    }
    smoothedPsd[x] = count > 0 ? sum / count : -80;
  }

  // Apply 5-point moving average for extra smoothing
  final smoothed2 = List<double>.filled(p.width, 0);
  for (int x = 0; x < p.width; x++) {
    double sum = 0;
    int count = 0;
    for (int dx = -2; dx <= 2; dx++) {
      final idx = x + dx;
      if (idx >= 0 && idx < p.width) {
        sum += smoothedPsd[idx];
        count++;
      }
    }
    smoothed2[x] = sum / count;
  }

  // Noise floor based scaling
  final sorted = List<double>.from(smoothed2)..sort();
  double noiseFloor = -60.0;
  if (sorted.isNotEmpty) {
    noiseFloor = sorted[sorted.length ~/ 2];   
  }

  final minPower = noiseFloor - 7;
  final maxPower = noiseFloor + 45;
  final dbRange = maxPower - minPower;

  // Draw each column with green gradient fill 
  for (int x = 0; x < p.width; x++) {
    final value = smoothed2[x];
    final normalized = ((value - minPower) / dbRange).clamp(0.0, 1.0);
    final peakY = ((1.0 - normalized) * (p.height - 1)).round().clamp(0, p.height - 1);       

    // Fill from peak to bottom with green gradient
    for (int y = peakY; y < p.height; y++) {   
      final idx = (y * p.width + x) * 4;       

      final fillNorm = 1.0 - ((y - peakY) / (p.height - peakY)).clamp(0.0, 1.0);
      final colorNorm = (fillNorm * normalized).clamp(0.0, 1.0);
      final colorIdx = (colorNorm * 255).round().clamp(0, 255);
      final rgb = viridisLut[colorIdx];        

      const alpha = 0.7;
      pixels[idx] = (rgb[0] * alpha + bgR * (1 - alpha)).round();
      pixels[idx + 1] = (rgb[1] * alpha + bgG * (1 - alpha)).round();
      pixels[idx + 2] = (rgb[2] * alpha + bgB * (1 - alpha)).round();
      pixels[idx + 3] = 255;
    }

    // Bright peak line
    if (peakY >= 0 && peakY < p.height) {      
      final colorIdx = (normalized * 255).round().clamp(0, 255);
      final rgb = viridisLut[colorIdx];        
      final idx = (peakY * p.width + x) * 4;   
      pixels[idx] = rgb[0];
      pixels[idx + 1] = rgb[1];
      pixels[idx + 2] = rgb[2];
      pixels[idx + 3] = 255;
    }
  }

  return pixels;
}

class _YAxisLabels extends StatelessWidget {   
  const _YAxisLabels();

  @override
  Widget build(BuildContext context) {
    return const Column(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      crossAxisAlignment: CrossAxisAlignment.end,
      children: [
        Text('-20', style: _style),
        Text('-50', style: _style),
        Text('-80', style: _style),
      ],
    );
  }
  static const _style = TextStyle(fontSize: 9, color: G20Colors.textSecondaryDark);
}

class _XAxisLabels extends StatelessWidget {   
  final double centerFreqMHz;
  final double bandwidthMHz;
  
  const _XAxisLabels({required this.centerFreqMHz, required this.bandwidthMHz});

  @override
  Widget build(BuildContext context) {
    final low = centerFreqMHz - bandwidthMHz / 2;
    final high = centerFreqMHz + bandwidthMHz / 2;
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        Text(low.toStringAsFixed(1), style: _style),
        Text('${centerFreqMHz.toStringAsFixed(1)} MHz', style: _style),
        Text(high.toStringAsFixed(1), style: _style),
      ],
    );
  }
  static const _style = TextStyle(fontSize: 9, color: G20Colors.textSecondaryDark);
}

// Debug counter for _DetectionOverlays
int _psdDebugCounter = 0;

/// Detection frequency overlays on PSD chart - thick vertical bands with 20% padding
class _DetectionOverlays extends ConsumerWidget {
  final List<Detection> detections;
  final double centerFreqMHz;
  final double bandwidthMHz;
  final double plotWidth;
  final double plotHeight;

  const _DetectionOverlays({
    required this.detections,
    required this.centerFreqMHz,
    required this.bandwidthMHz,
    required this.plotWidth,
    required this.plotHeight,
  });

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    if (detections.isEmpty || plotWidth <= 0) return const SizedBox.shrink();

    // Get waterfall state for age filtering
    final videoState = ref.watch(videoStreamProvider);
    final totalRows = videoState.totalRowsReceived;
    final bufferHeight = videoState.bufferHeight;

    // Filter to only detections visible on waterfall
    final visibleDetections = detections.where((det) {
      final rowsAgo = totalRows - det.absoluteRow;
      return rowsAgo >= 0 && rowsAgo < bufferHeight;
    }).toList();

    if (visibleDetections.isEmpty) return const SizedBox.shrink();

    _psdDebugCounter++;

    return Stack(
      clipBehavior: Clip.none,
      children: visibleDetections.map((det) {
        // Check visibility toggle
        final isVisible = ref.watch(soiVisibilityProvider(det.className));
        if (!isVisible) return const SizedBox.shrink();

        // y1/y2 = frequency position (horizontal) with 20% padding
        // NOTE: Frequency axis is FLIPPED (1.0 - y) to match waterfall display
        final detWidth = det.y2 - det.y1;
        final freqPadding = detWidth * 0.2;  // 20% padding each side
        final paddedY1 = (det.y1 - freqPadding).clamp(0.0, 1.0);
        final paddedY2 = (det.y2 + freqPadding).clamp(0.0, 1.0);
        
        // Flip: y2 becomes left, y1 becomes right
        final left = (1.0 - paddedY2) * plotWidth;       
        final right = (1.0 - paddedY1) * plotWidth;
        final width = right - left;

        if (width < 1 || paddedY2 <= 0 || paddedY1 >= 1) return const SizedBox.shrink();

        final color = getSOIColor(det.className);
        final isSelected = det.isSelected;
        
        return Positioned(
          left: left,
          top: 0,
          width: width.clamp(4.0, plotWidth),  
          height: plotHeight,
          child: Container(
            decoration: BoxDecoration(
              color: color.withOpacity(isSelected ? 0.25 : 0.15),
              border: Border(
                left: BorderSide(color: color.withOpacity(0.8), width: 1),
                right: BorderSide(color: color.withOpacity(0.8), width: 1),
              ),
            ),
          ),
        );
      }).toList(),
    );
  }
}
