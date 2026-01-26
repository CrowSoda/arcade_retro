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

/// Grouped detection band - merges overlapping detections per class
class _DetectionBand {
  final String className;
  double y1;  // Min freq (normalized 0-1)
  double y2;  // Max freq (normalized 0-1)
  int count;  // Number of detections merged
  bool isSelected;

  _DetectionBand({
    required this.className,
    required this.y1,
    required this.y2,
    this.count = 1,
    this.isSelected = false,
  });

  /// Merge another detection into this band if overlapping
  bool tryMerge(double otherY1, double otherY2, bool otherSelected) {
    // Check if overlapping (with 10% tolerance for "close" detections)
    final tolerance = 0.05;
    if (otherY1 <= y2 + tolerance && otherY2 >= y1 - tolerance) {
      y1 = y1 < otherY1 ? y1 : otherY1;  // min
      y2 = y2 > otherY2 ? y2 : otherY2;  // max
      count++;
      if (otherSelected) isSelected = true;
      return true;
    }
    return false;
  }
}

/// Detection frequency overlays on PSD chart - IMPROVED for busy environments
/// Groups overlapping detections per class, shows thin border markers instead of filled bands
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

    // GROUP DETECTIONS BY CLASS AND MERGE OVERLAPPING FREQUENCY RANGES
    // This dramatically reduces visual clutter in busy environments
    final bandsByClass = <String, List<_DetectionBand>>{};

    for (final det in visibleDetections) {
      // Check visibility toggle
      final isVisible = ref.watch(soiVisibilityProvider(det.className));
      if (!isVisible) continue;

      final bands = bandsByClass.putIfAbsent(det.className, () => []);
      
      // Try to merge with existing band
      bool merged = false;
      for (final band in bands) {
        if (band.tryMerge(det.y1, det.y2, det.isSelected)) {
          merged = true;
          break;
        }
      }
      
      // No overlap found - create new band
      if (!merged) {
        bands.add(_DetectionBand(
          className: det.className,
          y1: det.y1,
          y2: det.y2,
          isSelected: det.isSelected,
        ));
      }
    }

    // Build widgets for each merged band
    final widgets = <Widget>[];

    for (final entry in bandsByClass.entries) {
      final className = entry.key;
      final color = getSOIColor(className);
      
      for (final band in entry.value) {
        // Add small padding to each band
        final detWidth = band.y2 - band.y1;
        final freqPadding = detWidth * 0.1;  // 10% padding (reduced from 20%)
        final paddedY1 = (band.y1 - freqPadding).clamp(0.0, 1.0);
        final paddedY2 = (band.y2 + freqPadding).clamp(0.0, 1.0);
        
        // Flip: y2 becomes left, y1 becomes right
        final left = (1.0 - paddedY2) * plotWidth;       
        final right = (1.0 - paddedY1) * plotWidth;
        final width = right - left;

        if (width < 1 || paddedY2 <= 0 || paddedY1 >= 1) continue;

        // CLEANER LOOK: Just vertical lines at edges, no fill
        // More visible when selected
        final borderWidth = band.isSelected ? 2.0 : 1.5;
        final opacity = band.isSelected ? 0.9 : 0.7;
        
        widgets.add(
          Positioned(
            left: left,
            top: 0,
            width: width.clamp(3.0, plotWidth),  
            height: plotHeight,
            child: Container(
              decoration: BoxDecoration(
                // Very subtle fill only when selected
                color: band.isSelected ? color.withOpacity(0.08) : null,
                border: Border(
                  left: BorderSide(color: color.withOpacity(opacity), width: borderWidth),
                  right: BorderSide(color: color.withOpacity(opacity), width: borderWidth),
                ),
              ),
            ),
          ),
        );
      }
    }

    return Stack(
      clipBehavior: Clip.none,
      children: widgets,
    );
  }
}
