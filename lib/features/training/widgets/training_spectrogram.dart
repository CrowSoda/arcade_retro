import 'dart:async';
import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:ui' as ui;
import 'package:flutter/material.dart';
import 'package:flutter/gestures.dart';
import 'package:fftea/fftea.dart';
import '../../../core/config/theme.dart';
import '../../../core/utils/colormap.dart';
import '../../../core/services/rfcap_service.dart';
import '../../live_detection/providers/map_provider.dart' show getSOIColor;

/// Label box for training - stored permanently
class LabelBox {
  double x1, y1, x2, y2;  // Normalized 0-1 (x=time, y=freq after orientation swap)
  String className;
  final int id;
  bool isSelected;
  
  // Frequency/time bounds in real units
  double? freqStartMHz;
  double? freqEndMHz;
  double? timeStartSec;
  double? timeEndSec;

  LabelBox({
    required this.x1, required this.y1,
    required this.x2, required this.y2,
    required this.className,
    required this.id,
    this.isSelected = false,
    this.freqStartMHz,
    this.freqEndMHz,
    this.timeStartSec,
    this.timeEndSec,
  });

  Rect toRect(Size size) {
    final left = math.min(x1, x2) * size.width;
    final top = math.min(y1, y2) * size.height;
    final right = math.max(x1, x2) * size.width;
    final bottom = math.max(y1, y2) * size.height;
    return Rect.fromLTRB(left, top, right, bottom);
  }
  
  LabelBox copyWith({
    double? x1, double? y1, double? x2, double? y2,
    String? className, bool? isSelected,
  }) {
    return LabelBox(
      x1: x1 ?? this.x1,
      y1: y1 ?? this.y1,
      x2: x2 ?? this.x2,
      y2: y2 ?? this.y2,
      className: className ?? this.className,
      id: id,
      isSelected: isSelected ?? this.isSelected,
      freqStartMHz: freqStartMHz,
      freqEndMHz: freqEndMHz,
      timeStartSec: timeStartSec,
      timeEndSec: timeEndSec,
    );
  }
}

/// Training spectrogram widget - FreqHunter style
/// X-axis: Time, Y-axis: Frequency (swapped from traditional waterfall)
class TrainingSpectrogram extends StatefulWidget {
  final String? filepath;
  final RfcapHeader? header;
  final List<LabelBox> boxes;
  final Function(LabelBox)? onBoxCreated;
  final Function(LabelBox)? onBoxUpdated;
  final Function(int)? onBoxSelected;
  final Function(int)? onBoxDeleted;

  const TrainingSpectrogram({
    super.key,
    this.filepath,
    this.header,
    this.boxes = const [],
    this.onBoxCreated,
    this.onBoxUpdated,
    this.onBoxSelected,
    this.onBoxDeleted,
  });

  @override
  State<TrainingSpectrogram> createState() => TrainingSpectrogramState();
}

class TrainingSpectrogramState extends State<TrainingSpectrogram> {
  // Spectrogram data
  ui.Image? _displayImage;
  Uint8List? _pixelBuffer;
  Float32List? _spectrogramData;
  int _spectrogramWidth = 0;  // Time dimension
  int _spectrogramHeight = 0; // Frequency dimension
  
  // Full file bounds (set once on load)
  double _totalDurationSec = 0;
  double _totalBandwidthHz = 20e6;
  double _centerFreqHz = 0;
  double _sampleRate = 20e6;
  
  // ZOOM VIEW BOUNDS (these change with zoom)
  double _viewTimeStartSec = 0.0;
  double _viewTimeEndSec = 0.2;
  double _viewFreqStartHz = 0.0;  // Relative to center freq (-BW/2 to +BW/2)
  double _viewFreqEndHz = 20e6;   // Full bandwidth initially
  
  // For gesture scaling
  double _baseTimeSpan = 0.2;
  double _baseFreqSpan = 20e6;
  Offset _lastFocalPoint = Offset.zero;
  
  // FFT size for display
  int _fftSize = 2048;
  
  bool _isLoading = false;
  String? _error;
  
  // Drawing state
  bool _isDrawing = false;
  Offset? _drawStart;
  Offset? _drawCurrent;
  
  // Debounce timer for zoom recomputation
  Timer? _zoomDebounceTimer;
  
  // Raw IQ data cache for zoom recomputation
  Float32List? _cachedIqData;
  int _cachedOffsetSamples = 0;
  int _cachedNumSamples = 0;
  
  // FFT objects
  late FFT _fft;
  late Float64List _window;
  
  // Backward compatibility getters (map old names to new zoom vars)
  double get _windowStartSec => _viewTimeStartSec;
  double get _windowLengthSec => _viewTimeEndSec - _viewTimeStartSec;
  int get _windowFftSize => _fftSize;
  
  // Auto-detect params
  static const double _detectionThreshold = 10.0;  // dB above noise
  static const int _localWindowSize = 50;  // Size of local window for Otsu
  static const double _regionGrowK = 2.0;  // Tolerance multiplier for region growing
  static const int _chanVeseMaxIter = 100;  // Max iterations for Chan-Vese fallback

  @override
  void initState() {
    super.initState();
    _initFFT();
  }

  @override
  void didUpdateWidget(TrainingSpectrogram oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (widget.filepath != oldWidget.filepath && widget.filepath != null) {
      _loadAndCompute();
    }
  }

  void _initFFT() {
    _fft = FFT(_fftSize);
    _window = Float64List(_fftSize);
    for (int i = 0; i < _fftSize; i++) {
      _window[i] = 0.5 * (1.0 - math.cos(2.0 * math.pi * i / (_fftSize - 1)));
    }
    
    if (widget.filepath != null) {
      _loadAndCompute();
    }
  }

  Future<void> _loadAndCompute() async {
    if (widget.filepath == null) return;
    
    // Don't clear the image - keep showing old one until new one is ready
    setState(() {
      _isLoading = true;
      _error = null;
    });

    try {
      // Read file header to get metadata
      final header = await RfcapService.readHeader(widget.filepath!);
      if (header == null) {
        setState(() {
          _error = 'Failed to read file header';
          _isLoading = false;
        });
        return;
      }
      
      _sampleRate = header.sampleRate;
      _totalDurationSec = header.numSamples / _sampleRate;
      _totalBandwidthHz = header.bandwidthHz;
      _centerFreqHz = header.centerFreqHz;
      
      // Initialize view bounds if first load
      if (_viewTimeEndSec <= _viewTimeStartSec) {
        _viewTimeStartSec = 0.0;
        _viewTimeEndSec = math.min(0.2, _totalDurationSec);  // 200ms default
        _viewFreqStartHz = -_totalBandwidthHz / 2;
        _viewFreqEndHz = _totalBandwidthHz / 2;
      }
      
      // Clamp view bounds
      _viewTimeStartSec = _viewTimeStartSec.clamp(0.0, _totalDurationSec);
      _viewTimeEndSec = _viewTimeEndSec.clamp(_viewTimeStartSec + 0.001, _totalDurationSec);
      
      // Calculate time window to load
      final timeWindowSec = _viewTimeEndSec - _viewTimeStartSec;
      final offsetSamples = (_viewTimeStartSec * _sampleRate).toInt();
      final numSamples = (timeWindowSec * _sampleRate).toInt();
      
      debugPrint('Loading window: ${_viewTimeStartSec.toStringAsFixed(3)}s to ${_viewTimeEndSec.toStringAsFixed(3)}s '
          '($numSamples samples at offset $offsetSamples)');
      
      // Load only the window of IQ data
      final iqData = await RfcapService.readIqData(
        widget.filepath!,
        offsetSamples: offsetSamples,
        numSamples: numSamples,
      );

      if (iqData == null || iqData.isEmpty) {
        setState(() {
          _error = 'Failed to load IQ data';
          _isLoading = false;
        });
        return;
      }

      debugPrint('Loaded ${iqData.length ~/ 2} complex samples');
      
      // Cache for zoom recomputation
      _cachedIqData = iqData;
      _cachedOffsetSamples = offsetSamples;
      _cachedNumSamples = numSamples;
      
      // Compute spectrogram with frequency zoom (heterodyne + decimate)
      await _computeZoomSpectrogram(iqData);
      
      setState(() => _isLoading = false);
    } catch (e) {
      debugPrint('Error in _loadAndCompute: $e');
      setState(() {
        _error = 'Error: $e';
        _isLoading = false;
      });
    }
  }
  
  /// Compute spectrogram - ZOOM DISABLED to fix bounding box issues
  Future<void> _computeZoomSpectrogram(Float32List iqData) async {
    // ZOOM FFT DISABLED - just compute straight spectrogram
    // TODO: Re-implement zoom properly without breaking box addressing
    await _computeSpectrogram(iqData, _fftSize);
  }
  
  /// Heterodyne: mix signal to shift centerFreqHz to DC
  Float32List _heterodyne(Float32List iqData, double shiftHz, double sampleRate) {
    final numSamples = iqData.length ~/ 2;
    final result = Float32List(iqData.length);
    final twoPiOverSr = -2.0 * math.pi * shiftHz / sampleRate;
    
    for (int i = 0; i < numSamples; i++) {
      final phase = twoPiOverSr * i;
      final cos = math.cos(phase);
      final sin = math.sin(phase);
      
      final iVal = iqData[i * 2];
      final qVal = iqData[i * 2 + 1];
      
      // Complex multiply: (I + jQ) * (cos - j*sin) = (I*cos + Q*sin) + j(Q*cos - I*sin)
      result[i * 2] = (iVal * cos + qVal * sin).toDouble();
      result[i * 2 + 1] = (qVal * cos - iVal * sin).toDouble();
    }
    
    return result;
  }
  
  /// Decimate with simple moving average lowpass filter
  Float32List _decimateWithFilter(Float32List iqData, int factor) {
    final numInputSamples = iqData.length ~/ 2;
    final numOutputSamples = numInputSamples ~/ factor;
    final result = Float32List(numOutputSamples * 2);
    
    // Simple box filter (moving average) for anti-aliasing
    for (int i = 0; i < numOutputSamples; i++) {
      double sumI = 0, sumQ = 0;
      for (int j = 0; j < factor; j++) {
        final idx = (i * factor + j) * 2;
        if (idx + 1 < iqData.length) {
          sumI += iqData[idx];
          sumQ += iqData[idx + 1];
        }
      }
      result[i * 2] = sumI / factor;
      result[i * 2 + 1] = sumQ / factor;
    }
    
    return result;
  }

  Future<void> _computeSpectrogram(Float32List iqData, int fftSize) async {
    // IQ data is interleaved: I0, Q0, I1, Q1, ...
    final numComplexSamples = iqData.length ~/ 2;
    final hopSize = fftSize ~/ 4;  // 75% overlap
    final numTimeFrames = (numComplexSamples - fftSize) ~/ hopSize;
    
    if (numTimeFrames <= 0) {
      setState(() => _error = 'Not enough data for spectrogram');
      return;
    }
    
    // Create FFT and window for the specified size
    final fft = FFT(fftSize);
    final window = Float64List(fftSize);
    for (int i = 0; i < fftSize; i++) {
      window[i] = 0.5 * (1.0 - math.cos(2.0 * math.pi * i / (fftSize - 1)));
    }
    
    // FreqHunter style: width = time, height = frequency
    _spectrogramWidth = numTimeFrames;
    _spectrogramHeight = fftSize ~/ 2;  // Only positive frequencies
    _spectrogramData = Float32List(_spectrogramWidth * _spectrogramHeight);
    
    final fftIn = Float64x2List(fftSize);
    
    for (int timeFrame = 0; timeFrame < numTimeFrames; timeFrame++) {
      final sampleOffset = timeFrame * hopSize;
      
      // Window and load
      for (int i = 0; i < fftSize; i++) {
        final idx = (sampleOffset + i) * 2;
        if (idx + 1 >= iqData.length) break;
        
        final iVal = iqData[idx];
        final qVal = iqData[idx + 1];
        fftIn[i] = Float64x2(iVal * window[i], qVal * window[i]);
      }
      
      // FFT
      fft.inPlaceFft(fftIn);
      
      // Power spectrum - store as column (frequency vertical)
      // Frequency increases from bottom to top
      for (int freqBin = 0; freqBin < _spectrogramHeight; freqBin++) {
        // FFT shift: remap bins so DC is at center
        final fftBin = (freqBin + fftSize ~/ 2) % fftSize;
        final real = fftIn[fftBin].x;
        final imag = fftIn[fftBin].y;
        final power = real * real + imag * imag;
        final dB = 10.0 * math.log(power + 1e-12) / math.ln10;
        
        // Row = frequency (inverted so high freq at top)
        // Col = time
        final row = _spectrogramHeight - 1 - freqBin;
        _spectrogramData![row * _spectrogramWidth + timeFrame] = dB;
      }
    }
    
    await _renderToImage();
  }

  Future<void> _renderToImage() async {
    if (_spectrogramData == null) return;
    
    _pixelBuffer = Uint8List(_spectrogramWidth * _spectrogramHeight * 4);
    
    // Find dynamic range
    double minDb = double.infinity;
    double maxDb = double.negativeInfinity;
    for (final v in _spectrogramData!) {
      if (v.isFinite) {
        if (v < minDb) minDb = v;
        if (v > maxDb) maxDb = v;
      }
    }
    
    // Noise floor estimation (10th percentile)
    final sorted = List<double>.from(_spectrogramData!.where((v) => v.isFinite));
    sorted.sort();
    final noiseFloor = sorted.isNotEmpty ? sorted[sorted.length ~/ 10] : minDb;
    
    // Display range: noise floor - 5dB to noise floor + 50dB
    final displayMin = noiseFloor - 5;
    final displayMax = noiseFloor + 50;
    final dbRange = displayMax - displayMin;
    
    // Render pixels
    for (int row = 0; row < _spectrogramHeight; row++) {
      for (int col = 0; col < _spectrogramWidth; col++) {
        final idx = row * _spectrogramWidth + col;
        final value = _spectrogramData![idx];
        final normalized = ((value - displayMin) / dbRange).clamp(0.0, 1.0);
        final colorIdx = (normalized * 255).round().clamp(0, 255);
        final rgb = viridisLut[colorIdx];
        
        final pixelIdx = idx * 4;
        _pixelBuffer![pixelIdx] = rgb[0];
        _pixelBuffer![pixelIdx + 1] = rgb[1];
        _pixelBuffer![pixelIdx + 2] = rgb[2];
        _pixelBuffer![pixelIdx + 3] = 255;
      }
    }
    
    // Decode to image
    ui.decodeImageFromPixels(
      _pixelBuffer!,
      _spectrogramWidth,
      _spectrogramHeight,
      ui.PixelFormat.rgba8888,
      (image) {
        if (!mounted) {
          image.dispose();
          return;
        }
        final old = _displayImage;
        setState(() => _displayImage = image);
        old?.dispose();
      },
    );
  }

  /// Advanced auto-detect using hybrid approach:
  /// 1. Local Otsu threshold from click window
  /// 2. Adaptive seeded region growing
  /// 3. Morphological close/open refinement
  /// 4. Chan-Vese level set fallback for difficult signals
  void _autoDetect(Offset localPosition, Size size) {
    if (_spectrogramData == null) return;
    
    // Convert to spectrogram coordinates
    final col = (localPosition.dx / size.width * _spectrogramWidth).round().clamp(0, _spectrogramWidth - 1);
    final row = (localPosition.dy / size.height * _spectrogramHeight).round().clamp(0, _spectrogramHeight - 1);
    
    // Stage 1: Extract local window and compute Otsu threshold
    final localStats = _computeLocalStats(col, row, _localWindowSize);
    final otsuThreshold = _computeLocalOtsu(col, row, _localWindowSize);
    
    // Get power at tap point
    final centerPower = _spectrogramData![row * _spectrogramWidth + col];
    
    // Check if there's signal above local Otsu threshold
    if (centerPower < otsuThreshold) {
      // No significant energy - create small default box
      _createBoxAt(localPosition, size, 0.03, 0.05);
      return;
    }
    
    // Stage 2: Adaptive seeded region growing
    final mask = _adaptiveRegionGrow(col, row, otsuThreshold, localStats['stddev']!);
    
    // Stage 3: Morphological refinement (close then open)
    _morphClose(mask, 3);
    _morphOpen(mask, 2);
    
    // Extract bounding box from mask
    var bbox = _extractBoundingBox(mask);
    
    // Stage 4: If result is unreasonable, try Chan-Vese fallback
    if (bbox == null || _isUnreasonableBox(bbox)) {
      debugPrint('Primary detection failed, trying Chan-Vese fallback...');
      final chanVeseMask = _chanVeseLevelSet(col, row);
      if (chanVeseMask != null) {
        bbox = _extractBoundingBox(chanVeseMask);
      }
    }
    
    // If still no good result, fall back to simple flood fill
    if (bbox == null || _isUnreasonableBox(bbox)) {
      debugPrint('Fallback to simple flood fill...');
      bbox = _simpleFloodFill(col, row, otsuThreshold * 0.8);
    }
    
    // Create the box
    if (bbox != null) {
      // Add generous margin - better to be loose than tight for CNN training
      // 10% of box size or minimum 8 pixels
      final boxWidth = bbox['col2']! - bbox['col1']!;
      final boxHeight = bbox['row2']! - bbox['row1']!;
      final marginCol = math.max(8, (boxWidth * 0.15).round());
      final marginRow = math.max(8, (boxHeight * 0.15).round());
      
      final col1 = (bbox['col1']! - marginCol).clamp(0, _spectrogramWidth - 1);
      final col2 = (bbox['col2']! + marginCol).clamp(0, _spectrogramWidth - 1);
      final row1 = (bbox['row1']! - marginRow).clamp(0, _spectrogramHeight - 1);
      final row2 = (bbox['row2']! + marginRow).clamp(0, _spectrogramHeight - 1);
      
      _createBox(
        col1 / _spectrogramWidth,
        row1 / _spectrogramHeight,
        col2 / _spectrogramWidth,
        row2 / _spectrogramHeight,
      );
    } else {
      // Last resort: create small box at click
      _createBoxAt(localPosition, size, 0.03, 0.05);
    }
  }
  
  /// Compute local statistics around click point
  Map<String, double> _computeLocalStats(int col, int row, int windowSize) {
    final halfW = windowSize ~/ 2;
    final values = <double>[];
    
    for (int r = math.max(0, row - halfW); r < math.min(_spectrogramHeight, row + halfW); r++) {
      for (int c = math.max(0, col - halfW); c < math.min(_spectrogramWidth, col + halfW); c++) {
        final val = _spectrogramData![r * _spectrogramWidth + c];
        if (val.isFinite) values.add(val);
      }
    }
    
    if (values.isEmpty) return {'mean': 0, 'stddev': 1, 'min': 0, 'max': 0};
    
    final mean = values.reduce((a, b) => a + b) / values.length;
    final variance = values.map((v) => (v - mean) * (v - mean)).reduce((a, b) => a + b) / values.length;
    final stddev = math.sqrt(variance);
    
    values.sort();
    return {
      'mean': mean,
      'stddev': stddev,
      'min': values.first,
      'max': values.last,
      'median': values[values.length ~/ 2],
    };
  }
  
  /// Compute Otsu threshold on local window
  double _computeLocalOtsu(int col, int row, int windowSize) {
    final halfW = windowSize ~/ 2;
    final values = <double>[];
    
    for (int r = math.max(0, row - halfW); r < math.min(_spectrogramHeight, row + halfW); r++) {
      for (int c = math.max(0, col - halfW); c < math.min(_spectrogramWidth, col + halfW); c++) {
        final val = _spectrogramData![r * _spectrogramWidth + c];
        if (val.isFinite) values.add(val);
      }
    }
    
    if (values.isEmpty) return 0;
    
    // Build histogram (256 bins)
    values.sort();
    final minVal = values.first;
    final maxVal = values.last;
    final range = maxVal - minVal;
    if (range <= 0) return minVal;
    
    const numBins = 256;
    final histogram = List.filled(numBins, 0);
    for (final v in values) {
      final bin = ((v - minVal) / range * (numBins - 1)).round().clamp(0, numBins - 1);
      histogram[bin]++;
    }
    
    // Otsu's method - find threshold that maximizes between-class variance
    final total = values.length;
    double sumTotal = 0;
    for (int i = 0; i < numBins; i++) {
      sumTotal += i * histogram[i];
    }
    
    double sumB = 0;
    int wB = 0;
    double maxVariance = 0;
    int threshold = 0;
    
    for (int t = 0; t < numBins; t++) {
      wB += histogram[t];
      if (wB == 0) continue;
      
      final wF = total - wB;
      if (wF == 0) break;
      
      sumB += t * histogram[t];
      
      final mB = sumB / wB;
      final mF = (sumTotal - sumB) / wF;
      
      final variance = wB * wF * (mB - mF) * (mB - mF);
      if (variance > maxVariance) {
        maxVariance = variance;
        threshold = t;
      }
    }
    
    return minVal + (threshold / (numBins - 1)) * range;
  }
  
  /// Adaptive seeded region growing
  List<List<bool>> _adaptiveRegionGrow(int seedCol, int seedRow, double threshold, double localStddev) {
    final mask = List.generate(_spectrogramHeight, (_) => List.filled(_spectrogramWidth, false));
    final visited = List.generate(_spectrogramHeight, (_) => List.filled(_spectrogramWidth, false));
    
    final seedValue = _spectrogramData![seedRow * _spectrogramWidth + seedCol];
    double regionMean = seedValue;
    int regionCount = 1;
    final tolerance = localStddev * _regionGrowK;
    
    // BFS queue: [row, col]
    final queue = <List<int>>[[seedRow, seedCol]];
    mask[seedRow][seedCol] = true;
    visited[seedRow][seedCol] = true;
    
    while (queue.isNotEmpty) {
      final current = queue.removeAt(0);
      final r = current[0];
      final c = current[1];
      
      // Check 4-connected neighbors
      for (final delta in [[-1, 0], [1, 0], [0, -1], [0, 1]]) {
        final nr = r + delta[0];
        final nc = c + delta[1];
        
        if (nr < 0 || nr >= _spectrogramHeight || nc < 0 || nc >= _spectrogramWidth) continue;
        if (visited[nr][nc]) continue;
        
        visited[nr][nc] = true;
        final val = _spectrogramData![nr * _spectrogramWidth + nc];
        
        // Similarity criterion: above threshold AND similar to region mean
        if (val >= threshold && (val - regionMean).abs() < tolerance) {
          mask[nr][nc] = true;
          queue.add([nr, nc]);
          
          // Update running mean
          regionCount++;
          regionMean += (val - regionMean) / regionCount;
        }
      }
    }
    
    return mask;
  }
  
  /// Morphological closing (dilate then erode)
  void _morphClose(List<List<bool>> mask, int kernelSize) {
    _dilate(mask, kernelSize);
    _erode(mask, kernelSize);
  }
  
  /// Morphological opening (erode then dilate)
  void _morphOpen(List<List<bool>> mask, int kernelSize) {
    _erode(mask, kernelSize);
    _dilate(mask, kernelSize);
  }
  
  void _dilate(List<List<bool>> mask, int kernelSize) {
    final half = kernelSize ~/ 2;
    final copy = List.generate(_spectrogramHeight, (r) => List<bool>.from(mask[r]));
    
    for (int r = 0; r < _spectrogramHeight; r++) {
      for (int c = 0; c < _spectrogramWidth; c++) {
        if (!copy[r][c]) continue;
        // Set all neighbors within kernel to true
        for (int dr = -half; dr <= half; dr++) {
          for (int dc = -half; dc <= half; dc++) {
            final nr = r + dr;
            final nc = c + dc;
            if (nr >= 0 && nr < _spectrogramHeight && nc >= 0 && nc < _spectrogramWidth) {
              mask[nr][nc] = true;
            }
          }
        }
      }
    }
  }
  
  void _erode(List<List<bool>> mask, int kernelSize) {
    final half = kernelSize ~/ 2;
    final copy = List.generate(_spectrogramHeight, (r) => List<bool>.from(mask[r]));
    
    for (int r = 0; r < _spectrogramHeight; r++) {
      for (int c = 0; c < _spectrogramWidth; c++) {
        if (!copy[r][c]) continue;
        // Check if all neighbors within kernel are true
        bool allTrue = true;
        for (int dr = -half; dr <= half && allTrue; dr++) {
          for (int dc = -half; dc <= half && allTrue; dc++) {
            final nr = r + dr;
            final nc = c + dc;
            if (nr < 0 || nr >= _spectrogramHeight || nc < 0 || nc >= _spectrogramWidth || !copy[nr][nc]) {
              allTrue = false;
            }
          }
        }
        mask[r][c] = allTrue;
      }
    }
  }
  
  /// Extract bounding box from binary mask
  Map<String, int>? _extractBoundingBox(List<List<bool>> mask) {
    int minRow = _spectrogramHeight, maxRow = 0;
    int minCol = _spectrogramWidth, maxCol = 0;
    bool found = false;
    
    for (int r = 0; r < _spectrogramHeight; r++) {
      for (int c = 0; c < _spectrogramWidth; c++) {
        if (mask[r][c]) {
          found = true;
          if (r < minRow) minRow = r;
          if (r > maxRow) maxRow = r;
          if (c < minCol) minCol = c;
          if (c > maxCol) maxCol = c;
        }
      }
    }
    
    if (!found) return null;
    return {'row1': minRow, 'row2': maxRow, 'col1': minCol, 'col2': maxCol};
  }
  
  /// Check if bounding box is unreasonable (too small or too large)
  bool _isUnreasonableBox(Map<String, int> bbox) {
    final width = bbox['col2']! - bbox['col1']!;
    final height = bbox['row2']! - bbox['row1']!;
    
    // Too small (less than 3 pixels)
    if (width < 3 || height < 3) return true;
    
    // Too large (more than 80% of image)
    if (width > _spectrogramWidth * 0.8 || height > _spectrogramHeight * 0.8) return true;
    
    return false;
  }
  
  /// Simple flood fill fallback
  Map<String, int>? _simpleFloodFill(int seedCol, int seedRow, double threshold) {
    final visited = List.generate(_spectrogramHeight, (_) => List.filled(_spectrogramWidth, false));
    
    int minRow = seedRow, maxRow = seedRow;
    int minCol = seedCol, maxCol = seedCol;
    
    final queue = <List<int>>[[seedRow, seedCol]];
    visited[seedRow][seedCol] = true;
    
    while (queue.isNotEmpty) {
      final current = queue.removeAt(0);
      final r = current[0];
      final c = current[1];
      
      if (r < minRow) minRow = r;
      if (r > maxRow) maxRow = r;
      if (c < minCol) minCol = c;
      if (c > maxCol) maxCol = c;
      
      for (final delta in [[-1, 0], [1, 0], [0, -1], [0, 1]]) {
        final nr = r + delta[0];
        final nc = c + delta[1];
        
        if (nr < 0 || nr >= _spectrogramHeight || nc < 0 || nc >= _spectrogramWidth) continue;
        if (visited[nr][nc]) continue;
        
        visited[nr][nc] = true;
        final val = _spectrogramData![nr * _spectrogramWidth + nc];
        
        if (val >= threshold) {
          queue.add([nr, nc]);
        }
      }
    }
    
    return {'row1': minRow, 'row2': maxRow, 'col1': minCol, 'col2': maxCol};
  }
  
  /// Chan-Vese level set segmentation - fallback for difficult signals
  /// Simplified implementation that evolves a contour to minimize energy
  List<List<bool>>? _chanVeseLevelSet(int seedCol, int seedRow) {
    // Initialize level set as small circle around seed
    final phi = List.generate(_spectrogramHeight, (r) => 
      List.generate(_spectrogramWidth, (c) {
        final dist = math.sqrt(math.pow(r - seedRow, 2) + math.pow(c - seedCol, 2));
        return dist - 15.0;  // Circle of radius 15
      })
    );
    
    // Parameters
    const mu = 0.1;      // Length penalty
    const nu = 0.0;      // Area penalty  
    const lambda1 = 1.0; // Inside fitting
    const lambda2 = 1.0; // Outside fitting
    const dt = 0.5;      // Time step
    
    for (int iter = 0; iter < _chanVeseMaxIter; iter++) {
      // Compute inside/outside means
      double sumIn = 0, sumOut = 0;
      int countIn = 0, countOut = 0;
      
      for (int r = 0; r < _spectrogramHeight; r++) {
        for (int c = 0; c < _spectrogramWidth; c++) {
          final val = _spectrogramData![r * _spectrogramWidth + c];
          if (phi[r][c] <= 0) {
            sumIn += val;
            countIn++;
          } else {
            sumOut += val;
            countOut++;
          }
        }
      }
      
      final c1 = countIn > 0 ? sumIn / countIn : 0;
      final c2 = countOut > 0 ? sumOut / countOut : 0;
      
      // Update level set
      for (int r = 1; r < _spectrogramHeight - 1; r++) {
        for (int c = 1; c < _spectrogramWidth - 1; c++) {
          final val = _spectrogramData![r * _spectrogramWidth + c];
          
          // Compute curvature (simplified)
          final dx = (phi[r][c + 1] - phi[r][c - 1]) / 2;
          final dy = (phi[r + 1][c] - phi[r - 1][c]) / 2;
          final dxx = phi[r][c + 1] - 2 * phi[r][c] + phi[r][c - 1];
          final dyy = phi[r + 1][c] - 2 * phi[r][c] + phi[r - 1][c];
          
          final gradMag = math.sqrt(dx * dx + dy * dy + 1e-8);
          final curvature = (dxx + dyy) / gradMag;
          
          // Chan-Vese force
          final f1 = (val - c1) * (val - c1);
          final f2 = (val - c2) * (val - c2);
          final force = mu * curvature - nu - lambda1 * f1 + lambda2 * f2;
          
          // Dirac delta approximation
          final eps = 1.0;
          final dirac = eps / (math.pi * (eps * eps + phi[r][c] * phi[r][c]));
          
          phi[r][c] += dt * dirac * force;
        }
      }
    }
    
    // Convert level set to binary mask
    final mask = List.generate(_spectrogramHeight, (r) => 
      List.generate(_spectrogramWidth, (c) => phi[r][c] <= 0)
    );
    
    return mask;
  }

  void _createBoxAt(Offset center, Size size, double width, double height) {
    final cx = center.dx / size.width;
    final cy = center.dy / size.height;
    _createBox(
      (cx - width / 2).clamp(0.0, 1.0),
      (cy - height / 2).clamp(0.0, 1.0),
      (cx + width / 2).clamp(0.0, 1.0),
      (cy + height / 2).clamp(0.0, 1.0),
    );
  }

  void _createBox(double x1, double y1, double x2, double y2) {
    // Convert normalized coords to ABSOLUTE time (within current window)
    final absTimeStart = _windowStartSec + math.min(x1, x2) * _windowLengthSec;
    final absTimeEnd = _windowStartSec + math.max(x1, x2) * _windowLengthSec;
    
    // Calculate frequency bounds
    double? freqStartMHz, freqEndMHz;
    if (widget.header != null) {
      final bwHz = widget.header!.bandwidthHz;
      final cfHz = widget.header!.centerFreqHz;
      freqStartMHz = (cfHz - bwHz/2 + (1 - math.max(y1, y2)) * bwHz) / 1e6;
      freqEndMHz = (cfHz - bwHz/2 + (1 - math.min(y1, y2)) * bwHz) / 1e6;
    }
    
    final box = LabelBox(
      x1: x1, y1: y1, x2: x2, y2: y2,
      className: widget.header?.signalName ?? 'unknown',
      id: DateTime.now().millisecondsSinceEpoch,
      freqStartMHz: freqStartMHz,
      freqEndMHz: freqEndMHz,
      timeStartSec: absTimeStart,
      timeEndSec: absTimeEnd,
    );
    
    widget.onBoxCreated?.call(box);
  }

  /// Convert absolute time box to screen coordinates for current window
  /// Returns null if box is not visible in current window
  Rect? _boxToScreenRect(LabelBox box, Size size) {
    if (box.timeStartSec == null || box.timeEndSec == null) {
      return box.toRect(size);  // Fallback to normalized coords
    }
    
    final windowEnd = _windowStartSec + _windowLengthSec;
    
    // Check if box overlaps current window
    if (box.timeEndSec! < _windowStartSec || box.timeStartSec! > windowEnd) {
      return null;  // Box not visible
    }
    
    // Convert absolute time to screen x coords
    final x1 = ((box.timeStartSec! - _windowStartSec) / _windowLengthSec).clamp(0.0, 1.0);
    final x2 = ((box.timeEndSec! - _windowStartSec) / _windowLengthSec).clamp(0.0, 1.0);
    
    // Y coords stay the same (frequency doesn't change with panning)
    final y1 = math.min(box.y1, box.y2);
    final y2 = math.max(box.y1, box.y2);
    
    return Rect.fromLTRB(
      x1 * size.width,
      y1 * size.height,
      x2 * size.width,
      y2 * size.height,
    );
  }

  void _startDraw(Offset localPosition, Size size) {
    setState(() {
      _isDrawing = true;
      _drawStart = localPosition;
      _drawCurrent = localPosition;
    });
  }

  void _updateDraw(Offset localPosition, Size size) {
    if (!_isDrawing) return;
    setState(() {
      _drawCurrent = Offset(
        localPosition.dx.clamp(0, size.width),
        localPosition.dy.clamp(0, size.height),
      );
    });
  }

  void _endDraw(Size size) {
    if (!_isDrawing || _drawStart == null || _drawCurrent == null) {
      setState(() => _isDrawing = false);
      return;
    }
    
    final x1 = _drawStart!.dx / size.width;
    final y1 = _drawStart!.dy / size.height;
    final x2 = _drawCurrent!.dx / size.width;
    final y2 = _drawCurrent!.dy / size.height;
    
    // Only create if box is big enough
    if ((x2 - x1).abs() > 0.01 && (y2 - y1).abs() > 0.01) {
      _createBox(x1, y1, x2, y2);
    }
    
    setState(() {
      _isDrawing = false;
      _drawStart = null;
      _drawCurrent = null;
    });
  }

  void _onScroll(PointerScrollEvent event, Size size) {
    // DISABLED - scroll does nothing for now (zoom was breaking bounding boxes)
    // TODO: Re-implement zoom FFT properly without breaking box addressing
  }

  @override
  void dispose() {
    _displayImage?.dispose();
    super.dispose();
  }

  // Navigation methods - work with the new zoom bounds
  void _navigate(double deltaSec) {
    final currentLength = _viewTimeEndSec - _viewTimeStartSec;
    setState(() {
      _viewTimeStartSec = (_viewTimeStartSec + deltaSec).clamp(0.0, math.max(0.0, _totalDurationSec - currentLength));
      _viewTimeEndSec = _viewTimeStartSec + currentLength;
    });
    _loadAndCompute();
  }

  void _setWindowLength(double lengthSec) {
    setState(() {
      // Keep start, adjust end
      _viewTimeEndSec = _viewTimeStartSec + lengthSec;
      // Ensure we don't go past end
      if (_viewTimeEndSec > _totalDurationSec) {
        _viewTimeEndSec = _totalDurationSec;
        _viewTimeStartSec = math.max(0.0, _totalDurationSec - lengthSec);
      }
    });
    _loadAndCompute();
  }

  void _setFftSize(int size) {
    setState(() {
      _fftSize = size;
    });
    _loadAndCompute();
  }
  
  /// Zoom in/out centered on a point - this is the TRUE ZOOM method
  void _zoom(double zoomFactor, Offset? focalPoint, Size size) {
    final currentTimeSpan = _viewTimeEndSec - _viewTimeStartSec;
    final currentFreqSpan = _viewFreqEndHz - _viewFreqStartHz;
    
    // Calculate new spans
    final newTimeSpan = (currentTimeSpan / zoomFactor).clamp(0.01, _totalDurationSec);
    final newFreqSpan = (currentFreqSpan / zoomFactor).clamp(100000.0, _totalBandwidthHz);
    
    // Get focal point in data coordinates (default to center if null)
    double focalTimeSec, focalFreqHz;
    if (focalPoint != null) {
      // Convert screen position to data coordinates
      final normX = focalPoint.dx / size.width;
      final normY = focalPoint.dy / size.height;
      focalTimeSec = _viewTimeStartSec + normX * currentTimeSpan;
      focalFreqHz = _viewFreqEndHz - normY * currentFreqSpan;  // Y inverted
    } else {
      focalTimeSec = (_viewTimeStartSec + _viewTimeEndSec) / 2;
      focalFreqHz = (_viewFreqStartHz + _viewFreqEndHz) / 2;
    }
    
    // Recalculate bounds centered on focal point
    final newTimeStart = (focalTimeSec - newTimeSpan / 2).clamp(0.0, _totalDurationSec - newTimeSpan);
    final newTimeEnd = newTimeStart + newTimeSpan;
    
    final halfBw = _totalBandwidthHz / 2;
    final newFreqStart = (focalFreqHz - newFreqSpan / 2).clamp(-halfBw, halfBw - newFreqSpan);
    final newFreqEnd = newFreqStart + newFreqSpan;
    
    setState(() {
      _viewTimeStartSec = newTimeStart;
      _viewTimeEndSec = newTimeEnd;
      _viewFreqStartHz = newFreqStart;
      _viewFreqEndHz = newFreqEnd;
    });
    
    // Debounce the recomputation
    _zoomDebounceTimer?.cancel();
    _zoomDebounceTimer = Timer(const Duration(milliseconds: 100), () {
      _loadAndCompute();
    });
  }

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        // Main spectrogram area (left - takes most space)
        Expanded(
          child: _buildSpectrogramArea(),
        ),
        // Control panel (right - vertical strip)
        Container(
          width: 100,
          color: G20Colors.surfaceDark,
          child: _buildControlPanelVerticalRight(),
        ),
      ],
    );
  }

  Widget _buildSpectrogramArea() {
    // Show loading indicator only if we don't have an image yet
    if (_isLoading && _displayImage == null) {
      return Container(
        color: G20Colors.cardDark,
        child: const Center(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              CircularProgressIndicator(color: G20Colors.primary),
              SizedBox(height: 12),
              Text('Computing spectrogram...', style: TextStyle(color: G20Colors.textSecondaryDark)),
            ],
          ),
        ),
      );
    }

    if (_error != null) {
      return Container(
        color: G20Colors.cardDark,
        child: Center(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              const Icon(Icons.error_outline, color: G20Colors.error, size: 36),
              const SizedBox(height: 8),
              Text(_error!, style: const TextStyle(color: G20Colors.textSecondaryDark)),
              const SizedBox(height: 12),
              ElevatedButton(onPressed: _loadAndCompute, child: const Text('Retry')),
            ],
          ),
        ),
      );
    }

    if (widget.filepath == null || _displayImage == null) {
      return Container(
        color: G20Colors.cardDark,
        child: const Center(
          child: Text('Select a file to view spectrogram', 
            style: TextStyle(color: G20Colors.textSecondaryDark)),
        ),
      );
    }

    return LayoutBuilder(
      builder: (context, constraints) {
        final size = constraints.biggest;
        const leftMargin = 70.0;  // Freq axis - wider for "xxx.xx MHz"
        const bottomMargin = 20.0;  // Time axis only (no slider)
        const topMargin = 4.0;
        const rightMargin = 4.0;
        
        final plotRect = Rect.fromLTRB(
          leftMargin, topMargin,
          size.width - rightMargin, size.height - bottomMargin,
        );
        
        return Stack(
          clipBehavior: Clip.none,
          children: [
            Container(color: G20Colors.cardDark),
            // Spectrogram
            Positioned.fromRect(
              rect: plotRect,
              child: Listener(
                onPointerSignal: (event) {
                  if (event is PointerScrollEvent) {
                    _onScroll(event, plotRect.size);
                  }
                },
                child: GestureDetector(
                  onTapUp: (d) => _autoDetect(d.localPosition, plotRect.size),
                  onPanStart: (d) => _startDraw(d.localPosition, plotRect.size),
                  onPanUpdate: (d) => _updateDraw(d.localPosition, plotRect.size),
                  onPanEnd: (_) => _endDraw(plotRect.size),
                  child: Stack(
                    fit: StackFit.expand,
                    clipBehavior: Clip.hardEdge,
                    children: [
                      // Spectrogram image
                      RawImage(
                        image: _displayImage,
                        fit: BoxFit.fill,
                        filterQuality: FilterQuality.low,
                      ),
                      // Existing boxes - render with absolute time coords
                      ...widget.boxes.map((box) {
                        final screenRect = _boxToScreenRect(box, plotRect.size);
                        if (screenRect == null) return const SizedBox.shrink();
                        return _BoxWidgetAbsolute(
                          box: box,
                          screenRect: screenRect,
                          onTap: () => widget.onBoxSelected?.call(box.id),
                          onDelete: () => widget.onBoxDeleted?.call(box.id),
                        );
                      }),
                      // Current drawing box
                      if (_isDrawing && _drawStart != null && _drawCurrent != null)
                        _DrawingBox(start: _drawStart!, current: _drawCurrent!),
                    ],
                  ),
                ),
              ),
            ),
            // Frequency axis (left)
            Positioned(
              left: 0, top: topMargin, bottom: bottomMargin, width: leftMargin - 2,
              child: _FreqAxis(header: widget.header),
            ),
            // Time axis (bottom) - shows window position
            Positioned(
              left: leftMargin, right: rightMargin, bottom: 0, height: bottomMargin,
              child: _TimeAxisWindowed(
                windowStartSec: _windowStartSec,
                windowLengthSec: _windowLengthSec,
              ),
            ),
            // NO SLIDER - removed to prevent clipping
          ],
        );
      },
    );
  }

  Widget _buildControlPanel() {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(8),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          // Position display
          Container(
            padding: const EdgeInsets.symmetric(vertical: 8, horizontal: 4),
            decoration: BoxDecoration(
              color: G20Colors.backgroundDark,
              borderRadius: BorderRadius.circular(4),
            ),
            child: Column(
              children: [
                Text(
                  '${_windowStartSec.toStringAsFixed(1)}s',
                  style: const TextStyle(
                    color: G20Colors.primary,
                    fontSize: 16,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                Text(
                  'of ${_totalDurationSec.toStringAsFixed(0)}s',
                  style: const TextStyle(
                    color: G20Colors.textSecondaryDark,
                    fontSize: 10,
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(height: 12),
          
          // Navigation buttons - jump by window length
          const Text('Navigate', style: TextStyle(color: G20Colors.textSecondaryDark, fontSize: 9)),
          const SizedBox(height: 2),
          Row(
            children: [
              Expanded(
                child: _NavButtonSmall(
                  label: '◀',
                  onPressed: () => _navigate(-_windowLengthSec),
                ),
              ),
              const SizedBox(width: 2),
              Expanded(
                child: _NavButtonSmall(
                  label: '▶',
                  onPressed: () => _navigate(_windowLengthSec),
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          
          // Window length selector
          const Text('Window', style: TextStyle(color: G20Colors.textSecondaryDark, fontSize: 9)),
          const SizedBox(height: 2),
          Wrap(
            spacing: 2,
            runSpacing: 2,
            children: [
              _OptionButtonSmall(label: '0.1s', isSelected: _windowLengthSec == 0.1, onPressed: () => _setWindowLength(0.1)),
              _OptionButtonSmall(label: '0.2s', isSelected: _windowLengthSec == 0.2, onPressed: () => _setWindowLength(0.2)),
              _OptionButtonSmall(label: '0.5s', isSelected: _windowLengthSec == 0.5, onPressed: () => _setWindowLength(0.5)),
              _OptionButtonSmall(label: '1.0s', isSelected: _windowLengthSec == 1.0, onPressed: () => _setWindowLength(1.0)),
              _OptionButtonSmall(label: '2.0s', isSelected: _windowLengthSec == 2.0, onPressed: () => _setWindowLength(2.0)),
            ],
          ),
          const SizedBox(height: 8),
          
          // FFT Size selector
          const Text('FFT Size', style: TextStyle(color: G20Colors.textSecondaryDark, fontSize: 9)),
          const SizedBox(height: 2),
          Wrap(
            spacing: 2,
            runSpacing: 2,
            children: [
              _OptionButtonSmall(label: '1024', isSelected: _windowFftSize == 1024, onPressed: () => _setFftSize(1024)),
              _OptionButtonSmall(label: '2048', isSelected: _windowFftSize == 2048, onPressed: () => _setFftSize(2048)),
              _OptionButtonSmall(label: '4096', isSelected: _windowFftSize == 4096, onPressed: () => _setFftSize(4096)),
              _OptionButtonSmall(label: '8192', isSelected: _windowFftSize == 8192, onPressed: () => _setFftSize(8192)),
            ],
          ),
        ],
      ),
    );
  }

  /// Vertical control panel for RIGHT SIDE - Window, FFT, Navigation
  Widget _buildControlPanelVerticalRight() {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(8),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          // Position display
          Container(
            padding: const EdgeInsets.symmetric(vertical: 8),
            decoration: BoxDecoration(
              color: G20Colors.backgroundDark,
              borderRadius: BorderRadius.circular(4),
            ),
            child: Column(
              children: [
                Text(
                  '${_windowStartSec.toStringAsFixed(1)}s',
                  style: const TextStyle(
                    color: G20Colors.primary,
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                Text(
                  'of ${_totalDurationSec.toStringAsFixed(0)}s',
                  style: const TextStyle(
                    color: G20Colors.textSecondaryDark,
                    fontSize: 10,
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(height: 12),
          
          // Navigation buttons - full window (<</>>) and half window (</>)
          Row(
            children: [
              // << full window back
              Expanded(
                child: SizedBox(
                  height: 40,
                  child: ElevatedButton(
                    onPressed: () => _navigate(-_windowLengthSec),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: G20Colors.cardDark,
                      foregroundColor: G20Colors.textPrimaryDark,
                      padding: EdgeInsets.zero,
                      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(4)),
                    ),
                    child: const Text('◀◀', style: TextStyle(fontSize: 14)),
                  ),
                ),
              ),
              const SizedBox(width: 4),
              // >> full window forward
              Expanded(
                child: SizedBox(
                  height: 40,
                  child: ElevatedButton(
                    onPressed: () => _navigate(_windowLengthSec),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: G20Colors.cardDark,
                      foregroundColor: G20Colors.textPrimaryDark,
                      padding: EdgeInsets.zero,
                      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(4)),
                    ),
                    child: const Text('▶▶', style: TextStyle(fontSize: 14)),
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 4),
          // Half window navigation (< / >)
          Row(
            children: [
              Expanded(
                child: SizedBox(
                  height: 40,
                  child: ElevatedButton(
                    onPressed: () => _navigate(-_windowLengthSec / 2),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: G20Colors.cardDark,
                      foregroundColor: G20Colors.textPrimaryDark,
                      padding: EdgeInsets.zero,
                      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(4)),
                    ),
                    child: const Text('◀', style: TextStyle(fontSize: 16)),
                  ),
                ),
              ),
              const SizedBox(width: 4),
              Expanded(
                child: SizedBox(
                  height: 40,
                  child: ElevatedButton(
                    onPressed: () => _navigate(_windowLengthSec / 2),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: G20Colors.cardDark,
                      foregroundColor: G20Colors.textPrimaryDark,
                      padding: EdgeInsets.zero,
                      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(4)),
                    ),
                    child: const Text('▶', style: TextStyle(fontSize: 16)),
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 16),
          
          // Window length
          const Text('Window', style: TextStyle(color: G20Colors.textSecondaryDark, fontSize: 10)),
          const SizedBox(height: 4),
          _VerticalOptionButton(label: '0.2s', isSelected: _windowLengthSec == 0.2, onPressed: () => _setWindowLength(0.2)),
          const SizedBox(height: 4),
          _VerticalOptionButton(label: '0.5s', isSelected: _windowLengthSec == 0.5, onPressed: () => _setWindowLength(0.5)),
          const SizedBox(height: 4),
          _VerticalOptionButton(label: '1.0s', isSelected: _windowLengthSec == 1.0, onPressed: () => _setWindowLength(1.0)),
          const SizedBox(height: 4),
          _VerticalOptionButton(label: '2.0s', isSelected: _windowLengthSec == 2.0, onPressed: () => _setWindowLength(2.0)),
          const SizedBox(height: 16),
          
          // FFT Size
          const Text('FFT', style: TextStyle(color: G20Colors.textSecondaryDark, fontSize: 10)),
          const SizedBox(height: 4),
          _VerticalOptionButton(label: '1k', isSelected: _windowFftSize == 1024, onPressed: () => _setFftSize(1024)),
          const SizedBox(height: 4),
          _VerticalOptionButton(label: '2k', isSelected: _windowFftSize == 2048, onPressed: () => _setFftSize(2048)),
          const SizedBox(height: 4),
          _VerticalOptionButton(label: '4k', isSelected: _windowFftSize == 4096, onPressed: () => _setFftSize(4096)),
          const SizedBox(height: 4),
          _VerticalOptionButton(label: '8k', isSelected: _windowFftSize == 8192, onPressed: () => _setFftSize(8192)),
        ],
      ),
    );
  }

  /// Horizontal control panel for bottom strip layout - RIGHT SIDE for one-handed use
  Widget _buildControlPanelHorizontal() {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      child: Row(
        children: [
          // Position display (left)
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 4),
            decoration: BoxDecoration(
              color: G20Colors.backgroundDark,
              borderRadius: BorderRadius.circular(4),
            ),
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Text(
                  '${_windowStartSec.toStringAsFixed(1)}s',
                  style: const TextStyle(
                    color: G20Colors.primary,
                    fontSize: 20,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                Text(
                  'of ${_totalDurationSec.toStringAsFixed(0)}s',
                  style: const TextStyle(
                    color: G20Colors.textSecondaryDark,
                    fontSize: 12,
                  ),
                ),
              ],
            ),
          ),
          
          // Spacer to push controls to right
          const Spacer(),
          
          // Window length - RIGHT SIDE with 12px gaps
          Column(
            mainAxisAlignment: MainAxisAlignment.center,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Text('Window', style: TextStyle(color: G20Colors.textSecondaryDark, fontSize: 11)),
              const SizedBox(height: 4),
              Row(
                children: [
                  _TouchButtonWide(label: '0.2s', isSelected: _windowLengthSec == 0.2, onPressed: () => _setWindowLength(0.2)),
                  const SizedBox(width: 12),
                  _TouchButtonWide(label: '0.5s', isSelected: _windowLengthSec == 0.5, onPressed: () => _setWindowLength(0.5)),
                  const SizedBox(width: 12),
                  _TouchButtonWide(label: '1s', isSelected: _windowLengthSec == 1.0, onPressed: () => _setWindowLength(1.0)),
                  const SizedBox(width: 12),
                  _TouchButtonWide(label: '2s', isSelected: _windowLengthSec == 2.0, onPressed: () => _setWindowLength(2.0)),
                ],
              ),
            ],
          ),
          const SizedBox(width: 24),
          
          // FFT Size - RIGHT SIDE with 12px gaps
          Column(
            mainAxisAlignment: MainAxisAlignment.center,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Text('FFT', style: TextStyle(color: G20Colors.textSecondaryDark, fontSize: 11)),
              const SizedBox(height: 4),
              Row(
                children: [
                  _TouchButtonWide(label: '1k', isSelected: _windowFftSize == 1024, onPressed: () => _setFftSize(1024)),
                  const SizedBox(width: 12),
                  _TouchButtonWide(label: '2k', isSelected: _windowFftSize == 2048, onPressed: () => _setFftSize(2048)),
                  const SizedBox(width: 12),
                  _TouchButtonWide(label: '4k', isSelected: _windowFftSize == 4096, onPressed: () => _setFftSize(4096)),
                  const SizedBox(width: 12),
                  _TouchButtonWide(label: '8k', isSelected: _windowFftSize == 8192, onPressed: () => _setFftSize(8192)),
                ],
              ),
            ],
          ),
          const SizedBox(width: 24),
          
          // Navigation buttons - FAR RIGHT for thumb access
          SizedBox(
            width: 56,
            height: 56,
            child: ElevatedButton(
              onPressed: () => _navigate(-_windowLengthSec),
              style: ElevatedButton.styleFrom(
                backgroundColor: G20Colors.cardDark,
                foregroundColor: G20Colors.textPrimaryDark,
                padding: EdgeInsets.zero,
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(6)),
              ),
              child: const Text('◀', style: TextStyle(fontSize: 24)),
            ),
          ),
          const SizedBox(width: 8),
          SizedBox(
            width: 56,
            height: 56,
            child: ElevatedButton(
              onPressed: () => _navigate(_windowLengthSec),
              style: ElevatedButton.styleFrom(
                backgroundColor: G20Colors.cardDark,
                foregroundColor: G20Colors.textPrimaryDark,
                padding: EdgeInsets.zero,
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(6)),
              ),
              child: const Text('▶', style: TextStyle(fontSize: 24)),
            ),
          ),
        ],
      ),
    );
  }
}

/// Touch-friendly button (44x36 minimum)
class _TouchButton extends StatelessWidget {
  final String label;
  final bool isSelected;
  final VoidCallback onPressed;

  const _TouchButton({
    required this.label,
    required this.isSelected,
    required this.onPressed,
  });

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      height: 36,
      child: ElevatedButton(
        onPressed: onPressed,
        style: ElevatedButton.styleFrom(
          backgroundColor: isSelected ? G20Colors.primary : G20Colors.cardDark,
          foregroundColor: isSelected ? Colors.white : G20Colors.textSecondaryDark,
          padding: const EdgeInsets.symmetric(horizontal: 12),
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(4)),
        ),
        child: Text(
          label,
          style: TextStyle(
            fontSize: 12,
            fontWeight: isSelected ? FontWeight.bold : FontWeight.normal,
          ),
        ),
      ),
    );
  }
}

/// Wide touch-friendly button - 60px wide, 48px tall with proper spacing
class _TouchButtonWide extends StatelessWidget {
  final String label;
  final bool isSelected;
  final VoidCallback onPressed;

  const _TouchButtonWide({
    required this.label,
    required this.isSelected,
    required this.onPressed,
  });

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      height: 48,
      width: 60,
      child: ElevatedButton(
        onPressed: onPressed,
        style: ElevatedButton.styleFrom(
          backgroundColor: isSelected ? G20Colors.primary : G20Colors.cardDark,
          foregroundColor: isSelected ? Colors.white : G20Colors.textSecondaryDark,
          padding: EdgeInsets.zero,
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
        ),
        child: Text(
          label,
          style: TextStyle(
            fontSize: 14,
            fontWeight: isSelected ? FontWeight.bold : FontWeight.normal,
          ),
        ),
      ),
    );
  }
}

class _BoxWidget extends StatelessWidget {
  final LabelBox box;
  final Size size;
  final VoidCallback? onTap;
  final VoidCallback? onDelete;

  const _BoxWidget({required this.box, required this.size, this.onTap, this.onDelete});

  @override
  Widget build(BuildContext context) {
    final rect = box.toRect(size);
    final borderColor = box.isSelected ? G20Colors.primary : Colors.cyan;
    
    return Positioned.fromRect(
      rect: rect,
      child: GestureDetector(
        onTap: onTap,
        child: Container(
          decoration: BoxDecoration(
            border: Border.all(
              color: borderColor,
              width: box.isSelected ? 3 : 2,
            ),
            color: borderColor.withOpacity(0.15),
          ),
          child: Stack(
            clipBehavior: Clip.none,
            children: [
              // Label tag - positioned ABOVE the box for better visibility
              Positioned(
                top: -20, left: -2,
                child: Container(
                  padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 3),
                  decoration: BoxDecoration(
                    color: Colors.black.withOpacity(0.85),
                    borderRadius: BorderRadius.circular(3),
                    border: Border.all(color: borderColor, width: 1),
                  ),
                  child: Text(
                    box.className.toUpperCase(),
                    style: const TextStyle(
                      color: Colors.white,
                      fontSize: 11,
                      fontWeight: FontWeight.bold,
                      letterSpacing: 0.5,
                    ),
                  ),
                ),
              ),
              // Delete button
              if (box.isSelected)
                Positioned(
                  top: -8, right: -8,
                  child: GestureDetector(
                    onTap: onDelete,
                    child: Container(
                      padding: const EdgeInsets.all(4),
                      decoration: BoxDecoration(
                        color: G20Colors.error,
                        shape: BoxShape.circle,
                        border: Border.all(color: Colors.white, width: 1),
                      ),
                      child: const Icon(Icons.close, size: 10, color: Colors.white),
                    ),
                  ),
                ),
            ],
          ),
        ),
      ),
    );
  }
}

/// Box widget that uses pre-computed screen rect (for absolute time coords)
/// Uses SOI color for the signal class - same colors as live detection page
class _BoxWidgetAbsolute extends StatelessWidget {
  final LabelBox box;
  final Rect screenRect;
  final VoidCallback? onTap;
  final VoidCallback? onDelete;

  const _BoxWidgetAbsolute({
    required this.box,
    required this.screenRect,
    this.onTap,
    this.onDelete,
  });

  @override
  Widget build(BuildContext context) {
    // Use SOI color based on signal class name - consistent with live detection
    final soiColor = getSOIColor(box.className);
    final borderColor = box.isSelected ? Colors.white : soiColor;
    
    return Positioned.fromRect(
      rect: screenRect,
      child: GestureDetector(
        onTap: onTap,
        child: Container(
          decoration: BoxDecoration(
            border: Border.all(
              color: borderColor,
              width: box.isSelected ? 4 : 3,
            ),
            color: soiColor.withOpacity(0.2),
          ),
          // Delete button only when selected - positioned in corner
          child: box.isSelected ? Stack(
            clipBehavior: Clip.none,
            children: [
              Positioned(
                top: -12, right: -12,
                child: GestureDetector(
                  onTap: onDelete,
                  child: Container(
                    padding: const EdgeInsets.all(8),
                    decoration: BoxDecoration(
                      color: G20Colors.error,
                      shape: BoxShape.circle,
                      border: Border.all(color: Colors.white, width: 2),
                    ),
                    child: const Icon(Icons.close, size: 14, color: Colors.white),
                  ),
                ),
              ),
            ],
          ) : null,
        ),
      ),
    );
  }
}

class _DrawingBox extends StatelessWidget {
  final Offset start;
  final Offset current;

  const _DrawingBox({required this.start, required this.current});

  @override
  Widget build(BuildContext context) {
    final left = math.min(start.dx, current.dx);
    final top = math.min(start.dy, current.dy);
    final width = (start.dx - current.dx).abs();
    final height = (start.dy - current.dy).abs();
    
    return Positioned(
      left: left, top: top, width: width, height: height,
      child: Container(
        decoration: BoxDecoration(
          border: Border.all(color: G20Colors.primary, width: 2),
          color: G20Colors.primary.withOpacity(0.2),
        ),
      ),
    );
  }
}

class _FreqAxis extends StatelessWidget {
  final RfcapHeader? header;
  const _FreqAxis({this.header});

  @override
  Widget build(BuildContext context) {
    if (header == null) return const SizedBox();
    
    final low = header!.centerFreqMHz - header!.bandwidthMHz / 2;
    final high = header!.centerFreqMHz + header!.bandwidthMHz / 2;
    final mid = header!.centerFreqMHz;
    
    return Container(
      padding: const EdgeInsets.only(right: 4),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        crossAxisAlignment: CrossAxisAlignment.end,
        children: [
          Text('${high.toStringAsFixed(2)} MHz', style: const TextStyle(fontSize: 9, color: G20Colors.textSecondaryDark)),
          Text('${mid.toStringAsFixed(2)} MHz', style: const TextStyle(fontSize: 9, color: G20Colors.textSecondaryDark)),
          Text('${low.toStringAsFixed(2)} MHz', style: const TextStyle(fontSize: 9, color: G20Colors.textSecondaryDark)),
        ],
      ),
    );
  }
}

class _TimeAxis extends StatelessWidget {
  final double durationSec;
  const _TimeAxis({required this.durationSec});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.only(top: 4),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          const Text('0s', style: TextStyle(fontSize: 9, color: G20Colors.textSecondaryDark)),
          Text('${(durationSec / 2).toStringAsFixed(1)}s', style: const TextStyle(fontSize: 9, color: G20Colors.textSecondaryDark)),
          Text('${durationSec.toStringAsFixed(1)}s', style: const TextStyle(fontSize: 9, color: G20Colors.textSecondaryDark)),
        ],
      ),
    );
  }
}

/// Time axis showing window position
class _TimeAxisWindowed extends StatelessWidget {
  final double windowStartSec;
  final double windowLengthSec;
  const _TimeAxisWindowed({required this.windowStartSec, required this.windowLengthSec});

  @override
  Widget build(BuildContext context) {
    final endSec = windowStartSec + windowLengthSec;
    final midSec = windowStartSec + windowLengthSec / 2;
    
    return Container(
      padding: const EdgeInsets.only(top: 4),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text('${windowStartSec.toStringAsFixed(2)}s', style: const TextStyle(fontSize: 9, color: G20Colors.textSecondaryDark)),
          Text('${midSec.toStringAsFixed(2)}s', style: const TextStyle(fontSize: 9, color: G20Colors.textSecondaryDark)),
          Text('${endSec.toStringAsFixed(2)}s', style: const TextStyle(fontSize: 9, color: G20Colors.textSecondaryDark)),
        ],
      ),
    );
  }
}

/// Small navigation button for control panel
class _NavButtonSmall extends StatelessWidget {
  final String label;
  final VoidCallback onPressed;
  
  const _NavButtonSmall({required this.label, required this.onPressed});

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      height: 32,
      child: ElevatedButton(
        onPressed: onPressed,
        style: ElevatedButton.styleFrom(
          backgroundColor: G20Colors.cardDark,
          foregroundColor: G20Colors.textPrimaryDark,
          padding: EdgeInsets.zero,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(3),
          ),
        ),
        child: Text(label, style: const TextStyle(fontSize: 14)),
      ),
    );
  }
}

/// Small option button for control panel (window length, FFT size)
class _OptionButtonSmall extends StatelessWidget {
  final String label;
  final bool isSelected;
  final VoidCallback onPressed;
  
  const _OptionButtonSmall({
    required this.label,
    required this.isSelected,
    required this.onPressed,
  });

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      height: 24,
      child: ElevatedButton(
        onPressed: onPressed,
        style: ElevatedButton.styleFrom(
          backgroundColor: isSelected ? G20Colors.primary : G20Colors.cardDark,
          foregroundColor: isSelected ? Colors.white : G20Colors.textSecondaryDark,
          padding: const EdgeInsets.symmetric(horizontal: 6),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(3),
          ),
        ),
        child: Text(
          label,
          style: TextStyle(
            fontSize: 9,
            fontWeight: isSelected ? FontWeight.bold : FontWeight.normal,
          ),
        ),
      ),
    );
  }
}

/// Vertical option button for right sidebar
class _VerticalOptionButton extends StatelessWidget {
  final String label;
  final bool isSelected;
  final VoidCallback onPressed;
  
  const _VerticalOptionButton({
    required this.label,
    required this.isSelected,
    required this.onPressed,
  });

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      height: 36,
      child: ElevatedButton(
        onPressed: onPressed,
        style: ElevatedButton.styleFrom(
          backgroundColor: isSelected ? G20Colors.primary : G20Colors.cardDark,
          foregroundColor: isSelected ? Colors.white : G20Colors.textSecondaryDark,
          padding: EdgeInsets.zero,
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(4)),
        ),
        child: Text(
          label,
          style: TextStyle(
            fontSize: 12,
            fontWeight: isSelected ? FontWeight.bold : FontWeight.normal,
          ),
        ),
      ),
    );
  }
}

/// Position slider for navigating through the file
class _PositionSlider extends StatelessWidget {
  final double windowStartSec;
  final double windowLengthSec;
  final double totalDurationSec;
  final ValueChanged<double> onChanged;
  
  const _PositionSlider({
    required this.windowStartSec,
    required this.windowLengthSec,
    required this.totalDurationSec,
    required this.onChanged,
  });

  @override
  Widget build(BuildContext context) {
    if (totalDurationSec <= 0) return const SizedBox();
    
    // Calculate divisions - snap to half-window increments
    final stepSize = windowLengthSec / 2;
    final divisions = math.max(1, (totalDurationSec / stepSize).floor());
    final maxValue = math.max(0.0, totalDurationSec - windowLengthSec);
    
    return SliderTheme(
      data: SliderThemeData(
        trackHeight: 6,
        thumbShape: const RoundSliderThumbShape(enabledThumbRadius: 8),
        overlayShape: const RoundSliderOverlayShape(overlayRadius: 16),
        activeTrackColor: G20Colors.primary,
        inactiveTrackColor: G20Colors.cardDark,
        thumbColor: G20Colors.primary,
        overlayColor: G20Colors.primary.withOpacity(0.3),
        tickMarkShape: const RoundSliderTickMarkShape(tickMarkRadius: 2),
        activeTickMarkColor: G20Colors.primary.withOpacity(0.5),
        inactiveTickMarkColor: G20Colors.cardDark.withOpacity(0.5),
      ),
      child: Slider(
        value: windowStartSec.clamp(0.0, maxValue),
        min: 0.0,
        max: maxValue > 0 ? maxValue : 1.0,
        divisions: divisions,
        onChanged: maxValue > 0 ? (value) {
          // Snap to nearest step
          final snapped = (value / stepSize).round() * stepSize;
          onChanged(snapped.clamp(0.0, maxValue));
        } : null,
      ),
    );
  }
}
