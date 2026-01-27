/// Detection Converter - Converts between different detection formats
///
/// Handles conversion from VideoStream detections to Detection model
library;

import '../models/detection.dart';
import '../providers/video_stream_provider.dart' as video_stream;
import '../../../core/utils/dtg_formatter.dart';

/// Convert VideoStream Detection to DetectionProvider Detection
///
/// [vd] - The VideoDetection from the video stream
/// [pts] - The presentation timestamp
/// [centerFreqMHz] - The SDR center frequency (default 825 MHz)
/// [bandwidthMHz] - The SDR bandwidth (default 20 MHz)
Detection convertVideoDetection(
  video_stream.VideoDetection vd,
  double pts, {
  double centerFreqMHz = 825.0,
  double bandwidthMHz = 20.0,
}) {
  // Calculate frequency from normalized y coordinates (frequency axis)
  final freqStart = centerFreqMHz - (bandwidthMHz / 2);
  final freqMHz = freqStart + ((vd.y1 + vd.y2) / 2) * bandwidthMHz;
  final detBandwidthMHz = (vd.y2 - vd.y1) * bandwidthMHz;

  return Detection(
    id: 'vid_${vd.detectionId}_${pts.toStringAsFixed(2)}',
    classId: vd.classId,
    className: vd.className,
    confidence: vd.confidence,
    x1: vd.x1,
    y1: vd.y1,
    x2: vd.x2,
    y2: vd.y2,
    freqMHz: freqMHz,
    bandwidthMHz: detBandwidthMHz,
    mgrsLocation: '13SDE1234567890', // TODO: Get from GPS provider
    latitude: 39.7275,  // TODO: Get from GPS provider
    longitude: -104.7303,  // TODO: Get from GPS provider
    timestamp: DateTime.now(),
    trackId: vd.detectionId,
    pts: pts,
    absoluteRow: vd.absoluteRow,  // Pass through for row-based pruning
  );
}

/// Convert a list of VideoDetections to Detection models
List<Detection> convertVideoDetections(
  List<video_stream.VideoDetection> detections,
  double pts, {
  double centerFreqMHz = 825.0,
  double bandwidthMHz = 20.0,
}) {
  return detections.map((vd) => convertVideoDetection(
    vd,
    pts,
    centerFreqMHz: centerFreqMHz,
    bandwidthMHz: bandwidthMHz,
  )).toList();
}

/// Helper to create UNK detections with proper naming: unk_[DTG]_[freq]MHz
///
/// Example output: unk_220001ZJAN26_830.2MHz
Detection createUnkDetection({
  required String id,
  required double confidence,
  required double x1,
  required double y1,
  required double x2,
  required double y2,
  required double freqMHz,
  required double bandwidthMHz,
  String? mgrsLocation,
  double? latitude,
  double? longitude,
  DateTime? timestamp,
  int trackId = 0,
  double pts = 0.0,
  int absoluteRow = 0,
}) {
  final now = timestamp ?? DateTime.now();
  // Generate className as unk_DTG_freqMHz
  final className = generateUnkFilename(now, freqMHz.toStringAsFixed(1));

  return Detection(
    id: id,
    classId: 0,
    className: className,
    confidence: confidence,
    x1: x1,
    y1: y1,
    x2: x2,
    y2: y2,
    freqMHz: freqMHz,
    bandwidthMHz: bandwidthMHz,
    mgrsLocation: mgrsLocation ?? '13SDE1234567890',
    latitude: latitude ?? 39.7275,
    longitude: longitude ?? -104.7303,
    timestamp: now,
    trackId: trackId,
    pts: pts,
    absoluteRow: absoluteRow,
  );
}
