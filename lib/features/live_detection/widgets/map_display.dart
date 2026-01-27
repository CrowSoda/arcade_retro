import 'dart:async';
import 'dart:io';
import 'dart:math' show Point;
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_map/flutter_map.dart';
import 'package:latlong2/latlong.dart';
import 'package:vector_map_tiles/vector_map_tiles.dart';
import 'package:vector_map_tiles_pmtiles/vector_map_tiles_pmtiles.dart';
import '../../../core/config/theme.dart';
import '../../../core/database/signal_database.dart';
import '../../../core/utils/dtg_formatter.dart';
import '../providers/detection_provider.dart';
import '../providers/map_provider.dart';

/// Map display using Flutter Map with Vector Tiles + PMTiles
/// Supports Windows, Linux, macOS (pure Dart implementation)
class MapDisplay extends ConsumerStatefulWidget {
  const MapDisplay({super.key});

  @override
  ConsumerState<MapDisplay> createState() => _MapDisplayState();
}

class _MapDisplayState extends ConsumerState<MapDisplay> {
  final MapController _mapController = MapController();
  VectorTileProvider? _tileProvider;
  bool _isLoading = true;
  String? _error;

  // Debounce timer to prevent message queue flooding during fast zoom
  Timer? _positionDebouncer;

  // Freeze state during zoom to prevent tile loading spam
  bool _isZooming = false;
  Timer? _zoomEndTimer;
  double? _lastStableZoom;

  // Selected detection for popup
  Detection? _selectedDetection;

  @override
  void initState() {
    super.initState();
    _loadPmTiles();
  }

  @override
  void dispose() {
    _positionDebouncer?.cancel();
    _zoomEndTimer?.cancel();
    _mapController.dispose();
    super.dispose();
  }

  Future<void> _loadPmTiles() async {
    try {
      final pmtilesPath = _getPmtilesPath();

      // Check if file exists
      final file = File(pmtilesPath);
      if (!file.existsSync()) {
        setState(() {
          _error = 'PMTiles file not found:\n$pmtilesPath';
          _isLoading = false;
        });
        return;
      }

      // Load PMTiles from local file
      _tileProvider = await PmTilesVectorTileProvider.fromSource(pmtilesPath);
      setState(() => _isLoading = false);
    } catch (e) {
      setState(() {
        _error = 'Error loading PMTiles: $e';
        _isLoading = false;
      });
    }
  }

  String _getPmtilesPath() {
    final currentDir = Directory.current.path;
    // Try direct path first (when running from g20_demo/)
    final directPath = '$currentDir/data/map/20260119.pmtiles'.replaceAll('\\', '/');
    if (File(directPath).existsSync()) {
      return directPath;
    }
    // Fallback: running from parent directory
    return '$currentDir/g20_demo/data/map/20260119.pmtiles'.replaceAll('\\', '/');
  }

  /// Called during zoom gestures - sets freeze state and delays tile loading
  void _onZoomStart() {
    _zoomEndTimer?.cancel();
    if (!_isZooming) {
      _lastStableZoom = _mapController.camera.zoom;
      setState(() => _isZooming = true);
    }
  }

  /// Called when zoom gesture ends - waits before unfreezing to allow settle
  void _onZoomEnd() {
    _zoomEndTimer?.cancel();
    _zoomEndTimer = Timer(const Duration(milliseconds: 300), () {
      if (mounted) {
        setState(() => _isZooming = false);
      }
    });
  }

  /// Find detection near tap point
  Detection? _findDetectionAtPoint(LatLng tapPoint, List<Detection> detections) {
    const tapRadius = 0.002; // ~200m at equator
    for (final det in detections) {
      final dist = (det.latitude - tapPoint.latitude).abs() +
                   (det.longitude - tapPoint.longitude).abs();
      if (dist < tapRadius) {
        return det;
      }
    }
    return null;
  }

  @override
  Widget build(BuildContext context) {
    final mapState = ref.watch(mapStateProvider);
    final detections = ref.watch(detectionProvider);

    // Loading state
    if (_isLoading) {
      return const Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            CircularProgressIndicator(color: G20Colors.primary),
            SizedBox(height: 16),
            Text('Loading map tiles...', style: TextStyle(color: G20Colors.textSecondaryDark)),
          ],
        ),
      );
    }

    // Error state
    if (_error != null) {
      return Center(
        child: Container(
          padding: const EdgeInsets.all(24),
          margin: const EdgeInsets.all(16),
          decoration: BoxDecoration(
            color: G20Colors.surfaceDark,
            borderRadius: BorderRadius.circular(8),
            border: Border.all(color: Colors.red.withOpacity(0.5)),
          ),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              const Icon(Icons.error_outline, color: Colors.red, size: 48),
              const SizedBox(height: 16),
              Text(_error!, style: const TextStyle(color: G20Colors.textPrimaryDark)),
              const SizedBox(height: 16),
              ElevatedButton(
                onPressed: () {
                  setState(() {
                    _isLoading = true;
                    _error = null;
                  });
                  _loadPmTiles();
                },
                child: const Text('Retry'),
              ),
            ],
          ),
        ),
      );
    }

    // Filter visible detections
    final visibleDetections = detections
        .where((det) => !mapState.hiddenSOIs.contains(det.className))
        .toList();

    return Stack(
      children: [
        GestureDetector(
          onTapUp: (details) {
            // Convert screen point to lat/lng
            final point = _mapController.camera.pointToLatLng(
              Point(details.localPosition.dx, details.localPosition.dy),
            );
            final tapped = _findDetectionAtPoint(point, visibleDetections);
            setState(() => _selectedDetection = tapped);
          },
          child: FlutterMap(
            mapController: _mapController,
            options: MapOptions(
              initialCenter: mapState.mapCenter,
              initialZoom: mapState.mapZoom,
              minZoom: 2,
              maxZoom: 18,
              backgroundColor: const Color(0xFF293847),
              interactionOptions: const InteractionOptions(
                flags: InteractiveFlag.all,
                scrollWheelVelocity: 0.002,
              ),
              onPositionChanged: (position, hasGesture) {
                // Close popup on pan/zoom
                if (hasGesture && _selectedDetection != null) {
                  setState(() => _selectedDetection = null);
                }

                if (hasGesture && position.zoom != null) {
                  final currentZoom = position.zoom!;
                  final previousZoom = _lastStableZoom ?? mapState.mapZoom;
                  if ((currentZoom - previousZoom).abs() > 0.1) {
                    _onZoomStart();
                  }
                }

                _positionDebouncer?.cancel();
                _positionDebouncer = Timer(const Duration(milliseconds: 100), () {
                  if (mounted && position.center != null) {
                    if (!_isZooming) {
                      _lastStableZoom = position.zoom;
                    }
                    _onZoomEnd();
                    ref.read(mapStateProvider.notifier).updateMapPosition(
                      position.center!,
                      position.zoom ?? mapState.mapZoom,
                    );
                  }
                });
              },
            ),
            children: [
              VectorTileLayer(
                tileProviders: TileProviders({
                  'protomaps': _tileProvider!,
                }),
                theme: ProtomapsThemes.dark(),
                tileDelay: _isZooming
                  ? const Duration(milliseconds: 500)
                  : const Duration(milliseconds: 50),
                maximumTileSubstitutionDifference: 3,
                concurrency: _isZooming ? 1 : 4,
              ),
              // Detection markers
              CircleLayer(
                circles: visibleDetections.map((det) {
                  final isSelected = _selectedDetection?.id == det.id;
                  return CircleMarker(
                    point: LatLng(det.latitude, det.longitude),
                    radius: isSelected ? 14 : 10,
                    color: getSOIColor(det.className).withOpacity(0.9),
                    borderColor: isSelected ? Colors.white : Colors.black,
                    borderStrokeWidth: isSelected ? 3 : 2,
                  );
                }).toList(),
              ),
            ],
          ),
        ),
        // Detection popup
        if (_selectedDetection != null)
          _DetectionPopup(
            detection: _selectedDetection!,
            onClose: () => setState(() => _selectedDetection = null),
          ),
        // Zoom indicator when frozen
        if (_isZooming)
          Positioned(
            top: 8,
            left: 8,
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
              decoration: BoxDecoration(
                color: G20Colors.primary.withOpacity(0.8),
                borderRadius: BorderRadius.circular(4),
              ),
              child: const Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  SizedBox(
                    width: 12,
                    height: 12,
                    child: CircularProgressIndicator(
                      strokeWidth: 2,
                      color: Colors.white,
                    ),
                  ),
                  SizedBox(width: 6),
                  Text('Zooming...', style: TextStyle(color: Colors.white, fontSize: 10)),
                ],
              ),
            ),
          ),
        // Filter controls overlay
        Positioned(
          top: 8,
          right: 8,
          child: _FilterControls(),
        ),
        // Zoom controls
        Positioned(
          bottom: 16,
          right: 8,
          child: _ZoomControls(
            onZoomIn: () {
              _onZoomStart();
              final zoom = _mapController.camera.zoom;
              _mapController.move(_mapController.camera.center, zoom + 1);
              _onZoomEnd();
            },
            onZoomOut: () {
              _onZoomStart();
              final zoom = _mapController.camera.zoom;
              _mapController.move(_mapController.camera.center, zoom - 1);
              _onZoomEnd();
            },
          ),
        ),
      ],
    );
  }
}

/// Color-matched popup for detection details
class _DetectionPopup extends StatelessWidget {
  final Detection detection;
  final VoidCallback onClose;

  const _DetectionPopup({required this.detection, required this.onClose});

  @override
  Widget build(BuildContext context) {
    final color = getSOIColor(detection.className);
    final dtg = formatDTG(detection.timestamp);

    return Positioned(
      top: 60,
      left: 16,
      child: Material(
        color: Colors.transparent,
        child: Container(
          width: 220,
          padding: const EdgeInsets.all(12),
          decoration: BoxDecoration(
            color: G20Colors.surfaceDark.withOpacity(0.95),
            borderRadius: BorderRadius.circular(8),
            border: Border.all(color: color, width: 2),
            boxShadow: [
              BoxShadow(
                color: color.withOpacity(0.3),
                blurRadius: 8,
                offset: const Offset(0, 2),
              ),
            ],
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            mainAxisSize: MainAxisSize.min,
            children: [
              // Header with close button
              Row(
                children: [
                  Container(
                    width: 12,
                    height: 12,
                    decoration: BoxDecoration(
                      color: color,
                      shape: BoxShape.circle,
                    ),
                  ),
                  const SizedBox(width: 8),
                  Expanded(
                    child: Text(
                      detection.className,
                      style: TextStyle(
                        color: color,
                        fontWeight: FontWeight.bold,
                        fontSize: 13,
                      ),
                      overflow: TextOverflow.ellipsis,
                    ),
                  ),
                  GestureDetector(
                    onTap: onClose,
                    child: const Icon(Icons.close, color: G20Colors.textSecondaryDark, size: 18),
                  ),
                ],
              ),
              const SizedBox(height: 8),
              const Divider(color: G20Colors.cardDark, height: 1),
              const SizedBox(height: 8),
              // Info rows
              _InfoRow(label: 'Confidence', value: '${(detection.confidence * 100).toStringAsFixed(1)}%'),
              _InfoRow(label: 'Frequency', value: '${detection.freqMHz.toStringAsFixed(2)} MHz'),
              _InfoRow(label: 'Bandwidth', value: '${detection.bandwidthMHz.toStringAsFixed(1)} MHz'),
              _InfoRow(label: 'DTG', value: dtg),
              _InfoRow(label: 'MGRS', value: detection.mgrsLocation),
              _InfoRow(label: 'Track ID', value: '#${detection.trackId}'),
            ],
          ),
        ),
      ),
    );
  }
}

class _InfoRow extends StatelessWidget {
  final String label;
  final String value;

  const _InfoRow({required this.label, required this.value});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 2),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(label, style: const TextStyle(color: G20Colors.textSecondaryDark, fontSize: 11)),
          Text(value, style: const TextStyle(color: G20Colors.textPrimaryDark, fontSize: 11, fontWeight: FontWeight.w500)),
        ],
      ),
    );
  }
}

/// Filter controls overlay
class _FilterControls extends ConsumerWidget {
  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final mapState = ref.watch(mapStateProvider);
    final detections = ref.watch(detectionProvider);
    final allSOIs = detections.map((d) => d.className).toSet();

    return Container(
      padding: const EdgeInsets.all(8),
      decoration: BoxDecoration(
        color: G20Colors.surfaceDark.withOpacity(0.9),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: G20Colors.cardDark),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          _FilterButton(
            label: 'All',
            isActive: mapState.hiddenSOIs.isEmpty,
            onTap: () => ref.read(mapStateProvider.notifier).showAll(),
          ),
          const SizedBox(width: 4),
          _FilterButton(
            label: 'None',
            isActive: mapState.hiddenSOIs.length == allSOIs.length,
            onTap: () => ref.read(mapStateProvider.notifier).hideAll(allSOIs),
          ),
        ],
      ),
    );
  }
}

class _FilterButton extends StatelessWidget {
  final String label;
  final bool isActive;
  final VoidCallback onTap;

  const _FilterButton({required this.label, required this.isActive, required this.onTap});

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
        decoration: BoxDecoration(
          color: isActive ? G20Colors.primary : Colors.transparent,
          borderRadius: BorderRadius.circular(4),
        ),
        child: Text(
          label,
          style: TextStyle(
            color: isActive ? Colors.white : G20Colors.textSecondaryDark,
            fontSize: 11,
            fontWeight: isActive ? FontWeight.bold : FontWeight.normal,
          ),
        ),
      ),
    );
  }
}

/// Zoom controls
class _ZoomControls extends StatelessWidget {
  final VoidCallback onZoomIn;
  final VoidCallback onZoomOut;

  const _ZoomControls({required this.onZoomIn, required this.onZoomOut});

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        _ZoomButton(icon: Icons.add, onTap: onZoomIn),
        const SizedBox(height: 4),
        _ZoomButton(icon: Icons.remove, onTap: onZoomOut),
      ],
    );
  }
}

class _ZoomButton extends StatelessWidget {
  final IconData icon;
  final VoidCallback onTap;

  const _ZoomButton({required this.icon, required this.onTap});

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        width: 36,
        height: 36,
        decoration: BoxDecoration(
          color: G20Colors.surfaceDark.withOpacity(0.9),
          borderRadius: BorderRadius.circular(4),
          border: Border.all(color: G20Colors.cardDark),
        ),
        child: Icon(icon, color: G20Colors.textPrimaryDark, size: 20),
      ),
    );
  }
}
