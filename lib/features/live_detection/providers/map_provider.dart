import 'dart:math' as math;
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:latlong2/latlong.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'detection_provider.dart';

/// View mode for the main display area
enum DisplayMode {
  waterfall,  // Show waterfall + PSD
  map,        // Show map with detection markers
}

/// Map state including view mode, camera position, and SOI filters
class MapState {
  final DisplayMode displayMode;
  final LatLng mapCenter;
  final double mapZoom;
  final Set<String> hiddenSOIs;       // SOI names that are HIDDEN (not shown)
  final Set<String> visibleBoxSOIs;   // SOI names with visible bounding boxes in waterfall
  final String? selectedMarkerId;

  const MapState({
    this.displayMode = DisplayMode.waterfall,
    this.mapCenter = const LatLng(39.7275, -104.7303), // Default: Aurora, CO
    this.mapZoom = 13.0,
    this.hiddenSOIs = const {},       // Empty = all visible
    this.visibleBoxSOIs = const {},
    this.selectedMarkerId,
  });

  MapState copyWith({
    DisplayMode? displayMode,
    LatLng? mapCenter,
    double? mapZoom,
    Set<String>? hiddenSOIs,
    Set<String>? visibleBoxSOIs,
    String? selectedMarkerId,
  }) {
    return MapState(
      displayMode: displayMode ?? this.displayMode,
      mapCenter: mapCenter ?? this.mapCenter,
      mapZoom: mapZoom ?? this.mapZoom,
      hiddenSOIs: hiddenSOIs ?? this.hiddenSOIs,
      visibleBoxSOIs: visibleBoxSOIs ?? this.visibleBoxSOIs,
      selectedMarkerId: selectedMarkerId,
    );
  }
}

/// Map state notifier - manages display mode and filtering
class MapStateNotifier extends StateNotifier<MapState> {
  static const _prefsKeyHiddenSOIs = 'g20_hidden_sois';
  static const _prefsKeyMapLat = 'g20_map_lat';
  static const _prefsKeyMapLon = 'g20_map_lon';
  static const _prefsKeyMapZoom = 'g20_map_zoom';

  MapStateNotifier() : super(const MapState()) {
    _loadPersistedState();
  }

  /// Load persisted state from SharedPreferences
  Future<void> _loadPersistedState() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      
      // Load hidden SOIs
      final hiddenList = prefs.getStringList(_prefsKeyHiddenSOIs);
      final hidden = hiddenList != null ? Set<String>.from(hiddenList) : <String>{};
      
      // Load map position
      final lat = prefs.getDouble(_prefsKeyMapLat) ?? 39.7275;
      final lon = prefs.getDouble(_prefsKeyMapLon) ?? -104.7303;
      final zoom = prefs.getDouble(_prefsKeyMapZoom) ?? 13.0;
      
      state = state.copyWith(
        hiddenSOIs: hidden,
        mapCenter: LatLng(lat, lon),
        mapZoom: zoom,
      );
    } catch (e) {
      // Silently ignore persistence errors
    }
  }

  /// Save hidden SOIs to SharedPreferences
  Future<void> _saveHiddenSOIs() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      await prefs.setStringList(_prefsKeyHiddenSOIs, state.hiddenSOIs.toList());
    } catch (e) {
      // Silently ignore persistence errors
    }
  }

  /// Save map position to SharedPreferences
  Future<void> _saveMapPosition() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      await prefs.setDouble(_prefsKeyMapLat, state.mapCenter.latitude);
      await prefs.setDouble(_prefsKeyMapLon, state.mapCenter.longitude);
      await prefs.setDouble(_prefsKeyMapZoom, state.mapZoom);
    } catch (e) {
      // Silently ignore persistence errors
    }
  }

  /// Toggle between waterfall and map view
  void toggleDisplayMode() {
    state = state.copyWith(
      displayMode: state.displayMode == DisplayMode.waterfall 
          ? DisplayMode.map 
          : DisplayMode.waterfall,
    );
  }

  /// Set display mode explicitly
  void setDisplayMode(DisplayMode mode) {
    state = state.copyWith(displayMode: mode);
  }

  /// Update map camera position
  void updateMapPosition(LatLng center, double zoom) {
    state = state.copyWith(mapCenter: center, mapZoom: zoom);
    _saveMapPosition();  // Persist
  }

  /// Toggle visibility of a specific SOI - adds/removes from hidden set
  void toggleSOIVisibility(String soiName) {
    final hidden = Set<String>.from(state.hiddenSOIs);
    if (hidden.contains(soiName)) {
      hidden.remove(soiName);  // Was hidden, now show it
    } else {
      hidden.add(soiName);     // Was visible, now hide it
    }
    state = state.copyWith(hiddenSOIs: hidden);
    _saveHiddenSOIs();  // Persist
  }

  /// Toggle visibility of bounding box for SOI in waterfall
  void toggleBoxVisibility(String soiName) {
    final current = Set<String>.from(state.visibleBoxSOIs);
    if (current.contains(soiName)) {
      current.remove(soiName);
    } else {
      current.add(soiName);
    }
    state = state.copyWith(visibleBoxSOIs: current);
  }

  /// Show all markers (clear hidden set)
  void showAll() {
    state = state.copyWith(hiddenSOIs: const {});
  }

  /// Hide all markers with given names
  void hideAll(Set<String> allSOIs) {
    state = state.copyWith(hiddenSOIs: allSOIs);
  }

  /// Check if SOI should be visible (not in hidden set)
  bool isSOIVisible(String soiName) {
    return !state.hiddenSOIs.contains(soiName);
  }

  /// Check if bounding box should be visible
  bool isBoxVisible(String soiName) {
    return state.visibleBoxSOIs.contains(soiName);
  }

  /// Select a marker on the map
  void selectMarker(String? markerId) {
    state = state.copyWith(selectedMarkerId: markerId);
  }

  /// Zoom map to fit all detections in view
  void zoomToFitAllDetections(List<Detection> detections) {
    if (detections.isEmpty) return;

    // Find bounds
    var minLat = detections.first.latitude;
    var maxLat = detections.first.latitude;
    var minLon = detections.first.longitude;
    var maxLon = detections.first.longitude;

    for (final det in detections) {
      if (det.latitude < minLat) minLat = det.latitude;
      if (det.latitude > maxLat) maxLat = det.latitude;
      if (det.longitude < minLon) minLon = det.longitude;
      if (det.longitude > maxLon) maxLon = det.longitude;
    }

    // Add padding (10% of span)
    final latSpan = maxLat - minLat;
    final lonSpan = maxLon - minLon;
    final padding = math.max(latSpan, lonSpan) * 0.1;
    
    minLat -= padding;
    maxLat += padding;
    minLon -= padding;
    maxLon += padding;

    // Calculate center
    final centerLat = (minLat + maxLat) / 2;
    final centerLon = (minLon + maxLon) / 2;

    // Calculate zoom level based on span
    // Rough approximation: zoom = 15 - log2(span * 100)
    final maxSpan = math.max(maxLat - minLat, maxLon - minLon);
    double zoom;
    if (maxSpan <= 0.001) {
      zoom = 18.0;  // Very close
    } else if (maxSpan <= 0.01) {
      zoom = 15.0;
    } else if (maxSpan <= 0.1) {
      zoom = 12.0;
    } else if (maxSpan <= 1.0) {
      zoom = 9.0;
    } else {
      zoom = 6.0;  // Very spread out
    }

    state = state.copyWith(
      mapCenter: LatLng(centerLat, centerLon),
      mapZoom: zoom,
    );
    _saveMapPosition();
  }
}

/// Provider for map state
final mapStateProvider = StateNotifierProvider<MapStateNotifier, MapState>((ref) {
  return MapStateNotifier();
});

/// Provider for current display mode
final displayModeProvider = Provider<DisplayMode>((ref) {
  return ref.watch(mapStateProvider).displayMode;
});

/// Provider to check if a specific SOI is visible (not hidden)
final soiVisibilityProvider = Provider.family<bool, String>((ref, soiName) {
  final mapState = ref.watch(mapStateProvider);
  return !mapState.hiddenSOIs.contains(soiName);  // Visible if NOT in hidden set
});

/// Fixed color palette - NO REPEATS
/// Each known class gets a specific color, UNKs get a generated unique color
const Map<String, Color> _soiColorPalette = {
  // Threat signals - warm colors
  'creamy_chicken': Color(0xFFFF5722),  // Deep Orange
  
  // Known protocols - distinct colors
  'LTE_UPLINK': Color(0xFF2196F3),      // Blue
  'WIFI_24': Color(0xFF4CAF50),         // Green
  'BLUETOOTH': Color(0xFF9C27B0),       // Purple
  'ZIGBEE': Color(0xFF00BCD4),          // Cyan
  'LORA': Color(0xFFFFEB3B),            // Yellow
  'GPS_L1': Color(0xFF795548),          // Brown
  'ADSB': Color(0xFF607D8B),            // Blue Grey
  'DMR': Color(0xFFE91E63),             // Pink
  'P25': Color(0xFF673AB7),             // Deep Purple
  'TETRA': Color(0xFF009688),           // Teal
  'DECT': Color(0xFFFF9800),            // Orange
  'ISM_433': Color(0xFF8BC34A),         // Light Green
  'ISM_868': Color(0xFF03A9F4),         // Light Blue
  'ISM_915': Color(0xFFCDDC39),         // Lime
};

/// Golden ratio for generating visually distinct hues
const double _goldenRatio = 0.618033988749895;

/// Track used hues to avoid collisions for dynamic UNK signals
final Set<int> _usedHues = {};

/// Utility to generate consistent color for SOI name
/// NO COLOR REPEATS - each class gets a unique color
Color getSOIColor(String soiName) {
  final lowerName = soiName.toLowerCase();
  
  // Check known classes first (exact match on lowercase)
  for (final entry in _soiColorPalette.entries) {
    if (entry.key.toLowerCase() == lowerName) {
      return entry.value;
    }
  }
  
  // Check for partial matches (creamy_chicken variants)
  if (lowerName.contains('creamy') || lowerName.contains('chicken')) {
    return _soiColorPalette['creamy_chicken']!;
  }
  
  // For UNK signals (unk_DTG_freq format) - generate unique color based on hash
  // Use golden ratio to spread hues evenly and avoid collisions
  final hash = soiName.hashCode.abs();
  
  // Start with hash-based hue, then adjust if collision
  var hue = (hash * _goldenRatio * 360).round() % 360;
  
  // Avoid too-similar hues (within 15 degrees of used hues)
  int attempts = 0;
  while (_usedHues.any((used) => (hue - used).abs() < 15 || (hue - used).abs() > 345) && attempts < 24) {
    hue = (hue + 15) % 360;
    attempts++;
  }
  
  // Mark this hue as used
  _usedHues.add(hue);
  
  // UNK signals get a distinct red-tinted color scheme (hue 0-30 range shifted)
  // to visually distinguish them from known signals
  if (lowerName.startsWith('unk')) {
    // Shift towards red/magenta range for UNKs
    final unkHue = ((hash % 60) + 330) % 360; // 330-390 (330-360, 0-30) = red/magenta
    return HSLColor.fromAHSL(1.0, unkHue.toDouble(), 0.75, 0.50).toColor();
  }
  
  // Other unknown signals get full spectrum
  return HSLColor.fromAHSL(1.0, hue.toDouble(), 0.70, 0.45).toColor();
}
