/// lib/features/config/models/mission_config.dart
/// Mission configuration model for signal detection missions

import 'dart:io';

/// Configuration for a signal detection mission
class MissionConfig {
  /// Mission name
  final String name;

  /// Mission description
  final String description;

  /// Center frequency in MHz
  final double centerFreqMhz;

  /// Bandwidth in MHz
  final double bandwidthMhz;

  /// Dwell time in seconds
  final double dwellTimeSec;

  /// List of signal names to detect in this mission
  final List<String> effectiveSignals;

  /// When this mission was created
  final DateTime created;

  /// When this mission was last modified
  final DateTime modified;

  /// Path to the mission file (null if not saved)
  final String? filePath;

  const MissionConfig({
    required this.name,
    this.description = '',
    this.centerFreqMhz = 1000.0,
    this.bandwidthMhz = 20.0,
    this.dwellTimeSec = 1.0,
    this.effectiveSignals = const [],
    required this.created,
    required this.modified,
    this.filePath,
  });

  /// Create a default configuration
  static MissionConfig defaultConfig() {
    final now = DateTime.now();
    return MissionConfig(
      name: 'New Mission',
      description: 'A new signal detection mission',
      centerFreqMhz: 1000.0,
      bandwidthMhz: 20.0,
      dwellTimeSec: 1.0,
      effectiveSignals: [],
      created: now,
      modified: now,
    );
  }

  /// Load a mission from a YAML file
  static Future<MissionConfig> loadFromFile(String path) async {
    final file = File(path);
    final content = await file.readAsString();

    // Simple YAML parsing (for basic fields)
    final lines = content.split('\n');
    String name = 'Unknown';
    String description = '';
    double centerFreqMhz = 1000.0;
    double bandwidthMhz = 20.0;
    double dwellTimeSec = 1.0;
    List<String> signals = [];

    for (final line in lines) {
      final trimmed = line.trim();
      if (trimmed.startsWith('name:')) {
        name = trimmed.substring(5).trim();
      } else if (trimmed.startsWith('description:')) {
        description = trimmed.substring(12).trim();
      } else if (trimmed.startsWith('center_freq_mhz:')) {
        centerFreqMhz = double.tryParse(trimmed.substring(16).trim()) ?? 1000.0;
      } else if (trimmed.startsWith('bandwidth_mhz:')) {
        bandwidthMhz = double.tryParse(trimmed.substring(14).trim()) ?? 20.0;
      } else if (trimmed.startsWith('dwell_time_sec:')) {
        dwellTimeSec = double.tryParse(trimmed.substring(15).trim()) ?? 1.0;
      } else if (trimmed.startsWith('- ') && !trimmed.contains(':')) {
        // List item (signal name)
        signals.add(trimmed.substring(2).trim());
      }
    }

    final stat = await file.stat();
    return MissionConfig(
      name: name,
      description: description,
      centerFreqMhz: centerFreqMhz,
      bandwidthMhz: bandwidthMhz,
      dwellTimeSec: dwellTimeSec,
      effectiveSignals: signals,
      created: stat.changed,
      modified: stat.modified,
      filePath: path,
    );
  }

  /// Save this mission to a YAML file
  Future<void> saveToFile(String path) async {
    final buffer = StringBuffer();
    buffer.writeln('# Mission Configuration');
    buffer.writeln('name: $name');
    buffer.writeln('description: $description');
    buffer.writeln('center_freq_mhz: $centerFreqMhz');
    buffer.writeln('bandwidth_mhz: $bandwidthMhz');
    buffer.writeln('dwell_time_sec: $dwellTimeSec');
    buffer.writeln('signals:');
    for (final signal in effectiveSignals) {
      buffer.writeln('  - $signal');
    }

    final file = File(path);
    await file.writeAsString(buffer.toString());
  }

  /// Create a copy with updated fields
  MissionConfig copyWith({
    String? name,
    String? description,
    double? centerFreqMhz,
    double? bandwidthMhz,
    double? dwellTimeSec,
    List<String>? effectiveSignals,
    DateTime? created,
    DateTime? modified,
    String? filePath,
  }) {
    return MissionConfig(
      name: name ?? this.name,
      description: description ?? this.description,
      centerFreqMhz: centerFreqMhz ?? this.centerFreqMhz,
      bandwidthMhz: bandwidthMhz ?? this.bandwidthMhz,
      dwellTimeSec: dwellTimeSec ?? this.dwellTimeSec,
      effectiveSignals: effectiveSignals ?? this.effectiveSignals,
      created: created ?? this.created,
      modified: modified ?? this.modified,
      filePath: filePath ?? this.filePath,
    );
  }

  @override
  String toString() => 'MissionConfig(name: $name, freq: $centerFreqMhz MHz)';
}
