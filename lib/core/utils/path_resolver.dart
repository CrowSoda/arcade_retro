import 'dart:io';
import 'package:path/path.dart' as path;

/// Path resolution utilities for cross-platform data file access
/// Handles both development and deployed scenarios

/// Resolve a data file path, checking multiple candidate locations
/// Returns the first existing path, or the most likely path if none exist
String resolveDataPath(String relativePath) {
  // Normalize the relative path
  final normalized = relativePath.replaceAll('\\', '/');
  
  // Get the executable directory (for deployed apps)
  final execDir = path.dirname(Platform.resolvedExecutable);
  
  // Get current working directory (for development)
  final cwd = Directory.current.path;
  
  // Candidate paths in priority order
  final candidates = [
    // Development: running from g20_demo/
    path.join(cwd, normalized),
    // Development: running from parent directory
    path.join(cwd, 'g20_demo', normalized),
    // Deployed: relative to executable
    path.join(execDir, normalized),
    // Deployed: in data folder next to executable
    path.join(execDir, 'data', path.basename(normalized)),
  ];
  
  // Return first existing path
  for (final candidate in candidates) {
    final normalizedCandidate = candidate.replaceAll('\\', '/');
    if (File(normalizedCandidate).existsSync()) {
      return normalizedCandidate;
    }
    // Also check if it's a directory
    if (Directory(normalizedCandidate).existsSync()) {
      return normalizedCandidate;
    }
  }
  
  // Return the most likely dev path if none exist
  return path.join(cwd, normalized).replaceAll('\\', '/');
}

/// Resolve the IQ data file path
String resolveIqDataPath(String filename) {
  return resolveDataPath('data/$filename');
}

/// Resolve the map tiles path
String resolveMapPath(String filename) {
  return resolveDataPath('data/map/$filename');
}

/// Check if a data file exists
bool dataFileExists(String relativePath) {
  final resolved = resolveDataPath(relativePath);
  return File(resolved).existsSync();
}
