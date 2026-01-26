/// G20 Logger - Unified logging utility with levels and filtering
/// 
/// Usage:
///   G20Log.d('VideoStream', 'Connected to backend');  // Debug
///   G20Log.i('Scanner', 'Loaded mission');            // Info  
///   G20Log.w('RX', 'Connection unstable');            // Warning
///   G20Log.e('Inference', 'Model failed to load');    // Error
///   G20Log.perf('Waterfall', 'Frame: 7ms');           // Performance (disabled by default)
/// 
/// Configuration:
///   G20Log.enablePerf = true;  // Enable perf logging
///   G20Log.minLevel = LogLevel.info;  // Hide debug messages
///   G20Log.enabledModules = {'VideoStream', 'Scanner'};  // Filter modules
library;

import 'package:flutter/foundation.dart';

/// Log levels in order of severity
enum LogLevel {
  debug,   // Verbose debugging info
  info,    // Normal operation events
  perf,    // Performance metrics (special handling)
  warn,    // Warnings that don't stop operation
  error,   // Errors that need attention
}

/// Centralized logging for G20 app
class G20Log {
  G20Log._();  // Prevent instantiation
  
  // ============================================
  // CONFIGURATION - Set these at app startup
  // ============================================
  
  /// Minimum log level to display (messages below this are hidden)
  static LogLevel minLevel = kDebugMode ? LogLevel.debug : LogLevel.info;
  
  /// Enable performance logging (very verbose, disabled by default)
  static bool enablePerf = false;
  
  /// If non-empty, only show logs from these modules
  /// Empty set = show all modules
  static Set<String> enabledModules = {};
  
  /// If true, include timestamps in log output
  static bool showTimestamps = false;
  
  /// Performance log interval - only print every N calls
  /// Set to 1 for every call, 30 for ~once per second at 30fps
  static int perfLogInterval = 30;
  
  // Track perf call counts per tag
  static final Map<String, int> _perfCounts = {};
  
  // ============================================
  // LOGGING METHODS
  // ============================================
  
  /// Debug level - verbose info for development
  static void d(String module, String message) {
    _log(LogLevel.debug, module, message);
  }
  
  /// Info level - normal operation events
  static void i(String module, String message) {
    _log(LogLevel.info, module, message);
  }
  
  /// Warning level - non-fatal issues
  static void w(String module, String message) {
    _log(LogLevel.warn, module, message);
  }
  
  /// Error level - needs attention
  static void e(String module, String message) {
    _log(LogLevel.error, module, message);
  }
  
  /// Performance logging - throttled by perfLogInterval
  /// Returns true if the log was actually printed (for conditional work)
  static bool perf(String module, String message) {
    if (!enablePerf) return false;
    
    // Increment and check interval
    _perfCounts[module] = (_perfCounts[module] ?? 0) + 1;
    if (_perfCounts[module]! % perfLogInterval != 0) {
      return false;  // Skip this one
    }
    
    _log(LogLevel.perf, module, message);
    return true;
  }
  
  /// Reset perf counters (call on significant state changes)
  static void resetPerfCounters() {
    _perfCounts.clear();
  }
  
  // ============================================
  // INTERNAL
  // ============================================
  
  static void _log(LogLevel level, String module, String message) {
    // Check minimum level
    if (level.index < minLevel.index && level != LogLevel.perf) {
      return;
    }
    
    // Check module filter
    if (enabledModules.isNotEmpty && !enabledModules.contains(module)) {
      return;
    }
    
    // Build log string
    final buffer = StringBuffer();
    
    // Timestamp (optional)
    if (showTimestamps) {
      final now = DateTime.now();
      buffer.write('${now.hour.toString().padLeft(2, '0')}:'
          '${now.minute.toString().padLeft(2, '0')}:'
          '${now.second.toString().padLeft(2, '0')}.'
          '${now.millisecond.toString().padLeft(3, '0')} ');
    }
    
    // Level prefix
    buffer.write(_levelPrefix(level));
    
    // Module and message
    buffer.write('[$module] $message');
    
    debugPrint(buffer.toString());
  }
  
  static String _levelPrefix(LogLevel level) {
    switch (level) {
      case LogLevel.debug:
        return '';  // No prefix for debug
      case LogLevel.info:
        return '';  // No prefix for info
      case LogLevel.perf:
        return '[PERF] ';
      case LogLevel.warn:
        return '[WARN] ';
      case LogLevel.error:
        return '[ERR] ';
    }
  }
}

// ============================================
// CONVENIENCE EXTENSIONS
// ============================================

/// Extension to make logging even easier from any context
extension G20Logging on Object {
  void logDebug(String message) => G20Log.d(runtimeType.toString(), message);
  void logInfo(String message) => G20Log.i(runtimeType.toString(), message);
  void logWarn(String message) => G20Log.w(runtimeType.toString(), message);
  void logError(String message) => G20Log.e(runtimeType.toString(), message);
}

// ============================================
// PRESET CONFIGURATIONS
// ============================================

/// Apply production logging settings (minimal output)
void configureProductionLogging() {
  G20Log.minLevel = LogLevel.warn;
  G20Log.enablePerf = false;
  G20Log.enabledModules = {};
  G20Log.showTimestamps = false;
}

/// Apply development logging settings (verbose)
void configureDevelopmentLogging() {
  G20Log.minLevel = LogLevel.debug;
  G20Log.enablePerf = true;
  G20Log.perfLogInterval = 30;  // Every ~1 second at 30fps
  G20Log.enabledModules = {};  // All modules
  G20Log.showTimestamps = false;
}

/// Apply debug-specific module logging
void configureModuleDebugging(Set<String> modules) {
  G20Log.minLevel = LogLevel.debug;
  G20Log.enabledModules = modules;
  G20Log.showTimestamps = true;
}
