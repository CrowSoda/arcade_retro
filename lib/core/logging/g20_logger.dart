/// G20 Logger - Production-grade structured logging system
/// 
/// Features:
/// - Structured logging with metadata
/// - Session/correlation ID tracking
/// - Stack trace capture for errors
/// - JSON output for log aggregation
/// - Performance metrics collection
/// - Remote logging support (extensible)
/// - Ring buffer for recent logs
/// 
/// Usage:
///   final log = G20Logger.of('VideoStream');
///   log.info('Connected', data: {'port': 8765});
///   log.error('Connection failed', error: e, stackTrace: st);
library;

import 'dart:async';
import 'dart:collection';
import 'dart:convert';
import 'package:flutter/foundation.dart';

// ============================================
// LOG LEVELS
// ============================================

enum LogLevel {
  trace(0, 'TRACE'),
  debug(1, 'DEBUG'),
  info(2, 'INFO'),
  warn(3, 'WARN'),
  error(4, 'ERROR'),
  fatal(5, 'FATAL');

  const LogLevel(this.value, this.label);
  final int value;
  final String label;
}

// ============================================
// LOG ENTRY
// ============================================

/// Immutable log entry with all metadata
class LogEntry {
  final DateTime timestamp;
  final LogLevel level;
  final String module;
  final String message;
  final Map<String, dynamic>? data;
  final Object? error;
  final StackTrace? stackTrace;
  final String? sessionId;
  final String? correlationId;
  final Duration? duration; // For perf logs

  const LogEntry({
    required this.timestamp,
    required this.level,
    required this.module,
    required this.message,
    this.data,
    this.error,
    this.stackTrace,
    this.sessionId,
    this.correlationId,
    this.duration,
  });

  /// Convert to JSON for structured logging
  Map<String, dynamic> toJson() => {
    'ts': timestamp.toIso8601String(),
    'level': level.label,
    'module': module,
    'msg': message,
    if (data != null) 'data': data,
    if (error != null) 'error': error.toString(),
    if (stackTrace != null) 'stack': stackTrace.toString().split('\n').take(10).toList(),
    if (sessionId != null) 'session': sessionId,
    if (correlationId != null) 'correlation': correlationId,
    if (duration != null) 'duration_ms': duration!.inMilliseconds,
  };

  /// Human-readable format
  String toConsole({bool includeTimestamp = false}) {
    final buffer = StringBuffer();
    
    if (includeTimestamp) {
      buffer.write('${timestamp.hour.toString().padLeft(2, '0')}:'
          '${timestamp.minute.toString().padLeft(2, '0')}:'
          '${timestamp.second.toString().padLeft(2, '0')}.'
          '${timestamp.millisecond.toString().padLeft(3, '0')} ');
    }
    
    if (level.value >= LogLevel.warn.value) {
      buffer.write('[${level.label}] ');
    }
    
    buffer.write('[$module] $message');
    
    if (data != null && data!.isNotEmpty) {
      buffer.write(' ${_formatData(data!)}');
    }
    
    if (duration != null) {
      buffer.write(' (${duration!.inMilliseconds}ms)');
    }
    
    return buffer.toString();
  }

  String _formatData(Map<String, dynamic> data) {
    return data.entries.map((e) => '${e.key}=${e.value}').join(', ');
  }
}

// ============================================
// LOG OUTPUT HANDLER
// ============================================

/// Interface for log output destinations
abstract class LogHandler {
  void handle(LogEntry entry);
  Future<void> flush();
}

/// Default console output handler
class ConsoleLogHandler implements LogHandler {
  final bool useColors;
  final bool showTimestamps;

  ConsoleLogHandler({this.useColors = true, this.showTimestamps = false});

  @override
  void handle(LogEntry entry) {
    final msg = entry.toConsole(includeTimestamp: showTimestamps);
    
    if (useColors && !kReleaseMode) {
      debugPrint(_colorize(msg, entry.level));
    } else {
      debugPrint(msg);
    }
    
    // Print stack trace for errors
    if (entry.stackTrace != null && entry.level.value >= LogLevel.error.value) {
      debugPrint(entry.stackTrace.toString());
    }
  }

  String _colorize(String msg, LogLevel level) {
    // ANSI colors (work in most terminals)
    return switch (level) {
      LogLevel.trace => '\x1B[90m$msg\x1B[0m',  // Gray
      LogLevel.debug => msg,                      // Default
      LogLevel.info => '\x1B[36m$msg\x1B[0m',   // Cyan
      LogLevel.warn => '\x1B[33m$msg\x1B[0m',   // Yellow
      LogLevel.error => '\x1B[31m$msg\x1B[0m',  // Red
      LogLevel.fatal => '\x1B[41;37m$msg\x1B[0m', // White on red bg
    };
  }

  @override
  Future<void> flush() async {}
}

/// JSON file output handler (for log aggregation)
class JsonLogHandler implements LogHandler {
  final List<Map<String, dynamic>> _buffer = [];
  final int maxBufferSize;

  JsonLogHandler({this.maxBufferSize = 100});

  @override
  void handle(LogEntry entry) {
    _buffer.add(entry.toJson());
    if (_buffer.length > maxBufferSize) {
      _buffer.removeAt(0);
    }
  }

  @override
  Future<void> flush() async {
    // Could write to file or send to server here
  }

  String getJsonLogs() => jsonEncode(_buffer);
}

// ============================================
// RING BUFFER FOR RECENT LOGS
// ============================================

class LogRingBuffer {
  final Queue<LogEntry> _buffer;
  final int maxSize;

  LogRingBuffer({this.maxSize = 500}) : _buffer = Queue<LogEntry>();

  void add(LogEntry entry) {
    if (_buffer.length >= maxSize) {
      _buffer.removeFirst();
    }
    _buffer.add(entry);
  }

  List<LogEntry> getRecent({int count = 50, LogLevel? minLevel, String? module}) {
    var entries = _buffer.toList().reversed;
    
    if (minLevel != null) {
      entries = entries.where((e) => e.level.value >= minLevel.value);
    }
    
    if (module != null) {
      entries = entries.where((e) => e.module == module);
    }
    
    return entries.take(count).toList();
  }

  List<LogEntry> getErrors() => getRecent(minLevel: LogLevel.error);
}

// ============================================
// MAIN LOGGER CLASS
// ============================================

/// Central logging system
class G20LogManager {
  static G20LogManager? _instance;
  static G20LogManager get instance => _instance ??= G20LogManager._();

  G20LogManager._();

  // Configuration
  LogLevel minLevel = kDebugMode ? LogLevel.debug : LogLevel.info;
  bool enablePerf = false;
  int perfThrottle = 30;
  Set<String> enabledModules = {};
  String? sessionId;

  // Output handlers
  final List<LogHandler> _handlers = [ConsoleLogHandler()];
  
  // Ring buffer for recent logs
  final LogRingBuffer _buffer = LogRingBuffer(maxSize: 500);

  // Perf throttle counters
  final Map<String, int> _perfCounters = {};

  // Metrics aggregation
  final Map<String, List<int>> _perfMetrics = {};

  void addHandler(LogHandler handler) => _handlers.add(handler);
  void removeHandler(LogHandler handler) => _handlers.remove(handler);
  void clearHandlers() => _handlers.clear();

  /// Get recent log entries
  List<LogEntry> getRecentLogs({int count = 50, LogLevel? minLevel, String? module}) {
    return _buffer.getRecent(count: count, minLevel: minLevel, module: module);
  }

  /// Get recent errors for crash reporting
  List<LogEntry> getRecentErrors() => _buffer.getErrors();

  /// Get aggregated perf metrics
  Map<String, Map<String, num>> getPerfMetrics() {
    final result = <String, Map<String, num>>{};
    for (final entry in _perfMetrics.entries) {
      if (entry.value.isNotEmpty) {
        final sorted = List<int>.from(entry.value)..sort();
        result[entry.key] = {
          'count': entry.value.length,
          'min': sorted.first,
          'max': sorted.last,
          'avg': entry.value.reduce((a, b) => a + b) / entry.value.length,
          'p50': sorted[sorted.length ~/ 2],
          'p95': sorted[(sorted.length * 0.95).floor()],
        };
      }
    }
    return result;
  }

  void clearPerfMetrics() => _perfMetrics.clear();

  /// Log a message
  void log(
    LogLevel level,
    String module,
    String message, {
    Map<String, dynamic>? data,
    Object? error,
    StackTrace? stackTrace,
    String? correlationId,
    Duration? duration,
  }) {
    // Check level filter
    if (level.value < minLevel.value) return;
    
    // Check module filter
    if (enabledModules.isNotEmpty && !enabledModules.contains(module)) return;

    final entry = LogEntry(
      timestamp: DateTime.now(),
      level: level,
      module: module,
      message: message,
      data: data,
      error: error,
      stackTrace: stackTrace ?? (error != null ? StackTrace.current : null),
      sessionId: sessionId,
      correlationId: correlationId,
      duration: duration,
    );

    // Store in ring buffer
    _buffer.add(entry);

    // Send to all handlers
    for (final handler in _handlers) {
      try {
        handler.handle(entry);
      } catch (e) {
        // Don't let logging errors crash the app
      }
    }
  }

  /// Log performance metric with throttling
  bool logPerf(String module, String operation, Duration duration) {
    if (!enablePerf) return false;

    // Store metric for aggregation
    final key = '$module.$operation';
    _perfMetrics.putIfAbsent(key, () => []);
    _perfMetrics[key]!.add(duration.inMilliseconds);
    if (_perfMetrics[key]!.length > 1000) {
      _perfMetrics[key]!.removeAt(0);
    }

    // Throttle console output
    _perfCounters[key] = (_perfCounters[key] ?? 0) + 1;
    if (_perfCounters[key]! % perfThrottle != 0) return false;

    log(LogLevel.info, module, operation, duration: duration);
    return true;
  }

  /// Generate new session ID
  void startSession() {
    sessionId = DateTime.now().millisecondsSinceEpoch.toRadixString(36);
  }

  /// Flush all handlers
  Future<void> flush() async {
    for (final handler in _handlers) {
      await handler.flush();
    }
  }
}

// ============================================
// MODULE LOGGER (Convenience wrapper)
// ============================================

/// Logger instance for a specific module
class G20Logger {
  final String module;

  const G20Logger._(this.module);

  static G20Logger of(String module) => G20Logger._(module);

  G20LogManager get _manager => G20LogManager.instance;

  void trace(String msg, {Map<String, dynamic>? data}) =>
      _manager.log(LogLevel.trace, module, msg, data: data);

  void debug(String msg, {Map<String, dynamic>? data}) =>
      _manager.log(LogLevel.debug, module, msg, data: data);

  void info(String msg, {Map<String, dynamic>? data}) =>
      _manager.log(LogLevel.info, module, msg, data: data);

  void warn(String msg, {Map<String, dynamic>? data, Object? error}) =>
      _manager.log(LogLevel.warn, module, msg, data: data, error: error);

  void error(String msg, {Map<String, dynamic>? data, Object? error, StackTrace? stackTrace}) =>
      _manager.log(LogLevel.error, module, msg, data: data, error: error, stackTrace: stackTrace);

  void fatal(String msg, {Map<String, dynamic>? data, Object? error, StackTrace? stackTrace}) =>
      _manager.log(LogLevel.fatal, module, msg, data: data, error: error, stackTrace: stackTrace);

  /// Log performance with auto-throttling
  bool perf(String operation, Duration duration) =>
      _manager.logPerf(module, operation, duration);

  /// Measure execution time of async operation
  Future<T> measure<T>(String operation, Future<T> Function() fn) async {
    final stopwatch = Stopwatch()..start();
    try {
      return await fn();
    } finally {
      stopwatch.stop();
      perf(operation, stopwatch.elapsed);
    }
  }

  /// Measure execution time of sync operation
  T measureSync<T>(String operation, T Function() fn) {
    final stopwatch = Stopwatch()..start();
    try {
      return fn();
    } finally {
      stopwatch.stop();
      perf(operation, stopwatch.elapsed);
    }
  }
}

// ============================================
// PRESET CONFIGURATIONS
// ============================================

void configureLoggingProduction() {
  final mgr = G20LogManager.instance;
  mgr.minLevel = LogLevel.warn;
  mgr.enablePerf = false;
  mgr.startSession();
}

void configureLoggingDevelopment() {
  final mgr = G20LogManager.instance;
  mgr.minLevel = LogLevel.debug;
  mgr.enablePerf = false;  // Still disabled to reduce noise
  mgr.perfThrottle = 30;
  mgr.startSession();
}

void configureLoggingVerbose() {
  final mgr = G20LogManager.instance;
  mgr.minLevel = LogLevel.trace;
  mgr.enablePerf = true;
  mgr.perfThrottle = 1;  // Every call
  mgr.startSession();
}
