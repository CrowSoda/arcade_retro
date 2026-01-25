// lib/core/services/backend_launcher.dart
/// Auto-launch and manage the Python backend server on app startup.
/// 
/// SHUTDOWN HANDLING:
/// - Uses PID file tracking to detect/kill stale processes on startup
/// - Implements graceful shutdown with timeout fallback to force kill
/// - On Windows uses taskkill /F /T to kill process tree (including FFmpeg)

import 'dart:async';
import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:path/path.dart' as p;
import '../grpc/connection_manager.dart';

/// Backend process state
enum BackendState {
  stopped,
  starting,
  running,
  error,
}

/// Backend launcher state model
class BackendLauncherState {
  final BackendState state;
  final String? errorMessage;
  final int? pid;
  final int grpcPort;
  final int? wsPort; // WebSocket port (discovered from server stdout)
  final List<String> logs;

  const BackendLauncherState({
    this.state = BackendState.stopped,
    this.errorMessage,
    this.pid,
    this.grpcPort = 50051,
    this.wsPort,
    this.logs = const [],
  });

  bool get isRunning => state == BackendState.running;
  
  // Legacy getter for backward compatibility
  int get port => grpcPort;

  BackendLauncherState copyWith({
    BackendState? state,
    String? errorMessage,
    int? pid,
    int? grpcPort,
    int? wsPort,
    List<String>? logs,
  }) {
    return BackendLauncherState(
      state: state ?? this.state,
      errorMessage: errorMessage,
      pid: pid ?? this.pid,
      grpcPort: grpcPort ?? this.grpcPort,
      wsPort: wsPort ?? this.wsPort,
      logs: logs ?? this.logs,
    );
  }
}

/// Backend launcher - starts Python server automatically
class BackendLauncherNotifier extends StateNotifier<BackendLauncherState> {
  BackendLauncherNotifier(this._ref) : super(const BackendLauncherState());

  final Ref _ref;
  Process? _process;
  StreamSubscription? _stdoutSub;
  StreamSubscription? _stderrSub;
  bool _disposed = false;

  /// Path to the Python backend server
  String get _backendPath {
    // Try common locations
    final possiblePaths = [
      // Development: inside g20_demo
      p.join(Directory.current.path, 'backend'),
      // Bundled with app
      p.join(p.dirname(Platform.resolvedExecutable), 'backend'),
      // Relative to executable
      p.join(p.dirname(p.dirname(Platform.resolvedExecutable)), 'backend'),
    ];

    for (final path in possiblePaths) {
      final serverFile = File(p.join(path, 'server.py'));
      if (serverFile.existsSync()) {
        debugPrint('📁 Found backend at: $path');
        return path;
      }
    }

    // Fallback to current directory backend
    return p.join(Directory.current.path, 'backend');
  }

  /// Path to the PID file (used to track backend process for cleanup)
  String get _pidFilePath => p.join(_backendPath, '.backend.pid');

  /// Write the PID to a file for tracking
  Future<void> _writePidFile(int pid) async {
    try {
      final pidFile = File(_pidFilePath);
      await pidFile.writeAsString(pid.toString());
      debugPrint('📝 Wrote PID file: $pid');
    } catch (e) {
      debugPrint('⚠️ Failed to write PID file: $e');
    }
  }

  /// Delete the PID file
  Future<void> _deletePidFile() async {
    try {
      final pidFile = File(_pidFilePath);
      if (await pidFile.exists()) {
        await pidFile.delete();
        debugPrint('🗑️ Deleted PID file');
      }
    } catch (e) {
      debugPrint('⚠️ Failed to delete PID file: $e');
    }
  }

  /// Clean up any stale backend processes from previous runs
  Future<void> _cleanupStalePids() async {
    try {
      final pidFile = File(_pidFilePath);
      if (!await pidFile.exists()) return;

      final content = await pidFile.readAsString();
      final oldPid = int.tryParse(content.trim());
      
      if (oldPid == null) {
        await pidFile.delete();
        return;
      }

      debugPrint('🔍 Found stale PID file with PID: $oldPid');
      
      // Check if process is still running and kill it
      if (Platform.isWindows) {
        // On Windows, use taskkill to kill the process tree
        final result = await Process.run(
          'taskkill',
          ['/F', '/T', '/PID', oldPid.toString()],
        );
        if (result.exitCode == 0) {
          debugPrint('🧹 Killed stale backend process (PID: $oldPid)');
        } else {
          debugPrint('ℹ️ Stale process already gone (PID: $oldPid)');
        }
      } else {
        // On Unix, try SIGTERM then SIGKILL
        try {
          Process.killPid(oldPid, ProcessSignal.sigterm);
          await Future.delayed(const Duration(seconds: 2));
          Process.killPid(oldPid, ProcessSignal.sigkill);
          debugPrint('🧹 Killed stale backend process (PID: $oldPid)');
        } catch (e) {
          debugPrint('ℹ️ Stale process already gone (PID: $oldPid)');
        }
      }

      // Delete the stale PID file
      await pidFile.delete();
      debugPrint('🗑️ Cleaned up stale PID file');
      
      // Wait a moment for ports to be released
      await Future.delayed(const Duration(milliseconds: 500));
    } catch (e) {
      debugPrint('⚠️ Error cleaning up stale PIDs: $e');
    }
  }

  /// Start the backend server
  Future<bool> startBackend({int port = 50051}) async {
    if (_disposed) return false;
    if (state.state == BackendState.running) return true;
    if (state.state == BackendState.starting) return false;

    state = state.copyWith(
      state: BackendState.starting,
      errorMessage: null,
      grpcPort: port,
    );

    try {
      // Clean up any stale processes from previous runs first
      debugPrint('🧹 Checking for stale backend processes...');
      await _cleanupStalePids();
      
      // Find Python executable
      final pythonExe = await _findPython();
      if (pythonExe == null) {
        throw Exception('Python not found. Install Python 3.8+');
      }

      debugPrint('🐍 Found Python: $pythonExe');
      debugPrint('📁 Backend path: $_backendPath');

      // Build command - run server.py with both gRPC and WebSocket
      // Use --ws-port 0 for auto-discovery (OS picks free port, server prints it)
      final serverScript = p.join(_backendPath, 'server.py');
      final args = [serverScript, '--port', port.toString(), '--ws-port', '0'];

      // Start process from backend directory
      _process = await Process.start(
        pythonExe,
        args,
        workingDirectory: _backendPath,
      );

      final pid = _process!.pid;
      debugPrint('🚀 Backend started with PID: $pid');
      
      // Write PID file for tracking (used to cleanup stale processes on restart)
      await _writePidFile(pid);

      // Capture stdout
      _stdoutSub = _process!.stdout
          .transform(const SystemEncoding().decoder)
          .listen((data) {
        _addLog('[OUT] $data');
        debugPrint('🐍 $data');
        
        // Parse WS_PORT from server stdout (KISS auto-discovery)
        final wsPortMatch = RegExp(r'WS_PORT:(\d+)').firstMatch(data);
        if (wsPortMatch != null) {
          final discoveredPort = int.parse(wsPortMatch.group(1)!);
          debugPrint('🎯 Discovered WebSocket port: $discoveredPort');
          state = state.copyWith(wsPort: discoveredPort);
        }
        
        // Detect when server is ready (WebSocket or gRPC)
        if (data.contains('WebSocket server READY') || 
            data.contains('gRPC server started') || 
            data.contains('Serving on')) {
          state = state.copyWith(state: BackendState.running);
          _connectToBackend(port);
        }
      });

      // Capture stderr
      _stderrSub = _process!.stderr
          .transform(const SystemEncoding().decoder)
          .listen((data) {
        _addLog('[ERR] $data');
      });

      // Handle process exit
      _process!.exitCode.then((code) {
        if (!_disposed) {
          debugPrint('⚠️ Backend exited with code: $code');
          state = state.copyWith(
            state: BackendState.stopped,
            errorMessage: code != 0 ? 'Backend exited with code $code' : null,
            pid: null,
          );
        }
      });

      state = state.copyWith(pid: pid);

      // Wait a bit for server to start
      await Future.delayed(const Duration(seconds: 2));
      
      // If still starting, assume running
      if (state.state == BackendState.starting) {
        state = state.copyWith(state: BackendState.running);
        _connectToBackend(port);
      }

      return true;
    } catch (e) {
      debugPrint('❌ Failed to start backend: $e');
      state = state.copyWith(
        state: BackendState.error,
        errorMessage: e.toString(),
      );
      return false;
    }
  }

  /// Stop the backend server
  Future<void> stopBackend() async {
    _stdoutSub?.cancel();
    _stderrSub?.cancel();

    if (_process != null) {
      final pid = _process!.pid;
      debugPrint('🛑 Stopping backend (PID: $pid)');
      
      if (Platform.isWindows) {
        // Use taskkill /F /T to kill entire process tree (includes FFmpeg subprocesses)
        final result = await Process.run('taskkill', ['/F', '/T', '/PID', pid.toString()]);
        debugPrint('🛑 taskkill result: exit=${result.exitCode}');
      } else {
        _process!.kill(ProcessSignal.sigterm);
        try {
          await _process!.exitCode.timeout(const Duration(seconds: 3));
        } catch (_) {
          _process!.kill(ProcessSignal.sigkill);
        }
      }
      
      _process = null;
    }

    // Delete the PID file since process is stopped
    await _deletePidFile();

    state = state.copyWith(
      state: BackendState.stopped,
      pid: null,
    );
    
    debugPrint('✅ Backend stopped successfully');
  }

  /// Restart the backend
  Future<bool> restartBackend() async {
    await stopBackend();
    await Future.delayed(const Duration(milliseconds: 500));
    return startBackend(port: state.port);
  }

  void _connectToBackend(int port) {
    // Update connection config and connect
    final connectionManager = _ref.read(connectionManagerProvider.notifier);
    connectionManager.setConfig(ConnectionConfig(
      host: 'localhost',
      port: port,
    ));
    connectionManager.connect();
  }

  void _addLog(String message) {
    final logs = [...state.logs, message];
    // Keep last 100 lines
    if (logs.length > 100) {
      logs.removeRange(0, logs.length - 100);
    }
    state = state.copyWith(logs: logs);
  }

  Future<String?> _findPython() async {
    // Check common Python locations
    final candidates = Platform.isWindows
        ? ['python', 'python3', 'py']
        : ['python3', 'python'];

    for (final cmd in candidates) {
      try {
        final result = await Process.run(cmd, ['--version']);
        if (result.exitCode == 0) {
          return cmd;
        }
      } catch (_) {
        continue;
      }
    }
    return null;
  }

  @override
  void dispose() {
    _disposed = true;
    stopBackend();
    super.dispose();
  }
}

/// Provider for backend launcher
final backendLauncherProvider =
    StateNotifierProvider<BackendLauncherNotifier, BackendLauncherState>(
  (ref) => BackendLauncherNotifier(ref),
);

/// Auto-start backend on app initialization
Future<void> initializeBackend(WidgetRef ref) async {
  final launcher = ref.read(backendLauncherProvider.notifier);
  await launcher.startBackend();
}
