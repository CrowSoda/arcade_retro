// lib/core/services/backend_launcher.dart
/// Auto-launch and manage the Python backend server on app startup.
///
/// SHUTDOWN HANDLING:
/// - Uses PID file tracking to detect/kill stale processes on startup
/// - Implements graceful shutdown with timeout fallback to force kill
/// - On Windows uses taskkill /F /T to kill process tree (including FFmpeg)

import 'dart:async';
import 'dart:convert';
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
  final String? version; // Backend version (parsed from server stdout)
  final List<String> logs;

  const BackendLauncherState({
    this.state = BackendState.stopped,
    this.errorMessage,
    this.pid,
    this.grpcPort = 50051,
    this.wsPort,
    this.version,
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
    String? version,
    List<String>? logs,
  }) {
    return BackendLauncherState(
      state: state ?? this.state,
      errorMessage: errorMessage,
      pid: pid ?? this.pid,
      grpcPort: grpcPort ?? this.grpcPort,
      wsPort: wsPort ?? this.wsPort,
      version: version ?? this.version,
      logs: logs ?? this.logs,
    );
  }
}

/// Backend launcher - starts Python server automatically
class BackendLauncherNotifier extends StateNotifier<BackendLauncherState> {
  BackendLauncherNotifier(this._ref) : super(const BackendLauncherState());

  final Ref _ref;
  Process? _process;
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
        return path;
      }
    }

    // Fallback to current directory backend
    return p.join(Directory.current.path, 'backend');
  }

  /// Path to the PID file (used to track backend process for cleanup)
  String get _pidFilePath => p.join(_backendPath, '.backend.pid');

  /// Path to runtime/server.json (written by Python backend when ready)
  String get _serverInfoPath {
    // runtime/ is at project root level (parent of backend/)
    return p.join(p.dirname(_backendPath), 'runtime', 'server.json');
  }

  /// Read server info from runtime/server.json
  /// Returns null if file doesn't exist or is invalid
  Future<Map<String, dynamic>?> _readServerInfo() async {
    try {
      final file = File(_serverInfoPath);
      if (!await file.exists()) return null;
      final content = await file.readAsString();
      return jsonDecode(content) as Map<String, dynamic>;
    } catch (_) {
      return null;
    }
  }

  /// Poll for server.json until it appears or timeout
  Future<Map<String, dynamic>?> _waitForServerInfo({
    Duration timeout = const Duration(seconds: 30),
    Duration pollInterval = const Duration(milliseconds: 200),
  }) async {
    final stopwatch = Stopwatch()..start();
    while (stopwatch.elapsed < timeout) {
      final info = await _readServerInfo();
      if (info != null && info['ready'] == true) {
        return info;
      }
      await Future.delayed(pollInterval);
    }
    return null;
  }

  /// Delete server.json on startup (clean slate)
  Future<void> _deleteServerInfo() async {
    try {
      final file = File(_serverInfoPath);
      if (await file.exists()) {
        await file.delete();
      }
    } catch (_) {
      // Ignore errors
    }
  }

  /// Write the PID to a file for tracking
  Future<void> _writePidFile(int pid) async {
    try {
      final pidFile = File(_pidFilePath);
      await pidFile.writeAsString(pid.toString());
    } catch (_) {
      // Ignore write errors
    }
  }

  /// Delete the PID file
  Future<void> _deletePidFile() async {
    try {
      final pidFile = File(_pidFilePath);
      if (await pidFile.exists()) {
        await pidFile.delete();
      }
    } catch (_) {
      // Ignore delete errors
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

      // Check if process is still running and kill it
      if (Platform.isWindows) {
        // On Windows, use taskkill to kill the process tree
        await Process.run('taskkill', ['/F', '/T', '/PID', oldPid.toString()]);
      } else {
        // On Unix, try SIGTERM then SIGKILL
        try {
          Process.killPid(oldPid, ProcessSignal.sigterm);
          await Future.delayed(const Duration(seconds: 2));
          Process.killPid(oldPid, ProcessSignal.sigkill);
        } catch (_) {
          // Process already gone
        }
      }

      // Delete the stale PID file
      await pidFile.delete();

      // Wait a moment for ports to be released
      await Future.delayed(const Duration(milliseconds: 500));
    } catch (_) {
      // Ignore cleanup errors
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
      // Clean up any stale processes and server.json from previous runs
      await _cleanupStalePids();
      await _deleteServerInfo();

      // Find Python executable
      final pythonExe = await _findPython();
      if (pythonExe == null) {
        throw Exception('Python not found. Install Python 3.8+');
      }

      // Build command - run server.py with both gRPC and WebSocket
      // Use --ws-port 0 for auto-discovery (OS picks free port, writes to server.json)
      final serverScript = p.join(_backendPath, 'server.py');
      final args = [serverScript, '--port', port.toString(), '--ws-port', '0'];

      // Start process from backend directory
      _process = await Process.start(
        pythonExe,
        args,
        workingDirectory: _backendPath,
      );

      final pid = _process!.pid;

      // Write PID file for tracking (used to cleanup stale processes on restart)
      await _writePidFile(pid);

      // Capture stderr - PRINT TO CONSOLE for debugging (only errors/warnings)
      _stderrSub = _process!.stderr
          .transform(const SystemEncoding().decoder)
          .listen((data) {
        print('[Python ERR] $data');  // PRINT ERRORS SO WE CAN SEE THEM
        _addLog('[ERR] $data');
      });

      // Handle process exit
      _process!.exitCode.then((code) {
        if (!_disposed) {
          state = state.copyWith(
            state: BackendState.stopped,
            errorMessage: code != 0 ? 'Backend exited with code $code' : null,
            pid: null,
          );
        }
      });

      state = state.copyWith(pid: pid);

      // Wait for server.json to appear (replaces stdout parsing)
      // Server writes this file when WebSocket server is ready
      debugPrint('[Flutter] Waiting for server.json...');
      final serverInfo = await _waitForServerInfo(
        timeout: const Duration(seconds: 30),
      );

      if (serverInfo == null) {
        throw Exception('Backend failed to start (server.json not created within 30s)');
      }

      // Read ports from server.json
      final wsPort = serverInfo['ws_port'] as int?;
      final grpcPortFromFile = serverInfo['grpc_port'] as int? ?? port;

      debugPrint('[Flutter] Server ready: WS=$wsPort, gRPC=$grpcPortFromFile');

      state = state.copyWith(
        state: BackendState.running,
        wsPort: wsPort,
        grpcPort: grpcPortFromFile,
      );

      _connectToBackend(grpcPortFromFile);

      return true;
    } catch (e) {
      state = state.copyWith(
        state: BackendState.error,
        errorMessage: e.toString(),
      );
      return false;
    }
  }

  /// Stop the backend server
  Future<void> stopBackend() async {
    _stderrSub?.cancel();

    if (_process != null) {
      final pid = _process!.pid;

      if (Platform.isWindows) {
        // Use taskkill /F /T to kill entire process tree (includes FFmpeg subprocesses)
        await Process.run('taskkill', ['/F', '/T', '/PID', pid.toString()]);
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
