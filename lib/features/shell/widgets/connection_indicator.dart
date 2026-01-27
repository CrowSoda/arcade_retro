/// Connection Indicator Widget - Backend connection status dot
///
/// Shows green/yellow/red dot based on backend and stream connection state
library;

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../../../core/config/theme.dart';
import '../../../core/services/backend_launcher.dart';
import '../../live_detection/providers/video_stream_provider.dart';

/// Connection status indicator - wired to backend and video stream state
class ConnectionIndicator extends ConsumerWidget {
  const ConnectionIndicator({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    // Check both backend launcher and video stream connection
    final backendState = ref.watch(backendLauncherProvider);
    final videoState = ref.watch(videoStreamProvider);

    // Connected if backend is running (has wsPort) AND video stream is connected
    final isConnected = backendState.wsPort != null && videoState.isConnected;
    final isPartial = backendState.wsPort != null && !videoState.isConnected;

    final color = isConnected ? G20Colors.success
        : isPartial ? G20Colors.warning
        : G20Colors.error;

    final message = isConnected ? 'Connected to backend'
        : isPartial ? 'Backend running, stream disconnected'
        : 'Disconnected';

    return Tooltip(
      message: message,
      child: Container(
        width: 12,
        height: 12,
        decoration: BoxDecoration(
          color: color,
          shape: BoxShape.circle,
          boxShadow: [
            BoxShadow(
              color: color.withOpacity(0.5),
              blurRadius: 4,
              spreadRadius: 1,
            ),
          ],
        ),
      ),
    );
  }
}
