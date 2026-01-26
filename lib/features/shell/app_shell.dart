import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import '../../core/config/router.dart';
import '../../core/config/theme.dart';
import '../../core/services/backend_launcher.dart';
import '../config/providers/tuning_state_provider.dart';
import '../live_detection/providers/sdr_config_provider.dart';
import '../live_detection/providers/rx_state_provider.dart';
import '../live_detection/providers/video_stream_provider.dart';

/// Fixed color for recording indicator - cyan, neutral and visible
const _recordingColor = Color(0xFF00BCD4);

/// App shell with NavigationRail for desktop layout - 5 pages
class AppShell extends ConsumerStatefulWidget {
  final Widget child;

  const AppShell({super.key, required this.child});

  @override
  ConsumerState<AppShell> createState() => _AppShellState();
}

class _AppShellState extends ConsumerState<AppShell> {
  @override
  void initState() {
    super.initState();
  }

  @override
  Widget build(BuildContext context) {
    // Listen to capture state changes and update RX-2 accordingly
    ref.listen<ManualCaptureState>(manualCaptureProvider, (prev, next) {
      final multiRxNotifier = ref.read(multiRxProvider.notifier);
      
      if (next.phase == CapturePhase.capturing && prev?.phase != CapturePhase.capturing) {
        // Capture started - update RX-2 to manual mode with capture frequency
        final freqMHz = double.tryParse(next.targetFreqMHz ?? '825.0') ?? 825.0;
        multiRxNotifier.tuneRx2(freqMHz, 5.0, null);  // 5 MHz BW, no timeout
        debugPrint('[RX] RX-2 tuned to $freqMHz MHz for capture');
      } else if (next.phase == CapturePhase.idle && prev?.phase != CapturePhase.idle) {
        // Capture complete - return RX-2 to idle
        multiRxNotifier.rx2ResumeIdle();
        debugPrint('[RX] RX-2 returned to idle');
      }
    });
    
    final selectedIndex = ref.watch(navigationIndexProvider);

    return Scaffold(
      body: Row(
        children: [
          // Navigation Rail
          NavigationRail(
            selectedIndex: selectedIndex,
            onDestinationSelected: (index) {
              ref.read(navigationIndexProvider.notifier).state = index;
              switch (index) {
                case 0:
                  context.go(AppRoutes.live);
                  break;
                case 1:
                  context.go(AppRoutes.training);
                  break;
                case 2:
                  context.go(AppRoutes.mission);
                  break;
                case 3:
                  context.go(AppRoutes.database);
                  break;
                case 4:
                  context.go(AppRoutes.settings);
                  break;
              }
            },
            labelType: NavigationRailLabelType.all,
            leading: Padding(
              padding: const EdgeInsets.symmetric(vertical: 16),
              child: Column(
                children: [
                  Container(
                    width: 40,
                    height: 40,
                    decoration: BoxDecoration(
                      color: G20Colors.primary,
                      borderRadius: BorderRadius.circular(8),
                    ),
                    child: const Center(
                      child: Text(
                        'G20',
                        style: TextStyle(
                          color: Colors.white,
                          fontWeight: FontWeight.bold,
                          fontSize: 12,
                        ),
                      ),
                    ),
                  ),
                ],
              ),
            ),
            trailing: Expanded(
              child: Align(
                alignment: Alignment.bottomCenter,
                child: Padding(
                  padding: const EdgeInsets.only(bottom: 16),
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      // Recording indicator - shows when capturing (above freq display)
                      _RecordingIndicator(),
                      const SizedBox(height: 16),
                      // Frequency/Mode status display - always visible
                      _FrequencyStatusWidget(),
                      const SizedBox(height: 16),
                      // Connection status
                      _ConnectionIndicator(),
                    ],
                  ),
                ),
              ),
            ),
            destinations: const [
              NavigationRailDestination(
                icon: Icon(Icons.sensors_outlined),
                selectedIcon: Icon(Icons.sensors),
                label: Text('Live'),
              ),
              NavigationRailDestination(
                icon: Icon(Icons.model_training_outlined),
                selectedIcon: Icon(Icons.model_training),
                label: Text('Training'),
              ),
              NavigationRailDestination(
                icon: Icon(Icons.rocket_launch_outlined),
                selectedIcon: Icon(Icons.rocket_launch),
                label: Text('Mission'),
              ),
              NavigationRailDestination(
                icon: Icon(Icons.storage_outlined),
                selectedIcon: Icon(Icons.storage),
                label: Text('Database'),
              ),
              NavigationRailDestination(
                icon: Icon(Icons.settings_outlined),
                selectedIcon: Icon(Icons.settings),
                label: Text('Settings'),
              ),
            ],
          ),
          // Vertical divider
          const VerticalDivider(thickness: 1, width: 1),
          // Main content
          Expanded(child: widget.child),
        ],
      ),
    );
  }
}

/// Recording indicator - shows pulsing SOI color when capturing
class _RecordingIndicator extends ConsumerStatefulWidget {
  @override
  ConsumerState<_RecordingIndicator> createState() => _RecordingIndicatorState();
}

class _RecordingIndicatorState extends ConsumerState<_RecordingIndicator>
    with SingleTickerProviderStateMixin {
  late AnimationController _pulseController;
  late Animation<double> _pulseAnimation;

  @override
  void initState() {
    super.initState();
    _pulseController = AnimationController(
      duration: const Duration(milliseconds: 800),
      vsync: this,
    )..repeat(reverse: true);
    _pulseAnimation = Tween<double>(begin: 0.4, end: 1.0).animate(
      CurvedAnimation(parent: _pulseController, curve: Curves.easeInOut),
    );
  }

  @override
  void dispose() {
    _pulseController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final captureState = ref.watch(manualCaptureProvider);
    
    // Only show when actively capturing
    if (captureState.phase != CapturePhase.capturing) {
      return const SizedBox.shrink();
    }

    final signalName = captureState.signalName ?? 'unknown';
    final progress = captureState.captureProgress;
    final durationMin = captureState.captureDurationMinutes;
    final elapsedSec = (progress * durationMin * 60).round();
    final totalSec = durationMin * 60;

    return Tooltip(
      message: '[REC] RECORDING\n$signalName\n${_formatTime(elapsedSec)} / ${_formatTime(totalSec)}\n\nTap to cancel',
      waitDuration: const Duration(milliseconds: 300),
      preferBelow: false,
      child: GestureDetector(
        onTap: () {
          // Cancel capture on tap
          ref.read(manualCaptureProvider.notifier).cancel();
          // No snackbar
        },
        child: AnimatedBuilder(
          animation: _pulseAnimation,
          builder: (context, child) {
            final queueLen = captureState.queueLength;
            return Stack(
              clipBehavior: Clip.none,
              children: [
                Container(
                  width: 32,
                  height: 32,
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    color: G20Colors.cardDark,
                    border: Border.all(color: _recordingColor.withOpacity(0.3), width: 2),
                  ),
                  child: Stack(
                    alignment: Alignment.center,
                    children: [
                      // Progress ring
                      SizedBox(
                        width: 28,
                        height: 28,
                        child: CircularProgressIndicator(
                          value: progress,
                          strokeWidth: 3,
                          backgroundColor: G20Colors.backgroundDark,
                          valueColor: const AlwaysStoppedAnimation<Color>(_recordingColor),
                        ),
                      ),
                      // Pulsing center dot
                      Container(
                        width: 10,
                        height: 10,
                        decoration: BoxDecoration(
                          shape: BoxShape.circle,
                          color: _recordingColor.withOpacity(_pulseAnimation.value),
                          boxShadow: [
                            BoxShadow(
                              color: _recordingColor.withOpacity(_pulseAnimation.value * 0.5),
                              blurRadius: 6,
                              spreadRadius: 2,
                            ),
                          ],
                        ),
                      ),
                    ],
                  ),
                ),
                // Queue badge - show when anything in queue
                if (queueLen >= 1)
                  Positioned(
                    right: -4,
                    top: -4,
                    child: Container(
                      width: 16,
                      height: 16,
                      decoration: BoxDecoration(
                        color: G20Colors.warning,
                        shape: BoxShape.circle,
                        border: Border.all(color: G20Colors.cardDark, width: 1),
                      ),
                      child: Center(
                        child: Text(
                          '$queueLen',
                          style: const TextStyle(
                            color: Colors.black,
                            fontSize: 9,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                      ),
                    ),
                  ),
              ],
            );
          },
        ),
      ),
    );
  }

  String _formatTime(int totalSeconds) {
    final min = totalSeconds ~/ 60;
    final sec = totalSeconds % 60;
    return '${min}:${sec.toString().padLeft(2, '0')}';
  }
}

/// Multi-RX status display - shows status for each connected RX channel
class _FrequencyStatusWidget extends ConsumerWidget {
  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final rxState = ref.watch(multiRxProvider);
    final connectedChannels = rxState.connectedChannels;
    
    if (connectedChannels.isEmpty) {
      return _SingleRxCard(
        rxNumber: 0,
        centerMHz: 0,
        bwMHz: 0,
        modeColor: Colors.grey,
        modeText: 'NO RX',
        isConnected: false,
      );
    }

    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        for (final rx in connectedChannels) ...[
          _SingleRxCard(
            rxNumber: rx.rxNumber,
            centerMHz: rx.centerFreqMHz,
            bwMHz: rx.bandwidthMHz,
            modeColor: rx.modeColor,
            modeText: rx.modeDisplayString,
            isConnected: rx.isConnected,
          ),
          if (rx != connectedChannels.last) const SizedBox(height: 6),
        ],
      ],
    );
  }
}

/// Single RX channel status card
class _SingleRxCard extends StatelessWidget {
  final int rxNumber;
  final double centerMHz;
  final double bwMHz;
  final Color modeColor;
  final String modeText;
  final bool isConnected;

  const _SingleRxCard({
    required this.rxNumber,
    required this.centerMHz,
    required this.bwMHz,
    required this.modeColor,
    required this.modeText,
    required this.isConnected,
  });

  @override
  Widget build(BuildContext context) {
    return Tooltip(
      message: 'RX$rxNumber\nCenter: ${centerMHz.toStringAsFixed(1)} MHz\nBandwidth: ${bwMHz.toStringAsFixed(0)} MHz\nMode: $modeText',
      waitDuration: const Duration(milliseconds: 300),
      child: Container(
        width: 72,
        padding: const EdgeInsets.symmetric(vertical: 6, horizontal: 4),
        decoration: BoxDecoration(
          color: G20Colors.cardDark,
          borderRadius: BorderRadius.circular(6),
          border: Border.all(
            color: modeColor.withValues(alpha: 0.5),
            width: 1,
          ),
        ),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            // RX label
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 1),
              decoration: BoxDecoration(
                color: modeColor.withValues(alpha: 0.2),
                borderRadius: BorderRadius.circular(3),
              ),
              child: Text(
                'RX$rxNumber',
                style: TextStyle(
                  color: modeColor,
                  fontSize: 8,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ),
            const SizedBox(height: 3),
            // Frequency display with RF icon
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              mainAxisSize: MainAxisSize.min,
              children: [
                Icon(
                  Icons.sensors,
                  size: 10,
                  color: modeColor,
                ),
                const SizedBox(width: 2),
                Text(
                  '${centerMHz.toStringAsFixed(0)}',
                  style: const TextStyle(
                    color: G20Colors.textPrimaryDark,
                    fontSize: 12,
                    fontWeight: FontWeight.bold,
                    fontFeatures: [FontFeature.tabularFigures()],
                  ),
                ),
              ],
            ),
            Text(
              'MHz',
              style: TextStyle(
                color: G20Colors.textSecondaryDark.withValues(alpha: 0.7),
                fontSize: 7,
              ),
            ),
            // Bandwidth
            Text(
              'BW: ${bwMHz.toStringAsFixed(0)}',
              style: const TextStyle(
                color: G20Colors.textSecondaryDark,
                fontSize: 8,
              ),
            ),
            const SizedBox(height: 3),
            // Mode indicator
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 2),
              decoration: BoxDecoration(
                color: modeColor.withValues(alpha: 0.2),
                borderRadius: BorderRadius.circular(3),
              ),
              child: Text(
                modeText,
                style: TextStyle(
                  color: modeColor,
                  fontSize: 8,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

/// Connection status indicator - wired to backend and video stream state
class _ConnectionIndicator extends ConsumerWidget {
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
