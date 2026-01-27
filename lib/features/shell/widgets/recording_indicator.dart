/// Recording Indicator Widget - Animated recording status for nav rail
///
/// Shows pulsing recording indicator with progress, queue badge, and cancel on tap
library;

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../../../core/config/theme.dart';
import '../../config/providers/tuning_state_provider.dart';

/// Fixed color for recording indicator - cyan, neutral and visible
const recordingIndicatorColor = Color(0xFF00BCD4);

/// Recording indicator - shows pulsing SOI color when capturing
class RecordingIndicator extends ConsumerStatefulWidget {
  const RecordingIndicator({super.key});

  @override
  ConsumerState<RecordingIndicator> createState() => _RecordingIndicatorState();
}

class _RecordingIndicatorState extends ConsumerState<RecordingIndicator>
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
      message: 'RECORDING\n$signalName\n${_formatTime(elapsedSec)} / ${_formatTime(totalSec)}\n\nTap to cancel',
      waitDuration: const Duration(milliseconds: 300),
      preferBelow: false,
      child: GestureDetector(
        onTap: () {
          // Cancel capture on tap
          ref.read(manualCaptureProvider.notifier).cancel();
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
                    border: Border.all(color: recordingIndicatorColor.withOpacity(0.3), width: 2),
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
                          valueColor: const AlwaysStoppedAnimation<Color>(recordingIndicatorColor),
                        ),
                      ),
                      // Pulsing center dot
                      Container(
                        width: 10,
                        height: 10,
                        decoration: BoxDecoration(
                          shape: BoxShape.circle,
                          color: recordingIndicatorColor.withOpacity(_pulseAnimation.value),
                          boxShadow: [
                            BoxShadow(
                              color: recordingIndicatorColor.withOpacity(_pulseAnimation.value * 0.5),
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
