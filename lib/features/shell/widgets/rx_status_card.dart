/// RX Status Card Widget - Displays SDR channel status
/// 
/// Shows frequency, bandwidth, mode, and connection status for each RX channel
library;

import 'dart:ui';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../../../core/config/theme.dart';
import '../../live_detection/providers/rx_state_provider.dart';

/// Multi-RX status display - shows status for each connected RX channel
class RxStatusDisplay extends ConsumerWidget {
  const RxStatusDisplay({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final rxState = ref.watch(multiRxProvider);
    final connectedChannels = rxState.connectedChannels;
    
    if (connectedChannels.isEmpty) {
      return const RxStatusCard(
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
          RxStatusCard(
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
class RxStatusCard extends StatelessWidget {
  final int rxNumber;
  final double centerMHz;
  final double bwMHz;
  final Color modeColor;
  final String modeText;
  final bool isConnected;

  const RxStatusCard({
    super.key,
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
