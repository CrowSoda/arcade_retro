/// Waterfall PSD View - Combined waterfall display and PSD chart
///
/// Shows waterfall spectrogram (top 70%) and PSD line chart (bottom 30%)
library;

import 'package:flutter/material.dart';
import '../../../core/config/theme.dart';
import 'video_waterfall_display.dart';
import 'psd_chart.dart';
import 'map_display.dart';

/// Waterfall + PSD view (original layout)
class WaterfallPsdView extends StatelessWidget {
  const WaterfallPsdView({super.key});

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.fromLTRB(8, 0, 8, 8),
      decoration: BoxDecoration(
        color: G20Colors.surfaceDark,
        borderRadius: const BorderRadius.vertical(bottom: Radius.circular(8)),
        border: const Border(
          left: BorderSide(color: G20Colors.cardDark, width: 1),
          right: BorderSide(color: G20Colors.cardDark, width: 1),
          bottom: BorderSide(color: G20Colors.cardDark, width: 1),
        ),
      ),
      child: Column(
        children: [
          // Waterfall display (top 70%) - VIDEO STREAMING
          const Expanded(
            flex: 7,
            child: ClipRRect(
              borderRadius: BorderRadius.zero,
              child: VideoWaterfallDisplay(),
            ),
          ),
          // Divider
          Container(
            height: 1,
            color: G20Colors.cardDark,
          ),
          // PSD chart (bottom 30%)
          const Expanded(
            flex: 3,
            child: ClipRRect(
              borderRadius: BorderRadius.vertical(bottom: Radius.circular(7)),
              child: PsdChart(),
            ),
          ),
        ],
      ),
    );
  }
}

/// Map view (replaces waterfall when toggled)
class MapView extends StatelessWidget {
  const MapView({super.key});

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.fromLTRB(8, 0, 8, 8),
      decoration: BoxDecoration(
        color: G20Colors.surfaceDark,
        borderRadius: const BorderRadius.vertical(bottom: Radius.circular(8)),
        border: const Border(
          left: BorderSide(color: G20Colors.cardDark, width: 1),
          right: BorderSide(color: G20Colors.cardDark, width: 1),
          bottom: BorderSide(color: G20Colors.cardDark, width: 1),
        ),
      ),
      child: const ClipRRect(
        borderRadius: BorderRadius.vertical(bottom: Radius.circular(7)),
        child: MapDisplay(),
      ),
    );
  }
}
