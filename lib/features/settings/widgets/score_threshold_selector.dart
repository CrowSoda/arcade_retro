/// Score Threshold Selector Widget
///
/// Allows user to set detection confidence threshold
/// Only detections above this threshold will be displayed
library;

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../../../core/config/theme.dart';
import '../../live_detection/providers/video_stream_provider.dart';
import '../providers/settings_providers.dart';

/// Score threshold selector with slider and quick-select buttons
class ScoreThresholdSelector extends ConsumerWidget {
  const ScoreThresholdSelector({super.key});

  void _setScoreThreshold(WidgetRef ref, double newThreshold) {
    debugPrint('[Settings] Score threshold changed: ${(newThreshold * 100).round()}%');

    // Update provider state
    ref.read(scoreThresholdProvider.notifier).state = newThreshold;

    // DIRECT CALL to backend
    ref.read(videoStreamProvider.notifier).setScoreThreshold(newThreshold);
  }

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final threshold = ref.watch(scoreThresholdProvider);
    final percentage = (threshold * 100).round();

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            const Text(
              'Detection Confidence Threshold',
              style: TextStyle(
                fontSize: 14,
                color: G20Colors.textPrimaryDark,
              ),
            ),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
              decoration: BoxDecoration(
                color: G20Colors.primary.withOpacity(0.2),
                borderRadius: BorderRadius.circular(4),
              ),
              child: Text(
                '$percentage%',
                style: const TextStyle(
                  fontWeight: FontWeight.bold,
                  color: G20Colors.primary,
                ),
              ),
            ),
          ],
        ),
        const SizedBox(height: 4),
        Text(
          'Only show detections with confidence above this threshold',
          style: TextStyle(
            fontSize: 11,
            color: G20Colors.textSecondaryDark,
          ),
        ),
        const SizedBox(height: 8),
        Slider(
          value: threshold,
          min: 0.0,
          max: 1.0,
          divisions: 20,
          label: '$percentage%',
          onChanged: (v) => _setScoreThreshold(ref, v),
          onChangeEnd: (v) => _setScoreThreshold(ref, v),
        ),
        const SizedBox(height: 8),
        Row(
          children: [
            _ThresholdOption(
              label: '50%',
              value: 0.5,
              selected: (threshold - 0.5).abs() < 0.01,
              onTap: () => _setScoreThreshold(ref, 0.5),
            ),
            const SizedBox(width: 8),
            _ThresholdOption(
              label: '75%',
              value: 0.75,
              selected: (threshold - 0.75).abs() < 0.01,
              onTap: () => _setScoreThreshold(ref, 0.75),
            ),
            const SizedBox(width: 8),
            _ThresholdOption(
              label: '90%',
              value: 0.9,
              selected: (threshold - 0.9).abs() < 0.01,
              onTap: () => _setScoreThreshold(ref, 0.9),
            ),
            const SizedBox(width: 8),
            _ThresholdOption(
              label: '95%',
              value: 0.95,
              selected: (threshold - 0.95).abs() < 0.01,
              onTap: () => _setScoreThreshold(ref, 0.95),
            ),
          ],
        ),
      ],
    );
  }
}

class _ThresholdOption extends StatelessWidget {
  final String label;
  final double value;
  final bool selected;
  final VoidCallback onTap;

  const _ThresholdOption({
    required this.label,
    required this.value,
    required this.selected,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return Expanded(
      child: GestureDetector(
        onTap: onTap,
        child: Container(
          padding: const EdgeInsets.symmetric(vertical: 8),
          decoration: BoxDecoration(
            color: selected ? G20Colors.primary.withOpacity(0.2) : G20Colors.cardDark,
            borderRadius: BorderRadius.circular(6),
            border: Border.all(
              color: selected ? G20Colors.primary : G20Colors.cardDark,
              width: selected ? 2 : 1,
            ),
          ),
          child: Center(
            child: Text(
              label,
              style: TextStyle(
                color: selected ? G20Colors.primary : G20Colors.textSecondaryDark,
                fontWeight: selected ? FontWeight.bold : FontWeight.normal,
                fontSize: 13,
              ),
            ),
          ),
        ),
      ),
    );
  }
}
