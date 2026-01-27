/// FFT Size Selector Widget
///
/// Allows user to select FFT resolution for waterfall processing
/// GPU-accelerated with cuFFT kernel warmup on change
library;

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:shared_preferences/shared_preferences.dart';
import '../../../core/config/theme.dart';
import '../../live_detection/providers/video_stream_provider.dart';
import '../providers/settings_providers.dart';

/// FFT Size selector for waterfall resolution/performance tradeoff
/// GPU-accelerated - allows user to balance frequency resolution vs speed
class FftSizeSelector extends ConsumerWidget {
  const FftSizeSelector({super.key});

  Future<void> _setFftSize(WidgetRef ref, int newSize) async {
    debugPrint('[Settings] FFT size button tapped: $newSize');

    // Update provider state
    ref.read(waterfallFftSizeProvider.notifier).state = newSize;

    // Save to SharedPreferences for persistence
    try {
      final prefs = await SharedPreferences.getInstance();
      await prefs.setInt('waterfall_fft_size', newSize);
      debugPrint('[Settings] FFT size saved to preferences: $newSize');
    } catch (e) {
      debugPrint('[Settings] Failed to save FFT size: $e');
    }

    // Send to backend (includes cuFFT warmup - may take 100-500ms)
    ref.read(videoStreamProvider.notifier).setFftSize(newSize);
  }

  String _estimatedTime(int size) {
    switch (size) {
      case 8192: return '~2ms';
      case 16384: return '~4ms';
      case 32768: return '~6ms';
      case 65536: return '~10ms';
      default: return '?';
    }
  }

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final fftSize = ref.watch(waterfallFftSizeProvider);
    final option = fftSizeOptions[fftSize];

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            const Text(
              'FFT Resolution',
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
                option?.label ?? '${(fftSize / 1024).round()}K',
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
          'Higher = better frequency resolution, slower processing',
          style: TextStyle(
            fontSize: 11,
            color: G20Colors.textSecondaryDark,
          ),
        ),
        const SizedBox(height: 12),
        // FFT size buttons
        Row(
          children: fftSizeOptions.entries.map((entry) {
            final size = entry.key;
            final opt = entry.value;
            return Expanded(
              child: Padding(
                padding: EdgeInsets.only(
                  right: size != 65536 ? 6 : 0,
                ),
                child: _FftSizeOption(
                  label: opt.label,
                  sublabel: opt.sublabel,
                  selected: fftSize == size,
                  onTap: () => _setFftSize(ref, size),
                ),
              ),
            );
          }).toList(),
        ),
        const SizedBox(height: 8),
        // Info display showing resolution and timing
        Container(
          padding: const EdgeInsets.all(8),
          decoration: BoxDecoration(
            color: G20Colors.cardDark,
            borderRadius: BorderRadius.circular(6),
          ),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.spaceAround,
            children: [
              Column(
                children: [
                  Text(
                    'Resolution',
                    style: TextStyle(
                      fontSize: 10,
                      color: G20Colors.textSecondaryDark,
                    ),
                  ),
                  Text(
                    option?.resolution ?? '${(20000000 / fftSize).round()} Hz/bin',
                    style: const TextStyle(
                      fontSize: 12,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ],
              ),
              Column(
                children: [
                  Text(
                    'Est. Time',
                    style: TextStyle(
                      fontSize: 10,
                      color: G20Colors.textSecondaryDark,
                    ),
                  ),
                  Text(
                    _estimatedTime(fftSize),
                    style: const TextStyle(
                      fontSize: 12,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ],
              ),
              Column(
                children: [
                  Text(
                    'FFTs/frame',
                    style: TextStyle(
                      fontSize: 10,
                      color: G20Colors.textSecondaryDark,
                    ),
                  ),
                  Text(
                    '${((660000 - fftSize) ~/ (fftSize ~/ 2) + 1)}',
                    style: const TextStyle(
                      fontSize: 12,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ],
              ),
            ],
          ),
        ),
      ],
    );
  }
}

class _FftSizeOption extends StatelessWidget {
  final String label;
  final String sublabel;
  final bool selected;
  final VoidCallback onTap;

  const _FftSizeOption({
    required this.label,
    required this.sublabel,
    required this.selected,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
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
        child: Column(
          children: [
            Text(
              label,
              style: TextStyle(
                color: selected ? G20Colors.primary : G20Colors.textSecondaryDark,
                fontWeight: selected ? FontWeight.bold : FontWeight.normal,
                fontSize: 14,
              ),
            ),
            Text(
              sublabel,
              style: TextStyle(
                color: selected
                    ? G20Colors.primary.withOpacity(0.7)
                    : G20Colors.textSecondaryDark,
                fontSize: 10,
              ),
            ),
          ],
        ),
      ),
    );
  }
}
