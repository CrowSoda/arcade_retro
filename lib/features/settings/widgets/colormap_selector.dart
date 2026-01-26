/// Colormap Selector Widget
/// 
/// Allows user to select color palette for waterfall display
/// Sends command to backend which switches the LUT
library;

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../../../core/config/theme.dart';
import '../../live_detection/providers/video_stream_provider.dart';
import '../providers/settings_providers.dart';

/// Colormap selector - changes waterfall color palette
class ColormapSelector extends ConsumerWidget {
  const ColormapSelector({super.key});

  void _setColormap(WidgetRef ref, int newColormap) {
    debugPrint('[Settings] Colormap changed: ${colormapNames[newColormap]}');
    
    // Update provider state
    ref.read(waterfallColormapProvider.notifier).state = newColormap;
    
    // Send to backend
    ref.read(videoStreamProvider.notifier).setColormap(newColormap);
  }

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final colormap = ref.watch(waterfallColormapProvider);

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text(
          'Colormap',
          style: TextStyle(
            fontSize: 14,
            color: G20Colors.textPrimaryDark,
          ),
        ),
        const SizedBox(height: 4),
        Text(
          'Color palette for waterfall display',
          style: TextStyle(
            fontSize: 11,
            color: G20Colors.textSecondaryDark,
          ),
        ),
        const SizedBox(height: 12),
        // Colormap buttons in a row
        Row(
          children: List.generate(colormapNames.length, (i) {
            final isSelected = colormap == i;
            return Expanded(
              child: Padding(
                padding: EdgeInsets.only(right: i < colormapNames.length - 1 ? 6 : 0),
                child: GestureDetector(
                  onTap: () => _setColormap(ref, i),
                  child: Container(
                    padding: const EdgeInsets.symmetric(vertical: 10),
                    decoration: BoxDecoration(
                      color: isSelected 
                          ? G20Colors.primary.withOpacity(0.2) 
                          : G20Colors.cardDark,
                      borderRadius: BorderRadius.circular(6),
                      border: Border.all(
                        color: isSelected ? G20Colors.primary : G20Colors.cardDark,
                        width: isSelected ? 2 : 1,
                      ),
                    ),
                    child: Center(
                      child: Text(
                        colormapNames[i],
                        style: TextStyle(
                          color: isSelected ? G20Colors.primary : G20Colors.textSecondaryDark,
                          fontWeight: isSelected ? FontWeight.bold : FontWeight.normal,
                          fontSize: 11,
                        ),
                      ),
                    ),
                  ),
                ),
              ),
            );
          }),
        ),
      ],
    );
  }
}
