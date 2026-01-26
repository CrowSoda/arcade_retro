/// Display Mode Header - Toggle between Spectrum and Map views
/// 
/// Shows current view mode with toggle button for switching
library;

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../../../core/config/theme.dart';
import '../providers/waterfall_provider.dart';
import '../providers/map_provider.dart';

/// Header with toggle button between Waterfall/PSD and Map
class DisplayModeHeader extends ConsumerWidget {
  const DisplayModeHeader({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final displayMode = ref.watch(displayModeProvider);

    return Container(
      margin: const EdgeInsets.fromLTRB(8, 8, 8, 0),
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        color: G20Colors.surfaceDark,
        borderRadius: const BorderRadius.vertical(top: Radius.circular(8)),
        border: Border.all(color: G20Colors.cardDark, width: 1),
      ),
      child: Row(
        children: [
          // Title based on mode
          Icon(
            displayMode == DisplayMode.waterfall 
                ? Icons.waves 
                : Icons.map,
            color: G20Colors.primary,
            size: 20,
          ),
          const SizedBox(width: 8),
          Text(
            displayMode == DisplayMode.waterfall 
                ? 'Spectrum View' 
                : 'Detection Map',
            style: const TextStyle(
              fontSize: 14,
              fontWeight: FontWeight.bold,
              color: G20Colors.textPrimaryDark,
            ),
          ),
          const Spacer(),
          // Toggle button
          const ModeToggleButton(),
        ],
      ),
    );
  }
}

/// Toggle button between waterfall and map modes
class ModeToggleButton extends ConsumerWidget {
  const ModeToggleButton({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final displayMode = ref.watch(displayModeProvider);

    return Container(
      decoration: BoxDecoration(
        color: G20Colors.cardDark,
        borderRadius: BorderRadius.circular(6),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          _ToggleOption(
            icon: Icons.waves,
            label: 'Spectrum',
            isSelected: displayMode == DisplayMode.waterfall,
            onTap: () {
              ref.read(mapStateProvider.notifier).setDisplayMode(DisplayMode.waterfall);
              ref.read(waterfallProvider.notifier).resumeRendering();  // Resume visual updates
            },
          ),
          _ToggleOption(
            icon: Icons.map,
            label: 'Map',
            isSelected: displayMode == DisplayMode.map,
            onTap: () {
              ref.read(mapStateProvider.notifier).setDisplayMode(DisplayMode.map);
              ref.read(waterfallProvider.notifier).skipRendering();  // Skip renders but keep processing
            },
          ),
        ],
      ),
    );
  }
}

class _ToggleOption extends StatelessWidget {
  final IconData icon;
  final String label;
  final bool isSelected;
  final VoidCallback onTap;

  const _ToggleOption({
    required this.icon,
    required this.label,
    required this.isSelected,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
        decoration: BoxDecoration(
          color: isSelected ? G20Colors.primary : Colors.transparent,
          borderRadius: BorderRadius.circular(5),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(
              icon,
              size: 16,
              color: isSelected ? Colors.white : G20Colors.textSecondaryDark,
            ),
            const SizedBox(width: 6),
            Text(
              label,
              style: TextStyle(
                fontSize: 12,
                fontWeight: isSelected ? FontWeight.bold : FontWeight.normal,
                color: isSelected ? Colors.white : G20Colors.textSecondaryDark,
              ),
            ),
          ],
        ),
      ),
    );
  }
}
