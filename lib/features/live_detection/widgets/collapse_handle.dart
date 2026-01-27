/// Collapse Handle Widget - Panel collapse/expand control
///
/// Draggable handle for collapsing/expanding the right panel
library;

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../../../core/config/theme.dart';
import '../live_detection_screen.dart' show rightPanelCollapsedProvider;

/// Collapse/expand handle for right panel
class CollapseHandle extends ConsumerWidget {
  final bool isCollapsed;

  const CollapseHandle({super.key, required this.isCollapsed});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    return GestureDetector(
      onTap: () {
        ref.read(rightPanelCollapsedProvider.notifier).state = !isCollapsed;
      },
      child: Container(
        width: 20,
        margin: const EdgeInsets.symmetric(vertical: 8),
        decoration: BoxDecoration(
          color: G20Colors.cardDark,
          borderRadius: BorderRadius.circular(4),
        ),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            // Visual handle indicator
            Container(
              width: 4,
              height: 40,
              margin: const EdgeInsets.symmetric(vertical: 4),
              decoration: BoxDecoration(
                color: G20Colors.textSecondaryDark,
                borderRadius: BorderRadius.circular(2),
              ),
            ),
            Icon(
              isCollapsed ? Icons.chevron_left : Icons.chevron_right,
              color: G20Colors.textSecondaryDark,
              size: 16,
            ),
            const SizedBox(height: 4),
            RotatedBox(
              quarterTurns: 3,
              child: Text(
                isCollapsed ? 'SHOW' : 'HIDE',
                style: const TextStyle(
                  fontSize: 9,
                  fontWeight: FontWeight.bold,
                  color: G20Colors.textSecondaryDark,
                  letterSpacing: 1,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
