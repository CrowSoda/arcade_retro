import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../../../core/config/theme.dart';

/// Map display placeholder - Premium feature requires license
///
/// Contact vendor for map visualization capabilities:
/// - Real-time detection geolocation
/// - Interactive filtering by SOI
/// - Offline PMTiles support
class MapDisplay extends ConsumerWidget {
  const MapDisplay({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    return Container(
      color: const Color(0xFF293847), // Match map background
      child: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(
              Icons.map_outlined,
              size: 64,
              color: G20Colors.textSecondaryDark.withOpacity(0.5),
            ),
            const SizedBox(height: 16),
            Text(
              'Map Visualization',
              style: TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.bold,
                color: G20Colors.textPrimaryDark.withOpacity(0.7),
              ),
            ),
            const SizedBox(height: 8),
            Text(
              'Feature not included in this build',
              style: TextStyle(
                fontSize: 14,
                color: G20Colors.textSecondaryDark.withOpacity(0.6),
              ),
            ),
            const SizedBox(height: 24),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
              decoration: BoxDecoration(
                border: Border.all(
                  color: G20Colors.primary.withOpacity(0.3),
                ),
                borderRadius: BorderRadius.circular(4),
              ),
              child: Text(
                'Use Spectrum View for signal analysis',
                style: TextStyle(
                  fontSize: 12,
                  color: G20Colors.primary.withOpacity(0.8),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
