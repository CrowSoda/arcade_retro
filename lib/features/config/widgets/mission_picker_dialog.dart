/// Mission Picker Dialog - Shared widget for selecting missions
///
/// Used in both LiveDetectionScreen and InputsPanel to avoid duplication
library;

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../../../core/config/theme.dart';
import '../config_screen.dart' show Mission, missionsProvider, activeMissionProvider;

/// Show the mission picker dialog
///
/// Returns the selected mission or null if cancelled/cleared
Future<Mission?> showMissionPickerDialog(
  BuildContext context,
  WidgetRef ref, {
  VoidCallback? onMissionLoaded,
}) async {
  final missions = ref.read(missionsProvider);

  return showDialog<Mission?>(
    context: context,
    builder: (ctx) => _MissionPickerDialog(
      missions: missions,
      onSelect: (mission) {
        ref.read(activeMissionProvider.notifier).state = mission;
        Navigator.pop(ctx, mission);
        debugPrint('[Mission] Selected: ${mission.name}');
        onMissionLoaded?.call();
      },
      onClear: () {
        ref.read(activeMissionProvider.notifier).state = null;
        Navigator.pop(ctx, null);
        debugPrint('[Mission] Cleared active mission');
      },
    ),
  );
}

/// Mission picker dialog widget
class _MissionPickerDialog extends StatelessWidget {
  final List<Mission> missions;
  final ValueChanged<Mission> onSelect;
  final VoidCallback onClear;

  const _MissionPickerDialog({
    required this.missions,
    required this.onSelect,
    required this.onClear,
  });

  @override
  Widget build(BuildContext context) {
    return Dialog(
      backgroundColor: G20Colors.surfaceDark,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      child: Container(
        width: 400,
        constraints: const BoxConstraints(maxHeight: 500),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            // Header
            Container(
              padding: const EdgeInsets.all(20),
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  colors: [G20Colors.primary, G20Colors.primary.withOpacity(0.7)],
                  begin: Alignment.topLeft,
                  end: Alignment.bottomRight,
                ),
                borderRadius: const BorderRadius.vertical(top: Radius.circular(16)),
              ),
              child: Row(
                children: [
                  Container(
                    padding: const EdgeInsets.all(10),
                    decoration: BoxDecoration(
                      color: Colors.white.withOpacity(0.2),
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: const Icon(Icons.rocket_launch, color: Colors.white, size: 28),
                  ),
                  const SizedBox(width: 16),
                  const Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          'Load Mission',
                          style: TextStyle(
                            color: Colors.white,
                            fontSize: 20,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                        SizedBox(height: 4),
                        Text(
                          'Select a mission to activate',
                          style: TextStyle(color: Colors.white70, fontSize: 13),
                        ),
                      ],
                    ),
                  ),
                  IconButton(
                    icon: const Icon(Icons.close, color: Colors.white70),
                    onPressed: () => Navigator.pop(context),
                  ),
                ],
              ),
            ),

            // Mission list
            Flexible(
              child: missions.isEmpty
                  ? _EmptyMissionsPlaceholder()
                  : ListView.separated(
                      shrinkWrap: true,
                      padding: const EdgeInsets.all(16),
                      itemCount: missions.length,
                      separatorBuilder: (_, __) => const SizedBox(height: 12),
                      itemBuilder: (context, index) {
                        final mission = missions[index];
                        return MissionCard(
                          mission: mission,
                          onTap: () => onSelect(mission),
                        );
                      },
                    ),
            ),

            // Footer
            Container(
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: G20Colors.cardDark.withOpacity(0.5),
                borderRadius: const BorderRadius.vertical(bottom: Radius.circular(16)),
              ),
              child: Row(
                children: [
                  // Clear mission button
                  TextButton.icon(
                    onPressed: onClear,
                    icon: const Icon(Icons.clear, size: 18),
                    label: const Text('Clear Mission'),
                    style: TextButton.styleFrom(foregroundColor: Colors.red.shade400),
                  ),
                  const Spacer(),
                  TextButton(
                    onPressed: () => Navigator.pop(context),
                    child: const Text('Cancel'),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

/// Empty state placeholder for no missions
class _EmptyMissionsPlaceholder extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.all(40),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(Icons.inbox, size: 64, color: Colors.grey.shade600),
          const SizedBox(height: 16),
          const Text(
            'No missions created yet',
            style: TextStyle(color: G20Colors.textSecondaryDark, fontSize: 16),
          ),
          const SizedBox(height: 8),
          const Text(
            'Go to Mission tab to create one',
            style: TextStyle(color: G20Colors.textSecondaryDark, fontSize: 12),
          ),
        ],
      ),
    );
  }
}

/// Mission card widget - displays a single mission in the picker
class MissionCard extends StatelessWidget {
  final Mission mission;
  final VoidCallback onTap;

  const MissionCard({
    super.key,
    required this.mission,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return Material(
      color: G20Colors.backgroundDark,
      borderRadius: BorderRadius.circular(12),
      child: InkWell(
        borderRadius: BorderRadius.circular(12),
        onTap: onTap,
        child: Container(
          padding: const EdgeInsets.all(16),
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(12),
            border: Border.all(color: G20Colors.cardDark),
          ),
          child: Row(
            children: [
              // Icon
              Container(
                width: 48,
                height: 48,
                decoration: BoxDecoration(
                  color: G20Colors.primary.withOpacity(0.15),
                  borderRadius: BorderRadius.circular(10),
                ),
                child: const Icon(Icons.rocket_launch, color: G20Colors.primary, size: 24),
              ),
              const SizedBox(width: 16),
              // Info
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      mission.name,
                      style: const TextStyle(
                        color: G20Colors.textPrimaryDark,
                        fontSize: 16,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    const SizedBox(height: 4),
                    Row(
                      children: [
                        _MissionInfoChip(icon: Icons.radio, label: '${mission.freqRanges.length} ranges'),
                        const SizedBox(width: 8),
                        _MissionInfoChip(icon: Icons.psychology, label: '${mission.models.length} models'),
                        const SizedBox(width: 8),
                        _MissionInfoChip(icon: Icons.speed, label: '${mission.bandwidthMhz.toInt()} MHz'),
                      ],
                    ),
                  ],
                ),
              ),
              // Arrow
              const Icon(Icons.arrow_forward_ios, color: G20Colors.textSecondaryDark, size: 16),
            ],
          ),
        ),
      ),
    );
  }
}

/// Small info chip for mission details
class _MissionInfoChip extends StatelessWidget {
  final IconData icon;
  final String label;

  const _MissionInfoChip({required this.icon, required this.label});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
      decoration: BoxDecoration(
        color: G20Colors.cardDark,
        borderRadius: BorderRadius.circular(4),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 10, color: G20Colors.textSecondaryDark),
          const SizedBox(width: 4),
          Text(
            label,
            style: const TextStyle(color: G20Colors.textSecondaryDark, fontSize: 10),
          ),
        ],
      ),
    );
  }
}
