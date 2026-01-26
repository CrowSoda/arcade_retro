import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../../core/config/theme.dart';
import 'providers/waterfall_provider.dart';
import 'providers/detection_provider.dart';
import 'providers/map_provider.dart';
import 'providers/inference_provider.dart';
import 'providers/video_stream_provider.dart';
import '../settings/settings_screen.dart' show waterfallFpsProvider;
import '../config/config_screen.dart' show missionsProvider, activeMissionProvider, Mission;
import 'widgets/video_waterfall_display.dart';
import 'widgets/psd_chart.dart';
import 'widgets/detection_table.dart';
import 'widgets/inputs_panel.dart';
import 'widgets/map_display.dart';

/// Provider for right panel collapsed state
final rightPanelCollapsedProvider = StateProvider<bool>((ref) => false);

/// Live Detection Screen - Collapsible right panel layout
/// Layout:
///   LEFT:  Waterfall (top) + PSD (bottom) OR Map (full)
///   RIGHT: Collapsible panel with Detection Table (top) + Inputs (bottom)
class LiveDetectionScreen extends ConsumerStatefulWidget {
  const LiveDetectionScreen({super.key});

  @override
  ConsumerState<LiveDetectionScreen> createState() => _LiveDetectionScreenState();
}

class _LiveDetectionScreenState extends ConsumerState<LiveDetectionScreen> {
  int _lastPruneRow = 0;  // Track last prune to avoid excessive calls
  
  @override
  void initState() {
    super.initState();
    
    // Set up detection forwarding from video stream to detection provider
    WidgetsBinding.instance.addPostFrameCallback((_) {
      final videoNotifier = ref.read(videoStreamProvider.notifier);
      final detectionNotifier = ref.read(detectionProvider.notifier);
      
      videoNotifier.setDetectionCallback((detections, pts) {
        // Convert and add to detection provider
        final converted = detections.map((d) => 
          convertVideoDetection(d, pts)
        ).toList();
        
        detectionNotifier.addDetections(converted);
        
      });
      
      // Connect to video stream
      videoNotifier.connect('localhost', 8765);
    });
  }

  @override
  Widget build(BuildContext context) {
    // Trigger auto-start of inference when backend is ready
    ref.watch(autoStartInferenceProvider);
    
    final displayMode = ref.watch(displayModeProvider);
    final isCollapsed = ref.watch(rightPanelCollapsedProvider);
    
    // FPS CONTROL: Listen for FPS changes and send to backend
    ref.listen<int>(waterfallFpsProvider, (previous, next) {
      final currentState = ref.read(videoStreamProvider);
      if (previous != next && currentState.isConnected) {
        ref.read(videoStreamProvider.notifier).setFps(next);
      }
    });
    
    // PSD BOX LIFECYCLE: Prune detections when waterfall scrolls or buffer changes
    ref.listen<VideoStreamState>(videoStreamProvider, (previous, next) {
      final detectionNotifier = ref.read(detectionProvider.notifier);
      
      // Clear all detections on reconnect (connection state changed to connected)
      if (previous?.isConnected != true && next.isConnected) {
        detectionNotifier.clearAll();
        _lastPruneRow = 0;
        debugPrint('[PSD Lifecycle] Connection established - cleared all detection boxes');
        
        // Send initial FPS if not default
        final currentFps = ref.read(waterfallFpsProvider);
        if (currentFps != 30) {
          ref.read(videoStreamProvider.notifier).setFps(currentFps);
        }
        return;
      }
      
      // Prune detections that have scrolled off based on absoluteRow
      // Only prune every ~30 rows (about 1 frame) to avoid excessive overhead
      final currentRow = next.totalRowsReceived;
      final bufferHeight = next.bufferHeight;
      
      if (currentRow - _lastPruneRow >= 30 && bufferHeight > 0) {
        detectionNotifier.pruneByAbsoluteRow(currentRow, bufferHeight);
        _lastPruneRow = currentRow;
      }
    });

    return Scaffold(
      backgroundColor: G20Colors.backgroundDark,
      body: Row(
        children: [
          // LEFT SIDE - Waterfall + PSD OR Map (expands when right panel collapsed)
          Expanded(
            child: Column(
              children: [
                // Toggle header
                _DisplayModeHeader(),
                // Content based on mode
                Expanded(
                  child: displayMode == DisplayMode.waterfall
                      ? _WaterfallPsdView()
                      : _MapView(),
                ),
              ],
            ),
          ),
          // COLLAPSE HANDLE - always visible
          _CollapseHandle(isCollapsed: isCollapsed),
          // RIGHT SIDE - Detection Table + Inputs (collapsible)
          AnimatedContainer(
            duration: const Duration(milliseconds: 200),
            curve: Curves.easeInOut,
            width: isCollapsed ? 0 : 350,
            child: isCollapsed
                ? const SizedBox.shrink()
                : Column(
                    children: [
                      // Detection Table (top 60%)
                      Expanded(
                        flex: 6,
                        child: Container(
                          margin: const EdgeInsets.fromLTRB(0, 8, 8, 4),
                          decoration: BoxDecoration(
                            color: G20Colors.surfaceDark,
                            borderRadius: BorderRadius.circular(8),
                            border: Border.all(color: G20Colors.cardDark, width: 1),
                          ),
                          child: ClipRRect(
                            borderRadius: const BorderRadius.all(Radius.circular(8)),
                            child: _DetectionTableWithLongPress(),
                          ),
                        ),
                      ),
                      // Inputs panel (bottom 40%)
                      Expanded(
                        flex: 4,
                        child: Container(
                          margin: const EdgeInsets.fromLTRB(0, 4, 8, 8),
                          decoration: BoxDecoration(
                            color: G20Colors.surfaceDark,
                            borderRadius: BorderRadius.circular(8),
                            border: Border.all(color: G20Colors.cardDark, width: 1),
                          ),
                          child: const ClipRRect(
                            borderRadius: BorderRadius.all(Radius.circular(8)),
                            child: InputsPanel(),
                          ),
                        ),
                      ),
                    ],
                  ),
          ),
        ],
      ),
    );
  }
}

/// Collapse/expand handle for right panel
class _CollapseHandle extends ConsumerWidget {
  final bool isCollapsed;

  const _CollapseHandle({required this.isCollapsed});

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

/// Detection table wrapper with long-press to zoom map
class _DetectionTableWithLongPress extends ConsumerWidget {
  @override
  Widget build(BuildContext context, WidgetRef ref) {
    return GestureDetector(
      onLongPress: () {
        // Zoom map to show all detections
        final detections = ref.read(detectionProvider);
        if (detections.isEmpty) {
          debugPrint('[Map] No detections to show');
          return;
        }
        
        // Calculate bounds that fit all detections and switch to map view
        final currentMode = ref.read(mapStateProvider).displayMode;
        ref.read(mapStateProvider.notifier).zoomToFitAllDetections(detections);
        
        // Switch to map view if not already
        if (currentMode != DisplayMode.map) {
          ref.read(mapStateProvider.notifier).setDisplayMode(DisplayMode.map);
          ref.read(waterfallProvider.notifier).skipRendering();
        }
        
        debugPrint('[Map] Showing ${detections.length} detections on map');
      },
      child: const DetectionTable(),
    );
  }
}

/// Header with toggle button between Waterfall/PSD and Map
class _DisplayModeHeader extends ConsumerWidget {
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
          _ModeToggleButton(),
        ],
      ),
    );
  }
}

/// Toggle button between waterfall and map modes
class _ModeToggleButton extends ConsumerWidget {
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

/// Waterfall + PSD view (original layout)
class _WaterfallPsdView extends StatelessWidget {
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
          Expanded(
            flex: 7,
            child: const ClipRRect(
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
          Expanded(
            flex: 3,
            child: const ClipRRect(
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
class _MapView extends StatelessWidget {
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

/// Mission picker button - shows current mission and opens picker popup
class _MissionPickerButton extends ConsumerWidget {
  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final activeMission = ref.watch(activeMissionProvider);
    final missions = ref.watch(missionsProvider);
    
    return Material(
      color: activeMission != null ? Colors.green.shade700 : G20Colors.cardDark,
      borderRadius: BorderRadius.circular(8),
      child: InkWell(
        borderRadius: BorderRadius.circular(8),
        onTap: () => _showMissionPicker(context, ref, missions),
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              Icon(
                Icons.rocket_launch,
                size: 18,
                color: activeMission != null ? Colors.white : G20Colors.textSecondaryDark,
              ),
              const SizedBox(width: 6),
              Text(
                activeMission?.name ?? 'No Mission',
                style: TextStyle(
                  fontSize: 12,
                  fontWeight: FontWeight.w600,
                  color: activeMission != null ? Colors.white : G20Colors.textSecondaryDark,
                ),
              ),
              const SizedBox(width: 4),
              Icon(
                Icons.expand_more,
                size: 16,
                color: activeMission != null ? Colors.white70 : G20Colors.textSecondaryDark,
              ),
            ],
          ),
        ),
      ),
    );
  }

  void _showMissionPicker(BuildContext context, WidgetRef ref, List<Mission> missions) {
    showDialog(
      context: context,
      builder: (ctx) => _MissionPickerDialog(
        missions: missions,
        onSelect: (mission) {
          ref.read(activeMissionProvider.notifier).state = mission;
          Navigator.pop(ctx);
          
          // Log mission activation
          debugPrint('[Live] ════════════════════════════════════════');
          debugPrint('[Live] MISSION ACTIVATED: ${mission.name}');
          debugPrint('[Live] BW: ${mission.bandwidthMhz} MHz, Dwell: ${mission.dwellTimeSec} sec');
          debugPrint('[Live] Ranges: ${mission.freqRanges.length}, Models: ${mission.models.length}');
          debugPrint('[Live] ════════════════════════════════════════');
        },
        onClear: () {
          ref.read(activeMissionProvider.notifier).state = null;
          Navigator.pop(ctx);
        },
      ),
    );
  }
}

/// Swanky mission picker dialog
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
                  ? Padding(
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
                    )
                  : ListView.separated(
                      shrinkWrap: true,
                      padding: const EdgeInsets.all(16),
                      itemCount: missions.length,
                      separatorBuilder: (_, __) => const SizedBox(height: 12),
                      itemBuilder: (context, index) {
                        final mission = missions[index];
                        return _MissionCard(mission: mission, onTap: () => onSelect(mission));
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

/// Individual mission card in picker
class _MissionCard extends StatelessWidget {
  final Mission mission;
  final VoidCallback onTap;

  const _MissionCard({required this.mission, required this.onTap});

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
                        _InfoChip(icon: Icons.radio, label: '${mission.freqRanges.length} ranges'),
                        const SizedBox(width: 8),
                        _InfoChip(icon: Icons.psychology, label: '${mission.models.length} models'),
                        const SizedBox(width: 8),
                        _InfoChip(icon: Icons.speed, label: '${mission.bandwidthMhz.toInt()} MHz'),
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
class _InfoChip extends StatelessWidget {
  final IconData icon;
  final String label;

  const _InfoChip({required this.icon, required this.label});

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
