import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../../core/config/theme.dart';
import 'providers/waterfall_provider.dart';
import 'providers/detection_provider.dart';
import 'providers/map_provider.dart';
import 'providers/inference_provider.dart';
import 'providers/video_stream_provider.dart';
import '../settings/settings_screen.dart' show waterfallFpsProvider;
import 'widgets/waterfall_display.dart';
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
          debugPrint('No detections to show');
          return;
        }
        
        // Calculate bounds that fit all detections
        final mapState = ref.read(mapStateProvider);
        ref.read(mapStateProvider.notifier).zoomToFitAllDetections(detections);
        
        // Switch to map view if not already
        if (mapState.displayMode != DisplayMode.map) {
          ref.read(mapStateProvider.notifier).setDisplayMode(DisplayMode.map);
          ref.read(waterfallProvider.notifier).skipRendering();
        }
        
        debugPrint('🗺️ Showing ${detections.length} detections on map');
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
