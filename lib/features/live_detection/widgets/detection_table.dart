import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../../../core/config/theme.dart';
import '../../../core/utils/dtg_formatter.dart';
import '../providers/detection_provider.dart';
import '../providers/map_provider.dart';
import '../providers/sdr_config_provider.dart';

/// Detection table showing: Name, DTG, MGRS (device location)
/// Groups detections by className, shows only most recent per signal type
class DetectionTable extends ConsumerWidget {
  const DetectionTable({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final detections = ref.watch(detectionProvider);

    // Group by className, keep only most recent per type
    final uniqueDetections = <String, Detection>{};
    for (final det in detections) {
      if (!uniqueDetections.containsKey(det.className) ||
          det.timestamp.isAfter(uniqueDetections[det.className]!.timestamp)) {
        uniqueDetections[det.className] = det;
      }
    }
    final groupedList = uniqueDetections.values.toList()
      ..sort((a, b) => b.timestamp.compareTo(a.timestamp));  // Most recent first

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        // Header
        Container(
          padding: const EdgeInsets.all(12),
          decoration: const BoxDecoration(
            border: Border(bottom: BorderSide(color: G20Colors.cardDark, width: 1)),
          ),
          child: const Text(
            'Detections',
            style: TextStyle(
              fontSize: 16,
              fontWeight: FontWeight.bold,
              color: G20Colors.textPrimaryDark,
            ),
          ),
        ),
        // Table header row
        Container(
          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
          decoration: const BoxDecoration(
            color: G20Colors.cardDark,
          ),
          child: Row(
            children: const [
              SizedBox(width: 28), // Eye icon + Color indicator space
              Expanded(
                flex: 5,  // Name
                child: Text('Name', style: TextStyle(fontSize: 11, fontWeight: FontWeight.bold, color: G20Colors.textSecondaryDark)),
              ),
              Expanded(
                flex: 3,  // DTG (reduced)
                child: Text('DTG', style: TextStyle(fontSize: 11, fontWeight: FontWeight.bold, color: G20Colors.textSecondaryDark)),
              ),
              Expanded(
                flex: 5,  // MGRS (increased)
                child: Text('MGRS', style: TextStyle(fontSize: 11, fontWeight: FontWeight.bold, color: G20Colors.textSecondaryDark)),
              ),
            ],
          ),
        ),
        // Detection rows - touch/drag scrollable
        Expanded(
          child: groupedList.isEmpty
              ? const Center(
                  child: Text(
                    'No detections',
                    style: TextStyle(color: G20Colors.textSecondaryDark, fontSize: 12),
                  ),
                )
              : ListView.builder(
                  physics: const BouncingScrollPhysics(),  // Touch/drag scrolling
                  itemCount: groupedList.length,
                  itemBuilder: (context, index) {
                    final det = groupedList[index];
                    return _DetectionRow(detection: det);
                  },
                ),
        ),
      ],
    );
  }
}

class _DetectionRow extends ConsumerWidget {
  final Detection detection;

  const _DetectionRow({required this.detection});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final color = getSOIColor(detection.className); // Use consistent color based on name
    final isSelected = detection.isSelected;
    final isVisible = ref.watch(soiVisibilityProvider(detection.className));

    return GestureDetector(
      onTap: () => ref.read(mapStateProvider.notifier).toggleSOIVisibility(detection.className),
      onLongPress: () => _showCaptureDialog(context, ref),
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
        decoration: BoxDecoration(
          color: isSelected ? color.withOpacity(0.15) : Colors.transparent,
          border: const Border(bottom: BorderSide(color: G20Colors.cardDark, width: 0.5)),
        ),
        child: Row(
          children: [
            // Eye visibility toggle
            GestureDetector(
              onTap: () => ref.read(mapStateProvider.notifier).toggleSOIVisibility(detection.className),
              child: Icon(
                isVisible ? Icons.visibility : Icons.visibility_off,
                size: 16,
                color: isVisible ? color : G20Colors.textSecondaryDark.withOpacity(0.4),
              ),
            ),
            const SizedBox(width: 4),
            // Color indicator
            Container(
              width: 8,
              height: 8,
              decoration: BoxDecoration(
                color: isVisible ? color : color.withOpacity(0.3),
                shape: BoxShape.circle,
              ),
            ),
            const SizedBox(width: 8),
            // Name
            Expanded(
              flex: 5,  // Match header
              child: Text(
                detection.className,
                style: TextStyle(
                  fontSize: 12,
                  fontWeight: isSelected ? FontWeight.bold : FontWeight.normal,
                  color: isVisible ? color : color.withOpacity(0.4),
                ),
                overflow: TextOverflow.ellipsis,
              ),
            ),
            // DTG (Date Time Group)
            Expanded(
              flex: 3,  // Reduced to give MGRS more space
              child: Text(
                formatDTG(detection.timestamp),
                style: TextStyle(
                  fontSize: 10,
                  color: isVisible ? G20Colors.textSecondaryDark : G20Colors.textSecondaryDark.withOpacity(0.4),
                  fontFamily: 'monospace',
                ),
                maxLines: 1,
                overflow: TextOverflow.ellipsis,
              ),
            ),
            // MGRS (device location)
            Expanded(
              flex: 5,  // Increased to fit 15-char MGRS
              child: Text(
                detection.mgrsLocation,
                style: TextStyle(
                  fontSize: 10,
                  color: isVisible ? G20Colors.textSecondaryDark : G20Colors.textSecondaryDark.withOpacity(0.4),
                  fontFamily: 'monospace',
                ),
                maxLines: 1,
                overflow: TextOverflow.ellipsis,
              ),
            ),
          ],
        ),
      ),
    );
  }

  /// Show capture/labeling dialog for retraining
  void _showCaptureDialog(BuildContext context, WidgetRef ref) {
    final color = getSOIColor(detection.className);

    showDialog(
      context: context,
      builder: (context) => _CaptureDialog(
        detection: detection,
        color: color,
      ),
    );
  }
}

/// Capture dialog with duration selection - SAME WORKFLOW FOR ALL SIGNALS
class _CaptureDialog extends ConsumerStatefulWidget {
  final Detection detection;
  final Color color;

  const _CaptureDialog({
    required this.detection,
    required this.color,
  });

  @override
  ConsumerState<_CaptureDialog> createState() => _CaptureDialogState();
}

class _CaptureDialogState extends ConsumerState<_CaptureDialog> {
  int _durationMinutes = 5;  // Default 5 min

  static const _durations = [1, 2, 5, 10];  // minutes

  @override
  Widget build(BuildContext context) {
    return AlertDialog(
      backgroundColor: G20Colors.surfaceDark,
      title: Row(
        children: [
          Container(
            width: 12, height: 12,
            decoration: BoxDecoration(color: widget.color, shape: BoxShape.circle),
          ),
          const SizedBox(width: 8),
          Expanded(
            child: Text(
              'Capture: ${widget.detection.className}',
              style: const TextStyle(fontSize: 14, color: G20Colors.textPrimaryDark),
              overflow: TextOverflow.ellipsis,
            ),
          ),
        ],
      ),
      content: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Frequency
          Text(
            'Freq: ${widget.detection.freqMHz.toStringAsFixed(2)} MHz',
            style: const TextStyle(fontSize: 12, color: G20Colors.textSecondaryDark),
          ),
          const SizedBox(height: 16),

          // Duration selector
          const Text('Duration:', style: TextStyle(fontSize: 12, color: G20Colors.textPrimaryDark)),
          const SizedBox(height: 8),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceEvenly,
            children: _durations.map((d) => _DurationChip(
              minutes: d,
              isSelected: _durationMinutes == d,
              onTap: () => setState(() => _durationMinutes = d),
            )).toList(),
          ),

          const SizedBox(height: 16),
          Container(
            padding: const EdgeInsets.all(8),
            decoration: BoxDecoration(
              color: G20Colors.primary.withOpacity(0.1),
              borderRadius: BorderRadius.circular(6),
              border: Border.all(color: G20Colors.primary.withOpacity(0.3)),
            ),
            child: const Text(
              'Swipe on waterfall to select capture bandwidth',
              style: TextStyle(fontSize: 11, color: G20Colors.textSecondaryDark),
            ),
          ),
        ],
      ),
      actions: [
        TextButton(
          onPressed: () => Navigator.pop(context),
          child: const Text('Cancel', style: TextStyle(color: G20Colors.textSecondaryDark)),
        ),
        ElevatedButton(
          onPressed: () => _startCapture(context),
          style: ElevatedButton.styleFrom(backgroundColor: G20Colors.primary),
          child: const Text('Select & Draw'),
        ),
      ],
    );
  }

  void _startCapture(BuildContext context) {
    Navigator.pop(context);

    // Start drawing mode with selected duration
    ref.read(manualCaptureProvider.notifier).startDrawingMode(
      widget.detection.freqMHz.toStringAsFixed(2),
      durationMinutes: _durationMinutes,
    );
    // No snackbar - toast shows from drawing mode
  }
}

/// Duration chip button
class _DurationChip extends StatelessWidget {
  final int minutes;
  final bool isSelected;
  final VoidCallback onTap;

  const _DurationChip({
    required this.minutes,
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
          color: isSelected ? G20Colors.primary : G20Colors.cardDark,
          borderRadius: BorderRadius.circular(16),
        ),
        child: Text(
          '$minutes min',
          style: TextStyle(
            fontSize: 12,
            color: isSelected ? Colors.white : G20Colors.textSecondaryDark,
            fontWeight: isSelected ? FontWeight.bold : FontWeight.normal,
          ),
        ),
      ),
    );
  }
}
