import 'dart:typed_data';
import 'package:flutter/material.dart';

/// Data model for a crop to be reviewed
class CropReviewData {
  final String id;
  final Uint8List imageBytes;
  final double? modelConfidence;
  final Map<String, dynamic>? metadata;

  const CropReviewData({
    required this.id,
    required this.imageBytes,
    this.modelConfidence,
    this.metadata,
  });
}

/// Result of crop review
class CropReviewResult {
  final Map<String, bool> labels; // id -> isSignal
  final List<String> skipped;
  final int autoAccepted;
  final int autoRejected;

  const CropReviewResult({
    required this.labels,
    required this.skipped,
    required this.autoAccepted,
    required this.autoRejected,
  });

  int get totalLabeled => labels.length;
  int get signalCount => labels.values.where((v) => v).length;
  int get notSignalCount => labels.values.where((v) => !v).length;
}

/// Shows crop review popup dialog
/// Returns null if cancelled, CropReviewResult if completed
Future<CropReviewResult?> showCropReviewDialog({
  required BuildContext context,
  required List<CropReviewData> crops,
  double autoAcceptThreshold = 0.8,
  double autoRejectThreshold = 0.2,
  int targetLabels = 25,
}) {
  return showDialog<CropReviewResult>(
    context: context,
    barrierDismissible: false,
    builder: (context) => CropReviewDialog(
      crops: crops,
      autoAcceptThreshold: autoAcceptThreshold,
      autoRejectThreshold: autoRejectThreshold,
      targetLabels: targetLabels,
    ),
  );
}

/// Crop review dialog widget
class CropReviewDialog extends StatefulWidget {
  final List<CropReviewData> crops;
  final double autoAcceptThreshold;
  final double autoRejectThreshold;
  final int targetLabels;

  const CropReviewDialog({
    required this.crops,
    this.autoAcceptThreshold = 0.8,
    this.autoRejectThreshold = 0.2,
    this.targetLabels = 25,
    super.key,
  });

  @override
  State<CropReviewDialog> createState() => _CropReviewDialogState();
}

class _CropReviewDialogState extends State<CropReviewDialog> {
  late List<CropReviewData> _reviewCrops;
  late List<CropReviewData> _autoAccepted;
  late List<CropReviewData> _autoRejected;

  final Map<String, bool> _labels = {};
  final List<String> _skipped = [];
  final List<_UndoAction> _undoStack = [];

  int _currentIndex = 0;

  @override
  void initState() {
    super.initState();
    _categorizeCrops();
  }

  void _categorizeCrops() {
    _reviewCrops = [];
    _autoAccepted = [];
    _autoRejected = [];

    for (final crop in widget.crops) {
      if (crop.modelConfidence == null) {
        _reviewCrops.add(crop);
      } else if (crop.modelConfidence! >= widget.autoAcceptThreshold) {
        _autoAccepted.add(crop);
      } else if (crop.modelConfidence! <= widget.autoRejectThreshold) {
        _autoRejected.add(crop);
      } else {
        _reviewCrops.add(crop);
      }
    }
  }

  bool get _hasMore => _currentIndex < _reviewCrops.length;
  CropReviewData? get _currentCrop =>
      _hasMore ? _reviewCrops[_currentIndex] : null;
  double get _progress => _labels.length / widget.targetLabels;
  bool get _targetReached => _labels.length >= widget.targetLabels;

  void _label(bool isSignal) {
    if (!_hasMore) return;
    final crop = _currentCrop!;
    _undoStack.add(_UndoAction(
      cropId: crop.id,
      index: _currentIndex,
      wasLabel: true,
    ));
    _labels[crop.id] = isSignal;
    setState(() => _currentIndex++);
  }

  void _skip() {
    if (!_hasMore) return;
    final crop = _currentCrop!;
    _undoStack.add(_UndoAction(
      cropId: crop.id,
      index: _currentIndex,
      wasLabel: false,
    ));
    _skipped.add(crop.id);
    setState(() => _currentIndex++);
  }

  void _undo() {
    if (_undoStack.isEmpty) return;
    final action = _undoStack.removeLast();
    if (action.wasLabel) {
      _labels.remove(action.cropId);
    } else {
      _skipped.remove(action.cropId);
    }
    setState(() => _currentIndex = action.index);
  }

  void _finish() {
    Navigator.of(context).pop(CropReviewResult(
      labels: Map.unmodifiable(_labels),
      skipped: List.unmodifiable(_skipped),
      autoAccepted: _autoAccepted.length,
      autoRejected: _autoRejected.length,
    ));
  }

  void _cancel() {
    Navigator.of(context).pop(null);
  }

  @override
  Widget build(BuildContext context) {
    return Dialog(
      child: Container(
        width: 500,
        height: 600,
        padding: const EdgeInsets.all(16),
        child: Column(
          children: [
            // Header
            _buildHeader(),
            const SizedBox(height: 8),

            // Summary bar
            _buildSummaryBar(),
            const SizedBox(height: 16),

            // Main content
            Expanded(child: _buildContent()),

            // Controls
            const SizedBox(height: 16),
            _buildControls(),
          ],
        ),
      ),
    );
  }

  Widget _buildHeader() {
    return Row(
      children: [
        const Icon(Icons.rate_review, size: 24),
        const SizedBox(width: 8),
        const Text(
          'Review Uncertain Crops',
          style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
        ),
        const Spacer(),
        // Progress indicator
        SizedBox(
          width: 40,
          height: 40,
          child: Stack(
            alignment: Alignment.center,
            children: [
              CircularProgressIndicator(
                value: _progress.clamp(0.0, 1.0),
                backgroundColor: Colors.grey[300],
                valueColor: AlwaysStoppedAnimation<Color>(
                  _targetReached ? Colors.green : Colors.blue,
                ),
                strokeWidth: 4,
              ),
              Text(
                '${_labels.length}',
                style: const TextStyle(fontSize: 12, fontWeight: FontWeight.bold),
              ),
            ],
          ),
        ),
        IconButton(
          icon: const Icon(Icons.close),
          onPressed: _cancel,
          tooltip: 'Cancel',
        ),
      ],
    );
  }

  Widget _buildSummaryBar() {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        color: Colors.grey[100],
        borderRadius: BorderRadius.circular(8),
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceAround,
        children: [
          _SummaryItem(
            icon: Icons.check_circle,
            color: Colors.green,
            label: 'Auto-accepted',
            count: _autoAccepted.length,
          ),
          _SummaryItem(
            icon: Icons.help_outline,
            color: Colors.orange,
            label: 'Need review',
            count: _reviewCrops.length - _currentIndex,
          ),
          _SummaryItem(
            icon: Icons.cancel,
            color: Colors.red,
            label: 'Auto-rejected',
            count: _autoRejected.length,
          ),
        ],
      ),
    );
  }

  Widget _buildContent() {
    if (!_hasMore) {
      return _buildComplete();
    }

    final crop = _currentCrop!;
    return Column(
      children: [
        // Confidence badge
        if (crop.modelConfidence != null)
          _ConfidenceBadge(confidence: crop.modelConfidence!),
        const SizedBox(height: 8),

        // Crop image
        Expanded(
          child: _CropCard(
            crop: crop,
            onSwipeLeft: () => _label(false),
            onSwipeRight: () => _label(true),
            onSwipeUp: _skip,
          ),
        ),

        // Counter
        Text(
          '${_currentIndex + 1} / ${_reviewCrops.length}',
          style: TextStyle(color: Colors.grey[600]),
        ),
      ],
    );
  }

  Widget _buildComplete() {
    return Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(
            _targetReached ? Icons.check_circle : Icons.pending,
            size: 64,
            color: _targetReached ? Colors.green : Colors.orange,
          ),
          const SizedBox(height: 16),
          Text(
            _targetReached ? 'Target reached!' : 'Review complete',
            style: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
          ),
          const SizedBox(height: 8),
          Text(
            '${_labels.length} labeled (${_labels.values.where((v) => v).length} signal, '
            '${_labels.values.where((v) => !v).length} not signal)',
            style: TextStyle(color: Colors.grey[600]),
          ),
        ],
      ),
    );
  }

  Widget _buildControls() {
    return Column(
      children: [
        // Swipe buttons
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceEvenly,
          children: [
            // Undo
            IconButton(
              onPressed: _undoStack.isNotEmpty ? _undo : null,
              icon: const Icon(Icons.undo),
              tooltip: 'Undo',
            ),

            // Not Signal (swipe left)
            FloatingActionButton(
              heroTag: 'reject',
              onPressed: _hasMore ? () => _label(false) : null,
              backgroundColor: Colors.red[100],
              child: const Icon(Icons.close, color: Colors.red, size: 32),
            ),

            // Skip (swipe up)
            FloatingActionButton.small(
              heroTag: 'skip',
              onPressed: _hasMore ? _skip : null,
              backgroundColor: Colors.grey[300],
              child: const Icon(Icons.help_outline, color: Colors.grey),
            ),

            // Signal (swipe right)
            FloatingActionButton(
              heroTag: 'accept',
              onPressed: _hasMore ? () => _label(true) : null,
              backgroundColor: Colors.green[100],
              child: const Icon(Icons.check, color: Colors.green, size: 32),
            ),

            const SizedBox(width: 48),
          ],
        ),
        const SizedBox(height: 8),

        // Legend
        Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            _LegendItem(
                icon: Icons.close, color: Colors.red, label: '← Not Signal'),
            const SizedBox(width: 16),
            _LegendItem(
                icon: Icons.help_outline, color: Colors.grey, label: '↑ Skip'),
            const SizedBox(width: 16),
            _LegendItem(
                icon: Icons.check, color: Colors.green, label: 'Signal →'),
          ],
        ),
        const SizedBox(height: 16),

        // Done button
        SizedBox(
          width: double.infinity,
          child: ElevatedButton.icon(
            onPressed: _labels.isNotEmpty ? _finish : null,
            icon: const Icon(Icons.done),
            label: Text('Done (${_labels.length} labeled)'),
            style: ElevatedButton.styleFrom(
              padding: const EdgeInsets.symmetric(vertical: 12),
              backgroundColor: _targetReached ? Colors.green : null,
            ),
          ),
        ),
      ],
    );
  }
}

class _UndoAction {
  final String cropId;
  final int index;
  final bool wasLabel;

  _UndoAction({
    required this.cropId,
    required this.index,
    required this.wasLabel,
  });
}

class _SummaryItem extends StatelessWidget {
  final IconData icon;
  final Color color;
  final String label;
  final int count;

  const _SummaryItem({
    required this.icon,
    required this.color,
    required this.label,
    required this.count,
  });

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(icon, color: color, size: 16),
            const SizedBox(width: 4),
            Text(
              '$count',
              style: TextStyle(
                color: color,
                fontWeight: FontWeight.bold,
                fontSize: 16,
              ),
            ),
          ],
        ),
        Text(label, style: TextStyle(fontSize: 11, color: Colors.grey[600])),
      ],
    );
  }
}

class _ConfidenceBadge extends StatelessWidget {
  final double confidence;

  const _ConfidenceBadge({required this.confidence});

  @override
  Widget build(BuildContext context) {
    final color = confidence > 0.7
        ? Colors.green
        : confidence > 0.4
            ? Colors.orange
            : Colors.red;

    final label = confidence > 0.7
        ? 'Model thinks: SIGNAL'
        : confidence > 0.4
            ? 'Model unsure'
            : 'Model thinks: NOT SIGNAL';

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.1),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: color),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(Icons.psychology, color: color, size: 18),
          const SizedBox(width: 6),
          Text(
            '$label (${(confidence * 100).toInt()}%)',
            style: TextStyle(
                color: color, fontWeight: FontWeight.bold, fontSize: 13),
          ),
        ],
      ),
    );
  }
}

class _CropCard extends StatefulWidget {
  final CropReviewData crop;
  final VoidCallback onSwipeLeft;
  final VoidCallback onSwipeRight;
  final VoidCallback onSwipeUp;

  const _CropCard({
    required this.crop,
    required this.onSwipeLeft,
    required this.onSwipeRight,
    required this.onSwipeUp,
  });

  @override
  State<_CropCard> createState() => _CropCardState();
}

class _CropCardState extends State<_CropCard> {
  double _dragX = 0;
  double _dragY = 0;

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onPanUpdate: (details) {
        setState(() {
          _dragX += details.delta.dx;
          _dragY += details.delta.dy;
        });
      },
      onPanEnd: (details) {
        if (_dragX > 80) {
          widget.onSwipeRight();
        } else if (_dragX < -80) {
          widget.onSwipeLeft();
        } else if (_dragY < -80) {
          widget.onSwipeUp();
        }
        setState(() {
          _dragX = 0;
          _dragY = 0;
        });
      },
      child: Transform.translate(
        offset: Offset(_dragX, _dragY),
        child: Transform.rotate(
          angle: _dragX * 0.001,
          child: Card(
            elevation: 8,
            shape:
                RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
            child: Stack(
              children: [
                // Crop image
                ClipRRect(
                  borderRadius: BorderRadius.circular(12),
                  child: Image.memory(
                    widget.crop.imageBytes,
                    fit: BoxFit.contain,
                    width: double.infinity,
                    height: double.infinity,
                  ),
                ),

                // Swipe indicators
                if (_dragX > 50)
                  Positioned(
                    top: 20,
                    left: 20,
                    child: _buildIndicator('SIGNAL', Colors.green),
                  ),
                if (_dragX < -50)
                  Positioned(
                    top: 20,
                    right: 20,
                    child: _buildIndicator('NOT SIGNAL', Colors.red),
                  ),
                if (_dragY < -50)
                  Positioned(
                    top: 20,
                    left: 0,
                    right: 0,
                    child: Center(child: _buildIndicator('SKIP', Colors.grey)),
                  ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildIndicator(String text, Color color) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
      decoration: BoxDecoration(
        border: Border.all(color: color, width: 3),
        borderRadius: BorderRadius.circular(6),
        color: Colors.white,
      ),
      child: Text(
        text,
        style: TextStyle(
          color: color,
          fontSize: 16,
          fontWeight: FontWeight.bold,
        ),
      ),
    );
  }
}

class _LegendItem extends StatelessWidget {
  final IconData icon;
  final Color color;
  final String label;

  const _LegendItem({
    required this.icon,
    required this.color,
    required this.label,
  });

  @override
  Widget build(BuildContext context) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Icon(icon, color: color, size: 14),
        const SizedBox(width: 4),
        Text(label, style: TextStyle(fontSize: 11, color: Colors.grey[600])),
      ],
    );
  }
}
