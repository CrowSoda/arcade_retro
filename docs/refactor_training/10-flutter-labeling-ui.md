# Flutter Labeling UI

## Overview

The labeling interface must be fast (0.3-0.5s per decision) and intuitive for field use. Two primary modes: swipe (mobile) and grid (desktop).

---

## Labeling Workflow: Uncertain-Only Review

**Key insight:** User should NOT swipe through 500 crops. Auto-accept high confidence, auto-reject low confidence, show only uncertain.

### Auto-Threshold System

```
All Blob Detections (e.g., 500)
         │
         ▼
┌─────────────────────────────────┐
│   Classifier Scores All Crops    │
└─────────────────┬───────────────┘
                  │
    ┌─────────────┼─────────────┐
    ▼             ▼             ▼
conf > 0.8    0.2-0.8      conf < 0.2
AUTO-ACCEPT   REVIEW       AUTO-REJECT
  (45)         (12)          (443)
    │             │             │
    └─────────────┼─────────────┘
                  │
                  ▼
        User reviews ONLY 12 crops
```

### Summary Bar

Always show the user what's happening:

```
┌────────────────────────────────────────────┐
│  ✓ Auto-accepted: 45                       │
│  ? Need review: 12                         │
│  ✗ Auto-rejected: 443                      │
│                                            │
│  [Adjust thresholds] [View all]            │
└────────────────────────────────────────────┘
```

### Configuration

```yaml
# config/crop_classifier.yaml
labeling:
  auto_accept_threshold: 0.8   # Above this = auto-accept
  auto_reject_threshold: 0.2   # Below this = auto-reject
  # Only show crops between 0.2 and 0.8 for human review
```

---

## Swipe Interface (Mobile/Field)

The Tinder-style swipe pattern is optimal for mobile field devices:
- Innate gesture (natural hand motion)
- Fast decisions (0.3-0.5s per item)
- Familiar UX pattern

### Implementation

```dart
// pubspec.yaml dependencies:
// flutter_card_swiper: ^5.0.0

import 'package:flutter/material.dart';
import 'package:flutter_card_swiper/flutter_card_swiper.dart';

class SwipeLabelingScreen extends StatefulWidget {
  final List<CropData> crops;
  final Function(String cropId, bool isSignal) onLabel;
  final Function(String cropId) onSkip;

  const SwipeLabelingScreen({
    required this.crops,
    required this.onLabel,
    required this.onSkip,
    super.key,
  });

  @override
  State<SwipeLabelingScreen> createState() => _SwipeLabelingScreenState();
}

class _SwipeLabelingScreenState extends State<SwipeLabelingScreen> {
  final CardSwiperController _controller = CardSwiperController();
  int _currentIndex = 0;
  int _labeledCount = 0;
  final int _targetLabels = 25;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Label ${_currentIndex + 1}/${widget.crops.length}'),
        actions: [
          // Progress indicator
          Padding(
            padding: const EdgeInsets.all(8.0),
            child: CircularProgressIndicator(
              value: _labeledCount / _targetLabels,
              backgroundColor: Colors.grey[300],
              valueColor: AlwaysStoppedAnimation<Color>(
                _labeledCount >= _targetLabels ? Colors.green : Colors.blue,
              ),
            ),
          ),
        ],
      ),
      body: Column(
        children: [
          // Model confidence badge
          if (widget.crops.isNotEmpty)
            _ConfidenceBadge(
              confidence: widget.crops[_currentIndex].modelConfidence,
            ),

          // Swipe cards
          Expanded(
            child: CardSwiper(
              controller: _controller,
              cardsCount: widget.crops.length,
              numberOfCardsDisplayed: 3,
              backCardOffset: const Offset(20, 20),
              padding: const EdgeInsets.all(24.0),
              onSwipe: _onSwipe,
              onUndo: _onUndo,
              cardBuilder: (context, index, horizontalOffset, verticalOffset) {
                return _CropCard(
                  crop: widget.crops[index],
                  horizontalOffset: horizontalOffset,
                );
              },
            ),
          ),

          // Button fallbacks + legend
          _SwipeControls(
            onReject: () => _controller.swipe(CardSwiperDirection.left),
            onSkip: () => _controller.swipe(CardSwiperDirection.top),
            onAccept: () => _controller.swipe(CardSwiperDirection.right),
            onUndo: _controller.undo,
          ),

          const SizedBox(height: 16),
        ],
      ),
    );
  }

  bool _onSwipe(
    int previousIndex,
    int? currentIndex,
    CardSwiperDirection direction,
  ) {
    final crop = widget.crops[previousIndex];

    switch (direction) {
      case CardSwiperDirection.right:
        widget.onLabel(crop.id, true);  // Signal
        _labeledCount++;
        break;
      case CardSwiperDirection.left:
        widget.onLabel(crop.id, false); // Not signal
        _labeledCount++;
        break;
      case CardSwiperDirection.top:
        widget.onSkip(crop.id);         // Unsure
        break;
      default:
        return false;
    }

    setState(() {
      _currentIndex = currentIndex ?? _currentIndex;
    });

    return true;
  }

  bool _onUndo(
    int? previousIndex,
    int currentIndex,
    CardSwiperDirection direction,
  ) {
    // Allow undo
    setState(() {
      _currentIndex = currentIndex;
      if (direction != CardSwiperDirection.top) {
        _labeledCount = (_labeledCount - 1).clamp(0, _targetLabels);
      }
    });
    return true;
  }
}


class _ConfidenceBadge extends StatelessWidget {
  final double? confidence;

  const _ConfidenceBadge({this.confidence});

  @override
  Widget build(BuildContext context) {
    if (confidence == null) return const SizedBox.shrink();

    final color = confidence! > 0.7
        ? Colors.green
        : confidence! > 0.4
            ? Colors.orange
            : Colors.red;

    final label = confidence! > 0.7
        ? 'Model thinks: SIGNAL'
        : confidence! > 0.4
            ? 'Model unsure'
            : 'Model thinks: NOT SIGNAL';

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      margin: const EdgeInsets.all(8),
      decoration: BoxDecoration(
        color: color.withOpacity(0.1),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: color),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(Icons.psychology, color: color, size: 20),
          const SizedBox(width: 8),
          Text(
            '$label (${(confidence! * 100).toInt()}%)',
            style: TextStyle(color: color, fontWeight: FontWeight.bold),
          ),
        ],
      ),
    );
  }
}


class _CropCard extends StatelessWidget {
  final CropData crop;
  final double horizontalOffset;

  const _CropCard({required this.crop, required this.horizontalOffset});

  @override
  Widget build(BuildContext context) {
    // Color tint based on swipe direction
    final overlayColor = horizontalOffset > 0
        ? Colors.green.withOpacity(horizontalOffset.abs() * 0.003)
        : Colors.red.withOpacity(horizontalOffset.abs() * 0.003);

    return Card(
      elevation: 8,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      child: Stack(
        children: [
          // Crop image
          ClipRRect(
            borderRadius: BorderRadius.circular(16),
            child: Image.memory(
              crop.imageBytes,
              fit: BoxFit.contain,
              width: double.infinity,
              height: double.infinity,
            ),
          ),

          // Direction overlay
          Container(
            decoration: BoxDecoration(
              borderRadius: BorderRadius.circular(16),
              color: overlayColor,
            ),
          ),

          // Swipe indicators
          if (horizontalOffset > 50)
            Positioned(
              top: 20,
              left: 20,
              child: _SwipeIndicator(
                text: 'SIGNAL',
                color: Colors.green,
                rotation: -0.3,
              ),
            ),
          if (horizontalOffset < -50)
            Positioned(
              top: 20,
              right: 20,
              child: _SwipeIndicator(
                text: 'NOT SIGNAL',
                color: Colors.red,
                rotation: 0.3,
              ),
            ),
        ],
      ),
    );
  }
}


class _SwipeIndicator extends StatelessWidget {
  final String text;
  final Color color;
  final double rotation;

  const _SwipeIndicator({
    required this.text,
    required this.color,
    required this.rotation,
  });

  @override
  Widget build(BuildContext context) {
    return Transform.rotate(
      angle: rotation,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
        decoration: BoxDecoration(
          border: Border.all(color: color, width: 3),
          borderRadius: BorderRadius.circular(8),
        ),
        child: Text(
          text,
          style: TextStyle(
            color: color,
            fontSize: 24,
            fontWeight: FontWeight.bold,
          ),
        ),
      ),
    );
  }
}


class _SwipeControls extends StatelessWidget {
  final VoidCallback onReject;
  final VoidCallback onSkip;
  final VoidCallback onAccept;
  final VoidCallback onUndo;

  const _SwipeControls({
    required this.onReject,
    required this.onSkip,
    required this.onAccept,
    required this.onUndo,
  });

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 24),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceEvenly,
        children: [
          // Undo
          IconButton(
            onPressed: onUndo,
            icon: const Icon(Icons.undo),
            tooltip: 'Undo',
          ),

          // Reject (Not Signal)
          FloatingActionButton(
            heroTag: 'reject',
            onPressed: onReject,
            backgroundColor: Colors.red[100],
            child: const Icon(Icons.close, color: Colors.red, size: 32),
          ),

          // Skip (Unsure)
          FloatingActionButton.small(
            heroTag: 'skip',
            onPressed: onSkip,
            backgroundColor: Colors.grey[300],
            child: const Icon(Icons.help_outline, color: Colors.grey),
          ),

          // Accept (Signal)
          FloatingActionButton(
            heroTag: 'accept',
            onPressed: onAccept,
            backgroundColor: Colors.green[100],
            child: const Icon(Icons.check, color: Colors.green, size: 32),
          ),

          // Placeholder for symmetry
          const SizedBox(width: 48),
        ],
      ),
    );
  }
}
```

---

## Grid Interface (Desktop/Tablet)

Show 9-15 crops simultaneously for faster batch review.

```dart
class GridLabelingScreen extends StatefulWidget {
  final List<CropData> crops;
  final Function(Map<String, bool> labels) onBatchConfirm;

  const GridLabelingScreen({
    required this.crops,
    required this.onBatchConfirm,
    super.key,
  });

  @override
  State<GridLabelingScreen> createState() => _GridLabelingScreenState();
}

class _GridLabelingScreenState extends State<GridLabelingScreen> {
  final Map<String, bool?> _labels = {};  // null = unset

  @override
  void initState() {
    super.initState();
    // Pre-fill with model predictions
    for (final crop in widget.crops) {
      if (crop.modelConfidence != null) {
        _labels[crop.id] = crop.modelConfidence! > 0.5;
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Review Batch'),
        actions: [
          TextButton.icon(
            onPressed: _confirmBatch,
            icon: const Icon(Icons.check),
            label: Text('Confirm (${_labels.values.whereType<bool>().length})'),
          ),
        ],
      ),
      body: GridView.builder(
        padding: const EdgeInsets.all(16),
        gridDelegate: const SliverGridDelegateWithMaxCrossAxisExtent(
          maxCrossAxisExtent: 200,
          crossAxisSpacing: 12,
          mainAxisSpacing: 12,
        ),
        itemCount: widget.crops.length,
        itemBuilder: (context, index) {
          final crop = widget.crops[index];
          final label = _labels[crop.id];

          return _GridCropCard(
            crop: crop,
            label: label,
            onTap: () => _cycleLabel(crop.id),
          );
        },
      ),
      bottomNavigationBar: _GridLegend(),
    );
  }

  void _cycleLabel(String cropId) {
    setState(() {
      final current = _labels[cropId];
      // Cycle: null → true → false → null
      if (current == null) {
        _labels[cropId] = true;
      } else if (current == true) {
        _labels[cropId] = false;
      } else {
        _labels[cropId] = null;
      }
    });
  }

  void _confirmBatch() {
    final confirmed = Map<String, bool>.fromEntries(
      _labels.entries
          .where((e) => e.value != null)
          .map((e) => MapEntry(e.key, e.value!)),
    );

    if (confirmed.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Label at least one crop')),
      );
      return;
    }

    widget.onBatchConfirm(confirmed);
  }
}


class _GridCropCard extends StatelessWidget {
  final CropData crop;
  final bool? label;
  final VoidCallback onTap;

  const _GridCropCard({
    required this.crop,
    required this.label,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    final borderColor = label == true
        ? Colors.green
        : label == false
            ? Colors.red
            : Colors.grey[400]!;

    final borderWidth = label != null ? 4.0 : 1.0;

    return GestureDetector(
      onTap: onTap,
      child: Container(
        decoration: BoxDecoration(
          border: Border.all(color: borderColor, width: borderWidth),
          borderRadius: BorderRadius.circular(8),
        ),
        child: Stack(
          children: [
            // Crop image
            ClipRRect(
              borderRadius: BorderRadius.circular(6),
              child: Image.memory(
                crop.imageBytes,
                fit: BoxFit.cover,
                width: double.infinity,
                height: double.infinity,
              ),
            ),

            // Label indicator
            if (label != null)
              Positioned(
                top: 4,
                right: 4,
                child: Container(
                  padding: const EdgeInsets.all(4),
                  decoration: BoxDecoration(
                    color: label! ? Colors.green : Colors.red,
                    shape: BoxShape.circle,
                  ),
                  child: Icon(
                    label! ? Icons.check : Icons.close,
                    color: Colors.white,
                    size: 16,
                  ),
                ),
              ),

            // Model confidence (faded if user overrode)
            if (crop.modelConfidence != null)
              Positioned(
                bottom: 4,
                left: 4,
                child: Container(
                  padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                  decoration: BoxDecoration(
                    color: Colors.black54,
                    borderRadius: BorderRadius.circular(10),
                  ),
                  child: Text(
                    '${(crop.modelConfidence! * 100).toInt()}%',
                    style: const TextStyle(
                      color: Colors.white,
                      fontSize: 11,
                    ),
                  ),
                ),
              ),
          ],
        ),
      ),
    );
  }
}


class _GridLegend extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(16),
      color: Colors.grey[100],
      child: Row(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          _LegendItem(color: Colors.green, label: 'Signal'),
          const SizedBox(width: 24),
          _LegendItem(color: Colors.red, label: 'Not Signal'),
          const SizedBox(width: 24),
          _LegendItem(color: Colors.grey, label: 'Unset (tap to cycle)'),
        ],
      ),
    );
  }
}

class _LegendItem extends StatelessWidget {
  final Color color;
  final String label;

  const _LegendItem({required this.color, required this.label});

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        Container(
          width: 16,
          height: 16,
          decoration: BoxDecoration(
            border: Border.all(color: color, width: 3),
            borderRadius: BorderRadius.circular(4),
          ),
        ),
        const SizedBox(width: 8),
        Text(label),
      ],
    );
  }
}
```

---

## Data Models

```dart
class CropData {
  final String id;
  final Uint8List imageBytes;
  final double? modelConfidence;
  final Map<String, dynamic>? metadata;

  const CropData({
    required this.id,
    required this.imageBytes,
    this.modelConfidence,
    this.metadata,
  });
}
```

---

## Session Management

```dart
class LabelingSession {
  final List<CropData> _allCrops;
  final Map<String, bool> _labels = {};
  final List<String> _skipped = [];
  int _batchIndex = 0;
  final int _batchSize;

  LabelingSession({
    required List<CropData> crops,
    int batchSize = 10,
  })  : _allCrops = crops,
        _batchSize = batchSize;

  List<CropData> getNextBatch() {
    final start = _batchIndex * _batchSize;
    final end = (start + _batchSize).clamp(0, _allCrops.length);
    return _allCrops.sublist(start, end);
  }

  void recordLabels(Map<String, bool> labels) {
    _labels.addAll(labels);
    _batchIndex++;
  }

  void recordSkipped(String id) {
    _skipped.add(id);
  }

  bool get hasMoreBatches => _batchIndex * _batchSize < _allCrops.length;

  int get totalLabeled => _labels.length;

  Map<String, bool> get allLabels => Map.unmodifiable(_labels);
}
```
