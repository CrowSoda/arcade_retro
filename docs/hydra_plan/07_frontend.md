# Phase 6: Flutter Frontend

## New Providers

### `lib/features/training/providers/training_provider.dart`

```dart
import 'dart:async';
import 'package:flutter_riverpod/flutter_riverpod.dart';

final trainingProvider = StateNotifierProvider<TrainingNotifier, TrainingState>((ref) {
  return TrainingNotifier(ref);
});

class TrainingState {
  final bool isTraining;
  final String? currentSignal;
  final int epoch;
  final int totalEpochs;
  final double trainLoss;
  final double valLoss;
  final double f1Score;
  final double precision;
  final double recall;
  final bool isBest;
  final Duration elapsed;
  final TrainingResult? lastResult;
  final String? error;

  const TrainingState({
    this.isTraining = false,
    this.currentSignal,
    this.epoch = 0,
    this.totalEpochs = 50,
    this.trainLoss = 0,
    this.valLoss = 0,
    this.f1Score = 0,
    this.precision = 0,
    this.recall = 0,
    this.isBest = false,
    this.elapsed = Duration.zero,
    this.lastResult,
    this.error,
  });

  double get progress => totalEpochs > 0 ? epoch / totalEpochs : 0;
  
  TrainingState copyWith({...}) => TrainingState(...);
}

class TrainingResult {
  final String signalName;
  final int version;
  final int sampleCount;
  final int epochsTrained;
  final bool earlyStopped;
  final Map<String, double> metrics;
  final double trainingTimeSec;
  final int? previousVersion;
  final Map<String, double>? previousMetrics;
  final bool autoPromoted;
  final String? promotionReason;
}

class TrainingNotifier extends StateNotifier<TrainingState> {
  final Ref _ref;
  StreamSubscription? _progressSub;
  
  TrainingNotifier(this._ref) : super(const TrainingState());
  
  Future<void> trainNewSignal(String signalName, {String? notes}) async {
    state = state.copyWith(isTraining: true, currentSignal: signalName, error: null);
    
    final ws = _ref.read(webSocketProvider);
    ws.send({'command': 'train_signal', 'signal_name': signalName, 'notes': notes});
    
    _listenForProgress();
  }
  
  Future<void> extendSignal(String signalName, {String? notes}) async {
    state = state.copyWith(isTraining: true, currentSignal: signalName, error: null);
    
    final ws = _ref.read(webSocketProvider);
    ws.send({'command': 'extend_signal', 'signal_name': signalName, 'notes': notes});
    
    _listenForProgress();
  }
  
  void cancelTraining() {
    final ws = _ref.read(webSocketProvider);
    ws.send({'command': 'cancel_training'});
  }
  
  void _listenForProgress() {
    final ws = _ref.read(webSocketProvider);
    _progressSub = ws.messages.listen((msg) {
      if (msg['type'] == 'training_progress') {
        _handleProgress(msg);
      } else if (msg['type'] == 'training_complete') {
        _handleComplete(msg);
      } else if (msg['type'] == 'training_failed') {
        _handleFailed(msg);
      }
    });
  }
  
  void _handleProgress(Map<String, dynamic> data) {
    state = state.copyWith(
      epoch: data['epoch'],
      totalEpochs: data['total_epochs'],
      trainLoss: data['train_loss'],
      valLoss: data['val_loss'],
      f1Score: data['f1_score'],
      precision: data['precision'] ?? 0,
      recall: data['recall'] ?? 0,
      isBest: data['is_best'],
      elapsed: Duration(seconds: (data['elapsed_sec'] as num).toInt()),
    );
  }
  
  void _handleComplete(Map<String, dynamic> data) {
    _progressSub?.cancel();
    
    final result = TrainingResult(
      signalName: data['signal_name'],
      version: data['version'],
      sampleCount: data['sample_count'],
      epochsTrained: data['epochs_trained'],
      earlyStopped: data['early_stopped'],
      metrics: Map<String, double>.from(data['metrics']),
      trainingTimeSec: data['training_time_sec'],
      previousVersion: data['previous_version'],
      previousMetrics: data['previous_metrics'] != null 
          ? Map<String, double>.from(data['previous_metrics']) : null,
      autoPromoted: data['auto_promoted'],
      promotionReason: data['promotion_reason'],
    );
    
    state = state.copyWith(isTraining: false, lastResult: result);
  }
  
  void _handleFailed(Map<String, dynamic> data) {
    _progressSub?.cancel();
    state = state.copyWith(isTraining: false, error: data['error']);
  }
}
```

### `lib/features/training/providers/signal_versions_provider.dart`

```dart
final signalVersionsProvider = StateNotifierProvider<SignalVersionsNotifier, SignalVersionsState>((ref) {
  return SignalVersionsNotifier(ref);
});

class SignalVersionsState {
  final Map<String, SignalInfo> signals;
  final bool isLoading;
  final String? selectedSignal;
  
  const SignalVersionsState({
    this.signals = const {},
    this.isLoading = false,
    this.selectedSignal,
  });
}

class SignalInfo {
  final String name;
  final int activeVersion;
  final int sampleCount;
  final double f1Score;
  final DateTime lastTrained;
  final List<VersionInfo> versions;
}

class VersionInfo {
  final int version;
  final DateTime createdAt;
  final int sampleCount;
  final double f1Score;
  final double? precision;
  final double? recall;
  final bool isActive;
  final String? notes;
  final String? promotionReason;
}

class SignalVersionsNotifier extends StateNotifier<SignalVersionsState> {
  final Ref _ref;
  
  SignalVersionsNotifier(this._ref) : super(const SignalVersionsState());
  
  Future<void> loadRegistry() async {
    state = state.copyWith(isLoading: true);
    
    final ws = _ref.read(webSocketProvider);
    ws.send({'command': 'get_registry'});
    
    // Response handled by message listener
  }
  
  Future<void> loadVersionHistory(String signalName) async {
    final ws = _ref.read(webSocketProvider);
    ws.send({'command': 'get_version_history', 'signal_name': signalName});
  }
  
  Future<void> promoteVersion(String signalName, int version) async {
    final ws = _ref.read(webSocketProvider);
    ws.send({'command': 'promote_version', 'signal_name': signalName, 'version': version});
  }
  
  Future<void> rollback(String signalName) async {
    final ws = _ref.read(webSocketProvider);
    ws.send({'command': 'rollback_signal', 'signal_name': signalName});
  }
  
  void handleRegistryResponse(Map<String, dynamic> data) {
    final signals = <String, SignalInfo>{};
    
    for (final entry in (data['signals'] as Map).entries) {
      signals[entry.key] = SignalInfo(
        name: entry.key,
        activeVersion: entry.value['active_version'],
        sampleCount: entry.value['sample_count'],
        f1Score: entry.value['f1_score'],
        lastTrained: DateTime.parse(entry.value['last_trained']),
        versions: [], // Loaded separately
      );
    }
    
    state = state.copyWith(signals: signals, isLoading: false);
  }
}
```

---

## Training Screen UI Redesign

### Layout Structure

```
┌─────────────────────────────────────────────────────────────────────┐
│  🎯 Train Signal Detector                              [Settings]   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─ LEFT: SPECTROGRAM (60%) ──────────────────────────────────┐    │
│  │  Source: [▼ MAN_024307ZJAN26_2430.rfcap]   [+ Add Files]   │    │
│  │                                                             │    │
│  │  ┌─────────────────────────────────────────────────────┐   │    │
│  │  │                                                     │   │    │
│  │  │              TRAINING SPECTROGRAM                   │   │    │
│  │  │          (existing widget unchanged)                │   │    │
│  │  │                                                     │   │    │
│  │  └─────────────────────────────────────────────────────┘   │    │
│  │                                                             │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                     │
│  ┌─ RIGHT: SIGNAL PANEL (40%) ────────────────────────────────┐    │
│  │                                                             │    │
│  │  Signal: [▼ creamy_chicken    ]   [+ New Signal]           │    │
│  │                                                             │    │
│  │  ┌─ INFO CARD ─────────────────────────────────────────┐   │    │
│  │  │  Active: v2          F1: 0.93                       │   │    │
│  │  │  Samples: 200 (train: 160, val: 40)                 │   │    │
│  │  │  Pending: 15 new samples                            │   │    │
│  │  └─────────────────────────────────────────────────────┘   │    │
│  │                                                             │    │
│  │  Notes (optional):                                         │    │
│  │  ┌─────────────────────────────────────────────────────┐   │    │
│  │  │ Added low SNR samples from urban collection         │   │    │
│  │  └─────────────────────────────────────────────────────┘   │    │
│  │                                                             │    │
│  │  [🚀 Train v3]   (215 total samples)                       │    │
│  │                                                             │    │
│  │  ─────────────────────────────────────────────────────     │    │
│  │                                                             │    │
│  │  Version History:                                          │    │
│  │  ┌─────┬───────┬───────┬──────────────────┬────────────┐   │    │
│  │  │ v2● │ 0.93  │ 200   │ "low SNR edges"  │ [Active]   │   │    │
│  │  │ v1  │ 0.91  │ 127   │ "initial"        │ [Promote]  │   │    │
│  │  └─────┴───────┴───────┴──────────────────┴────────────┘   │    │
│  │                                                             │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                     │
│  ┌─ BOTTOM: SAMPLES ────────────────────────────────────────────┐  │
│  │  Samples (215)                                    [View All]  │  │
│  │  ┌─────┬────────────┬─────────────┬─────────────┬──────────┐ │  │
│  │  │ NEW │ 0.05-0.15s │ 2429.5 MHz  │ Today 10:15 │ [View]   │ │  │
│  │  │ trn │ 0.10-0.20s │ 2429.3 MHz  │ Jan 23      │ [View]   │ │  │
│  │  │ val │ 0.50-0.60s │ 2429.6 MHz  │ Jan 20      │ [View]   │ │  │
│  │  └─────┴────────────┴─────────────┴─────────────┴──────────┘ │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### Training Progress Overlay

```dart
class TrainingProgressOverlay extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Consumer(builder: (context, ref, _) {
      final state = ref.watch(trainingProvider);
      
      if (!state.isTraining) return const SizedBox.shrink();
      
      return Container(
        color: Colors.black54,
        child: Center(
          child: Card(
            child: Padding(
              padding: const EdgeInsets.all(32),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Text('Training ${state.currentSignal}',
                    style: Theme.of(context).textTheme.headlineSmall),
                  const SizedBox(height: 24),
                  
                  // Progress bar
                  LinearProgressIndicator(value: state.progress),
                  const SizedBox(height: 8),
                  Text('Epoch ${state.epoch} / ${state.totalEpochs}'),
                  
                  const SizedBox(height: 16),
                  
                  // Metrics
                  Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      _MetricChip('Train Loss', state.trainLoss.toStringAsFixed(4)),
                      _MetricChip('Val Loss', state.valLoss.toStringAsFixed(4)),
                      _MetricChip('F1', state.f1Score.toStringAsFixed(3),
                        highlight: state.isBest),
                    ],
                  ),
                  
                  const SizedBox(height: 16),
                  Text('Elapsed: ${state.elapsed.inMinutes}m ${state.elapsed.inSeconds % 60}s'),
                  
                  const SizedBox(height: 24),
                  TextButton(
                    onPressed: () => ref.read(trainingProvider.notifier).cancelTraining(),
                    child: const Text('Cancel'),
                  ),
                ],
              ),
            ),
          ),
        ),
      );
    });
  }
}
```

### Training Complete Dialog

```dart
class TrainingCompleteDialog extends StatelessWidget {
  final TrainingResult result;
  
  @override
  Widget build(BuildContext context) {
    final improved = result.previousMetrics != null &&
        result.metrics['f1_score']! > result.previousMetrics!['f1_score']!;
    
    return AlertDialog(
      title: Row(
        children: [
          Icon(improved ? Icons.check_circle : Icons.warning,
            color: improved ? Colors.green : Colors.orange),
          const SizedBox(width: 8),
          Text(improved ? 'Training Complete!' : 'Training Complete'),
        ],
      ),
      content: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          // Comparison table
          if (result.previousMetrics != null) ...[
            _ComparisonRow('F1 Score',
              result.previousMetrics!['f1_score']!,
              result.metrics['f1_score']!),
            _ComparisonRow('Precision',
              result.previousMetrics!['precision'] ?? 0,
              result.metrics['precision']!),
            _ComparisonRow('Recall',
              result.previousMetrics!['recall'] ?? 0,
              result.metrics['recall']!),
            _ComparisonRow('Samples',
              result.previousVersion != null ? 0 : 0, // Would need prev sample count
              result.sampleCount.toDouble()),
          ],
          
          const SizedBox(height: 16),
          
          // Promotion status
          if (result.autoPromoted)
            Chip(
              avatar: const Icon(Icons.check, size: 16),
              label: Text('Auto-promoted: ${result.promotionReason}'),
              backgroundColor: Colors.green.shade100,
            )
          else
            Chip(
              avatar: const Icon(Icons.info, size: 16),
              label: const Text('v${result.version} saved, not promoted'),
              backgroundColor: Colors.orange.shade100,
            ),
        ],
      ),
      actions: [
        if (!result.autoPromoted)
          TextButton(
            onPressed: () {
              // Promote manually
              context.read(signalVersionsProvider.notifier)
                .promoteVersion(result.signalName, result.version);
              Navigator.pop(context);
            },
            child: const Text('Promote Anyway'),
          ),
        ElevatedButton(
          onPressed: () => Navigator.pop(context),
          child: const Text('Done'),
        ),
      ],
    );
  }
}
```

### Version History Widget

```dart
class VersionHistoryWidget extends StatelessWidget {
  final String signalName;
  final List<VersionInfo> versions;
  final int activeVersion;
  
  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text('Version History', style: Theme.of(context).textTheme.titleSmall),
        const SizedBox(height: 8),
        ...versions.map((v) => _VersionRow(
          version: v,
          isActive: v.version == activeVersion,
          onPromote: () => context.read(signalVersionsProvider.notifier)
            .promoteVersion(signalName, v.version),
        )),
      ],
    );
  }
}

class _VersionRow extends StatelessWidget {
  final VersionInfo version;
  final bool isActive;
  final VoidCallback onPromote;
  
  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(vertical: 8, horizontal: 12),
      decoration: BoxDecoration(
        color: isActive ? G20Colors.primary.withOpacity(0.1) : null,
        borderRadius: BorderRadius.circular(4),
      ),
      child: Row(
        children: [
          // Version badge
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
            decoration: BoxDecoration(
              color: isActive ? G20Colors.primary : G20Colors.cardDark,
              borderRadius: BorderRadius.circular(4),
            ),
            child: Text('v${version.version}',
              style: TextStyle(
                color: isActive ? Colors.white : G20Colors.textSecondaryDark,
                fontWeight: FontWeight.bold,
              )),
          ),
          
          const SizedBox(width: 12),
          
          // F1 Score
          Text('F1: ${version.f1Score.toStringAsFixed(2)}'),
          
          const SizedBox(width: 12),
          
          // Sample count
          Text('${version.sampleCount} smp',
            style: const TextStyle(color: G20Colors.textSecondaryDark)),
          
          const Spacer(),
          
          // Notes (truncated)
          if (version.notes != null)
            Expanded(
              child: Text(version.notes!,
                maxLines: 1,
                overflow: TextOverflow.ellipsis,
                style: const TextStyle(
                  color: G20Colors.textSecondaryDark,
                  fontSize: 12,
                )),
            ),
          
          // Action button
          if (isActive)
            const Chip(label: Text('Active'), backgroundColor: Colors.green)
          else
            TextButton(
              onPressed: onPromote,
              child: const Text('Promote'),
            ),
        ],
      ),
    );
  }
}
```
