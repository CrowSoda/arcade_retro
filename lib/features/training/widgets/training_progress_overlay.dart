import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../../../core/config/theme.dart';
import '../providers/training_provider.dart';

/// Overlay showing training progress
class TrainingProgressOverlay extends ConsumerWidget {
  final VoidCallback? onCancel;
  final VoidCallback? onDismiss;

  const TrainingProgressOverlay({
    super.key,
    this.onCancel,
    this.onDismiss,
  });

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final state = ref.watch(trainingProvider);

    if (!state.isTraining && state.lastResult == null) {
      return const SizedBox.shrink();
    }

    return Container(
      color: Colors.black54,
      child: Center(
        child: state.isTraining
            ? _buildProgressCard(context, ref, state)
            : _buildResultCard(context, ref, state),
      ),
    );
  }

  Widget _buildProgressCard(BuildContext context, WidgetRef ref, TrainingState state) {
    final progress = state.progress;

    return Container(
      width: 400,
      padding: const EdgeInsets.all(24),
      decoration: BoxDecoration(
        color: G20Colors.surfaceDark,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: G20Colors.primary),
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          // Header
          Row(
            children: [
              const Icon(Icons.model_training, color: G20Colors.primary, size: 24),
              const SizedBox(width: 12),
              Expanded(
                child: Text(
                  'Training ${state.currentSignal ?? "..."} v${(progress?.epoch ?? 0) > 0 ? "?" : "1"}',
                  style: const TextStyle(
                    color: G20Colors.textPrimaryDark,
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 24),

          // Progress bar
          ClipRRect(
            borderRadius: BorderRadius.circular(8),
            child: LinearProgressIndicator(
              value: progress?.progressPercent ?? 0,
              backgroundColor: G20Colors.cardDark,
              valueColor: const AlwaysStoppedAnimation(G20Colors.primary),
              minHeight: 12,
            ),
          ),
          const SizedBox(height: 8),
          Text(
            '${((progress?.progressPercent ?? 0) * 100).toInt()}%',
            style: const TextStyle(
              color: G20Colors.primary,
              fontSize: 24,
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(height: 16),

          // Stats
          if (progress != null) ...[
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                _StatBox('Epoch', '${progress.epoch}/${progress.totalEpochs}'),
                _StatBox('Train Loss', progress.trainLoss.toStringAsFixed(4)),
                _StatBox('Val Loss', progress.valLoss.toStringAsFixed(4)),
              ],
            ),
            const SizedBox(height: 12),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                _StatBox(
                  'F1',
                  '${(progress.f1Score * 100).toStringAsFixed(1)}%',
                  highlight: progress.isBest,
                ),
                _StatBox('Precision', '${(progress.precision * 100).toStringAsFixed(1)}%'),
                _StatBox('Recall', '${(progress.recall * 100).toStringAsFixed(1)}%'),
              ],
            ),
            const SizedBox(height: 12),
            if (progress.isBest)
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
                decoration: BoxDecoration(
                  color: G20Colors.success.withOpacity(0.2),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: const Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Icon(Icons.star, color: G20Colors.success, size: 16),
                    SizedBox(width: 4),
                    Text(
                      'Best so far!',
                      style: TextStyle(color: G20Colors.success, fontSize: 12, fontWeight: FontWeight.w600),
                    ),
                  ],
                ),
              ),
            const SizedBox(height: 8),
            Text(
              'Elapsed: ${_formatDuration(progress.elapsedSec)}',
              style: const TextStyle(color: G20Colors.textSecondaryDark, fontSize: 12),
            ),
          ],
          const SizedBox(height: 24),

          // Cancel button
          ElevatedButton(
            onPressed: () {
              ref.read(trainingProvider.notifier).cancelTraining();
              onCancel?.call();
            },
            style: ElevatedButton.styleFrom(
              backgroundColor: G20Colors.error,
              padding: const EdgeInsets.symmetric(horizontal: 32, vertical: 12),
            ),
            child: const Text('Cancel'),
          ),
        ],
      ),
    );
  }

  Widget _buildResultCard(BuildContext context, WidgetRef ref, TrainingState state) {
    final result = state.lastResult;
    if (result == null) return const SizedBox.shrink();

    final improved = result.f1Improvement > 0;
    final autoPromoted = result.autoPromoted;

    return Container(
      width: 450,
      padding: const EdgeInsets.all(24),
      decoration: BoxDecoration(
        color: G20Colors.surfaceDark,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: improved ? G20Colors.success : G20Colors.warning),
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          // Header
          Row(
            children: [
              Icon(
                improved ? Icons.check_circle : Icons.warning_amber,
                color: improved ? G20Colors.success : G20Colors.warning,
                size: 28,
              ),
              const SizedBox(width: 12),
              Expanded(
                child: Text(
                  improved ? 'Training Complete!' : 'Training Complete',
                  style: TextStyle(
                    color: improved ? G20Colors.success : G20Colors.warning,
                    fontSize: 20,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 20),

          // Comparison table
          Container(
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: G20Colors.backgroundDark,
              borderRadius: BorderRadius.circular(8),
            ),
            child: Column(
              children: [
                // Header row
                Row(
                  children: [
                    const Expanded(flex: 2, child: SizedBox()),
                    Expanded(
                      flex: 3,
                      child: Text(
                        result.previousVersion != null ? 'v${result.previousVersion}' : 'Before',
                        style: const TextStyle(color: G20Colors.textSecondaryDark, fontSize: 12),
                        textAlign: TextAlign.center,
                      ),
                    ),
                    const SizedBox(width: 8),
                    const Icon(Icons.arrow_forward, size: 16, color: G20Colors.textSecondaryDark),
                    const SizedBox(width: 8),
                    Expanded(
                      flex: 3,
                      child: Text(
                        'v${result.version}',
                        style: const TextStyle(color: G20Colors.primary, fontSize: 12, fontWeight: FontWeight.bold),
                        textAlign: TextAlign.center,
                      ),
                    ),
                  ],
                ),
                const Divider(color: G20Colors.cardDark),
                _ComparisonRow(
                  'F1',
                  result.previousF1,
                  result.f1Score,
                  isPercentage: true,
                ),
                _ComparisonRow(
                  'Precision',
                  result.previousMetrics?['precision'] ?? 0,
                  result.metrics['precision'] ?? 0,
                  isPercentage: true,
                ),
                _ComparisonRow(
                  'Recall',
                  result.previousMetrics?['recall'] ?? 0,
                  result.metrics['recall'] ?? 0,
                  isPercentage: true,
                ),
                _ComparisonRow(
                  'Samples',
                  null,
                  result.sampleCount.toDouble(),
                  isCount: true,
                ),
              ],
            ),
          ),
          const SizedBox(height: 16),

          // Promotion status
          if (autoPromoted)
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
              decoration: BoxDecoration(
                color: G20Colors.success.withOpacity(0.2),
                borderRadius: BorderRadius.circular(8),
              ),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  const Icon(Icons.check, color: G20Colors.success, size: 18),
                  const SizedBox(width: 8),
                  Text(
                    'Auto-promoted: ${result.promotionReason ?? "Improved"}',
                    style: const TextStyle(color: G20Colors.success, fontSize: 12),
                  ),
                ],
              ),
            )
          else if (result.previousVersion != null)
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
              decoration: BoxDecoration(
                color: G20Colors.warning.withOpacity(0.2),
                borderRadius: BorderRadius.circular(8),
              ),
              child: const Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Icon(Icons.info_outline, color: G20Colors.warning, size: 18),
                  SizedBox(width: 8),
                  Text(
                    'Not auto-promoted (< 2% improvement)',
                    style: TextStyle(color: G20Colors.warning, fontSize: 12),
                  ),
                ],
              ),
            ),
          const SizedBox(height: 8),

          // Stats
          Text(
            '${result.epochsTrained} epochs • ${result.earlyStopped ? "Early stopped" : "Complete"} • ${_formatDuration(result.trainingTimeSec)}',
            style: const TextStyle(color: G20Colors.textSecondaryDark, fontSize: 11),
          ),
          const SizedBox(height: 20),

          // Action buttons
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              if (!autoPromoted && result.previousVersion != null) ...[
                OutlinedButton(
                  onPressed: () {
                    // Keep old version active
                    onDismiss?.call();
                  },
                  style: OutlinedButton.styleFrom(
                    foregroundColor: G20Colors.textPrimaryDark,
                  ),
                  child: Text('Keep v${result.previousVersion}'),
                ),
                const SizedBox(width: 12),
                ElevatedButton(
                  onPressed: () {
                    // Promote new version
                    // TODO: Call promote API
                    onDismiss?.call();
                  },
                  style: ElevatedButton.styleFrom(
                    backgroundColor: G20Colors.primary,
                  ),
                  child: const Text('Promote Anyway'),
                ),
              ] else
                ElevatedButton(
                  onPressed: () => onDismiss?.call(),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: G20Colors.primary,
                    padding: const EdgeInsets.symmetric(horizontal: 32, vertical: 12),
                  ),
                  child: const Text('Done'),
                ),
            ],
          ),
        ],
      ),
    );
  }

  String _formatDuration(double seconds) {
    final mins = (seconds / 60).floor();
    final secs = (seconds % 60).round();
    if (mins > 0) {
      return '${mins}m ${secs}s';
    }
    return '${secs}s';
  }
}

/// Stat box for progress view
class _StatBox extends StatelessWidget {
  final String label;
  final String value;
  final bool highlight;

  const _StatBox(this.label, this.value, {this.highlight = false});

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Text(
          label,
          style: const TextStyle(color: G20Colors.textSecondaryDark, fontSize: 10),
        ),
        const SizedBox(height: 2),
        Text(
          value,
          style: TextStyle(
            color: highlight ? G20Colors.success : G20Colors.textPrimaryDark,
            fontSize: 14,
            fontWeight: FontWeight.w600,
          ),
        ),
      ],
    );
  }
}

/// Comparison row for result view
class _ComparisonRow extends StatelessWidget {
  final String label;
  final double? before;
  final double after;
  final bool isPercentage;
  final bool isCount;

  const _ComparisonRow(
    this.label,
    this.before,
    this.after, {
    this.isPercentage = false,
    this.isCount = false,
  });

  @override
  Widget build(BuildContext context) {
    String formatValue(double? v) {
      if (v == null) return '-';
      if (isCount) return v.toInt().toString();
      if (isPercentage) return '${(v * 100).toStringAsFixed(1)}%';
      return v.toStringAsFixed(3);
    }

    final diff = before != null ? after - before! : 0.0;
    final improved = diff > 0;
    final diffStr = diff != 0
        ? '${improved ? "+" : ""}${isPercentage ? "${(diff * 100).toStringAsFixed(1)}%" : diff.toStringAsFixed(2)}'
        : '';

    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        children: [
          Expanded(
            flex: 2,
            child: Text(
              label,
              style: const TextStyle(color: G20Colors.textSecondaryDark, fontSize: 12),
            ),
          ),
          Expanded(
            flex: 3,
            child: Text(
              formatValue(before),
              style: const TextStyle(color: G20Colors.textSecondaryDark, fontSize: 12),
              textAlign: TextAlign.center,
            ),
          ),
          const SizedBox(width: 8),
          SizedBox(
            width: 16,
            child: diffStr.isNotEmpty
                ? Icon(
                    improved ? Icons.arrow_upward : Icons.arrow_downward,
                    size: 12,
                    color: improved ? G20Colors.success : G20Colors.error,
                  )
                : null,
          ),
          const SizedBox(width: 8),
          Expanded(
            flex: 3,
            child: RichText(
              textAlign: TextAlign.center,
              text: TextSpan(
                children: [
                  TextSpan(
                    text: formatValue(after),
                    style: const TextStyle(color: G20Colors.textPrimaryDark, fontSize: 12, fontWeight: FontWeight.w600),
                  ),
                  if (diffStr.isNotEmpty)
                    TextSpan(
                      text: ' $diffStr',
                      style: TextStyle(
                        color: improved ? G20Colors.success : G20Colors.error,
                        fontSize: 10,
                      ),
                    ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}
