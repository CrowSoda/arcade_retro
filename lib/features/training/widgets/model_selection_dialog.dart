/// Model selection dialog for training with DURATION selection.
///
/// User selects:
/// 1. New model or extend existing
/// 2. How much of the file to scan for blobs (10s, 30s, 60s, Full)
/// Then backend runs blob detection, high-confidence auto-accept,
/// low-confidence auto-reject, medium shows for tinder swipe.
library;

import 'package:flutter/material.dart';

/// Result of model selection
class ModelSelectionResult {
  /// Selected model name
  final String modelName;

  /// True if creating new model, false if extending existing
  final bool isNew;

  /// Duration to scan for blob detection (seconds, or -1 for full file)
  final double scanDurationSec;

  const ModelSelectionResult({
    required this.modelName,
    required this.isNew,
    required this.scanDurationSec,
  });
}

/// Shows model selection dialog.
///
/// [currentClassName] - the class name from the training screen text field.
/// [existingModels] - list of available models to extend.
/// [fileDurationSec] - total file duration for "Full" option.
///
/// Returns [ModelSelectionResult] if user proceeds, null if cancelled.
Future<ModelSelectionResult?> showModelSelectionDialog({
  required BuildContext context,
  required String currentClassName,
  required List<String> existingModels,
  double fileDurationSec = 60.0,
}) async {
  return showDialog<ModelSelectionResult>(
    context: context,
    barrierDismissible: true,
    builder: (context) => _ModelSelectionDialog(
      currentClassName: currentClassName,
      existingModels: existingModels,
      fileDurationSec: fileDurationSec,
    ),
  );
}

class _ModelSelectionDialog extends StatefulWidget {
  final String currentClassName;
  final List<String> existingModels;
  final double fileDurationSec;

  const _ModelSelectionDialog({
    required this.currentClassName,
    required this.existingModels,
    required this.fileDurationSec,
  });

  @override
  State<_ModelSelectionDialog> createState() => _ModelSelectionDialogState();
}

class _ModelSelectionDialogState extends State<_ModelSelectionDialog> {
  String? _selectedExisting;
  double _scanDuration = 30.0; // Default 30s

  @override
  void initState() {
    super.initState();
    if (widget.existingModels.isNotEmpty) {
      _selectedExisting = widget.existingModels.first;
    }
    // Cap default to file duration
    if (_scanDuration > widget.fileDurationSec) {
      _scanDuration = widget.fileDurationSec;
    }
  }

  void _createNew() {
    Navigator.of(context).pop(ModelSelectionResult(
      modelName: widget.currentClassName,
      isNew: true,
      scanDurationSec: _scanDuration,
    ));
  }

  void _extendExisting() {
    if (_selectedExisting == null) return;
    Navigator.of(context).pop(ModelSelectionResult(
      modelName: _selectedExisting!,
      isNew: false,
      scanDurationSec: _scanDuration,
    ));
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final colorScheme = theme.colorScheme;
    final hasExisting = widget.existingModels.isNotEmpty;

    return AlertDialog(
      title: Row(
        children: [
          Icon(Icons.model_training, color: colorScheme.primary),
          const SizedBox(width: 12),
          const Text('Train Model'),
        ],
      ),
      content: SingleChildScrollView(
        child: SizedBox(
          width: 420,
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              // Show the model name
              Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: colorScheme.surfaceContainerHighest,
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Row(
                  children: [
                    Icon(Icons.label, size: 20, color: colorScheme.primary),
                    const SizedBox(width: 8),
                    Text('Model: ', style: theme.textTheme.bodyMedium),
                    Text(
                      widget.currentClassName,
                      style: theme.textTheme.titleMedium?.copyWith(
                        fontWeight: FontWeight.bold,
                        color: colorScheme.primary,
                      ),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 20),

              // ═══════════════════════════════════════════════════════════════
              // SCAN DURATION - Big touch-friendly buttons
              // ═══════════════════════════════════════════════════════════════
              Text(
                '🔍 Scan Duration (blob detection)',
                style: theme.textTheme.titleSmall?.copyWith(
                  fontWeight: FontWeight.bold,
                ),
              ),
              const SizedBox(height: 8),
              Text(
                'How much of the file to scan for signals:',
                style: theme.textTheme.bodySmall?.copyWith(
                  color: colorScheme.onSurfaceVariant,
                ),
              ),
              const SizedBox(height: 12),

              // Duration buttons - LARGE for finger tap
              Row(
                children: [
                  _DurationButton(
                    label: '10s',
                    isSelected: _scanDuration == 10.0,
                    onTap: () => setState(() => _scanDuration = 10.0),
                  ),
                  const SizedBox(width: 8),
                  _DurationButton(
                    label: '30s',
                    isSelected: _scanDuration == 30.0,
                    onTap: () => setState(() => _scanDuration = 30.0),
                  ),
                  const SizedBox(width: 8),
                  _DurationButton(
                    label: '60s',
                    isSelected: _scanDuration == 60.0,
                    onTap: widget.fileDurationSec >= 60
                        ? () => setState(() => _scanDuration = 60.0)
                        : null,
                  ),
                  const SizedBox(width: 8),
                  _DurationButton(
                    label: 'Full\n${widget.fileDurationSec.toInt()}s',
                    isSelected: _scanDuration == widget.fileDurationSec,
                    onTap: () => setState(() => _scanDuration = widget.fileDurationSec),
                  ),
                ],
              ),
              const SizedBox(height: 8),
              Text(
                '• High confidence blobs → Auto-accepted ✅\n'
                '• Low confidence → Auto-rejected ❌\n'
                '• Medium confidence → You swipe to label',
                style: theme.textTheme.bodySmall?.copyWith(
                  color: colorScheme.onSurfaceVariant,
                  fontSize: 11,
                ),
              ),

              const SizedBox(height: 24),

              // ═══════════════════════════════════════════════════════════════
              // NEW MODEL - Big button
              // ═══════════════════════════════════════════════════════════════
              SizedBox(
                height: 56,
                child: FilledButton.icon(
                  onPressed: _createNew,
                  icon: const Icon(Icons.add_circle, size: 24),
                  label: Text(
                    hasExisting ? 'Create New Model' : 'Create Model',
                    style: const TextStyle(fontSize: 16),
                  ),
                  style: FilledButton.styleFrom(
                    backgroundColor: colorScheme.primary,
                  ),
                ),
              ),

              // EXTEND EXISTING section (only if models exist)
              if (hasExisting) ...[
                const SizedBox(height: 16),
                const Row(
                  children: [
                    Expanded(child: Divider()),
                    Padding(
                      padding: EdgeInsets.symmetric(horizontal: 12),
                      child: Text('or', style: TextStyle(color: Colors.grey)),
                    ),
                    Expanded(child: Divider()),
                  ],
                ),
                const SizedBox(height: 16),
                Text(
                  'Extend existing model:',
                  style: theme.textTheme.bodySmall?.copyWith(
                    color: colorScheme.onSurfaceVariant,
                  ),
                ),
                const SizedBox(height: 8),
                Row(
                  children: [
                    Expanded(
                      child: DropdownButtonFormField<String>(
                        value: _selectedExisting,
                        decoration: InputDecoration(
                          border: const OutlineInputBorder(),
                          isDense: true,
                          filled: true,
                          fillColor: colorScheme.surface,
                        ),
                        items: widget.existingModels.map((name) {
                          return DropdownMenuItem(
                            value: name,
                            child: Text(name),
                          );
                        }).toList(),
                        onChanged: (value) {
                          setState(() => _selectedExisting = value);
                        },
                      ),
                    ),
                    const SizedBox(width: 8),
                    SizedBox(
                      height: 48,
                      child: OutlinedButton(
                        onPressed: _extendExisting,
                        child: const Text('Extend'),
                      ),
                    ),
                  ],
                ),
              ],
            ],
          ),
        ),
      ),
      actions: [
        TextButton(
          onPressed: () => Navigator.of(context).pop(),
          child: const Text('Cancel'),
        ),
      ],
    );
  }
}

/// Large touch-friendly duration button
class _DurationButton extends StatelessWidget {
  final String label;
  final bool isSelected;
  final VoidCallback? onTap;

  const _DurationButton({
    required this.label,
    required this.isSelected,
    this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final colorScheme = theme.colorScheme;
    final isDisabled = onTap == null;

    return Expanded(
      child: GestureDetector(
        onTap: onTap,
        child: Container(
          height: 64,
          decoration: BoxDecoration(
            color: isSelected
                ? colorScheme.primary.withValues(alpha: 0.2)
                : isDisabled
                    ? colorScheme.surfaceContainerHighest.withValues(alpha: 0.3)
                    : colorScheme.surfaceContainerHighest,
            borderRadius: BorderRadius.circular(12),
            border: Border.all(
              color: isSelected
                  ? colorScheme.primary
                  : isDisabled
                      ? colorScheme.outlineVariant.withValues(alpha: 0.3)
                      : colorScheme.outlineVariant,
              width: isSelected ? 2 : 1,
            ),
          ),
          child: Center(
            child: Text(
              label,
              textAlign: TextAlign.center,
              style: TextStyle(
                fontSize: 14,
                fontWeight: isSelected ? FontWeight.bold : FontWeight.normal,
                color: isSelected
                    ? colorScheme.primary
                    : isDisabled
                        ? colorScheme.onSurface.withValues(alpha: 0.3)
                        : colorScheme.onSurface,
              ),
            ),
          ),
        ),
      ),
    );
  }
}
