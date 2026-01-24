import 'dart:async';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:file_picker/file_picker.dart';
import '../../../core/config/theme.dart';
import '../../../core/database/signal_database.dart';
import '../providers/waterfall_provider.dart';
import '../providers/sdr_config_provider.dart';
import '../providers/rx_state_provider.dart';
import '../../config/providers/tuning_state_provider.dart';
import '../../settings/settings_screen.dart' show autoTuneDelayProvider;

/// Show capture warning dialog - returns true if user confirms, false if cancelled
Future<bool> showCaptureWarningDialog(BuildContext context, ManualCaptureState captureState) async {
  final result = await showDialog<bool>(
    context: context,
    builder: (ctx) => AlertDialog(
      backgroundColor: G20Colors.surfaceDark,
      title: const Row(
        children: [
          Icon(Icons.warning_amber, color: G20Colors.warning, size: 24),
          SizedBox(width: 8),
          Text('Capture in Progress', 
            style: TextStyle(color: G20Colors.textPrimaryDark, fontSize: 16)),
        ],
      ),
      content: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text('RX-2 is currently capturing ${captureState.signalName ?? "data"}.',
            style: const TextStyle(color: G20Colors.textSecondaryDark)),
          const SizedBox(height: 12),
          const Text('Tuning will end and save the current capture.',
            style: TextStyle(color: G20Colors.textPrimaryDark)),
          if (captureState.queueLength > 0)
            Padding(
              padding: const EdgeInsets.only(top: 8),
              child: Text(
                'Queued captures (${captureState.queueLength}) will resume once RX-2 is free.',
                style: const TextStyle(color: G20Colors.textSecondaryDark, fontSize: 12),
              ),
            ),
        ],
      ),
      actions: [
        TextButton(
          onPressed: () => Navigator.pop(ctx, false),
          child: const Text('Cancel', style: TextStyle(color: G20Colors.textSecondaryDark)),
        ),
        ElevatedButton(
          onPressed: () => Navigator.pop(ctx, true),
          style: ElevatedButton.styleFrom(backgroundColor: G20Colors.warning),
          child: const Text('Continue'),
        ),
      ],
    ),
  );
  return result == true;
}

/// Simplified inputs panel - Center Freq, BW, Config
class InputsPanel extends ConsumerStatefulWidget {
  const InputsPanel({super.key});

  @override
  ConsumerState<InputsPanel> createState() => _InputsPanelState();
}

class _InputsPanelState extends ConsumerState<InputsPanel> {
  final _freqController = TextEditingController();
  final _bwController = TextEditingController();
  bool _initialized = false;
  int? _selectedTimeout = 60; // Default to 60s
  Timer? _autoTuneTimer;  // Debounce timer for auto-tune on typing

  @override
  void dispose() {
    _autoTuneTimer?.cancel();
    _freqController.dispose();
    _bwController.dispose();
    super.dispose();
  }

  /// Validate frequency is within valid range (30 MHz - 6000 MHz for NV100)
  bool _isValidFrequency(double? freq) {
    if (freq == null) return false;
    return freq >= 30.0 && freq <= 6000.0;
  }

  /// Start/restart auto-tune debounce timer
  void _onFrequencyChanged(String value) {
    _autoTuneTimer?.cancel();
    
    final autoTuneDelay = ref.read(autoTuneDelayProvider);
    if (autoTuneDelay == null) return;  // Auto-tune disabled
    
    final freq = double.tryParse(value);
    if (!_isValidFrequency(freq)) return;  // Invalid frequency, don't auto-tune
    
    _autoTuneTimer = Timer(Duration(seconds: autoTuneDelay), () {
      if (mounted) _handleGo();
    });
  }

  void _handleGo() async {
    _autoTuneTimer?.cancel();  // Cancel any pending auto-tune
    
    final centerMHz = double.tryParse(_freqController.text);
    final bwMHz = ref.read(sdrConfigProvider).bandwidthMHz;

    if (!_isValidFrequency(centerMHz)) {
      // Show warning dialog for invalid frequency
      await showDialog(
        context: context,
        builder: (ctx) => AlertDialog(
          backgroundColor: G20Colors.surfaceDark,
          title: const Row(
            children: [
              Icon(Icons.warning_amber, color: G20Colors.error, size: 24),
              SizedBox(width: 8),
              Text('Invalid Frequency', 
                style: TextStyle(color: G20Colors.textPrimaryDark, fontSize: 16)),
            ],
          ),
          content: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                centerMHz == null 
                  ? 'Please enter a valid frequency'
                  : 'Frequency ${centerMHz.toStringAsFixed(1)} MHz is outside NV100 range.',
                style: const TextStyle(color: G20Colors.textSecondaryDark),
              ),
              const SizedBox(height: 12),
              const Text(
                'Sidekiq NV100 Specifications:',
                style: TextStyle(color: G20Colors.textPrimaryDark, fontWeight: FontWeight.bold),
              ),
              const SizedBox(height: 4),
              const Text('• RF Tuning Range: 30 MHz - 6 GHz',
                style: TextStyle(color: G20Colors.textSecondaryDark, fontSize: 12)),
              const Text('• Max Bandwidth: 50 MHz',
                style: TextStyle(color: G20Colors.textSecondaryDark, fontSize: 12)),
            ],
          ),
          actions: [
            ElevatedButton(
              onPressed: () => Navigator.pop(ctx),
              style: ElevatedButton.styleFrom(backgroundColor: G20Colors.primary),
              child: const Text('OK'),
            ),
          ],
        ),
      );
      return;
    }

    // Check if capture in progress - warn user
    final captureState = ref.read(manualCaptureProvider);
    if (captureState.isCapturing) {
      final proceed = await showCaptureWarningDialog(context, captureState);
      if (!proceed) return;
      // Cancel current capture (it will save) and clear queue
      ref.read(manualCaptureProvider.notifier).cancel();
    }

    // Update both providers
    ref.read(sdrConfigProvider.notifier).setFrequency(centerMHz!);
    ref.read(waterfallProvider.notifier).setCenterFrequency(centerMHz);
    
    // Update RX-2 in command bar
    ref.read(multiRxProvider.notifier).tuneRx2(centerMHz, bwMHz, null);

    await ref.read(tuningStateProvider.notifier).updateTuning(
      centerMHz: centerMHz,
      bwMHz: bwMHz,
    );
    debugPrint('📻 Tuned to ${centerMHz.toStringAsFixed(1)} MHz');
  }

  void _handleTimeoutSelect(int? timeout) async {
    setState(() => _selectedTimeout = timeout);
    await ref.read(tuningStateProvider.notifier).setManualMode(timeout);
    debugPrint('⏱️ Manual mode: ${timeout != null ? "${timeout}s timeout" : "permanent"}');
  }

  void _handleResumeAuto() async {
    await ref.read(tuningStateProvider.notifier).resumeAuto();
    debugPrint('🤖 Auto scan resumed');
  }

  @override
  Widget build(BuildContext context) {
    final sdrConfig = ref.watch(sdrConfigProvider);
    final dbState = ref.watch(signalDatabaseProvider);
    final activeMission = dbState.activeMissionConfig;
    final tuningState = ref.watch(tuningStateProvider);
    final isManual = tuningState.mode == TuningMode.manual;

    // Initialize controllers once
    if (!_initialized) {
      _freqController.text = sdrConfig.centerFreqMHz.toStringAsFixed(1);
      _bwController.text = sdrConfig.bandwidthMHz.toStringAsFixed(1);
      _initialized = true;
    }

    return Padding(
      padding: const EdgeInsets.all(12),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          // Header with mode indicator
          Row(
            children: [
              const Text(
                'Config',
                style: TextStyle(
                  fontSize: 16,
                  fontWeight: FontWeight.bold,
                  color: G20Colors.textPrimaryDark,
                ),
              ),
              const Spacer(),
              // Mode indicator
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                decoration: BoxDecoration(
                  color: isManual 
                      ? G20Colors.warning.withValues(alpha: 0.2) 
                      : G20Colors.success.withValues(alpha: 0.2),
                  borderRadius: BorderRadius.circular(4),
                ),
                child: Text(
                  tuningState.modeDisplayString,
                  style: TextStyle(
                    color: isManual ? G20Colors.warning : G20Colors.success,
                    fontSize: 10,
                    fontWeight: FontWeight.w600,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          
          // Active config display
          Container(
            padding: const EdgeInsets.all(8),
            decoration: BoxDecoration(
              color: G20Colors.cardDark,
              borderRadius: BorderRadius.circular(6),
            ),
            child: Row(
              children: [
                Icon(
                  activeMission != null ? Icons.check_circle : Icons.warning_amber,
                  size: 14,
                  color: activeMission != null ? G20Colors.success : G20Colors.warning,
                ),
                const SizedBox(width: 8),
                Expanded(
                  child: Text(
                    activeMission?.name ?? 'No config loaded',
                    style: TextStyle(
                      fontSize: 11,
                      color: activeMission != null ? G20Colors.textPrimaryDark : G20Colors.textSecondaryDark,
                    ),
                  ),
                ),
                // Load Config button (icon only now)
                SizedBox(
                  width: 32,
                  height: 24,
                  child: IconButton(
                    onPressed: () => _loadConfig(context, ref),
                    icon: const Icon(Icons.folder_open, size: 14),
                    padding: EdgeInsets.zero,
                    color: G20Colors.primary,
                    tooltip: 'Load Config',
                  ),
                ),
              ],
            ),
          ),
          
          const SizedBox(height: 8),
          
          // Frequency + BW inputs
          Row(
            children: [
              // Center Frequency
              Expanded(
                child: _EditableField(
                  label: 'Center (MHz)',
                  controller: _freqController,
                  onSubmit: (_) => _handleGo(),
                  onChanged: _onFrequencyChanged,
                ),
              ),
              const SizedBox(width: 8),
              // Bandwidth dropdown
              Expanded(
                child: _BandwidthDropdown(
                  value: sdrConfig.bandwidthMHz,
                  onChanged: (bw) {
                    ref.read(sdrConfigProvider.notifier).setBandwidth(bw);
                    ref.read(waterfallProvider.notifier).setBandwidth(bw);
                  },
                ),
              ),
            ],
          ),
          
          const SizedBox(height: 8),
          
          // Timeout buttons row
          Row(
            children: [
              const Text('Timeout:', style: TextStyle(fontSize: 9, color: G20Colors.textSecondaryDark)),
              const SizedBox(width: 6),
              _TimeoutButton(label: '60s', isSelected: _selectedTimeout == 60 && isManual, onTap: () => _handleTimeoutSelect(60)),
              const SizedBox(width: 4),
              _TimeoutButton(label: '120s', isSelected: _selectedTimeout == 120 && isManual, onTap: () => _handleTimeoutSelect(120)),
              const SizedBox(width: 4),
              _TimeoutButton(label: '300s', isSelected: _selectedTimeout == 300 && isManual, onTap: () => _handleTimeoutSelect(300)),
              const SizedBox(width: 4),
              _TimeoutButton(label: '∞', isSelected: _selectedTimeout == null && isManual, onTap: () => _handleTimeoutSelect(null)),
            ],
          ),
          
          const SizedBox(height: 8),
          
          // GO button - full width
          SizedBox(
            height: 40,
            child: ElevatedButton(
              onPressed: tuningState.isTuning ? null : _handleGo,
              style: ElevatedButton.styleFrom(
                backgroundColor: G20Colors.primary,
                foregroundColor: Colors.white,
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(6)),
              ),
              child: tuningState.isTuning
                  ? const SizedBox(
                      width: 18,
                      height: 18,
                      child: CircularProgressIndicator(
                        strokeWidth: 2,
                        color: Colors.white,
                      ),
                    )
                  : const Text('GO', style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16)),
            ),
          ),
          
          // Resume Auto button (only when in manual mode)
          if (isManual) ...[
            const SizedBox(height: 8),
            SizedBox(
              height: 36,
              child: OutlinedButton.icon(
                onPressed: _handleResumeAuto,
                icon: const Icon(Icons.autorenew, size: 16),
                label: const Text('Resume Auto Scan', style: TextStyle(fontWeight: FontWeight.bold, fontSize: 11)),
                style: OutlinedButton.styleFrom(
                  foregroundColor: G20Colors.success,
                  side: const BorderSide(color: G20Colors.success),
                  padding: const EdgeInsets.symmetric(horizontal: 8),
                  shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(4)),
                ),
              ),
            ),
          ],
          
          // Error message if any
          if (sdrConfig.errorMessage != null || tuningState.errorMessage != null) ...[
            const SizedBox(height: 6),
            Text(
              sdrConfig.errorMessage ?? tuningState.errorMessage ?? '',
              style: const TextStyle(fontSize: 9, color: G20Colors.error),
            ),
          ],
        ],
      ),
    );
  }

  void _loadConfig(BuildContext context, WidgetRef ref) async {
    // Check if capture in progress - warn user (loading config may change freq)
    final captureState = ref.read(manualCaptureProvider);
    if (captureState.isCapturing) {
      final proceed = await showCaptureWarningDialog(context, captureState);
      if (!proceed) return;
      ref.read(manualCaptureProvider.notifier).cancel();
    }
    
    final result = await FilePicker.platform.pickFiles(
      type: FileType.custom,
      allowedExtensions: ['json', 'yaml', 'yml'],
      dialogTitle: 'Load Mission Config',
    );

    if (result != null && result.files.single.path != null) {
      ref.read(sdrConfigProvider.notifier).loadConfigFile(
        result.files.single.path!,
        result.files.single.name,
      );
      debugPrint('📄 Loaded config: ${result.files.single.name}');
    }
  }
}

/// Simple editable text field with optional auto-tune support
class _EditableField extends StatelessWidget {
  final String label;
  final TextEditingController controller;
  final Function(String) onSubmit;
  final Function(String)? onChanged;

  const _EditableField({
    required this.label,
    required this.controller,
    required this.onSubmit,
    this.onChanged,
  });

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(label, style: const TextStyle(fontSize: 10, color: G20Colors.textSecondaryDark)),
        const SizedBox(height: 4),
        SizedBox(
          height: 32,
          child: TextField(
            controller: controller,
            style: const TextStyle(fontSize: 12, color: G20Colors.textPrimaryDark),
            keyboardType: const TextInputType.numberWithOptions(decimal: true),
            inputFormatters: [FilteringTextInputFormatter.allow(RegExp(r'[\d.]'))],
            decoration: InputDecoration(
              filled: true,
              fillColor: G20Colors.cardDark,
              contentPadding: const EdgeInsets.symmetric(horizontal: 8, vertical: 6),
              border: OutlineInputBorder(
                borderRadius: BorderRadius.circular(4),
                borderSide: BorderSide.none,
              ),
              focusedBorder: OutlineInputBorder(
                borderRadius: BorderRadius.circular(4),
                borderSide: const BorderSide(color: G20Colors.primary, width: 1),
              ),
            ),
            onChanged: onChanged,
            onSubmitted: onSubmit,
            onEditingComplete: () => onSubmit(controller.text),
          ),
        ),
      ],
    );
  }
}

/// Bandwidth dropdown (NV100 supported values)
class _BandwidthDropdown extends StatelessWidget {
  final double value;
  final Function(double) onChanged;

  const _BandwidthDropdown({required this.value, required this.onChanged});

  // NV100 supported bandwidths
  static const _bandwidths = [5.0, 10.0, 20.0, 40.0, 50.0];

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text('BW (MHz)', style: TextStyle(fontSize: 10, color: G20Colors.textSecondaryDark)),
        const SizedBox(height: 4),
        SizedBox(
          height: 32,
          child: Container(
            padding: const EdgeInsets.symmetric(horizontal: 8),
            decoration: BoxDecoration(
              color: G20Colors.cardDark,
              borderRadius: BorderRadius.circular(4),
            ),
            child: DropdownButtonHideUnderline(
              child: DropdownButton<double>(
                value: _bandwidths.contains(value) ? value : 20.0,
                isExpanded: true,
                dropdownColor: G20Colors.cardDark,
                style: const TextStyle(fontSize: 12, color: G20Colors.textPrimaryDark),
                items: _bandwidths.map((bw) => DropdownMenuItem(
                  value: bw,
                  child: Text('${bw.toInt()}'),
                )).toList(),
                onChanged: (v) => v != null ? onChanged(v) : null,
              ),
            ),
          ),
        ),
      ],
    );
  }
}

/// Timeout selection button (compact)
class _TimeoutButton extends StatelessWidget {
  final String label;
  final bool isSelected;
  final VoidCallback onTap;

  const _TimeoutButton({
    required this.label,
    required this.isSelected,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return Expanded(
      child: GestureDetector(
        onTap: onTap,
        child: Container(
          padding: const EdgeInsets.symmetric(vertical: 6),
          decoration: BoxDecoration(
            color: isSelected ? G20Colors.warning.withValues(alpha: 0.3) : G20Colors.cardDark,
            borderRadius: BorderRadius.circular(4),
            border: Border.all(
              color: isSelected ? G20Colors.warning : G20Colors.cardDark,
              width: isSelected ? 2 : 1,
            ),
          ),
          child: Center(
            child: Text(
              label,
              style: TextStyle(
                color: isSelected ? G20Colors.warning : G20Colors.textSecondaryDark,
                fontSize: 10,
                fontWeight: isSelected ? FontWeight.bold : FontWeight.normal,
              ),
            ),
          ),
        ),
      ),
    );
  }
}
