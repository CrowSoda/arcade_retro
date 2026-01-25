import 'dart:async';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../../../core/config/theme.dart';
import '../providers/waterfall_provider.dart';
import '../providers/sdr_config_provider.dart';
import '../providers/rx_state_provider.dart';
import '../providers/scanner_provider.dart';
import '../../config/providers/tuning_state_provider.dart';
import '../../config/config_screen.dart' show missionsProvider, activeMissionProvider, Mission;
import '../../settings/settings_screen.dart' show autoTuneDelayProvider;

/// Fading toast overlay - disappears after duration
class _FadingToast extends StatefulWidget {
  final String message;
  final IconData icon;
  final Color color;
  final Duration duration;
  final VoidCallback onComplete;

  const _FadingToast({
    required this.message,
    required this.icon,
    required this.color,
    this.duration = const Duration(seconds: 2),
    required this.onComplete,
  });

  @override
  State<_FadingToast> createState() => _FadingToastState();
}

class _FadingToastState extends State<_FadingToast> with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<double> _fadeAnimation;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(vsync: this, duration: const Duration(milliseconds: 300));
    _fadeAnimation = Tween<double>(begin: 0, end: 1).animate(CurvedAnimation(parent: _controller, curve: Curves.easeOut));
    
    // Fade in
    _controller.forward();
    
    // Wait then fade out
    Future.delayed(widget.duration - const Duration(milliseconds: 300), () {
      if (mounted) {
        _controller.reverse().then((_) => widget.onComplete());
      }
    });
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return FadeTransition(
      opacity: _fadeAnimation,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
        decoration: BoxDecoration(
          color: widget.color,
          borderRadius: BorderRadius.circular(8),
          boxShadow: [BoxShadow(color: Colors.black38, blurRadius: 8, offset: const Offset(0, 2))],
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(widget.icon, color: Colors.white, size: 20),
            const SizedBox(width: 8),
            Text(widget.message, style: const TextStyle(color: Colors.white, fontWeight: FontWeight.w600)),
          ],
        ),
      ),
    );
  }
}

/// Show a fading toast overlay
void showFadingToast(BuildContext context, String message, {IconData icon = Icons.check_circle, Color color = Colors.green}) {
  final overlay = Overlay.of(context);
  late OverlayEntry entry;
  
  entry = OverlayEntry(
    builder: (context) => Positioned(
      top: MediaQuery.of(context).size.height * 0.15,
      left: 0,
      right: 0,
      child: Center(
        child: Material(
          color: Colors.transparent,
          child: _FadingToast(
            message: message,
            icon: icon,
            color: color.withOpacity(0.9),
            onComplete: () => entry.remove(),
          ),
        ),
      ),
    ),
  );
  
  overlay.insert(entry);
}

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
    final activeMission = ref.watch(activeMissionProvider);  // Watch the actual active mission
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
                'Mission',
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
                // Load Mission button - touch-friendly (48x48)
                Material(
                  color: G20Colors.primary,
                  borderRadius: BorderRadius.circular(8),
                  child: InkWell(
                    borderRadius: BorderRadius.circular(8),
                    onTap: () => _loadConfig(context, ref),
                    child: const SizedBox(
                      width: 48,
                      height: 32,
                      child: Icon(Icons.rocket_launch, size: 18, color: Colors.white),
                    ),
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
    
    // Show mission picker popup
    final missions = ref.read(missionsProvider);
    _showMissionPickerPopup(context, ref, missions);
  }

  void _showMissionPickerPopup(BuildContext context, WidgetRef ref, List<Mission> missions) {
    showDialog(
      context: context,
      builder: (ctx) => Dialog(
        backgroundColor: G20Colors.surfaceDark,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
        child: Container(
          width: 360,
          constraints: const BoxConstraints(maxHeight: 450),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              // Header
              Container(
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  gradient: LinearGradient(
                    colors: [G20Colors.primary, G20Colors.primary.withOpacity(0.7)],
                  ),
                  borderRadius: const BorderRadius.vertical(top: Radius.circular(16)),
                ),
                child: Row(
                  children: [
                    const Icon(Icons.rocket_launch, color: Colors.white, size: 24),
                    const SizedBox(width: 12),
                    const Expanded(
                      child: Text('Load Mission', style: TextStyle(color: Colors.white, fontSize: 18, fontWeight: FontWeight.bold)),
                    ),
                    IconButton(
                      icon: const Icon(Icons.close, color: Colors.white70),
                      onPressed: () => Navigator.pop(ctx),
                    ),
                  ],
                ),
              ),
              
              // Mission list
              Flexible(
                child: missions.isEmpty
                    ? Padding(
                        padding: const EdgeInsets.all(32),
                        child: Column(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            Icon(Icons.inbox, size: 48, color: Colors.grey.shade600),
                            const SizedBox(height: 12),
                            const Text('No missions created', style: TextStyle(color: G20Colors.textSecondaryDark)),
                            const SizedBox(height: 8),
                            const Text('Go to Mission tab to create one', style: TextStyle(color: G20Colors.textSecondaryDark, fontSize: 11)),
                          ],
                        ),
                      )
                    : ListView.builder(
                        shrinkWrap: true,
                        padding: const EdgeInsets.all(12),
                        itemCount: missions.length,
                        itemBuilder: (context, index) {
                          final mission = missions[index];
                          return Padding(
                            padding: const EdgeInsets.only(bottom: 8),
                            child: Material(
                              color: G20Colors.backgroundDark,
                              borderRadius: BorderRadius.circular(10),
                              child: InkWell(
                                borderRadius: BorderRadius.circular(10),
                                onTap: () {
                                  // Set mission as active
                                  ref.read(activeMissionProvider.notifier).state = mission;
                                  
                                  // Load mission into scanner and start stepped scanning
                                  final scanner = ref.read(scannerProvider.notifier);
                                  scanner.loadMission(mission);
                                  scanner.startScanning();
                                  
                                  // Get the first step's center frequency for display
                                  final firstStep = ref.read(scannerProvider).currentStep;
                                  final centerFreq = firstStep?.centerMHz ?? 
                                      (mission.freqRanges.isNotEmpty 
                                          ? mission.freqRanges.first.startMhz + mission.bandwidthMhz / 2
                                          : ref.read(sdrConfigProvider).centerFreqMHz);
                                  
                                  Navigator.pop(ctx);
                                  
                                  // Show fading toast with step count
                                  final stepCount = ref.read(scannerProvider).totalSteps;
                                  showFadingToast(
                                    context, 
                                    'Mission "${mission.name}" - $stepCount steps', 
                                    icon: Icons.rocket_launch, 
                                    color: Colors.green.shade700,
                                  );
                                  
                                  debugPrint('[Mission] Loaded: ${mission.name} with $stepCount steps');
                                  
                                  // Update freq text field
                                  setState(() {
                                    _freqController.text = centerFreq.toStringAsFixed(1);
                                  });
                                },
                                child: Container(
                                  padding: const EdgeInsets.all(12),
                                  decoration: BoxDecoration(
                                    borderRadius: BorderRadius.circular(10),
                                    border: Border.all(color: G20Colors.cardDark),
                                  ),
                                  child: Row(
                                    children: [
                                      Container(
                                        width: 40,
                                        height: 40,
                                        decoration: BoxDecoration(
                                          color: G20Colors.primary.withOpacity(0.15),
                                          borderRadius: BorderRadius.circular(8),
                                        ),
                                        child: const Icon(Icons.rocket_launch, color: G20Colors.primary, size: 20),
                                      ),
                                      const SizedBox(width: 12),
                                      Expanded(
                                        child: Column(
                                          crossAxisAlignment: CrossAxisAlignment.start,
                                          children: [
                                            Text(mission.name, style: const TextStyle(color: G20Colors.textPrimaryDark, fontWeight: FontWeight.bold)),
                                            const SizedBox(height: 2),
                                            Text(
                                              '${mission.freqRanges.length} ranges • ${mission.models.length} models • ${mission.bandwidthMhz.toInt()} MHz',
                                              style: const TextStyle(color: G20Colors.textSecondaryDark, fontSize: 10),
                                            ),
                                          ],
                                        ),
                                      ),
                                      const Icon(Icons.arrow_forward_ios, color: G20Colors.textSecondaryDark, size: 14),
                                    ],
                                  ),
                                ),
                              ),
                            ),
                          );
                        },
                      ),
              ),
              
              // Footer
              Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: G20Colors.cardDark.withOpacity(0.5),
                  borderRadius: const BorderRadius.vertical(bottom: Radius.circular(16)),
                ),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    TextButton.icon(
                      onPressed: () {
                        ref.read(activeMissionProvider.notifier).state = null;
                        Navigator.pop(ctx);
                      },
                      icon: const Icon(Icons.clear, size: 16),
                      label: const Text('Clear'),
                      style: TextButton.styleFrom(foregroundColor: Colors.red.shade400),
                    ),
                    TextButton(
                      onPressed: () => Navigator.pop(ctx),
                      child: const Text('Cancel'),
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
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
          height: 40,  // Touch-friendly height
          child: TextField(
            controller: controller,
            style: const TextStyle(fontSize: 14, color: G20Colors.textPrimaryDark),
            keyboardType: const TextInputType.numberWithOptions(decimal: true),
            inputFormatters: [FilteringTextInputFormatter.allow(RegExp(r'[\d.]'))],
            decoration: InputDecoration(
              filled: true,
              fillColor: G20Colors.cardDark,
              contentPadding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
              border: OutlineInputBorder(
                borderRadius: BorderRadius.circular(6),
                borderSide: BorderSide.none,
              ),
              focusedBorder: OutlineInputBorder(
                borderRadius: BorderRadius.circular(6),
                borderSide: const BorderSide(color: G20Colors.primary, width: 2),
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

/// Bandwidth dropdown (NV100 supported values) - touch-friendly
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
          height: 40,  // Touch-friendly height
          child: Container(
            padding: const EdgeInsets.symmetric(horizontal: 12),
            decoration: BoxDecoration(
              color: G20Colors.cardDark,
              borderRadius: BorderRadius.circular(6),
            ),
            child: DropdownButtonHideUnderline(
              child: DropdownButton<double>(
                value: _bandwidths.contains(value) ? value : 20.0,
                isExpanded: true,
                dropdownColor: G20Colors.cardDark,
                style: const TextStyle(fontSize: 14, color: G20Colors.textPrimaryDark),
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

/// Timeout selection button - touch-friendly (minimum 40px height)
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
      child: Material(
        color: isSelected ? G20Colors.warning.withValues(alpha: 0.3) : G20Colors.cardDark,
        borderRadius: BorderRadius.circular(6),
        child: InkWell(
          borderRadius: BorderRadius.circular(6),
          onTap: onTap,
          child: Container(
            height: 40,  // Touch-friendly minimum
            decoration: BoxDecoration(
              borderRadius: BorderRadius.circular(6),
              border: Border.all(
                color: isSelected ? G20Colors.warning : Colors.transparent,
                width: 2,
              ),
            ),
            child: Center(
              child: Text(
                label,
                style: TextStyle(
                  color: isSelected ? G20Colors.warning : G20Colors.textSecondaryDark,
                  fontSize: 12,
                  fontWeight: isSelected ? FontWeight.bold : FontWeight.w500,
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }
}
