import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:shared_preferences/shared_preferences.dart';
import '../../core/config/theme.dart';
import '../live_detection/providers/video_stream_provider.dart';

// Re-export all providers for backward compatibility with existing imports
export 'providers/settings_providers.dart';

// Import providers from the new centralized location
import 'providers/settings_providers.dart';

/// Settings Screen - Configuration and connection settings
class SettingsScreen extends ConsumerStatefulWidget {
  const SettingsScreen({super.key});

  @override
  ConsumerState<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends ConsumerState<SettingsScreen> {
  // Connection settings
  final _hostController = TextEditingController(text: '192.168.1.100');
  final _grpcPortController = TextEditingController(text: '50051');
  final _udpPortController = TextEditingController(text: '5000');
  bool _autoConnect = true;

  // Display settings (colormap only - dB is now provider-based)
  int _colormap = 0;

  // System info
  final String _appVersion = '1.0.0';

  @override
  void dispose() {
    _hostController.dispose();
    _grpcPortController.dispose();
    _udpPortController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Connection Settings
          _buildSection(
            title: 'Connection',
            icon: Icons.wifi,
            children: [
              _buildTextField(
                label: 'Host',
                controller: _hostController,
                hint: '192.168.1.100',
              ),
              const SizedBox(height: 12),
              Row(
                children: [
                  Expanded(
                    child: _buildTextField(
                      label: 'gRPC Port',
                      controller: _grpcPortController,
                      hint: '50051',
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: _buildTextField(
                      label: 'UDP Port',
                      controller: _udpPortController,
                      hint: '5000',
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 12),
              SwitchListTile(
                title: const Text('Auto-connect on startup'),
                value: _autoConnect,
                onChanged: (v) => setState(() => _autoConnect = v),
                contentPadding: EdgeInsets.zero,
              ),
              const SizedBox(height: 8),
              Row(
                children: [
                  ElevatedButton.icon(
                    onPressed: _testConnection,
                    icon: const Icon(Icons.network_check),
                    label: const Text('Test Connection'),
                  ),
                  const SizedBox(width: 8),
                  ElevatedButton.icon(
                    onPressed: _saveConnectionSettings,
                    icon: const Icon(Icons.save),
                    label: const Text('Save'),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: G20Colors.success,
                    ),
                  ),
                ],
              ),
            ],
          ),
          const SizedBox(height: 24),
          // Display Settings
          _buildSection(
            title: 'Display',
            icon: Icons.display_settings,
            children: [
              // FFT Size Selector - GPU-accelerated FFT resolution control
              const _FftSizeSelector(),
              const SizedBox(height: 16),
              // Colormap selector - wired to provider and backend
              const _ColormapSelector(),
              const SizedBox(height: 16),
              // Stats overlay toggle
              const _StatsOverlayToggle(),
            ],
          ),
          const SizedBox(height: 24),
          // Model Settings - just threshold, model info moved to Mission
          _buildSection(
            title: 'Model',
            icon: Icons.model_training,
            children: [
              const _ScoreThresholdSelector(),
            ],
          ),
          const SizedBox(height: 24),
          // About
          _buildSection(
            title: 'About',
            icon: Icons.info,
            children: [
              ListTile(
                contentPadding: EdgeInsets.zero,
                title: const Text('App Version'),
                trailing: Text(_appVersion),
              ),
              // Backend version - wired to provider
              Consumer(
                builder: (context, ref, _) {
                  final version = ref.watch(backendVersionProvider);
                  final isConnected = version != 'Not Connected';
                  return ListTile(
                    contentPadding: EdgeInsets.zero,
                    title: const Text('Backend Version'),
                    trailing: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Container(
                          width: 8,
                          height: 8,
                          margin: const EdgeInsets.only(right: 8),
                          decoration: BoxDecoration(
                            shape: BoxShape.circle,
                            color: isConnected ? G20Colors.success : G20Colors.error,
                          ),
                        ),
                        Text(
                          version,
                          style: TextStyle(
                            color: isConnected ? G20Colors.textPrimaryDark : G20Colors.error,
                          ),
                        ),
                      ],
                    ),
                  );
                },
              ),
              const SizedBox(height: 8),
              Row(
                children: [
                  TextButton.icon(
                    onPressed: () {
                      // TODO: Open documentation
                    },
                    icon: const Icon(Icons.book),
                    label: const Text('Documentation'),
                  ),
                  const SizedBox(width: 8),
                  TextButton.icon(
                    onPressed: () {
                      // TODO: Open logs
                    },
                    icon: const Icon(Icons.description),
                    label: const Text('View Logs'),
                  ),
                ],
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildSection({
    required String title,
    required IconData icon,
    required List<Widget> children,
  }) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(icon, color: G20Colors.primary),
                const SizedBox(width: 8),
                Text(
                  title,
                  style: const TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ],
            ),
            const Divider(),
            ...children,
          ],
        ),
      ),
    );
  }

  Widget _buildTextField({
    required String label,
    required TextEditingController controller,
    required String hint,
  }) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          label,
          style: const TextStyle(
            fontSize: 12,
            color: G20Colors.textSecondaryDark,
          ),
        ),
        const SizedBox(height: 4),
        TextField(
          controller: controller,
          decoration: InputDecoration(
            hintText: hint,
            isDense: true,
          ),
        ),
      ],
    );
  }

  Widget _buildSlider({
    required String label,
    required double value,
    required double min,
    required double max,
    required int divisions,
    required String suffix,
    required void Function(double) onChanged,
  }) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Text(
              label,
              style: const TextStyle(
                fontSize: 12,
                color: G20Colors.textSecondaryDark,
              ),
            ),
            Text(
              '${value.toStringAsFixed(0)}$suffix',
              style: const TextStyle(
                fontWeight: FontWeight.bold,
              ),
            ),
          ],
        ),
        Slider(
          value: value,
          min: min,
          max: max,
          divisions: divisions,
          onChanged: onChanged,
        ),
      ],
    );
  }

  Widget _buildDropdown({
    required String label,
    required int value,
    required List<String> items,
    required void Function(int?) onChanged,
  }) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          label,
          style: const TextStyle(
            fontSize: 12,
            color: G20Colors.textSecondaryDark,
          ),
        ),
        const SizedBox(height: 4),
        DropdownButton<int>(
          value: value,
          isExpanded: true,
          items: items
              .asMap()
              .entries
              .map((e) => DropdownMenuItem(
                    value: e.key,
                    child: Text(e.value),
                  ))
              .toList(),
          onChanged: onChanged,
        ),
      ],
    );
  }

  void _testConnection() {
    // TODO: Implement connection test
    debugPrint('[Settings] Testing connection...');
  }

  void _saveConnectionSettings() {
    // TODO: Save settings to SharedPreferences
    debugPrint('[Settings] Settings saved');
  }
}

/// Auto-tune delay selector widget
class _AutoTuneDelaySelector extends ConsumerWidget {
  const _AutoTuneDelaySelector();

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final delay = ref.watch(autoTuneDelayProvider);

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text(
          'Auto-tune after typing',
          style: TextStyle(
            fontSize: 14,
            color: G20Colors.textPrimaryDark,
          ),
        ),
        const SizedBox(height: 4),
        Text(
          'Automatically tune to frequency after you stop typing',
          style: TextStyle(
            fontSize: 11,
            color: G20Colors.textSecondaryDark,
          ),
        ),
        const SizedBox(height: 12),
        Row(
          children: [
            _DelayOption(
              label: 'Off',
              value: null,
              selected: delay == null,
              onTap: () => ref.read(autoTuneDelayProvider.notifier).setValue(null),
            ),
            const SizedBox(width: 8),
            _DelayOption(
              label: '1s',
              value: 1,
              selected: delay == 1,
              onTap: () => ref.read(autoTuneDelayProvider.notifier).setValue(1),
            ),
            const SizedBox(width: 8),
            _DelayOption(
              label: '2s',
              value: 2,
              selected: delay == 2,
              onTap: () => ref.read(autoTuneDelayProvider.notifier).setValue(2),
            ),
            const SizedBox(width: 8),
            _DelayOption(
              label: '5s',
              value: 5,
              selected: delay == 5,
              onTap: () => ref.read(autoTuneDelayProvider.notifier).setValue(5),
            ),
          ],
        ),
      ],
    );
  }
}

class _DelayOption extends StatelessWidget {
  final String label;
  final int? value;
  final bool selected;
  final VoidCallback onTap;

  const _DelayOption({
    required this.label,
    required this.value,
    required this.selected,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return Expanded(
      child: GestureDetector(
        onTap: onTap,
        child: Container(
          padding: const EdgeInsets.symmetric(vertical: 10),
          decoration: BoxDecoration(
            color: selected ? G20Colors.primary.withValues(alpha: 0.2) : G20Colors.cardDark,
            borderRadius: BorderRadius.circular(6),
            border: Border.all(
              color: selected ? G20Colors.primary : G20Colors.cardDark,
              width: selected ? 2 : 1,
            ),
          ),
          child: Center(
            child: Text(
              label,
              style: TextStyle(
                color: selected ? G20Colors.primary : G20Colors.textSecondaryDark,
                fontWeight: selected ? FontWeight.bold : FontWeight.normal,
              ),
            ),
          ),
        ),
      ),
    );
  }
}

/// Waterfall time span selector - controls how many seconds of data to show
class _WaterfallTimeSpanSelector extends ConsumerWidget {
  const _WaterfallTimeSpanSelector();

  String _formatTimeSpan(double seconds) {
    if (seconds < 1.0) {
      return '${(seconds * 1000).round()}ms';
    } else {
      return '${seconds.toStringAsFixed(0)}s';
    }
  }

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final timeSpan = ref.watch(waterfallTimeSpanProvider);

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            const Text(
              'Waterfall Time Span',
              style: TextStyle(
                fontSize: 14,
                color: G20Colors.textPrimaryDark,
              ),
            ),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
              decoration: BoxDecoration(
                color: G20Colors.primary.withValues(alpha: 0.2),
                borderRadius: BorderRadius.circular(4),
              ),
              child: Text(
                _formatTimeSpan(timeSpan),
                style: const TextStyle(
                  fontWeight: FontWeight.bold,
                  color: G20Colors.primary,
                ),
              ),
            ),
          ],
        ),
        const SizedBox(height: 4),
        Text(
          'How many seconds of history the waterfall displays',
          style: TextStyle(
            fontSize: 11,
            color: G20Colors.textSecondaryDark,
          ),
        ),
        const SizedBox(height: 12),
        // First row - short durations
        Row(
          children: [
            _TimeSpanOption(
              label: '200ms',
              value: 0.2,
              selected: (timeSpan - 0.2).abs() < 0.05,
              onTap: () => ref.read(waterfallTimeSpanProvider.notifier).setValue(0.2),
            ),
            const SizedBox(width: 6),
            _TimeSpanOption(
              label: '500ms',
              value: 0.5,
              selected: (timeSpan - 0.5).abs() < 0.05,
              onTap: () => ref.read(waterfallTimeSpanProvider.notifier).setValue(0.5),
            ),
            const SizedBox(width: 6),
            _TimeSpanOption(
              label: '1s',
              value: 1.0,
              selected: (timeSpan - 1.0).abs() < 0.1,
              onTap: () => ref.read(waterfallTimeSpanProvider.notifier).setValue(1.0),
            ),
            const SizedBox(width: 6),
            _TimeSpanOption(
              label: '2s',
              value: 2.0,
              selected: (timeSpan - 2.0).abs() < 0.1,
              onTap: () => ref.read(waterfallTimeSpanProvider.notifier).setValue(2.0),
            ),
          ],
        ),
        const SizedBox(height: 6),
        // Second row - longer durations
        Row(
          children: [
            _TimeSpanOption(
              label: '5s',
              value: 5.0,
              selected: (timeSpan - 5.0).abs() < 0.1,
              onTap: () => ref.read(waterfallTimeSpanProvider.notifier).setValue(5.0),
            ),
            const SizedBox(width: 6),
            _TimeSpanOption(
              label: '10s',
              value: 10.0,
              selected: (timeSpan - 10.0).abs() < 0.1,
              onTap: () => ref.read(waterfallTimeSpanProvider.notifier).setValue(10.0),
            ),
            const SizedBox(width: 6),
            _TimeSpanOption(
              label: '30s',
              value: 30.0,
              selected: (timeSpan - 30.0).abs() < 0.1,
              onTap: () => ref.read(waterfallTimeSpanProvider.notifier).setValue(30.0),
            ),
            const SizedBox(width: 6),
            _TimeSpanOption(
              label: '60s',
              value: 60.0,
              selected: (timeSpan - 60.0).abs() < 0.1,
              onTap: () => ref.read(waterfallTimeSpanProvider.notifier).setValue(60.0),
            ),
          ],
        ),
      ],
    );
  }
}

class _TimeSpanOption extends StatelessWidget {
  final String label;
  final double value;
  final bool selected;
  final VoidCallback onTap;

  const _TimeSpanOption({
    required this.label,
    required this.value,
    required this.selected,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return Expanded(
      child: GestureDetector(
        onTap: onTap,
        child: Container(
          padding: const EdgeInsets.symmetric(vertical: 10),
          decoration: BoxDecoration(
            color: selected ? G20Colors.primary.withValues(alpha: 0.2) : G20Colors.cardDark,
            borderRadius: BorderRadius.circular(6),
            border: Border.all(
              color: selected ? G20Colors.primary : G20Colors.cardDark,
              width: selected ? 2 : 1,
            ),
          ),
          child: Center(
            child: Text(
              label,
              style: TextStyle(
                color: selected ? G20Colors.primary : G20Colors.textSecondaryDark,
                fontWeight: selected ? FontWeight.bold : FontWeight.normal,
              ),
            ),
          ),
        ),
      ),
    );
  }
}

/// Waterfall FPS selector - controls streaming speed for debugging
class _WaterfallFpsSelector extends ConsumerWidget {
  const _WaterfallFpsSelector();

  void _setFps(WidgetRef ref, int newFps) {
    debugPrint('[Settings] FPS button tapped: $newFps');
    
    // Update provider state (persisted)
    ref.read(waterfallFpsProvider.notifier).setValue(newFps);
    
    // DIRECT CALL to backend - no listener needed!
    ref.read(videoStreamProvider.notifier).setFps(newFps);
  }

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final fps = ref.watch(waterfallFpsProvider);

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            const Text(
              'Stream FPS',
              style: TextStyle(
                fontSize: 14,
                color: G20Colors.textPrimaryDark,
              ),
            ),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
              decoration: BoxDecoration(
                color: fps < 30 
                    ? Colors.orange.withValues(alpha: 0.2)
                    : G20Colors.primary.withValues(alpha: 0.2),
                borderRadius: BorderRadius.circular(4),
              ),
              child: Text(
                '${fps}fps',
                style: TextStyle(
                  fontWeight: FontWeight.bold,
                  color: fps < 30 ? Colors.orange : G20Colors.primary,
                ),
              ),
            ),
          ],
        ),
        const SizedBox(height: 4),
        Text(
          'Slow down waterfall for debugging (affects data rate)',
          style: TextStyle(
            fontSize: 11,
            color: G20Colors.textSecondaryDark,
          ),
        ),
        const SizedBox(height: 12),
        Row(
          children: [
            _FpsOption(
              label: '1',
              value: 1,
              selected: fps == 1,
              onTap: () => _setFps(ref, 1),
            ),
            const SizedBox(width: 6),
            _FpsOption(
              label: '5',
              value: 5,
              selected: fps == 5,
              onTap: () => _setFps(ref, 5),
            ),
            const SizedBox(width: 6),
            _FpsOption(
              label: '10',
              value: 10,
              selected: fps == 10,
              onTap: () => _setFps(ref, 10),
            ),
            const SizedBox(width: 6),
            _FpsOption(
              label: '15',
              value: 15,
              selected: fps == 15,
              onTap: () => _setFps(ref, 15),
            ),
            const SizedBox(width: 6),
            _FpsOption(
              label: '30',
              value: 30,
              selected: fps == 30,
              onTap: () => _setFps(ref, 30),
            ),
          ],
        ),
      ],
    );
  }
}

class _FpsOption extends StatelessWidget {
  final String label;
  final int value;
  final bool selected;
  final VoidCallback onTap;

  const _FpsOption({
    required this.label,
    required this.value,
    required this.selected,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    // Highlight slow FPS options with orange
    final isSlowFps = value < 30;
    final activeColor = isSlowFps ? Colors.orange : G20Colors.primary;
    
    return Expanded(
      child: GestureDetector(
        onTap: onTap,
        child: Container(
          padding: const EdgeInsets.symmetric(vertical: 10),
          decoration: BoxDecoration(
            color: selected ? activeColor.withValues(alpha: 0.2) : G20Colors.cardDark,
            borderRadius: BorderRadius.circular(6),
            border: Border.all(
              color: selected ? activeColor : G20Colors.cardDark,
              width: selected ? 2 : 1,
            ),
          ),
          child: Center(
            child: Text(
              label,
              style: TextStyle(
                color: selected ? activeColor : G20Colors.textSecondaryDark,
                fontWeight: selected ? FontWeight.bold : FontWeight.normal,
              ),
            ),
          ),
        ),
      ),
    );
  }
}

/// Score threshold selector widget with slider and quick-select buttons
class _ScoreThresholdSelector extends ConsumerWidget {
  const _ScoreThresholdSelector();

  void _setScoreThreshold(WidgetRef ref, double newThreshold) {
    debugPrint('[Settings] Score threshold changed: ${(newThreshold * 100).round()}%');
    
    // Update provider state (persisted)
    ref.read(scoreThresholdProvider.notifier).setValue(newThreshold);
    
    // DIRECT CALL to backend - no listener needed!
    ref.read(videoStreamProvider.notifier).setScoreThreshold(newThreshold);
  }

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final threshold = ref.watch(scoreThresholdProvider);
    final percentage = (threshold * 100).round();

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            const Text(
              'Detection Confidence Threshold',
              style: TextStyle(
                fontSize: 14,
                color: G20Colors.textPrimaryDark,
              ),
            ),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
              decoration: BoxDecoration(
                color: G20Colors.primary.withValues(alpha: 0.2),
                borderRadius: BorderRadius.circular(4),
              ),
              child: Text(
                '$percentage%',
                style: const TextStyle(
                  fontWeight: FontWeight.bold,
                  color: G20Colors.primary,
                ),
              ),
            ),
          ],
        ),
        const SizedBox(height: 4),
        Text(
          'Only show detections with confidence above this threshold',
          style: TextStyle(
            fontSize: 11,
            color: G20Colors.textSecondaryDark,
          ),
        ),
        const SizedBox(height: 8),
        Slider(
          value: threshold,
          min: 0.0,
          max: 1.0,
          divisions: 20,
          label: '$percentage%',
          onChanged: (v) => _setScoreThreshold(ref, v),
          onChangeEnd: (v) => _setScoreThreshold(ref, v),  // Send on release
        ),
        const SizedBox(height: 8),
        Row(
          children: [
            _ThresholdOption(
              label: '50%',
              value: 0.5,
              selected: (threshold - 0.5).abs() < 0.01,
              onTap: () => _setScoreThreshold(ref, 0.5),
            ),
            const SizedBox(width: 8),
            _ThresholdOption(
              label: '75%',
              value: 0.75,
              selected: (threshold - 0.75).abs() < 0.01,
              onTap: () => _setScoreThreshold(ref, 0.75),
            ),
            const SizedBox(width: 8),
            _ThresholdOption(
              label: '90%',
              value: 0.9,
              selected: (threshold - 0.9).abs() < 0.01,
              onTap: () => _setScoreThreshold(ref, 0.9),
            ),
            const SizedBox(width: 8),
            _ThresholdOption(
              label: '95%',
              value: 0.95,
              selected: (threshold - 0.95).abs() < 0.01,
              onTap: () => _setScoreThreshold(ref, 0.95),
            ),
          ],
        ),
      ],
    );
  }
}

class _ThresholdOption extends StatelessWidget {
  final String label;
  final double value;
  final bool selected;
  final VoidCallback onTap;

  const _ThresholdOption({
    required this.label,
    required this.value,
    required this.selected,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return Expanded(
      child: GestureDetector(
        onTap: onTap,
        child: Container(
          padding: const EdgeInsets.symmetric(vertical: 8),
          decoration: BoxDecoration(
            color: selected ? G20Colors.primary.withValues(alpha: 0.2) : G20Colors.cardDark,
            borderRadius: BorderRadius.circular(6),
            border: Border.all(
              color: selected ? G20Colors.primary : G20Colors.cardDark,
              width: selected ? 2 : 1,
            ),
          ),
          child: Center(
            child: Text(
              label,
              style: TextStyle(
                color: selected ? G20Colors.primary : G20Colors.textSecondaryDark,
                fontWeight: selected ? FontWeight.bold : FontWeight.normal,
                fontSize: 13,
              ),
            ),
          ),
        ),
      ),
    );
  }
}

/// FFT Size selector for waterfall resolution/performance tradeoff
/// GPU-accelerated - allows user to balance frequency resolution vs speed
class _FftSizeSelector extends ConsumerWidget {
  const _FftSizeSelector();

  void _setFftSize(WidgetRef ref, int newSize) {
    debugPrint('[Settings] FFT size button tapped: $newSize');
    
    // Update provider state (persisted automatically)
    ref.read(waterfallFftSizeProvider.notifier).setValue(newSize);
    
    // Send to backend (includes cuFFT warmup - may take 100-500ms)
    ref.read(videoStreamProvider.notifier).setFftSize(newSize);
  }

  String _estimatedTime(int size) {
    switch (size) {
      case 8192: return '~2ms';
      case 16384: return '~4ms';
      case 32768: return '~6ms';
      case 65536: return '~10ms';
      default: return '?';
    }
  }

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final fftSize = ref.watch(waterfallFftSizeProvider);
    final option = fftSizeOptions[fftSize];

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            const Text(
              'FFT Resolution',
              style: TextStyle(
                fontSize: 14,
                color: G20Colors.textPrimaryDark,
              ),
            ),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
              decoration: BoxDecoration(
                color: G20Colors.primary.withValues(alpha: 0.2),
                borderRadius: BorderRadius.circular(4),
              ),
              child: Text(
                option?.label ?? '${(fftSize / 1024).round()}K',
                style: const TextStyle(
                  fontWeight: FontWeight.bold,
                  color: G20Colors.primary,
                ),
              ),
            ),
          ],
        ),
        const SizedBox(height: 4),
        Text(
          'Higher = better frequency resolution, slower processing',
          style: TextStyle(
            fontSize: 11,
            color: G20Colors.textSecondaryDark,
          ),
        ),
        const SizedBox(height: 12),
        // FFT size buttons
        Row(
          children: fftSizeOptions.entries.map((entry) {
            final size = entry.key;
            final opt = entry.value;
            return Expanded(
              child: Padding(
                padding: EdgeInsets.only(
                  right: size != 65536 ? 6 : 0,
                ),
                child: _FftSizeOption(
                  label: opt.label,
                  sublabel: opt.sublabel,
                  selected: fftSize == size,
                  onTap: () => _setFftSize(ref, size),
                ),
              ),
            );
          }).toList(),
        ),
      ],
    );
  }
}

class _FftSizeOption extends StatelessWidget {
  final String label;
  final String sublabel;
  final bool selected;
  final VoidCallback onTap;

  const _FftSizeOption({
    required this.label,
    required this.sublabel,
    required this.selected,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.symmetric(vertical: 8),
        decoration: BoxDecoration(
          color: selected ? G20Colors.primary.withValues(alpha: 0.2) : G20Colors.cardDark,
          borderRadius: BorderRadius.circular(6),
          border: Border.all(
            color: selected ? G20Colors.primary : G20Colors.cardDark,
            width: selected ? 2 : 1,
          ),
        ),
        child: Column(
          children: [
            Text(
              label,
              style: TextStyle(
                color: selected ? G20Colors.primary : G20Colors.textSecondaryDark,
                fontWeight: selected ? FontWeight.bold : FontWeight.normal,
                fontSize: 14,
              ),
            ),
            Text(
              sublabel,
              style: TextStyle(
                color: selected 
                    ? G20Colors.primary.withValues(alpha: 0.7) 
                    : G20Colors.textSecondaryDark,
                fontSize: 10,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

/// dB Range selector - controls min/max dB for waterfall display
/// Wired to providers and sends to backend
class _DbRangeSelector extends ConsumerWidget {
  const _DbRangeSelector();

  void _setDbRange(WidgetRef ref, {double? minDb, double? maxDb}) {
    if (minDb != null) {
      ref.read(waterfallMinDbProvider.notifier).setValue(minDb);
    }
    if (maxDb != null) {
      ref.read(waterfallMaxDbProvider.notifier).setValue(maxDb);
    }
    
    // Send to backend
    final currentMin = ref.read(waterfallMinDbProvider);
    final currentMax = ref.read(waterfallMaxDbProvider);
    ref.read(videoStreamProvider.notifier).setDbRange(currentMin, currentMax);
    debugPrint('[Settings] dB range changed: $currentMin to $currentMax dB');
  }

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final minDb = ref.watch(waterfallMinDbProvider);
    final maxDb = ref.watch(waterfallMaxDbProvider);
    
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text(
          'Display Dynamic Range',
          style: TextStyle(
            fontSize: 14,
            color: G20Colors.textPrimaryDark,
          ),
        ),
        const SizedBox(height: 4),
        Text(
          'Adjust noise floor (min) and peak (max) display levels',
          style: TextStyle(
            fontSize: 11,
            color: G20Colors.textSecondaryDark,
          ),
        ),
        const SizedBox(height: 12),
        Row(
          children: [
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      Text(
                        'Min dB',
                        style: TextStyle(
                          fontSize: 12,
                          color: G20Colors.textSecondaryDark,
                        ),
                      ),
                      Text(
                        '${minDb.toStringAsFixed(0)} dB',
                        style: const TextStyle(fontWeight: FontWeight.bold),
                      ),
                    ],
                  ),
                  Slider(
                    value: minDb,
                    min: -120,
                    max: -60,
                    divisions: 12,
                    onChanged: (v) => _setDbRange(ref, minDb: v),
                  ),
                ],
              ),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      Text(
                        'Max dB',
                        style: TextStyle(
                          fontSize: 12,
                          color: G20Colors.textSecondaryDark,
                        ),
                      ),
                      Text(
                        '${maxDb.toStringAsFixed(0)} dB',
                        style: const TextStyle(fontWeight: FontWeight.bold),
                      ),
                    ],
                  ),
                  Slider(
                    value: maxDb,
                    min: -60,
                    max: 0,
                    divisions: 12,
                    onChanged: (v) => _setDbRange(ref, maxDb: v),
                  ),
                ],
              ),
            ),
          ],
        ),
        // Dynamic range indicator
        Container(
          padding: const EdgeInsets.all(8),
          decoration: BoxDecoration(
            color: G20Colors.cardDark,
            borderRadius: BorderRadius.circular(6),
          ),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Text(
                'Dynamic Range: ',
                style: TextStyle(
                  fontSize: 11,
                  color: G20Colors.textSecondaryDark,
                ),
              ),
              Text(
                '${(maxDb - minDb).toStringAsFixed(0)} dB',
                style: const TextStyle(
                  fontSize: 12,
                  fontWeight: FontWeight.bold,
                  color: G20Colors.primary,
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }
}

/// Colormap selector - changes waterfall color palette
/// Sends command to backend which switches the LUT
class _ColormapSelector extends ConsumerWidget {
  const _ColormapSelector();

  void _setColormap(WidgetRef ref, int newColormap) {
    debugPrint('[Settings] Colormap changed: ${colormapNames[newColormap]}');
    
    // Update provider state (persisted)
    ref.read(waterfallColormapProvider.notifier).setValue(newColormap);
    
    // Send to backend
    ref.read(videoStreamProvider.notifier).setColormap(newColormap);
  }

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final colormap = ref.watch(waterfallColormapProvider);

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text(
          'Colormap',
          style: TextStyle(
            fontSize: 14,
            color: G20Colors.textPrimaryDark,
          ),
        ),
        const SizedBox(height: 4),
        Text(
          'Color palette for waterfall display',
          style: TextStyle(
            fontSize: 11,
            color: G20Colors.textSecondaryDark,
          ),
        ),
        const SizedBox(height: 12),
        // Colormap buttons in a row
        Row(
          children: List.generate(colormapNames.length, (i) {
            final isSelected = colormap == i;
            return Expanded(
              child: Padding(
                padding: EdgeInsets.only(right: i < colormapNames.length - 1 ? 6 : 0),
                child: GestureDetector(
                  onTap: () => _setColormap(ref, i),
                  child: Container(
                    padding: const EdgeInsets.symmetric(vertical: 10),
                    decoration: BoxDecoration(
                      color: isSelected 
                          ? G20Colors.primary.withValues(alpha: 0.2) 
                          : G20Colors.cardDark,
                      borderRadius: BorderRadius.circular(6),
                      border: Border.all(
                        color: isSelected ? G20Colors.primary : G20Colors.cardDark,
                        width: isSelected ? 2 : 1,
                      ),
                    ),
                    child: Center(
                      child: Text(
                        colormapNames[i],
                        style: TextStyle(
                          color: isSelected ? G20Colors.primary : G20Colors.textSecondaryDark,
                          fontWeight: isSelected ? FontWeight.bold : FontWeight.normal,
                          fontSize: 11,
                        ),
                      ),
                    ),
                  ),
                ),
              ),
            );
          }),
        ),
      ],
    );
  }
}

/// Stats overlay toggle - show/hide FPS, resolution, row count on waterfall
class _StatsOverlayToggle extends ConsumerWidget {
  const _StatsOverlayToggle();

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final showStats = ref.watch(showStatsOverlayProvider);
    
    return SwitchListTile(
      title: const Text('Show Stats Overlay'),
      subtitle: Text(
        'Display FPS, resolution, and row count on waterfall',
        style: TextStyle(
          fontSize: 11,
          color: G20Colors.textSecondaryDark,
        ),
      ),
      value: showStats,
      onChanged: (v) {
        ref.read(showStatsOverlayProvider.notifier).setValue(v);
      },
      contentPadding: EdgeInsets.zero,
    );
  }
}

/// Skip first waterfall frame toggle
/// Useful to avoid garbage/initialization data on initial connection
class _SkipFirstFrameToggle extends ConsumerWidget {
  const _SkipFirstFrameToggle();

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final skipFirst = ref.watch(skipFirstWaterfallFrameProvider);
    
    return SwitchListTile(
      title: const Text('Skip First Waterfall Frame'),
      subtitle: Text(
        'Discard initial frame on connection (avoids garbage data)',
        style: TextStyle(
          fontSize: 11,
          color: G20Colors.textSecondaryDark,
        ),
      ),
      value: skipFirst,
      onChanged: (v) {
        ref.read(skipFirstWaterfallFrameProvider.notifier).setValue(v);
        // Notify video stream provider
        ref.read(videoStreamProvider.notifier).setSkipFirstFrame(v);
        debugPrint('[Settings] Skip first frame: $v');
      },
      contentPadding: EdgeInsets.zero,
    );
  }
}
