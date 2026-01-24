import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../../core/config/theme.dart';

/// Auto-tune delay setting (null = disabled, otherwise seconds)
final autoTuneDelayProvider = StateProvider<int?>((ref) => null);  // Disabled by default

/// Score threshold for detection filtering (0.0 - 1.0)
/// Default 0.9 (90%) - only show high confidence detections
final scoreThresholdProvider = StateProvider<double>((ref) => 0.9);

/// Waterfall display time span in seconds
/// Controls how many seconds of data the waterfall shows
/// Default 5s - good balance of detail and history
final waterfallTimeSpanProvider = StateProvider<double>((ref) => 5.0);

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

  // Display settings
  double _minDb = -100;
  double _maxDb = -20;
  int _colormap = 0;

  // System info
  final String _appVersion = '1.0.0';
  final String _backendVersion = 'Not Connected';

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
              const _WaterfallTimeSpanSelector(),
              const SizedBox(height: 16),
              Row(
                children: [
                  Expanded(
                    child: _buildSlider(
                      label: 'Min dB',
                      value: _minDb,
                      min: -120,
                      max: -60,
                      divisions: 12,
                      suffix: ' dB',
                      onChanged: (v) => setState(() => _minDb = v),
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: _buildSlider(
                      label: 'Max dB',
                      value: _maxDb,
                      min: -60,
                      max: 0,
                      divisions: 12,
                      suffix: ' dB',
                      onChanged: (v) => setState(() => _maxDb = v),
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 12),
              _buildDropdown(
                label: 'Colormap',
                value: _colormap,
                items: const [
                  'Viridis',
                  'Plasma',
                  'Inferno',
                  'Grayscale',
                ],
                onChanged: (v) => setState(() => _colormap = v ?? 0),
              ),
            ],
          ),
          const SizedBox(height: 24),
          // Tuning Settings
          _buildSection(
            title: 'Tuning',
            icon: Icons.tune,
            children: [
              _AutoTuneDelaySelector(),
            ],
          ),
          const SizedBox(height: 24),
          // Model Settings
          _buildSection(
            title: 'Model',
            icon: Icons.model_training,
            children: [
              const _ScoreThresholdSelector(),
              const SizedBox(height: 16),
              const Divider(),
              ListTile(
                contentPadding: EdgeInsets.zero,
                title: const Text('Active Model'),
                subtitle: const Text('modern_burst_gap_fold3.pth'),
                trailing: TextButton(
                  onPressed: () {
                    // TODO: Open model selector
                  },
                  child: const Text('Change'),
                ),
              ),
              ListTile(
                contentPadding: EdgeInsets.zero,
                title: const Text('Model Hash'),
                subtitle: const Text('a1b2c3d4e5f6...'),
              ),
              ListTile(
                contentPadding: EdgeInsets.zero,
                title: const Text('Classes'),
                subtitle: const Text('6 classes trained'),
              ),
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
              ListTile(
                contentPadding: EdgeInsets.zero,
                title: const Text('Backend Version'),
                trailing: Text(_backendVersion),
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
    debugPrint('🔗 Testing connection...');
  }

  void _saveConnectionSettings() {
    // TODO: Save settings to SharedPreferences
    debugPrint('💾 Settings saved');
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
              onTap: () => ref.read(autoTuneDelayProvider.notifier).state = null,
            ),
            const SizedBox(width: 8),
            _DelayOption(
              label: '1s',
              value: 1,
              selected: delay == 1,
              onTap: () => ref.read(autoTuneDelayProvider.notifier).state = 1,
            ),
            const SizedBox(width: 8),
            _DelayOption(
              label: '2s',
              value: 2,
              selected: delay == 2,
              onTap: () => ref.read(autoTuneDelayProvider.notifier).state = 2,
            ),
            const SizedBox(width: 8),
            _DelayOption(
              label: '5s',
              value: 5,
              selected: delay == 5,
              onTap: () => ref.read(autoTuneDelayProvider.notifier).state = 5,
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
                '${timeSpan.toStringAsFixed(0)}s',
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
        Row(
          children: [
            _TimeSpanOption(
              label: '1s',
              value: 1.0,
              selected: (timeSpan - 1.0).abs() < 0.1,
              onTap: () => ref.read(waterfallTimeSpanProvider.notifier).state = 1.0,
            ),
            const SizedBox(width: 8),
            _TimeSpanOption(
              label: '2s',
              value: 2.0,
              selected: (timeSpan - 2.0).abs() < 0.1,
              onTap: () => ref.read(waterfallTimeSpanProvider.notifier).state = 2.0,
            ),
            const SizedBox(width: 8),
            _TimeSpanOption(
              label: '5s',
              value: 5.0,
              selected: (timeSpan - 5.0).abs() < 0.1,
              onTap: () => ref.read(waterfallTimeSpanProvider.notifier).state = 5.0,
            ),
            const SizedBox(width: 8),
            _TimeSpanOption(
              label: '10s',
              value: 10.0,
              selected: (timeSpan - 10.0).abs() < 0.1,
              onTap: () => ref.read(waterfallTimeSpanProvider.notifier).state = 10.0,
            ),
            const SizedBox(width: 8),
            _TimeSpanOption(
              label: '30s',
              value: 30.0,
              selected: (timeSpan - 30.0).abs() < 0.1,
              onTap: () => ref.read(waterfallTimeSpanProvider.notifier).state = 30.0,
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

/// Score threshold selector widget with slider and quick-select buttons
class _ScoreThresholdSelector extends ConsumerWidget {
  const _ScoreThresholdSelector();

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
          onChanged: (v) => ref.read(scoreThresholdProvider.notifier).state = v,
        ),
        const SizedBox(height: 8),
        Row(
          children: [
            _ThresholdOption(
              label: '50%',
              value: 0.5,
              selected: (threshold - 0.5).abs() < 0.01,
              onTap: () => ref.read(scoreThresholdProvider.notifier).state = 0.5,
            ),
            const SizedBox(width: 8),
            _ThresholdOption(
              label: '75%',
              value: 0.75,
              selected: (threshold - 0.75).abs() < 0.01,
              onTap: () => ref.read(scoreThresholdProvider.notifier).state = 0.75,
            ),
            const SizedBox(width: 8),
            _ThresholdOption(
              label: '90%',
              value: 0.9,
              selected: (threshold - 0.9).abs() < 0.01,
              onTap: () => ref.read(scoreThresholdProvider.notifier).state = 0.9,
            ),
            const SizedBox(width: 8),
            _ThresholdOption(
              label: '95%',
              value: 0.95,
              selected: (threshold - 0.95).abs() < 0.01,
              onTap: () => ref.read(scoreThresholdProvider.notifier).state = 0.95,
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
