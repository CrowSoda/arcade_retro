import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../../core/config/theme.dart';
import '../../core/utils/dtg_formatter.dart';

/// Available models for config selection
class ModelInfo {
  final String id;
  final String name;
  final String path;
  final double? f1Score;
  final bool isSelected;

  const ModelInfo({
    required this.id,
    required this.name,
    required this.path,
    this.f1Score,
    this.isSelected = false,
  });

  ModelInfo copyWith({bool? isSelected}) => ModelInfo(
    id: id,
    name: name,
    path: path,
    f1Score: f1Score,
    isSelected: isSelected ?? this.isSelected,
  );
}

/// Saved config info
class SavedConfig {
  final String name;
  final String filename;
  final int modelCount;
  final DateTime createdAt;

  const SavedConfig({
    required this.name,
    required this.filename,
    required this.modelCount,
    required this.createdAt,
  });
}

/// Provider for available models
final availableModelsProvider = StateProvider<List<ModelInfo>>((ref) {
  return [
    ModelInfo(id: '1', name: 'creamy_chicken', path: 'models/creamy_chicken_v3.trt', f1Score: 0.91),
    ModelInfo(id: '2', name: 'lte_uplink', path: 'models/lte_uplink_v2.trt', f1Score: 0.87),
    ModelInfo(id: '3', name: 'wifi_24', path: 'models/wifi_24_v1.trt', f1Score: 0.82),
    ModelInfo(id: '4', name: 'bluetooth', path: 'models/bluetooth_v1.trt', f1Score: 0.79),
  ];
});

/// Provider for saved configs
final savedConfigsProvider = StateProvider<List<SavedConfig>>((ref) {
  return [
    SavedConfig(name: 'alpha_strike', filename: 'config_alpha_strike_220001ZJAN26_2.json', modelCount: 2, createdAt: DateTime.now().subtract(const Duration(days: 1))),
    SavedConfig(name: 'full_spectrum', filename: 'config_full_spectrum_211530ZJAN26_4.json', modelCount: 4, createdAt: DateTime.now().subtract(const Duration(days: 3))),
  ];
});

/// Config screen - Generate mission configs from available models
class ConfigScreen extends ConsumerStatefulWidget {
  const ConfigScreen({super.key});

  @override
  ConsumerState<ConfigScreen> createState() => _ConfigScreenState();
}

class _ConfigScreenState extends ConsumerState<ConfigScreen> {
  final _configNameController = TextEditingController();
  Set<String> _selectedModelIds = {};

  @override
  void dispose() {
    _configNameController.dispose();
    super.dispose();
  }

  void _toggleModel(String id) {
    setState(() {
      if (_selectedModelIds.contains(id)) {
        _selectedModelIds.remove(id);
      } else {
        _selectedModelIds.add(id);
      }
    });
  }

  void _generateConfig() {
    final name = _configNameController.text.trim();
    if (name.isEmpty) {
      debugPrint('Please enter a config name');
      return;
    }
    if (_selectedModelIds.isEmpty) {
      debugPrint('Please select at least one model');
      return;
    }

    final dtg = formatDTG(DateTime.now()).replaceAll(' ', '').replaceAll(':', '');
    final filename = 'config_${name}_${dtg}_${_selectedModelIds.length}.json';

    // Add to saved configs
    final newConfig = SavedConfig(
      name: name,
      filename: filename,
      modelCount: _selectedModelIds.length,
      createdAt: DateTime.now(),
    );

    ref.read(savedConfigsProvider.notifier).update((state) => [...state, newConfig]);

    // Clear form
    _configNameController.clear();
    setState(() => _selectedModelIds = {});

    debugPrint('✅ Config "$filename" created!');
  }

  @override
  Widget build(BuildContext context) {
    final models = ref.watch(availableModelsProvider);
    final savedConfigs = ref.watch(savedConfigsProvider);

    return Scaffold(
      backgroundColor: G20Colors.backgroundDark,
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Header
            Row(
              children: [
                const Icon(Icons.inventory_2, color: G20Colors.primary),
                const SizedBox(width: 8),
                const Text(
                  'Mission Config Generator',
                  style: TextStyle(
                    color: G20Colors.textPrimaryDark,
                    fontSize: 20,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 16),
            
            // Main content
            Expanded(
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  // Left: Model selection
                  Expanded(
                    flex: 2,
                    child: Container(
                      padding: const EdgeInsets.all(16),
                      decoration: BoxDecoration(
                        color: G20Colors.surfaceDark,
                        borderRadius: BorderRadius.circular(8),
                        border: Border.all(color: G20Colors.cardDark),
                      ),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          const Text(
                            'Available Models',
                            style: TextStyle(
                              color: G20Colors.textPrimaryDark,
                              fontSize: 16,
                              fontWeight: FontWeight.w600,
                            ),
                          ),
                          const SizedBox(height: 12),
                          Expanded(
                            child: ListView.builder(
                              itemCount: models.length,
                              itemBuilder: (context, index) {
                                final model = models[index];
                                final isSelected = _selectedModelIds.contains(model.id);
                                return _ModelTile(
                                  model: model,
                                  isSelected: isSelected,
                                  onTap: () => _toggleModel(model.id),
                                );
                              },
                            ),
                          ),
                          const Divider(color: G20Colors.cardDark),
                          const SizedBox(height: 8),
                          // Config name input
                          TextField(
                            controller: _configNameController,
                            style: const TextStyle(color: G20Colors.textPrimaryDark),
                            decoration: InputDecoration(
                              labelText: 'Config Name',
                              labelStyle: const TextStyle(color: G20Colors.textSecondaryDark),
                              hintText: 'e.g., alpha_strike',
                              hintStyle: TextStyle(color: G20Colors.textSecondaryDark.withOpacity(0.5)),
                              filled: true,
                              fillColor: G20Colors.backgroundDark,
                              border: OutlineInputBorder(
                                borderRadius: BorderRadius.circular(4),
                                borderSide: const BorderSide(color: G20Colors.cardDark),
                              ),
                              enabledBorder: OutlineInputBorder(
                                borderRadius: BorderRadius.circular(4),
                                borderSide: const BorderSide(color: G20Colors.cardDark),
                              ),
                              focusedBorder: OutlineInputBorder(
                                borderRadius: BorderRadius.circular(4),
                                borderSide: const BorderSide(color: G20Colors.primary),
                              ),
                            ),
                          ),
                          const SizedBox(height: 12),
                          // Generate button
                          SizedBox(
                            width: double.infinity,
                            child: ElevatedButton.icon(
                              onPressed: _generateConfig,
                              icon: const Icon(Icons.save),
                              label: Text('Generate Config (${_selectedModelIds.length} models)'),
                              style: ElevatedButton.styleFrom(
                                backgroundColor: G20Colors.primary,
                                foregroundColor: Colors.white,
                                padding: const EdgeInsets.symmetric(vertical: 12),
                              ),
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                  const SizedBox(width: 16),
                  // Right column: Saved configs + Tuning controls
                  Expanded(
                    flex: 1,
                    child: Column(
                      children: [
                        // Saved configs (top)
                        Expanded(
                          flex: 1,
                          child: Container(
                            padding: const EdgeInsets.all(16),
                            decoration: BoxDecoration(
                              color: G20Colors.surfaceDark,
                              borderRadius: BorderRadius.circular(8),
                              border: Border.all(color: G20Colors.cardDark),
                            ),
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                const Text(
                                  'Saved Configs',
                                  style: TextStyle(
                                    color: G20Colors.textPrimaryDark,
                                    fontSize: 16,
                                    fontWeight: FontWeight.w600,
                                  ),
                                ),
                                const SizedBox(height: 12),
                                Expanded(
                                  child: savedConfigs.isEmpty
                                      ? const Center(
                                          child: Text(
                                            'No configs saved yet',
                                            style: TextStyle(color: G20Colors.textSecondaryDark),
                                          ),
                                        )
                                      : ListView.builder(
                                          itemCount: savedConfigs.length,
                                          itemBuilder: (context, index) {
                                            final config = savedConfigs[index];
                                            return _SavedConfigTile(config: config);
                                          },
                                        ),
                                ),
                              ],
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _ModelTile extends StatelessWidget {
  final ModelInfo model;
  final bool isSelected;
  final VoidCallback onTap;

  const _ModelTile({
    required this.model,
    required this.isSelected,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return InkWell(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
        margin: const EdgeInsets.only(bottom: 8),
        decoration: BoxDecoration(
          color: isSelected ? G20Colors.primary.withOpacity(0.2) : G20Colors.backgroundDark,
          borderRadius: BorderRadius.circular(6),
          border: Border.all(
            color: isSelected ? G20Colors.primary : G20Colors.cardDark,
            width: isSelected ? 2 : 1,
          ),
        ),
        child: Row(
          children: [
            Icon(
              isSelected ? Icons.check_box : Icons.check_box_outline_blank,
              color: isSelected ? G20Colors.primary : G20Colors.textSecondaryDark,
              size: 20,
            ),
            const SizedBox(width: 12),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    model.name,
                    style: const TextStyle(
                      color: G20Colors.textPrimaryDark,
                      fontWeight: FontWeight.w500,
                    ),
                  ),
                  Text(
                    model.path,
                    style: const TextStyle(
                      color: G20Colors.textSecondaryDark,
                      fontSize: 11,
                    ),
                  ),
                ],
              ),
            ),
            if (model.f1Score != null)
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                decoration: BoxDecoration(
                  color: G20Colors.success.withOpacity(0.2),
                  borderRadius: BorderRadius.circular(4),
                ),
                child: Text(
                  'F1: ${model.f1Score!.toStringAsFixed(2)}',
                  style: const TextStyle(
                    color: G20Colors.success,
                    fontSize: 11,
                    fontWeight: FontWeight.w500,
                  ),
                ),
              ),
          ],
        ),
      ),
    );
  }
}

class _SavedConfigTile extends StatelessWidget {
  final SavedConfig config;

  const _SavedConfigTile({required this.config});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(12),
      margin: const EdgeInsets.only(bottom: 8),
      decoration: BoxDecoration(
        color: G20Colors.backgroundDark,
        borderRadius: BorderRadius.circular(6),
        border: Border.all(color: G20Colors.cardDark),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              const Icon(Icons.description, color: G20Colors.primary, size: 16),
              const SizedBox(width: 8),
              Expanded(
                child: Text(
                  config.name,
                  style: const TextStyle(
                    color: G20Colors.textPrimaryDark,
                    fontWeight: FontWeight.w500,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 4),
          Text(
            '${config.modelCount} models',
            style: const TextStyle(color: G20Colors.textSecondaryDark, fontSize: 11),
          ),
          Text(
            formatDTG(config.createdAt),
            style: const TextStyle(color: G20Colors.textSecondaryDark, fontSize: 10),
          ),
        ],
      ),
    );
  }
}
