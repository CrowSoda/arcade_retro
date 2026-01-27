import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:path/path.dart' as p;
import '../../core/config/theme.dart';
import '../../core/database/signal_database.dart';

// Re-export for backwards compatibility
export '../../core/database/signal_database.dart' show SignalEntry, kModTypes, signalDatabaseProvider;

/// Database screen - View and manage signals and models
class DatabaseScreen extends ConsumerStatefulWidget {
  const DatabaseScreen({super.key});

  @override
  ConsumerState<DatabaseScreen> createState() => _DatabaseScreenState();
}

class _DatabaseScreenState extends ConsumerState<DatabaseScreen> with SingleTickerProviderStateMixin {
  late TabController _tabController;
  String _searchQuery = '';
  SignalEntry? _selectedEntry;
  List<ModelInfo> _models = [];
  bool _loadingModels = false;

  @override
  void initState() {
    super.initState();
    _tabController = TabController(length: 2, vsync: this);
    _tabController.addListener(() {
      if (_tabController.index == 1 && _models.isEmpty) {
        _loadModels();
      }
    });
  }

  @override
  void dispose() {
    _tabController.dispose();
    super.dispose();
  }

  Future<void> _loadModels() async {
    setState(() => _loadingModels = true);

    final models = <ModelInfo>[];
    final modelsDir = Directory('models');

    if (await modelsDir.exists()) {
      await for (final entity in modelsDir.list()) {
        if (entity is File) {
          final ext = p.extension(entity.path).toLowerCase();
          if (ext == '.pt' || ext == '.pth' || ext == '.onnx' || ext == '.engine') {
            final stat = await entity.stat();
            models.add(ModelInfo(
              name: p.basename(entity.path),
              path: entity.path,
              size: stat.size,
              modified: stat.modified,
              type: ext.replaceFirst('.', '').toUpperCase(),
            ));
          }
        }
      }
    }

    // Sort by modified date (newest first)
    models.sort((a, b) => b.modified.compareTo(a.modified));

    setState(() {
      _models = models;
      _loadingModels = false;
    });
  }

  void _showEditDialog(SignalEntry entry) {
    setState(() => _selectedEntry = entry);
    showDialog(
      context: context,
      builder: (context) => _EditSignalDialog(
        entry: entry,
        onSave: (updated) {
          ref.read(signalDatabaseProvider.notifier).updateSignal(entry.id, updated);
          setState(() => _selectedEntry = null);
        },
        onCancel: () => setState(() => _selectedEntry = null),
      ),
    );
  }

  void _confirmDeleteSignal(SignalEntry entry) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        backgroundColor: G20Colors.surfaceDark,
        title: const Text('Delete Signal', style: TextStyle(color: G20Colors.textPrimaryDark)),
        content: Text(
          'Are you sure you want to delete "${entry.name}"?\n\nThis will remove the signal from the database but NOT delete any training samples.',
          style: const TextStyle(color: G20Colors.textSecondaryDark),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Cancel'),
          ),
          ElevatedButton(
            onPressed: () {
              ref.read(signalDatabaseProvider.notifier).deleteSignal(entry.id);
              Navigator.pop(context);
            },
            style: ElevatedButton.styleFrom(
              backgroundColor: G20Colors.error,
              foregroundColor: Colors.white,
            ),
            child: const Text('Delete'),
          ),
        ],
      ),
    );
  }

  void _confirmDeleteModel(ModelInfo model) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        backgroundColor: G20Colors.surfaceDark,
        title: const Text('Delete Model', style: TextStyle(color: G20Colors.textPrimaryDark)),
        content: Text(
          'Are you sure you want to delete "${model.name}"?\n\nThis cannot be undone.',
          style: const TextStyle(color: G20Colors.textSecondaryDark),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Cancel'),
          ),
          ElevatedButton(
            onPressed: () async {
              Navigator.pop(context);
              try {
                await File(model.path).delete();
                _loadModels();
                if (mounted) {
                  ScaffoldMessenger.of(context).showSnackBar(
                    SnackBar(content: Text('Deleted ${model.name}'), backgroundColor: G20Colors.success),
                  );
                }
              } catch (e) {
                if (mounted) {
                  ScaffoldMessenger.of(context).showSnackBar(
                    SnackBar(content: Text('Error: $e'), backgroundColor: G20Colors.error),
                  );
                }
              }
            },
            style: ElevatedButton.styleFrom(
              backgroundColor: G20Colors.error,
              foregroundColor: Colors.white,
            ),
            child: const Text('Delete'),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: G20Colors.backgroundDark,
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Header with tabs
            Row(
              children: [
                const Icon(Icons.storage, color: G20Colors.primary),
                const SizedBox(width: 8),
                const Text(
                  'Database',
                  style: TextStyle(
                    color: G20Colors.textPrimaryDark,
                    fontSize: 20,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                const SizedBox(width: 24),
                // Tabs
                Container(
                  decoration: BoxDecoration(
                    color: G20Colors.surfaceDark,
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: TabBar(
                    controller: _tabController,
                    isScrollable: true,
                    indicator: BoxDecoration(
                      color: G20Colors.primary,
                      borderRadius: BorderRadius.circular(8),
                    ),
                    indicatorSize: TabBarIndicatorSize.tab,
                    labelColor: Colors.white,
                    unselectedLabelColor: G20Colors.textSecondaryDark,
                    labelStyle: const TextStyle(fontWeight: FontWeight.bold, fontSize: 14),
                    unselectedLabelStyle: const TextStyle(fontSize: 14),
                    dividerHeight: 0,
                    tabs: const [
                      Tab(text: '  Signals  ', icon: Icon(Icons.signal_cellular_alt, size: 18)),
                      Tab(text: '  Models  ', icon: Icon(Icons.model_training, size: 18)),
                    ],
                  ),
                ),
                const Spacer(),
                // Refresh button (for models tab)
                if (_tabController.index == 1)
                  IconButton(
                    icon: const Icon(Icons.refresh, color: G20Colors.primary),
                    onPressed: _loadModels,
                    tooltip: 'Refresh models',
                  ),
              ],
            ),
            const SizedBox(height: 16),

            // Tab content
            Expanded(
              child: TabBarView(
                controller: _tabController,
                children: [
                  _buildSignalsTab(),
                  _buildModelsTab(),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildSignalsTab() {
    final entries = ref.watch(signalDatabaseProvider);
    final filteredEntries = _searchQuery.isEmpty
        ? entries
        : entries.where((e) => e.name.toLowerCase().contains(_searchQuery.toLowerCase())).toList();

    return Column(
      children: [
        // Search box
        Row(
          children: [
            SizedBox(
              width: 300,
              child: TextField(
                onChanged: (v) => setState(() => _searchQuery = v),
                style: const TextStyle(color: G20Colors.textPrimaryDark, fontSize: 14),
                decoration: InputDecoration(
                  hintText: 'Search signals...',
                  hintStyle: TextStyle(color: G20Colors.textSecondaryDark.withOpacity(0.5)),
                  prefixIcon: const Icon(Icons.search, color: G20Colors.textSecondaryDark, size: 20),
                  filled: true,
                  fillColor: G20Colors.surfaceDark,
                  contentPadding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                  border: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(6),
                    borderSide: BorderSide.none,
                  ),
                ),
              ),
            ),
          ],
        ),
        const SizedBox(height: 12),

        // Table header
        Container(
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
          decoration: BoxDecoration(
            color: G20Colors.surfaceDark,
            borderRadius: const BorderRadius.vertical(top: Radius.circular(8)),
            border: Border.all(color: G20Colors.cardDark),
          ),
          child: Row(
            children: [
              Expanded(flex: 3, child: Text('Name', style: _headerStyle)),
              Expanded(flex: 2, child: Text('Mod Type', style: _headerStyle)),
              Expanded(flex: 2, child: Text('Mod Rate', style: _headerStyle)),
              Expanded(flex: 1, child: Text('Samples', style: _headerStyle)),
              Expanded(flex: 1, child: Text('F1', style: _headerStyle)),
              Expanded(flex: 1, child: Text('>90%', style: _headerStyle)),
              const SizedBox(width: 100), // Edit + Delete buttons
            ],
          ),
        ),

        // Table body
        Expanded(
          child: Container(
            decoration: BoxDecoration(
              color: G20Colors.backgroundDark,
              borderRadius: const BorderRadius.vertical(bottom: Radius.circular(8)),
              border: Border.all(color: G20Colors.cardDark),
            ),
            child: ListView.builder(
              itemCount: filteredEntries.length,
              itemBuilder: (context, index) {
                final entry = filteredEntries[index];
                final isSelected = _selectedEntry?.id == entry.id;
                return Container(
                  padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
                  decoration: BoxDecoration(
                    color: isSelected ? G20Colors.primary.withOpacity(0.1) : (index.isEven ? G20Colors.surfaceDark.withOpacity(0.3) : Colors.transparent),
                    border: Border(bottom: BorderSide(color: G20Colors.cardDark.withOpacity(0.5))),
                  ),
                  child: Row(
                    children: [
                      Expanded(
                        flex: 3,
                        child: Text(
                          entry.name,
                          style: const TextStyle(color: G20Colors.textPrimaryDark, fontWeight: FontWeight.w500),
                          overflow: TextOverflow.ellipsis,
                        ),
                      ),
                      Expanded(
                        flex: 2,
                        child: Text(
                          entry.modType,
                          style: TextStyle(
                            color: entry.modType == '--' ? G20Colors.textSecondaryDark : G20Colors.textPrimaryDark,
                          ),
                        ),
                      ),
                      Expanded(
                        flex: 2,
                        child: Text(
                          entry.modRate != null ? '${entry.modRate!.toStringAsFixed(0)} sps' : '--',
                          style: TextStyle(
                            color: entry.modRate == null ? G20Colors.textSecondaryDark : G20Colors.textPrimaryDark,
                          ),
                        ),
                      ),
                      Expanded(
                        flex: 1,
                        child: Text(
                          '${entry.sampleCount}',
                          style: const TextStyle(color: G20Colors.textPrimaryDark),
                        ),
                      ),
                      Expanded(
                        flex: 1,
                        child: Text(
                          entry.f1Score != null ? entry.f1Score!.toStringAsFixed(2) : '--',
                          style: TextStyle(
                            color: entry.f1Score != null ? G20Colors.success : G20Colors.textSecondaryDark,
                            fontWeight: entry.f1Score != null ? FontWeight.w600 : FontWeight.normal,
                          ),
                        ),
                      ),
                      Expanded(
                        flex: 1,
                        child: Text(
                          '${entry.timesAbove90}',
                          style: const TextStyle(color: G20Colors.textPrimaryDark),
                        ),
                      ),
                      SizedBox(
                        width: 100,
                        child: Row(
                          mainAxisAlignment: MainAxisAlignment.end,
                          children: [
                            IconButton(
                              icon: const Icon(Icons.edit, size: 18),
                              color: G20Colors.primary,
                              onPressed: () => _showEditDialog(entry),
                              tooltip: 'Edit',
                            ),
                            IconButton(
                              icon: const Icon(Icons.delete, size: 18),
                              color: G20Colors.error,
                              onPressed: () => _confirmDeleteSignal(entry),
                              tooltip: 'Delete',
                            ),
                          ],
                        ),
                      ),
                    ],
                  ),
                );
              },
            ),
          ),
        ),

        // Footer info
        Padding(
          padding: const EdgeInsets.only(top: 8),
          child: Text(
            '${filteredEntries.length} signals • Click edit to modify, delete to remove',
            style: const TextStyle(color: G20Colors.textSecondaryDark, fontSize: 12),
          ),
        ),
      ],
    );
  }

  Widget _buildModelsTab() {
    if (_loadingModels) {
      return const Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            CircularProgressIndicator(color: G20Colors.primary),
            SizedBox(height: 16),
            Text('Loading models...', style: TextStyle(color: G20Colors.textSecondaryDark)),
          ],
        ),
      );
    }

    if (_models.isEmpty) {
      return Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Icon(Icons.model_training, size: 64, color: G20Colors.textSecondaryDark),
            const SizedBox(height: 16),
            const Text('No models found', style: TextStyle(color: G20Colors.textPrimaryDark, fontSize: 18)),
            const SizedBox(height: 8),
            const Text('Train a model to see it here', style: TextStyle(color: G20Colors.textSecondaryDark)),
            const SizedBox(height: 16),
            ElevatedButton.icon(
              onPressed: _loadModels,
              icon: const Icon(Icons.refresh),
              label: const Text('Refresh'),
              style: ElevatedButton.styleFrom(
                backgroundColor: G20Colors.primary,
                foregroundColor: Colors.white,
              ),
            ),
          ],
        ),
      );
    }

    return Column(
      children: [
        // Table header
        Container(
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
          decoration: BoxDecoration(
            color: G20Colors.surfaceDark,
            borderRadius: const BorderRadius.vertical(top: Radius.circular(8)),
            border: Border.all(color: G20Colors.cardDark),
          ),
          child: Row(
            children: [
              Expanded(flex: 4, child: Text('Model Name', style: _headerStyle)),
              Expanded(flex: 1, child: Text('Type', style: _headerStyle)),
              Expanded(flex: 2, child: Text('Size', style: _headerStyle)),
              Expanded(flex: 3, child: Text('Last Modified', style: _headerStyle)),
              const SizedBox(width: 50), // Delete button
            ],
          ),
        ),

        // Table body
        Expanded(
          child: Container(
            decoration: BoxDecoration(
              color: G20Colors.backgroundDark,
              borderRadius: const BorderRadius.vertical(bottom: Radius.circular(8)),
              border: Border.all(color: G20Colors.cardDark),
            ),
            child: ListView.builder(
              itemCount: _models.length,
              itemBuilder: (context, index) {
                final model = _models[index];
                return Container(
                  padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
                  decoration: BoxDecoration(
                    color: index.isEven ? G20Colors.surfaceDark.withOpacity(0.3) : Colors.transparent,
                    border: Border(bottom: BorderSide(color: G20Colors.cardDark.withOpacity(0.5))),
                  ),
                  child: Row(
                    children: [
                      Expanded(
                        flex: 4,
                        child: Row(
                          children: [
                            Icon(
                              _getModelIcon(model.type),
                              size: 18,
                              color: _getModelColor(model.type),
                            ),
                            const SizedBox(width: 8),
                            Expanded(
                              child: Text(
                                model.name,
                                style: const TextStyle(color: G20Colors.textPrimaryDark, fontWeight: FontWeight.w500),
                                overflow: TextOverflow.ellipsis,
                              ),
                            ),
                          ],
                        ),
                      ),
                      Expanded(
                        flex: 1,
                        child: Container(
                          padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
                          decoration: BoxDecoration(
                            color: _getModelColor(model.type).withOpacity(0.2),
                            borderRadius: BorderRadius.circular(4),
                          ),
                          child: Text(
                            model.type,
                            style: TextStyle(
                              color: _getModelColor(model.type),
                              fontSize: 11,
                              fontWeight: FontWeight.bold,
                            ),
                            textAlign: TextAlign.center,
                          ),
                        ),
                      ),
                      Expanded(
                        flex: 2,
                        child: Text(
                          _formatFileSize(model.size),
                          style: const TextStyle(color: G20Colors.textPrimaryDark),
                        ),
                      ),
                      Expanded(
                        flex: 3,
                        child: Text(
                          _formatDate(model.modified),
                          style: const TextStyle(color: G20Colors.textSecondaryDark),
                        ),
                      ),
                      SizedBox(
                        width: 50,
                        child: IconButton(
                          icon: const Icon(Icons.delete, size: 18),
                          color: G20Colors.error,
                          onPressed: () => _confirmDeleteModel(model),
                          tooltip: 'Delete model',
                        ),
                      ),
                    ],
                  ),
                );
              },
            ),
          ),
        ),

        // Footer info
        Padding(
          padding: const EdgeInsets.only(top: 8),
          child: Text(
            '${_models.length} models • Looking in models/ directory',
            style: const TextStyle(color: G20Colors.textSecondaryDark, fontSize: 12),
          ),
        ),
      ],
    );
  }

  IconData _getModelIcon(String type) {
    switch (type) {
      case 'ONNX': return Icons.hub;
      case 'ENGINE': return Icons.bolt;
      case 'PT':
      case 'PTH': return Icons.memory;
      default: return Icons.model_training;
    }
  }

  Color _getModelColor(String type) {
    switch (type) {
      case 'ONNX': return Colors.orange;
      case 'ENGINE': return Colors.green;
      case 'PT':
      case 'PTH': return Colors.blue;
      default: return G20Colors.textSecondaryDark;
    }
  }

  String _formatFileSize(int bytes) {
    if (bytes < 1024) return '$bytes B';
    if (bytes < 1024 * 1024) return '${(bytes / 1024).toStringAsFixed(1)} KB';
    if (bytes < 1024 * 1024 * 1024) return '${(bytes / 1024 / 1024).toStringAsFixed(1)} MB';
    return '${(bytes / 1024 / 1024 / 1024).toStringAsFixed(2)} GB';
  }

  String _formatDate(DateTime dt) {
    return '${dt.year}-${dt.month.toString().padLeft(2, '0')}-${dt.day.toString().padLeft(2, '0')} '
           '${dt.hour.toString().padLeft(2, '0')}:${dt.minute.toString().padLeft(2, '0')}';
  }

  TextStyle get _headerStyle => const TextStyle(
    color: G20Colors.textSecondaryDark,
    fontSize: 12,
    fontWeight: FontWeight.w600,
  );
}

/// Model info for the models tab
class ModelInfo {
  final String name;
  final String path;
  final int size;
  final DateTime modified;
  final String type;

  ModelInfo({
    required this.name,
    required this.path,
    required this.size,
    required this.modified,
    required this.type,
  });
}

/// Edit dialog for signal metadata
class _EditSignalDialog extends StatefulWidget {
  final SignalEntry entry;
  final Function(SignalEntry) onSave;
  final VoidCallback onCancel;

  const _EditSignalDialog({
    required this.entry,
    required this.onSave,
    required this.onCancel,
  });

  @override
  State<_EditSignalDialog> createState() => _EditSignalDialogState();
}

class _EditSignalDialogState extends State<_EditSignalDialog> {
  late TextEditingController _nameController;
  late TextEditingController _modRateController;
  late TextEditingController _bandwidthController;
  late TextEditingController _notesController;
  late String _selectedModType;

  @override
  void initState() {
    super.initState();
    _nameController = TextEditingController(text: widget.entry.name);
    _modRateController = TextEditingController(text: widget.entry.modRate?.toString() ?? '');
    _bandwidthController = TextEditingController(text: widget.entry.bandwidth?.toString() ?? '');
    _notesController = TextEditingController(text: widget.entry.notes ?? '');
    _selectedModType = widget.entry.modType;
  }

  @override
  void dispose() {
    _nameController.dispose();
    _modRateController.dispose();
    _bandwidthController.dispose();
    _notesController.dispose();
    super.dispose();
  }

  void _save() {
    final updated = widget.entry.copyWith(
      name: _nameController.text,
      modType: _selectedModType,
      modRate: double.tryParse(_modRateController.text),
      bandwidth: double.tryParse(_bandwidthController.text),
      notes: _notesController.text.isEmpty ? null : _notesController.text,
    );
    widget.onSave(updated);
    Navigator.of(context).pop();
  }

  @override
  Widget build(BuildContext context) {
    return Dialog(
      backgroundColor: G20Colors.surfaceDark,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      child: Container(
        width: 400,
        padding: const EdgeInsets.all(24),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Header
            Row(
              children: [
                const Icon(Icons.edit, color: G20Colors.primary),
                const SizedBox(width: 8),
                Expanded(
                  child: Text(
                    'Edit: ${widget.entry.name}',
                    style: const TextStyle(
                      color: G20Colors.textPrimaryDark,
                      fontSize: 18,
                      fontWeight: FontWeight.bold,
                    ),
                    overflow: TextOverflow.ellipsis,
                  ),
                ),
                IconButton(
                  icon: const Icon(Icons.close, color: G20Colors.textSecondaryDark),
                  onPressed: () {
                    widget.onCancel();
                    Navigator.of(context).pop();
                  },
                ),
              ],
            ),
            const SizedBox(height: 20),

            // Form fields
            _buildTextField('Name', _nameController),
            const SizedBox(height: 12),

            // Mod Type dropdown
            const Text('Mod Type', style: TextStyle(color: G20Colors.textSecondaryDark, fontSize: 12)),
            const SizedBox(height: 4),
            DropdownButtonFormField<String>(
              value: _selectedModType,
              dropdownColor: G20Colors.surfaceDark,
              style: const TextStyle(color: G20Colors.textPrimaryDark),
              decoration: _inputDecoration,
              items: kModTypes.map((t) => DropdownMenuItem(value: t, child: Text(t))).toList(),
              onChanged: (v) => setState(() => _selectedModType = v ?? '--'),
            ),
            const SizedBox(height: 12),

            Row(
              children: [
                Expanded(child: _buildTextField('Mod Rate (sps)', _modRateController, isNumber: true)),
                const SizedBox(width: 12),
                Expanded(child: _buildTextField('Bandwidth (kHz)', _bandwidthController, isNumber: true)),
              ],
            ),
            const SizedBox(height: 12),

            _buildTextField('Notes', _notesController, maxLines: 2),
            const SizedBox(height: 20),

            // Stats (read-only)
            Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: G20Colors.backgroundDark,
                borderRadius: BorderRadius.circular(6),
              ),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceAround,
                children: [
                  _StatItem(label: 'Samples', value: '${widget.entry.sampleCount}'),
                  _StatItem(label: 'F1 Score', value: widget.entry.f1Score?.toStringAsFixed(2) ?? '--'),
                  _StatItem(label: '>90% Count', value: '${widget.entry.timesAbove90}'),
                ],
              ),
            ),
            const SizedBox(height: 20),

            // Buttons
            Row(
              mainAxisAlignment: MainAxisAlignment.end,
              children: [
                TextButton(
                  onPressed: () {
                    widget.onCancel();
                    Navigator.of(context).pop();
                  },
                  child: const Text('Cancel'),
                ),
                const SizedBox(width: 12),
                ElevatedButton.icon(
                  onPressed: _save,
                  icon: const Icon(Icons.save, size: 18),
                  label: const Text('Save'),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: G20Colors.primary,
                    foregroundColor: Colors.white,
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildTextField(String label, TextEditingController controller, {bool isNumber = false, int maxLines = 1}) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(label, style: const TextStyle(color: G20Colors.textSecondaryDark, fontSize: 12)),
        const SizedBox(height: 4),
        TextField(
          controller: controller,
          style: const TextStyle(color: G20Colors.textPrimaryDark),
          keyboardType: isNumber ? TextInputType.number : TextInputType.text,
          maxLines: maxLines,
          decoration: _inputDecoration,
        ),
      ],
    );
  }

  InputDecoration get _inputDecoration => InputDecoration(
    filled: true,
    fillColor: G20Colors.backgroundDark,
    contentPadding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
    border: OutlineInputBorder(
      borderRadius: BorderRadius.circular(6),
      borderSide: const BorderSide(color: G20Colors.cardDark),
    ),
    enabledBorder: OutlineInputBorder(
      borderRadius: BorderRadius.circular(6),
      borderSide: const BorderSide(color: G20Colors.cardDark),
    ),
    focusedBorder: OutlineInputBorder(
      borderRadius: BorderRadius.circular(6),
      borderSide: const BorderSide(color: G20Colors.primary),
    ),
  );
}

class _StatItem extends StatelessWidget {
  final String label;
  final String value;

  const _StatItem({required this.label, required this.value});

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Text(value, style: const TextStyle(color: G20Colors.textPrimaryDark, fontWeight: FontWeight.bold, fontSize: 16)),
        Text(label, style: const TextStyle(color: G20Colors.textSecondaryDark, fontSize: 11)),
      ],
    );
  }
}
