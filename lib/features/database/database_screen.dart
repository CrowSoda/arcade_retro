import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../../core/config/theme.dart';

/// Modulation types for dropdown
const List<String> kModTypes = [
  '--',
  'BPSK',
  'QPSK',
  'OQPSK',
  '8PSK',
  '16QAM',
  '64QAM',
  'OFDM',
  'FSK',
  'GFSK',
  'MSK',
  'GMSK',
  'FHSS',
  'DSSS',
  'AM',
  'FM',
  'Chirp',
  'Burst',
  'Unknown',
];

/// Signal entry for the database
class SignalEntry {
  final String id;
  String name;
  String modType;
  double? modRate;
  double? bandwidth;
  String? notes;
  int sampleCount;
  double? f1Score;
  int timesAbove90;

  SignalEntry({
    required this.id,
    required this.name,
    this.modType = '--',
    this.modRate,
    this.bandwidth,
    this.notes,
    this.sampleCount = 0,
    this.f1Score,
    this.timesAbove90 = 0,
  });

  SignalEntry copyWith({
    String? name,
    String? modType,
    double? modRate,
    double? bandwidth,
    String? notes,
    int? sampleCount,
    double? f1Score,
    int? timesAbove90,
  }) {
    return SignalEntry(
      id: id,
      name: name ?? this.name,
      modType: modType ?? this.modType,
      modRate: modRate ?? this.modRate,
      bandwidth: bandwidth ?? this.bandwidth,
      notes: notes ?? this.notes,
      sampleCount: sampleCount ?? this.sampleCount,
      f1Score: f1Score ?? this.f1Score,
      timesAbove90: timesAbove90 ?? this.timesAbove90,
    );
  }
}

/// Provider for signal database entries
final signalEntriesProvider = StateNotifierProvider<SignalEntriesNotifier, List<SignalEntry>>((ref) {
  return SignalEntriesNotifier();
});

class SignalEntriesNotifier extends StateNotifier<List<SignalEntry>> {
  SignalEntriesNotifier() : super([
    SignalEntry(id: '1', name: 'creamy_chicken', modType: '--', sampleCount: 127, f1Score: 0.91, timesAbove90: 47),
    SignalEntry(id: '2', name: 'lte_uplink', modType: 'OFDM', modRate: 15000, bandwidth: 10000, sampleCount: 89, f1Score: 0.87, timesAbove90: 23),
    SignalEntry(id: '3', name: 'wifi_24', modType: 'OFDM', modRate: 20000, bandwidth: 20000, sampleCount: 156, f1Score: 0.82, timesAbove90: 56),
    SignalEntry(id: '4', name: 'bluetooth', modType: 'GFSK', modRate: 1000000, bandwidth: 1000, sampleCount: 78, f1Score: 0.79, timesAbove90: 18),
    SignalEntry(id: '5', name: 'unk_220001ZJAN26_825', modType: '--', sampleCount: 34, timesAbove90: 8),
  ]);

  void updateEntry(String id, SignalEntry updated) {
    state = [
      for (final entry in state)
        entry.id == id ? updated : entry,
    ];
  }

  void deleteEntry(String id) {
    state = state.where((e) => e.id != id).toList();
  }
}

/// Database screen - View and edit signal metadata
class DatabaseScreen extends ConsumerStatefulWidget {
  const DatabaseScreen({super.key});

  @override
  ConsumerState<DatabaseScreen> createState() => _DatabaseScreenState();
}

class _DatabaseScreenState extends ConsumerState<DatabaseScreen> {
  String _searchQuery = '';
  SignalEntry? _selectedEntry;

  void _showEditDialog(SignalEntry entry) {
    setState(() => _selectedEntry = entry);
    showDialog(
      context: context,
      builder: (context) => _EditSignalDialog(
        entry: entry,
        onSave: (updated) {
          ref.read(signalEntriesProvider.notifier).updateEntry(entry.id, updated);
          setState(() => _selectedEntry = null);
        },
        onCancel: () => setState(() => _selectedEntry = null),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final entries = ref.watch(signalEntriesProvider);
    final filteredEntries = _searchQuery.isEmpty
        ? entries
        : entries.where((e) => e.name.toLowerCase().contains(_searchQuery.toLowerCase())).toList();

    return Scaffold(
      backgroundColor: G20Colors.backgroundDark,
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Header with search
            Row(
              children: [
                const Icon(Icons.storage, color: G20Colors.primary),
                const SizedBox(width: 8),
                const Text(
                  'Signal Database',
                  style: TextStyle(
                    color: G20Colors.textPrimaryDark,
                    fontSize: 20,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                const Spacer(),
                // Search box
                SizedBox(
                  width: 250,
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
            const SizedBox(height: 16),
            
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
                  const SizedBox(width: 50), // Edit button space
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
                    return InkWell(
                      onTap: () => _showEditDialog(entry),
                      child: Container(
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
                              width: 50,
                              child: IconButton(
                                icon: const Icon(Icons.edit, size: 18),
                                color: G20Colors.primary,
                                onPressed: () => _showEditDialog(entry),
                              ),
                            ),
                          ],
                        ),
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
                '${filteredEntries.length} signals • Click row to edit',
                style: const TextStyle(color: G20Colors.textSecondaryDark, fontSize: 12),
              ),
            ),
          ],
        ),
      ),
    );
  }

  TextStyle get _headerStyle => const TextStyle(
    color: G20Colors.textSecondaryDark,
    fontSize: 12,
    fontWeight: FontWeight.w600,
  );
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
