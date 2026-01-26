// lib/features/config/config_screen.dart
/// Config screen - Create, edit, and manage missions
/// 
/// A Mission defines:
/// - Frequency ranges to scan (e.g., 80-120 MHz, 1000-1200 MHz)
/// - RX Bandwidth
/// - Dwell time per frequency range
/// - Models with priority order (drag-drop to set signal priority)

import 'dart:convert';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:path/path.dart' as p;
import '../../core/config/theme.dart';
import '../live_detection/widgets/inputs_panel.dart' show showFadingToast;
import '../live_detection/providers/video_stream_provider.dart';

// ============================================================================
// DATA MODELS
// ============================================================================

/// Frequency range to scan
class FreqRange {
  final double startMhz;
  final double endMhz;

  const FreqRange({required this.startMhz, required this.endMhz});

  String get label => '${startMhz.toInt()}-${endMhz.toInt()} MHz';
  
  FreqRange copyWith({double? startMhz, double? endMhz}) => FreqRange(
    startMhz: startMhz ?? this.startMhz,
    endMhz: endMhz ?? this.endMhz,
  );
}

/// Model with priority for signal collection
class ModelPriority {
  final String id;
  final String name;
  final String filePath;
  final String? signalType;
  final int priority;

  const ModelPriority({
    required this.id,
    required this.name,
    required this.filePath,
    this.signalType,
    required this.priority,
  });

  ModelPriority copyWith({int? priority, String? signalType}) => ModelPriority(
    id: id,
    name: name,
    filePath: filePath,
    signalType: signalType ?? this.signalType,
    priority: priority ?? this.priority,
  );
}

/// Complete mission configuration
class Mission {
  final String id;
  final String name;
  final String description;
  final double bandwidthMhz;
  final double dwellTimeSec;
  final List<FreqRange> freqRanges;
  final List<ModelPriority> models;
  final DateTime created;
  final DateTime modified;

  const Mission({
    required this.id,
    required this.name,
    this.description = '',
    this.bandwidthMhz = 20.0,
    this.dwellTimeSec = 5.0,
    this.freqRanges = const [],
    this.models = const [],
    required this.created,
    required this.modified,
  });

  Mission copyWith({
    String? name,
    String? description,
    double? bandwidthMhz,
    double? dwellTimeSec,
    List<FreqRange>? freqRanges,
    List<ModelPriority>? models,
    DateTime? modified,
  }) => Mission(
    id: id,
    name: name ?? this.name,
    description: description ?? this.description,
    bandwidthMhz: bandwidthMhz ?? this.bandwidthMhz,
    dwellTimeSec: dwellTimeSec ?? this.dwellTimeSec,
    freqRanges: freqRanges ?? this.freqRanges,
    models: models ?? this.models,
    created: created,
    modified: modified ?? DateTime.now(),
  );
}

// ============================================================================
// PROVIDERS
// ============================================================================

/// Available Hydra heads - READ FROM REGISTRY.JSON
/// 
/// registry.json is THE source of truth maintained by the backend.
/// Contains all signals with validated metrics (F1, sample_count, version, etc.)
/// 
/// Structure:
/// {
///   "backbone_version": 1,
///   "signals": {
///     "creamy_chicken": {
///       "active_head_version": 1,
///       "sample_count": 200,
///       "f1_score": 0.93,
///       "head_path": "heads/creamy_chicken/active.pth",
///       ...
///     }
///   }
/// }
final availableModelsProvider = FutureProvider<List<ModelPriority>>((ref) async {
  final registryFile = File('models/registry.json');
  if (!await registryFile.exists()) {
    debugPrint('[Models] No registry.json found - run backend to generate');
    return [];
  }
  
  try {
    final registryJson = await registryFile.readAsString();
    final registry = json.decode(registryJson) as Map<String, dynamic>;
    final signals = registry['signals'] as Map<String, dynamic>? ?? {};
    
    final models = <ModelPriority>[];
    
    for (final entry in signals.entries) {
      final signalName = entry.key;
      final data = entry.value as Map<String, dynamic>;
      
      // Build metrics display string
      final f1Score = (data['f1_score'] as num?)?.toDouble() ?? 0.0;
      final sampleCount = data['sample_count'] as int? ?? 0;
      final version = data['active_head_version'] as int? ?? 1;
      
      String? metricsInfo;
      if (f1Score > 0) {
        metricsInfo = 'v$version • F1: ${(f1Score * 100).toInt()}% • $sampleCount samples';
      } else if (sampleCount > 0) {
        metricsInfo = 'v$version • $sampleCount samples';
      } else {
        metricsInfo = 'v$version';
      }
      
      final headPath = data['head_path'] as String? ?? 'heads/$signalName/active.pth';
      
      models.add(ModelPriority(
        id: signalName,
        name: signalName,
        filePath: headPath,
        signalType: metricsInfo,
        priority: 0,
      ));
    }
    
    models.sort((a, b) => a.name.compareTo(b.name));
    debugPrint('[Models] Loaded ${models.length} heads from registry: ${models.map((m) => m.name).join(', ')}');
    return models;
    
  } catch (e) {
    debugPrint('[Models] Error reading registry.json: $e');
    return [];
  }
});

/// All saved missions - persisted to config/missions.json
final missionsProvider = StateNotifierProvider<MissionsNotifier, List<Mission>>((ref) {
  final notifier = MissionsNotifier();
  return notifier;
});

class MissionsNotifier extends StateNotifier<List<Mission>> {
  static const _filePath = 'config/missions.json';
  
  MissionsNotifier() : super(_loadFromDiskSync());  // Load synchronously at startup

  void addMission(Mission mission) {
    state = [...state, mission];
    _saveToDisk();
  }

  void updateMission(Mission mission) {
    state = state.map((m) => m.id == mission.id ? mission : m).toList();
    _saveToDisk();
  }

  void deleteMission(String id) {
    state = state.where((m) => m.id != id).toList();
    _saveToDisk();
  }
  
  /// Synchronously load from disk at startup so list is ready immediately
  static List<Mission> _loadFromDiskSync() {
    try {
      final file = File(_filePath);
      if (file.existsSync()) {
        final jsonStr = file.readAsStringSync();
        final List<dynamic> jsonList = json.decode(jsonStr);
        final missions = jsonList.map((j) => _missionFromJsonStatic(j)).toList();
        debugPrint('[Missions] Loaded ${missions.length} missions from disk (sync)');
        return missions;
      }
    } catch (e) {
      debugPrint('[Missions] Error loading from disk: $e');
    }
    return [];
  }
  
  static Mission _missionFromJsonStatic(Map<String, dynamic> j) => Mission(
    id: j['id'],
    name: j['name'],
    description: j['description'] ?? '',
    bandwidthMhz: (j['bandwidthMhz'] as num).toDouble(),
    dwellTimeSec: (j['dwellTimeSec'] as num).toDouble(),
    freqRanges: (j['freqRanges'] as List).map((r) => FreqRange(startMhz: (r['startMhz'] as num).toDouble(), endMhz: (r['endMhz'] as num).toDouble())).toList(),
    models: (j['models'] as List).map((p) => ModelPriority(id: p['id'], name: p['name'], filePath: p['filePath'], signalType: p['signalType'], priority: p['priority'])).toList(),
    created: DateTime.parse(j['created']),
    modified: DateTime.parse(j['modified']),
  );
  
  Future<void> _saveToDisk() async {
    try {
      final dir = Directory('config');
      if (!await dir.exists()) {
        await dir.create(recursive: true);
      }
      final file = File(_filePath);
      final jsonList = state.map((m) => _missionToJson(m)).toList();
      await file.writeAsString(json.encode(jsonList));
      debugPrint('[Missions] Saved ${state.length} missions to disk');
    } catch (e) {
      debugPrint('[Missions] Error saving to disk: $e');
    }
  }
  
  Map<String, dynamic> _missionToJson(Mission m) => {
    'id': m.id,
    'name': m.name,
    'description': m.description,
    'bandwidthMhz': m.bandwidthMhz,
    'dwellTimeSec': m.dwellTimeSec,
    'freqRanges': m.freqRanges.map((r) => {'startMhz': r.startMhz, 'endMhz': r.endMhz}).toList(),
    'models': m.models.map((p) => {'id': p.id, 'name': p.name, 'filePath': p.filePath, 'signalType': p.signalType, 'priority': p.priority}).toList(),
    'created': m.created.toIso8601String(),
    'modified': m.modified.toIso8601String(),
  };
  
  Mission _missionFromJson(Map<String, dynamic> j) => Mission(
    id: j['id'],
    name: j['name'],
    description: j['description'] ?? '',
    bandwidthMhz: (j['bandwidthMhz'] as num).toDouble(),
    dwellTimeSec: (j['dwellTimeSec'] as num).toDouble(),
    freqRanges: (j['freqRanges'] as List).map((r) => FreqRange(startMhz: (r['startMhz'] as num).toDouble(), endMhz: (r['endMhz'] as num).toDouble())).toList(),
    models: (j['models'] as List).map((p) => ModelPriority(id: p['id'], name: p['name'], filePath: p['filePath'], signalType: p['signalType'], priority: p['priority'])).toList(),
    created: DateTime.parse(j['created']),
    modified: DateTime.parse(j['modified']),
  );
}

/// Currently selected mission for editing
final selectedMissionProvider = StateProvider<Mission?>((ref) => null);

/// Currently ACTIVE mission (loaded for live operation)
final activeMissionProvider = StateProvider<Mission?>((ref) => null);

/// Sidekiq NV100 specs:
/// - RF Tuning Range: 30 MHz to 6 GHz
/// - Max Channel Bandwidth: 50 MHz
/// - Sample Rates: Up to 61.44 Msamples/sec

/// Common bandwidth options (MHz) - max 50 MHz per Sidekiq NV100
const kBandwidthOptions = [5.0, 10.0, 20.0, 25.0, 40.0, 50.0];

/// Common dwell time options (seconds)
const kDwellTimeOptions = [1.0, 2.0, 3.0, 5.0, 10.0, 15.0, 30.0, 60.0];

/// RF tuning limits (MHz) - Sidekiq NV100
const kMinFreqMhz = 30.0;
const kMaxFreqMhz = 6000.0;

// ============================================================================
// CONFIG SCREEN
// ============================================================================

class ConfigScreen extends ConsumerWidget {
  const ConfigScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final missions = ref.watch(missionsProvider);
    final selectedMission = ref.watch(selectedMissionProvider);
    final availableModelsAsync = ref.watch(availableModelsProvider);

    return Scaffold(
      backgroundColor: G20Colors.backgroundDark,
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // LEFT: Mission list
            SizedBox(
              width: 280,
              child: _MissionListPanel(
                missions: missions,
                selectedId: selectedMission?.id,
                onSelect: (m) => ref.read(selectedMissionProvider.notifier).state = m,
                onNew: () => _showNewMissionDialog(context, ref),
                onDelete: (id) => ref.read(missionsProvider.notifier).deleteMission(id),
              ),
            ),
            const SizedBox(width: 16),
            
            // RIGHT: Mission editor
            Expanded(
              child: selectedMission == null
                  ? const _NoMissionSelected()
                  : availableModelsAsync.when(
                      data: (models) => _MissionEditor(
                        mission: selectedMission,
                        availableModels: models,
                        onSave: (updated) {
                          ref.read(missionsProvider.notifier).updateMission(updated);
                          ref.read(selectedMissionProvider.notifier).state = updated;
                        },
                      ),
                      loading: () => const Center(child: CircularProgressIndicator()),
                      error: (e, _) => Center(child: Text('Error loading models: $e')),
                    ),
            ),
          ],
        ),
      ),
    );
  }

  void _showNewMissionDialog(BuildContext context, WidgetRef ref) {
    final nameController = TextEditingController();
    
    showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Create New Mission'),
        content: TextField(
          controller: nameController,
          decoration: const InputDecoration(
            labelText: 'Mission Name',
            hintText: 'e.g., ISM Band Hunt',
          ),
          autofocus: true,
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx),
            child: const Text('Cancel'),
          ),
          ElevatedButton(
            onPressed: () {
              final name = nameController.text.trim();
              if (name.isNotEmpty) {
                final now = DateTime.now();
                final newMission = Mission(
                  id: 'mission_${now.millisecondsSinceEpoch}',
                  name: name,
                  created: now,
                  modified: now,
                );
                ref.read(missionsProvider.notifier).addMission(newMission);
                ref.read(selectedMissionProvider.notifier).state = newMission;
                Navigator.pop(ctx);
              }
            },
            style: ElevatedButton.styleFrom(backgroundColor: G20Colors.primary),
            child: const Text('Create'),
          ),
        ],
      ),
    );
  }
}

// ============================================================================
// MISSION LIST PANEL
// ============================================================================

class _MissionListPanel extends StatelessWidget {
  final List<Mission> missions;
  final String? selectedId;
  final ValueChanged<Mission> onSelect;
  final VoidCallback onNew;
  final ValueChanged<String> onDelete;

  const _MissionListPanel({
    required this.missions,
    this.selectedId,
    required this.onSelect,
    required this.onNew,
    required this.onDelete,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: G20Colors.surfaceDark,
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: G20Colors.cardDark),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              const Icon(Icons.folder, color: G20Colors.primary, size: 20),
              const SizedBox(width: 8),
              const Expanded(
                child: Text('Missions', style: TextStyle(color: G20Colors.textPrimaryDark, fontSize: 16, fontWeight: FontWeight.w600)),
              ),
              IconButton(icon: const Icon(Icons.add, color: G20Colors.primary), tooltip: 'New Mission', onPressed: onNew),
            ],
          ),
          const Divider(),
          Expanded(
            child: missions.isEmpty
                ? Center(
                    child: Column(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Icon(Icons.inbox, size: 48, color: Colors.grey.shade700),
                        const SizedBox(height: 8),
                        Text('No missions yet', style: TextStyle(color: G20Colors.textSecondaryDark)),
                        const SizedBox(height: 8),
                        ElevatedButton.icon(onPressed: onNew, icon: const Icon(Icons.add, size: 16), label: const Text('Create Mission'), style: ElevatedButton.styleFrom(backgroundColor: G20Colors.primary)),
                      ],
                    ),
                  )
                : ListView.builder(
                    itemCount: missions.length,
                    itemBuilder: (context, index) {
                      final mission = missions[index];
                      final isSelected = mission.id == selectedId;
                      return ListTile(
                        selected: isSelected,
                        selectedTileColor: G20Colors.primary.withOpacity(0.15),
                        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(6)),
                        leading: Icon(Icons.rocket_launch, color: isSelected ? G20Colors.primary : G20Colors.textSecondaryDark, size: 20),
                        title: Text(mission.name, style: TextStyle(color: isSelected ? G20Colors.primary : G20Colors.textPrimaryDark, fontWeight: isSelected ? FontWeight.bold : FontWeight.normal)),
                        subtitle: Text('${mission.freqRanges.length} ranges • ${mission.models.length} models', style: TextStyle(color: G20Colors.textSecondaryDark, fontSize: 11)),
                        trailing: IconButton(icon: const Icon(Icons.delete_outline, size: 18), color: Colors.red.shade400, onPressed: () => onDelete(mission.id)),
                        onTap: () => onSelect(mission),
                      );
                    },
                  ),
          ),
        ],
      ),
    );
  }
}

// ============================================================================
// NO MISSION SELECTED
// ============================================================================

class _NoMissionSelected extends StatelessWidget {
  const _NoMissionSelected();

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(32),
      decoration: BoxDecoration(color: G20Colors.surfaceDark, borderRadius: BorderRadius.circular(8), border: Border.all(color: G20Colors.cardDark)),
      child: Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(Icons.touch_app, size: 64, color: Colors.grey.shade700),
            const SizedBox(height: 16),
            Text('Select a mission to edit', style: TextStyle(color: G20Colors.textSecondaryDark, fontSize: 16)),
            const SizedBox(height: 8),
            Text('Or create a new mission from the left panel', style: TextStyle(color: G20Colors.textSecondaryDark, fontSize: 12)),
          ],
        ),
      ),
    );
  }
}

// ============================================================================
// MISSION EDITOR
// ============================================================================

class _MissionEditor extends ConsumerStatefulWidget {
  final Mission mission;
  final List<ModelPriority> availableModels;
  final ValueChanged<Mission> onSave;

  const _MissionEditor({required this.mission, required this.availableModels, required this.onSave});

  @override
  ConsumerState<_MissionEditor> createState() => _MissionEditorState();
}

class _MissionEditorState extends ConsumerState<_MissionEditor> {
  late TextEditingController _nameController;
  late TextEditingController _descController;
  late double _bandwidth;
  late double _dwellTime;
  late List<FreqRange> _freqRanges;
  late List<ModelPriority> _selectedModels;
  int? _editingFreqIndex;

  @override
  void initState() {
    super.initState();
    _initFromMission();
  }

  @override
  void didUpdateWidget(_MissionEditor oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.mission.id != widget.mission.id) {
      _initFromMission();
    }
  }

  void _initFromMission() {
    _nameController = TextEditingController(text: widget.mission.name);
    _descController = TextEditingController(text: widget.mission.description);
    _bandwidth = widget.mission.bandwidthMhz;
    _dwellTime = widget.mission.dwellTimeSec;
    _freqRanges = List.from(widget.mission.freqRanges);
    _selectedModels = List.from(widget.mission.models);
    _editingFreqIndex = null;
  }

  void _save() {
    final updated = widget.mission.copyWith(
      name: _nameController.text.trim(),
      description: _descController.text.trim(),
      bandwidthMhz: _bandwidth,
      dwellTimeSec: _dwellTime,
      freqRanges: _freqRanges,
      models: _selectedModels,
      modified: DateTime.now(),
    );
    widget.onSave(updated);
    
    // Show fading toast for save confirmation
    showFadingToast(context, 'Mission "${updated.name}" saved', icon: Icons.save, color: Colors.green.shade700);
    
    debugPrint('[Mission] Saved: ${updated.name}');
    
    // If this mission is currently ACTIVE, reload heads to pick up changes
    final activeMission = ref.read(activeMissionProvider);
    if (activeMission != null && activeMission.id == updated.id) {
      final signals = updated.models.map((m) => m.id).toList();
      debugPrint('[Mission] Active mission updated - reloading heads: $signals');
      ref.read(videoStreamProvider.notifier).loadHeads(signals);
    }
  }

  /// Load mission and send configuration to backend
  void _loadMission() {
    // First save current edits
    _save();
    
    // Build the updated mission
    final updatedMission = widget.mission.copyWith(
      name: _nameController.text.trim(),
      description: _descController.text.trim(),
      bandwidthMhz: _bandwidth,
      dwellTimeSec: _dwellTime,
      freqRanges: _freqRanges,
      models: _selectedModels,
    );
    
    // Set as active mission
    ref.read(activeMissionProvider.notifier).state = updatedMission;
    
    // Log mission config
    debugPrint('[Config] ════════════════════════════════════════');
    debugPrint('[Config] LOADING MISSION: ${_nameController.text}');
    debugPrint('[Config] ════════════════════════════════════════');
    debugPrint('[Config] RX Bandwidth: $_bandwidth MHz');
    debugPrint('[Config] Dwell Time: $_dwellTime sec');
    debugPrint('[Config] Frequency Ranges: ${_freqRanges.length}');
    for (final r in _freqRanges) {
      debugPrint('[Config]   • ${r.startMhz.toInt()}-${r.endMhz.toInt()} MHz');
    }
    debugPrint('[Config] Models (priority order): ${_selectedModels.length}');
    for (final m in _selectedModels) {
      debugPrint('[Config]   ${m.priority + 1}. ${m.name}');
    }
    debugPrint('[Config] ════════════════════════════════════════');
    
    // LOAD HEADS VIA BACKEND API
    final signals = _selectedModels.map((m) => m.id).toList();
    if (signals.isNotEmpty) {
      debugPrint('[Config] Loading heads via backend: $signals');
      ref.read(videoStreamProvider.notifier).loadHeads(signals);
      showFadingToast(context, 'Mission loaded - ${signals.length} detectors active', icon: Icons.rocket_launch, color: G20Colors.primary);
    } else {
      debugPrint('[Config] No models in mission - unloading all heads');
      ref.read(videoStreamProvider.notifier).unloadHeads();
      showFadingToast(context, 'Mission loaded - no detectors', icon: Icons.warning, color: Colors.orange);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(color: G20Colors.surfaceDark, borderRadius: BorderRadius.circular(8), border: Border.all(color: G20Colors.cardDark)),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Header
          Row(
            children: [
              const Icon(Icons.edit, color: G20Colors.primary, size: 20),
              const SizedBox(width: 8),
              Expanded(child: Text('Edit: ${widget.mission.name}', style: const TextStyle(color: G20Colors.textPrimaryDark, fontSize: 16, fontWeight: FontWeight.w600))),
              const SizedBox(width: 8),
              ElevatedButton.icon(onPressed: _save, icon: const Icon(Icons.save, size: 18), label: const Text('Save'), style: ElevatedButton.styleFrom(backgroundColor: G20Colors.primary)),
              const SizedBox(width: 8),
              ElevatedButton.icon(onPressed: _loadMission, icon: const Icon(Icons.rocket_launch, size: 18), label: const Text('Load'), style: ElevatedButton.styleFrom(backgroundColor: Colors.green.shade700)),
            ],
          ),
          const Divider(),
          
          // Scrollable content
          Expanded(
            child: SingleChildScrollView(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  // Name & Description
                  Row(
                    children: [
                      Expanded(flex: 2, child: TextField(controller: _nameController, decoration: const InputDecoration(labelText: 'Mission Name'))),
                      const SizedBox(width: 16),
                      Expanded(flex: 3, child: TextField(controller: _descController, decoration: const InputDecoration(labelText: 'Description'))),
                    ],
                  ),
                  const SizedBox(height: 16),
                  
                  // BW & Dwell DROPDOWNS
                  Row(
                    children: [
                      Expanded(
                        child: DropdownButtonFormField<double>(
                          value: kBandwidthOptions.contains(_bandwidth) ? _bandwidth : kBandwidthOptions.first,
                          decoration: const InputDecoration(labelText: 'RX Bandwidth', border: OutlineInputBorder()),
                          items: kBandwidthOptions.map((bw) => DropdownMenuItem(value: bw, child: Text('$bw MHz'))).toList(),
                          onChanged: (v) => setState(() => _bandwidth = v ?? _bandwidth),
                        ),
                      ),
                      const SizedBox(width: 16),
                      Expanded(
                        child: DropdownButtonFormField<double>(
                          value: kDwellTimeOptions.contains(_dwellTime) ? _dwellTime : kDwellTimeOptions.first,
                          decoration: const InputDecoration(labelText: 'Dwell Time', border: OutlineInputBorder()),
                          items: kDwellTimeOptions.map((dt) => DropdownMenuItem(value: dt, child: Text('$dt sec'))).toList(),
                          onChanged: (v) => setState(() => _dwellTime = v ?? _dwellTime),
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 24),
                  
                  // Frequency Ranges TABLE
                  _buildFreqRangesTable(),
                  const SizedBox(height: 24),
                  
                  // Models with Priority (drag-drop)
                  _buildModelsSection(),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildFreqRangesTable() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            const Icon(Icons.radio, color: G20Colors.primary, size: 18),
            const SizedBox(width: 8),
            const Text('Frequency Ranges', style: TextStyle(color: G20Colors.textPrimaryDark, fontWeight: FontWeight.w600)),
            const Spacer(),
            // Touch-friendly add button (48x48 minimum)
            Material(
              color: G20Colors.primary,
              borderRadius: BorderRadius.circular(8),
              child: InkWell(
                borderRadius: BorderRadius.circular(8),
                onTap: () => setState(() => _freqRanges.add(const FreqRange(startMhz: 0, endMhz: 0))),
                child: const Padding(
                  padding: EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                  child: Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Icon(Icons.add, color: Colors.white, size: 20),
                      SizedBox(width: 4),
                      Text('Add', style: TextStyle(color: Colors.white, fontWeight: FontWeight.w600)),
                    ],
                  ),
                ),
              ),
            ),
          ],
        ),
        const SizedBox(height: 8),
        Container(
          decoration: BoxDecoration(
            color: G20Colors.backgroundDark,
            borderRadius: BorderRadius.circular(6),
            border: Border.all(color: G20Colors.cardDark),
          ),
          child: _freqRanges.isEmpty
              ? const Padding(
                  padding: EdgeInsets.all(16),
                  child: Center(child: Text('No ranges - click + to add', style: TextStyle(color: Colors.grey))),
                )
              : DataTable(
                  columnSpacing: 24,
                  headingRowColor: WidgetStateProperty.all(G20Colors.cardDark),
                  columns: const [
                    DataColumn(label: Text('#', style: TextStyle(fontWeight: FontWeight.bold))),
                    DataColumn(label: Text('Start (MHz)', style: TextStyle(fontWeight: FontWeight.bold))),
                    DataColumn(label: Text('End (MHz)', style: TextStyle(fontWeight: FontWeight.bold))),
                    DataColumn(label: Text('', style: TextStyle(fontWeight: FontWeight.bold))),
                  ],
                  rows: _freqRanges.asMap().entries.map((e) {
                    final idx = e.key;
                    final range = e.value;
                    final isEditing = _editingFreqIndex == idx;
                    
                    return DataRow(
                      selected: isEditing,
                      onSelectChanged: (_) => setState(() => _editingFreqIndex = isEditing ? null : idx),
                      cells: [
                        DataCell(Text('${idx + 1}')),
                        DataCell(
                          isEditing
                              ? SizedBox(
                                  width: 100,
                                  child: TextFormField(
                                    autofocus: true,
                                    keyboardType: TextInputType.number,
                                    initialValue: range.startMhz > 0 ? range.startMhz.toInt().toString() : '',
                                    decoration: const InputDecoration(isDense: true, contentPadding: EdgeInsets.symmetric(horizontal: 8, vertical: 8), hintText: '30-6000'),
                                    onChanged: (v) {
                                      final val = double.tryParse(v);
                                      if (val != null) {
                                        _freqRanges[idx] = range.copyWith(startMhz: val);
                                      }
                                    },
                                  ),
                                )
                              : Text('${range.startMhz.toInt()}'),
                        ),
                        DataCell(
                          isEditing
                              ? SizedBox(
                                  width: 100,
                                  child: TextFormField(
                                    keyboardType: TextInputType.number,
                                    initialValue: range.endMhz > 0 ? range.endMhz.toInt().toString() : '',
                                    decoration: const InputDecoration(isDense: true, contentPadding: EdgeInsets.symmetric(horizontal: 8, vertical: 8), hintText: '30-6000'),
                                    onChanged: (v) {
                                      final val = double.tryParse(v);
                                      if (val != null) {
                                        _freqRanges[idx] = range.copyWith(endMhz: val);
                                      }
                                    },
                                  ),
                                )
                              : Text('${range.endMhz.toInt()}'),
                        ),
                        DataCell(
                          IconButton(
                            icon: const Icon(Icons.delete_outline, size: 18, color: Colors.red),
                            onPressed: () => setState(() => _freqRanges.removeAt(idx)),
                          ),
                        ),
                      ],
                    );
                  }).toList(),
                ),
        ),
      ],
    );
  }

  Widget _buildModelsSection() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            const Icon(Icons.psychology, color: G20Colors.primary, size: 18),
            const SizedBox(width: 8),
            const Text('Signal Priority (Models)', style: TextStyle(color: G20Colors.textPrimaryDark, fontWeight: FontWeight.w600)),
            const Spacer(),
            // Touch-friendly add button
            Material(
              color: G20Colors.primary,
              borderRadius: BorderRadius.circular(8),
              child: InkWell(
                borderRadius: BorderRadius.circular(8),
                onTap: _addModel,
                child: const Padding(
                  padding: EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                  child: Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Icon(Icons.add, color: Colors.white, size: 20),
                      SizedBox(width: 4),
                      Text('Add', style: TextStyle(color: Colors.white, fontWeight: FontWeight.w600)),
                    ],
                  ),
                ),
              ),
            ),
          ],
        ),
        const SizedBox(height: 8),
        _selectedModels.isEmpty
            ? GestureDetector(
                onTap: _addModel,
                child: Container(
                  padding: const EdgeInsets.all(16),
                  decoration: BoxDecoration(color: G20Colors.backgroundDark, borderRadius: BorderRadius.circular(6), border: Border.all(color: G20Colors.cardDark)),
                  child: const Center(child: Text('Click to add models', style: TextStyle(color: Colors.grey))),
                ),
              )
            : ReorderableListView.builder(
                shrinkWrap: true,
                physics: const NeverScrollableScrollPhysics(),
                itemCount: _selectedModels.length,
                onReorder: (oldIndex, newIndex) {
                  setState(() {
                    if (newIndex > oldIndex) newIndex--;
                    final item = _selectedModels.removeAt(oldIndex);
                    _selectedModels.insert(newIndex, item);
                    for (int i = 0; i < _selectedModels.length; i++) {
                      _selectedModels[i] = _selectedModels[i].copyWith(priority: i);
                    }
                  });
                },
                itemBuilder: (context, index) {
                  final model = _selectedModels[index];
                  return ListTile(
                    key: ValueKey(model.id),
                    tileColor: G20Colors.primary.withOpacity(0.1),
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(6)),
                    leading: Container(
                      width: 28,
                      height: 28,
                      decoration: BoxDecoration(color: G20Colors.primary, borderRadius: BorderRadius.circular(4)),
                      child: Center(child: Text('${index + 1}', style: const TextStyle(color: Colors.white, fontWeight: FontWeight.bold))),
                    ),
                    title: Text(model.name, style: const TextStyle(color: G20Colors.textPrimaryDark)),
                    subtitle: model.signalType != null 
                        ? Text(model.signalType!, style: TextStyle(color: G20Colors.textSecondaryDark, fontSize: 10))
                        : null,
                    trailing: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        IconButton(icon: const Icon(Icons.delete_outline, size: 18, color: Colors.red), onPressed: () => setState(() => _selectedModels.removeAt(index))),
                        const Icon(Icons.drag_handle, color: G20Colors.textSecondaryDark),
                      ],
                    ),
                  );
                },
              ),
      ],
    );
  }

  void _addModel() {
    final available = widget.availableModels.where((m) => !_selectedModels.any((s) => s.id == m.id)).toList();

    if (available.isEmpty) {
      showDialog(
        context: context,
        builder: (ctx) => AlertDialog(
          title: const Text('All Models Added'),
          content: const Text('All available signal detectors are already in this mission.'),
          actions: [
            ElevatedButton(
              onPressed: () => Navigator.pop(ctx),
              child: const Text('OK'),
            ),
          ],
        ),
      );
      return;
    }

    showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Add Signal Detector'),
        content: SizedBox(
          width: 400,
          child: ListView.builder(
            shrinkWrap: true,
            itemCount: available.length,
            itemBuilder: (context, index) {
              final model = available[index];
              return ListTile(
                leading: const Icon(Icons.psychology),
                title: Text(model.name),
                subtitle: model.signalType != null 
                    ? Text(model.signalType!, style: TextStyle(fontSize: 11, color: G20Colors.textSecondaryDark))
                    : null,
                onTap: () {
                  setState(() => _selectedModels.add(model.copyWith(priority: _selectedModels.length)));
                  Navigator.pop(ctx);
                },
              );
            },
          ),
        ),
        actions: [TextButton(onPressed: () => Navigator.pop(ctx), child: const Text('Cancel'))],
      ),
    );
  }

  @override
  void dispose() {
    _nameController.dispose();
    _descController.dispose();
    super.dispose();
  }
}
