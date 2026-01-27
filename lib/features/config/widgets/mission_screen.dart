// lib/features/config/widgets/mission_screen.dart
/// Mission configuration screen
/// Load, create, edit, and delete mission configurations

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:path/path.dart' as path;
import '../../../core/config/theme.dart';
import '../../live_detection/providers/video_stream_provider.dart';
import '../../settings/settings_screen.dart';
import '../models/mission_config.dart';
import '../providers/mission_provider.dart';

/// Main mission configuration screen
class MissionScreen extends ConsumerWidget {
  const MissionScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final state = ref.watch(missionProvider);
    final activeMission = state.activeMission;

    return SingleChildScrollView(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Active Mission Section
          _buildSection(
            title: 'Active Mission',
            icon: Icons.rocket_launch,
            children: [
              // Mission selector dropdown
              Row(
                children: [
                  Expanded(
                    child: _MissionDropdown(
                      missions: state.availableMissions,
                      selected: activeMission?.filePath,
                      onChanged: (path) {
                        if (path != null) {
                          ref.read(missionProvider.notifier).loadMission(path);
                        }
                      },
                    ),
                  ),
                  const SizedBox(width: 8),
                  IconButton(
                    icon: const Icon(Icons.refresh),
                    tooltip: 'Rescan missions',
                    onPressed: () {
                      ref.read(missionProvider.notifier).scanMissions();
                    },
                  ),
                ],
              ),
              const SizedBox(height: 16),

              // Mission summary
              if (activeMission != null) ...[
                _MissionSummaryCard(mission: activeMission),
                const SizedBox(height: 16),
              ],

              // Action buttons
              Wrap(
                spacing: 8,
                runSpacing: 8,
                children: [
                  ElevatedButton.icon(
                    onPressed: () => _showNewMissionDialog(context, ref),
                    icon: const Icon(Icons.add, size: 18),
                    label: const Text('New'),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: G20Colors.primary,
                    ),
                  ),
                  ElevatedButton.icon(
                    onPressed: activeMission != null
                        ? () => _showEditMissionDialog(context, ref, activeMission)
                        : null,
                    icon: const Icon(Icons.edit, size: 18),
                    label: const Text('Edit'),
                  ),
                  ElevatedButton.icon(
                    onPressed: activeMission != null
                        ? () => _showDuplicateDialog(context, ref)
                        : null,
                    icon: const Icon(Icons.copy, size: 18),
                    label: const Text('Duplicate'),
                  ),
                  ElevatedButton.icon(
                    onPressed: activeMission?.filePath != null &&
                            !activeMission!.filePath!.contains('default')
                        ? () => _showDeleteDialog(context, ref, activeMission.filePath!)
                        : null,
                    icon: const Icon(Icons.delete, size: 18),
                    label: const Text('Delete'),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.red.shade700,
                    ),
                  ),
                ],
              ),
            ],
          ),
          const SizedBox(height: 24),

          // Available Missions List
          _buildSection(
            title: 'Available Missions',
            icon: Icons.folder,
            children: [
              if (state.isLoading)
                const Center(child: CircularProgressIndicator())
              else if (state.availableMissions.isEmpty)
                const Padding(
                  padding: EdgeInsets.all(16),
                  child: Text(
                    'No mission files found',
                    style: TextStyle(color: Colors.grey),
                  ),
                )
              else
                ListView.separated(
                  shrinkWrap: true,
                  physics: const NeverScrollableScrollPhysics(),
                  itemCount: state.availableMissions.length,
                  separatorBuilder: (_, __) => const Divider(height: 1),
                  itemBuilder: (context, index) {
                    final filePath = state.availableMissions[index];
                    final fileName = path.basename(filePath);
                    final isActive = activeMission?.filePath == filePath;

                    return ListTile(
                      leading: Icon(
                        isActive ? Icons.radio_button_checked : Icons.radio_button_unchecked,
                        color: isActive ? G20Colors.primary : Colors.grey,
                      ),
                      title: Text(
                        fileName,
                        style: TextStyle(
                          fontWeight: isActive ? FontWeight.bold : FontWeight.normal,
                          color: isActive ? G20Colors.primary : null,
                        ),
                      ),
                      trailing: isActive
                          ? const Chip(
                              label: Text('ACTIVE', style: TextStyle(fontSize: 10)),
                              backgroundColor: G20Colors.primary,
                              padding: EdgeInsets.zero,
                            )
                          : null,
                      onTap: () {
                        ref.read(missionProvider.notifier).loadMission(filePath);
                      },
                    );
                  },
                ),
            ],
          ),

          // Error display
          if (state.error != null) ...[
            const SizedBox(height: 16),
            Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: Colors.red.shade900.withOpacity(0.3),
                borderRadius: BorderRadius.circular(8),
                border: Border.all(color: Colors.red.shade700),
              ),
              child: Row(
                children: [
                  const Icon(Icons.error, color: Colors.red),
                  const SizedBox(width: 8),
                  Expanded(child: Text(state.error!, style: const TextStyle(color: Colors.red))),
                ],
              ),
            ),
          ],
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

  void _showNewMissionDialog(BuildContext context, WidgetRef ref) {
    final nameController = TextEditingController();
    final descController = TextEditingController();

    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Create New Mission'),
        content: SizedBox(
          width: 400,
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              TextField(
                controller: nameController,
                decoration: const InputDecoration(
                  labelText: 'Mission Name',
                  hintText: 'e.g., ISM Band Hunt',
                ),
                autofocus: true,
              ),
              const SizedBox(height: 16),
              TextField(
                controller: descController,
                decoration: const InputDecoration(
                  labelText: 'Description (optional)',
                  hintText: 'Brief description of the mission',
                ),
                maxLines: 2,
              ),
            ],
          ),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Cancel'),
          ),
          ElevatedButton(
            onPressed: () {
              final name = nameController.text.trim();
              if (name.isNotEmpty) {
                ref.read(missionProvider.notifier).createNewMission(
                  name: name,
                  description: descController.text.trim(),
                );
                Navigator.pop(context);
              }
            },
            style: ElevatedButton.styleFrom(backgroundColor: G20Colors.primary),
            child: const Text('Create'),
          ),
        ],
      ),
    );
  }

  void _showDuplicateDialog(BuildContext context, WidgetRef ref) {
    final nameController = TextEditingController();

    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Duplicate Mission'),
        content: TextField(
          controller: nameController,
          decoration: const InputDecoration(
            labelText: 'New Mission Name',
          ),
          autofocus: true,
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Cancel'),
          ),
          ElevatedButton(
            onPressed: () {
              final name = nameController.text.trim();
              if (name.isNotEmpty) {
                ref.read(missionProvider.notifier).duplicateMission(newName: name);
                Navigator.pop(context);
              }
            },
            style: ElevatedButton.styleFrom(backgroundColor: G20Colors.primary),
            child: const Text('Duplicate'),
          ),
        ],
      ),
    );
  }

  void _showDeleteDialog(BuildContext context, WidgetRef ref, String filePath) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Delete Mission'),
        content: Text('Are you sure you want to delete ${path.basename(filePath)}?'),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Cancel'),
          ),
          ElevatedButton(
            onPressed: () {
              ref.read(missionProvider.notifier).deleteMission(filePath);
              Navigator.pop(context);
            },
            style: ElevatedButton.styleFrom(backgroundColor: Colors.red),
            child: const Text('Delete'),
          ),
        ],
      ),
    );
  }

  void _showEditMissionDialog(BuildContext context, WidgetRef ref, MissionConfig mission) {
    showDialog(
      context: context,
      builder: (context) => MissionEditDialog(mission: mission),
    );
  }
}

/// Dropdown for mission selection
class _MissionDropdown extends StatelessWidget {
  final List<String> missions;
  final String? selected;
  final ValueChanged<String?> onChanged;

  const _MissionDropdown({
    required this.missions,
    this.selected,
    required this.onChanged,
  });

  @override
  Widget build(BuildContext context) {
    return DropdownButtonFormField<String>(
      value: selected,
      decoration: const InputDecoration(
        labelText: 'Select Mission',
        border: OutlineInputBorder(),
        contentPadding: EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      ),
      items: missions.map((filePath) {
        return DropdownMenuItem(
          value: filePath,
          child: Text(path.basename(filePath)),
        );
      }).toList(),
      onChanged: onChanged,
    );
  }
}

/// Summary card showing active mission details
class _MissionSummaryCard extends StatelessWidget {
  final MissionConfig mission;

  const _MissionSummaryCard({required this.mission});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: G20Colors.cardDark,
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: G20Colors.primary.withOpacity(0.3)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            mission.name,
            style: const TextStyle(
              fontSize: 16,
              fontWeight: FontWeight.bold,
              color: G20Colors.primary,
            ),
          ),
          if (mission.description.isNotEmpty) ...[
            const SizedBox(height: 4),
            Text(
              mission.description,
              style: TextStyle(fontSize: 12, color: Colors.grey.shade400),
            ),
          ],
          const Divider(height: 16),
          Wrap(
            spacing: 16,
            runSpacing: 8,
            children: [
              _SummaryChip(icon: Icons.radio, label: '${mission.centerFreqMhz} MHz'),
              _SummaryChip(icon: Icons.settings_input_antenna, label: 'BW: ${mission.bandwidthMhz} MHz'),
              _SummaryChip(icon: Icons.timer, label: 'Dwell: ${mission.dwellTimeSec}s'),
              _SummaryChip(icon: Icons.psychology, label: mission.modelName),
              _SummaryChip(icon: Icons.percent, label: '${(mission.confidenceThreshold * 100).round()}%'),
              _SummaryChip(
                icon: mission.autoRecordDetections ? Icons.fiber_manual_record : Icons.fiber_manual_record_outlined,
                label: mission.autoRecordDetections ? 'Auto-record ON' : 'Auto-record OFF',
                color: mission.autoRecordDetections ? Colors.red : Colors.grey,
              ),
            ],
          ),
        ],
      ),
    );
  }
}

class _SummaryChip extends StatelessWidget {
  final IconData icon;
  final String label;
  final Color? color;

  const _SummaryChip({required this.icon, required this.label, this.color});

  @override
  Widget build(BuildContext context) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Icon(icon, size: 14, color: color ?? Colors.grey.shade400),
        const SizedBox(width: 4),
        Text(label, style: TextStyle(fontSize: 12, color: color ?? Colors.grey.shade300)),
      ],
    );
  }
}

/// Dialog for editing mission parameters
class MissionEditDialog extends ConsumerStatefulWidget {
  final MissionConfig mission;

  const MissionEditDialog({super.key, required this.mission});

  @override
  ConsumerState<MissionEditDialog> createState() => _MissionEditDialogState();
}

class _MissionEditDialogState extends ConsumerState<MissionEditDialog> {
  late TextEditingController _nameController;
  late TextEditingController _descController;
  late TextEditingController _centerFreqController;
  late TextEditingController _bandwidthController;
  late TextEditingController _dwellTimeController;
  late double _confidenceThreshold;
  late bool _autoRecord;

  @override
  void initState() {
    super.initState();
    _nameController = TextEditingController(text: widget.mission.name);
    _descController = TextEditingController(text: widget.mission.description);
    _centerFreqController = TextEditingController(text: widget.mission.centerFreqMhz.toString());
    _bandwidthController = TextEditingController(text: widget.mission.bandwidthMhz.toString());
    _dwellTimeController = TextEditingController(text: widget.mission.dwellTimeSec.toString());
    _confidenceThreshold = widget.mission.confidenceThreshold;
    _autoRecord = widget.mission.autoRecordDetections;
  }

  @override
  void dispose() {
    _nameController.dispose();
    _descController.dispose();
    _centerFreqController.dispose();
    _bandwidthController.dispose();
    _dwellTimeController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AlertDialog(
      title: const Text('Edit Mission'),
      content: SizedBox(
        width: 450,
        child: SingleChildScrollView(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              TextField(
                controller: _nameController,
                decoration: const InputDecoration(labelText: 'Name'),
              ),
              const SizedBox(height: 12),
              TextField(
                controller: _descController,
                decoration: const InputDecoration(labelText: 'Description'),
                maxLines: 2,
              ),
              const SizedBox(height: 16),
              const Text('─── Frequency ───', style: TextStyle(color: Colors.grey)),
              const SizedBox(height: 8),
              Row(
                children: [
                  Expanded(
                    child: TextField(
                      controller: _centerFreqController,
                      decoration: const InputDecoration(labelText: 'Center (MHz)'),
                      keyboardType: TextInputType.number,
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: TextField(
                      controller: _bandwidthController,
                      decoration: const InputDecoration(labelText: 'BW (MHz)'),
                      keyboardType: TextInputType.number,
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 16),
              const Text('─── Scan ───', style: TextStyle(color: Colors.grey)),
              const SizedBox(height: 8),
              TextField(
                controller: _dwellTimeController,
                decoration: const InputDecoration(labelText: 'Dwell Time (sec)'),
                keyboardType: TextInputType.number,
              ),
              const SizedBox(height: 16),
              const Text('─── Detection ───', style: TextStyle(color: Colors.grey)),
              const SizedBox(height: 8),
              Row(
                children: [
                  const Text('Confidence: '),
                  Expanded(
                    child: Slider(
                      value: _confidenceThreshold,
                      min: 0.1,
                      max: 0.99,
                      divisions: 18,
                      label: '${(_confidenceThreshold * 100).round()}%',
                      onChanged: (v) => setState(() => _confidenceThreshold = v),
                    ),
                  ),
                  Text('${(_confidenceThreshold * 100).round()}%'),
                ],
              ),
              const SizedBox(height: 8),
              SwitchListTile(
                title: const Text('Auto-record detections'),
                value: _autoRecord,
                onChanged: (v) => setState(() => _autoRecord = v),
                contentPadding: EdgeInsets.zero,
              ),
            ],
          ),
        ),
      ),
      actions: [
        TextButton(
          onPressed: () => Navigator.pop(context),
          child: const Text('Cancel'),
        ),
        ElevatedButton(
          onPressed: _saveChanges,
          style: ElevatedButton.styleFrom(backgroundColor: G20Colors.primary),
          child: const Text('Save'),
        ),
      ],
    );
  }

  void _saveChanges() {
    final notifier = ref.read(missionProvider.notifier);

    notifier.updateActiveMission((m) => m.copyWith(
      name: _nameController.text.trim(),
      description: _descController.text.trim(),
      centerFreqMhz: double.tryParse(_centerFreqController.text) ?? m.centerFreqMhz,
      bandwidthMhz: double.tryParse(_bandwidthController.text) ?? m.bandwidthMhz,
      dwellTimeSec: double.tryParse(_dwellTimeController.text) ?? m.dwellTimeSec,
      confidenceThreshold: _confidenceThreshold,
      autoRecordDetections: _autoRecord,
    ));

    notifier.saveMission();

    // Apply hot-reloadable settings to backend
    _applyMissionToBackend(ref, _confidenceThreshold);

    Navigator.pop(context);
  }

  /// Apply hot-reloadable mission settings to the backend via WebSocket
  void _applyMissionToBackend(WidgetRef ref, double confidenceThreshold) {
    final videoStream = ref.read(videoStreamProvider.notifier);

    // Apply confidence threshold (hot-reloadable)
    videoStream.setScoreThreshold(confidenceThreshold);

    // Also sync with settings providers
    ref.read(scoreThresholdProvider.notifier).state = confidenceThreshold;

    debugPrint('[MissionEdit] Applied settings - confidence: ${(confidenceThreshold * 100).round()}%');
  }
}
