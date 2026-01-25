import 'dart:async';
import 'dart:io';
import 'dart:math' as math;
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:path/path.dart' as path;
import '../../core/config/theme.dart';
import '../../core/database/signal_database.dart';
import '../../core/services/rfcap_service.dart';
import '../live_detection/providers/map_provider.dart' show getSOIColor;
import '../live_detection/providers/sdr_config_provider.dart';
import 'widgets/training_spectrogram.dart';

/// Training Screen - Label spectrogram data + Train
class TrainingScreen extends ConsumerStatefulWidget {
  const TrainingScreen({super.key});

  @override
  ConsumerState<TrainingScreen> createState() => _TrainingScreenState();
}

class _TrainingScreenState extends ConsumerState<TrainingScreen> {
  String? _selectedFile;
  RfcapHeader? _loadedHeader;
  bool _isTraining = false;
  double _trainingProgress = 0.0;
  bool _isLoadingFiles = true;
  
  final _classNameController = TextEditingController();
  List<String> _availableFiles = [];
  
  // Persistent label boxes PER FILE - keyed by filepath
  final Map<String, List<LabelBox>> _boxesByFile = {};
  int? _selectedBoxId;
  
  // Auto-refresh timer
  Timer? _refreshTimer;

  @override
  void initState() {
    super.initState();
    _loadAvailableFiles();
    // Auto-refresh file list every 5 seconds (will pick up new captures)
    _refreshTimer = Timer.periodic(const Duration(seconds: 5), (_) => _loadAvailableFiles());
  }

  @override
  void dispose() {
    _refreshTimer?.cancel();
    _classNameController.dispose();
    super.dispose();
  }
  
  /// Get current boxes for the selected file
  List<LabelBox> get _labelBoxes {
    if (_selectedFile == null) return [];
    return _boxesByFile[_selectedFile!] ??= [];
  }

  Future<void> _loadAvailableFiles() async {
    setState(() => _isLoadingFiles = true);
    
    final currentDir = Directory.current.path;
    final capturesDir = Directory('$currentDir/data/captures');
    
    if (!await capturesDir.exists()) {
      final altDir = Directory('$currentDir/g20_demo/data/captures');
      if (await altDir.exists()) {
        await _loadFilesFromDir(altDir);
        return;
      }
    } else {
      await _loadFilesFromDir(capturesDir);
    }
    
    setState(() => _isLoadingFiles = false);
  }

  Future<void> _loadFilesFromDir(Directory dir) async {
    final files = <String>[];
    await for (final entity in dir.list()) {
      if (entity.path.endsWith('.rfcap')) {
        // Only show "man_" files (manual captures awaiting labeling)
        // Once trained, files get renamed to their signal name
        final filename = path.basename(entity.path).toLowerCase();
        if (filename.startsWith('man_')) {
          files.add(entity.path);
        }
      }
    }
    
    // Sort by most recent (modification time)
    files.sort((a, b) {
      try {
        final aTime = File(a).statSync().modified;
        final bTime = File(b).statSync().modified;
        return bTime.compareTo(aTime);  // Most recent first
      } catch (_) {
        return 0;
      }
    });
    
    setState(() {
      _availableFiles = files;
      _isLoadingFiles = false;
      // Only auto-select if currently selected file is gone
      if (_selectedFile != null && !files.contains(_selectedFile)) {
        _selectedFile = files.isNotEmpty ? files.first : null;
        if (_selectedFile != null) _loadFileHeader(_selectedFile!);
      } else if (files.isNotEmpty && _selectedFile == null) {
        _selectedFile = files.first;
        _loadFileHeader(files.first);
      }
    });
  }

  Future<void> _loadFileHeader(String filepath) async {
    final header = await RfcapService.readHeader(filepath);
    if (header != null) {
      setState(() {
        _loadedHeader = header;
        // DON'T clear boxes - they persist per file in _boxesByFile
        _selectedBoxId = null;
        _classNameController.text = header.signalName;
      });
      print('Loaded RFCAP: $header');
    }
  }

  /// Delete a capture file
  Future<void> _deleteFile(String filepath) async {
    final confirm = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        backgroundColor: G20Colors.surfaceDark,
        title: const Text('Delete File?', style: TextStyle(color: G20Colors.textPrimaryDark)),
        content: Text('Delete ${path.basename(filepath)}?', 
          style: const TextStyle(color: G20Colors.textSecondaryDark)),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context, false),
            child: const Text('Cancel'),
          ),
          ElevatedButton(
            onPressed: () => Navigator.pop(context, true),
            style: ElevatedButton.styleFrom(backgroundColor: G20Colors.error),
            child: const Text('Delete'),
          ),
        ],
      ),
    );
    
    if (confirm == true) {
      try {
        await File(filepath).delete();
        // Remove from boxes cache
        _boxesByFile.remove(filepath);
        // Refresh file list
        if (_selectedFile == filepath) {
          _selectedFile = null;
          _loadedHeader = null;
        }
        _loadAvailableFiles();
        debugPrint('🗑️ File deleted');
      } catch (e) {
        debugPrint('Error deleting file: $e');
      }
    }
  }

  void _onFileSelected(String? filepath) {
    if (filepath != null && filepath != _selectedFile) {
      setState(() => _selectedFile = filepath);
      _loadFileHeader(filepath);
    }
  }

  void _onBoxCreated(LabelBox box) {
    // Use class name from text field
    box.className = _classNameController.text.trim().isEmpty 
        ? 'unknown' 
        : _classNameController.text.trim();
    
    setState(() {
      _labelBoxes.add(box);
      _selectedBoxId = box.id;
    });
  }

  void _onBoxSelected(int id) {
    setState(() {
      _selectedBoxId = id;
      for (var box in _labelBoxes) {
        box.isSelected = box.id == id;
      }
    });
  }

  void _onBoxDeleted(int id) {
    setState(() {
      _labelBoxes.removeWhere((b) => b.id == id);
      if (_selectedBoxId == id) _selectedBoxId = null;
    });
  }

  void _updateAllBoxClasses(String className) {
    if (className.trim().isEmpty) return;
    setState(() {
      // Update ALL boxes with the new class name
      for (var box in _labelBoxes) {
        box.className = className.trim();
      }
    });
  }

  void _startTraining() {
    if (_labelBoxes.isEmpty) {
      debugPrint('Cannot train: no label boxes');
      return;
    }
    _doTraining();
  }

  void _doTraining() {
    setState(() {
      _isTraining = true;
      _trainingProgress = 0.0;
    });
    _simulateTraining();
  }

  void _stopTraining() {
    setState(() => _isTraining = false);
  }

  /// Extract signal name from filename (e.g., "CREAMY_CHICKEN_210051ZJAN26.rfcap" -> "creamy_chicken")
  String _extractSignalName(String filename) {
    // Remove .rfcap extension
    var name = filename.replaceAll('.rfcap', '');
    
    // Split by underscore and take all parts except the last one (which is the timestamp)
    final parts = name.split('_');
    if (parts.length >= 2) {
      // Last part is usually the timestamp (like "210051ZJAN26")
      // Join all parts except the last one
      parts.removeLast();
      name = parts.join('_');
    }
    
    return name.toLowerCase();
  }

  void _simulateTraining() async {
    for (int i = 0; i <= 100 && _isTraining; i++) {
      await Future.delayed(const Duration(milliseconds: 100));
      if (mounted && _isTraining) {
        setState(() => _trainingProgress = i / 100);
      }
    }
    if (mounted && _isTraining) {
      setState(() => _isTraining = false);
      
      // Save training results to database
      _saveTrainingResults();
      
      debugPrint('✅ Training complete!');
    }
  }
  
  /// Save training results to the signal database
  void _saveTrainingResults() {
    final className = _classNameController.text.trim();
    if (className.isEmpty) {
      debugPrint('[Training] No class name specified, skipping DB update');
      return;
    }
    
    // Simulate realistic F1 scores based on number of labels
    // More labels = higher F1 score (with some randomness)
    final numLabels = _labelBoxes.length;
    final random = math.Random();
    
    // Base F1 starts at 0.65 and increases with more labels
    // maxes out around 0.95 with 20+ labels
    final baseF1 = 0.65 + (0.30 * (1 - math.exp(-numLabels / 10)));
    final noise = (random.nextDouble() - 0.5) * 0.08; // +/- 4% noise
    final f1Score = (baseF1 + noise).clamp(0.5, 0.98);
    
    // Precision and recall derived from F1 with some variance
    final precision = (f1Score + (random.nextDouble() - 0.5) * 0.1).clamp(0.4, 0.99);
    final recall = (f1Score + (random.nextDouble() - 0.5) * 0.1).clamp(0.4, 0.99);
    
    // Create training result
    final result = TrainingResult(
      timestamp: DateTime.now(),
      dataLabels: numLabels,
      f1Score: f1Score,
      precision: precision,
      recall: recall,
      epochs: 100,
      loss: 0.1 + random.nextDouble() * 0.3, // 0.1 - 0.4
      modelPath: 'models/${className.toLowerCase()}.engine',
    );
    
    // Save to database
    ref.read(signalDatabaseProvider.notifier).addTrainingResult(className, result);
    
    // Show success toast
    if (mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Training saved: $className (F1: ${f1Score.toStringAsFixed(2)}, ${numLabels} labels)'),
          backgroundColor: G20Colors.success,
          duration: const Duration(seconds: 3),
        ),
      );
    }
    
    debugPrint('[Training] Saved result for "$className": F1=${f1Score.toStringAsFixed(3)}, labels=$numLabels');
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: G20Colors.backgroundDark,
      body: Padding(
        padding: const EdgeInsets.all(12),
        child: Row(
          children: [
            // LEFT: Spectrogram (takes most width)
            Expanded(
              flex: 3,
              child: Container(
                decoration: BoxDecoration(
                  color: G20Colors.surfaceDark,
                  borderRadius: BorderRadius.circular(8),
                  border: Border.all(color: G20Colors.cardDark),
                ),
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(8),
                  child: TrainingSpectrogram(
                    filepath: _selectedFile,
                    header: _loadedHeader,
                    boxes: _labelBoxes,
                    onBoxCreated: _onBoxCreated,
                    onBoxSelected: _onBoxSelected,
                    onBoxDeleted: _onBoxDeleted,
                  ),
                ),
              ),
            ),
            const SizedBox(width: 8),
            
            // RIGHT: Controls sidebar
            SizedBox(
              width: 280,
              child: Column(
                children: [
                  // File selector
                  Container(
                    padding: const EdgeInsets.all(12),
                    decoration: BoxDecoration(
                      color: G20Colors.surfaceDark,
                      borderRadius: BorderRadius.circular(8),
                      border: Border.all(color: G20Colors.cardDark),
                    ),
                    child: _buildFileSelector(),
                  ),
                  const SizedBox(height: 8),
                  
                  // SOI info
                  if (_loadedHeader != null)
                    Container(
                      padding: const EdgeInsets.all(12),
                      decoration: BoxDecoration(
                        color: G20Colors.surfaceDark,
                        borderRadius: BorderRadius.circular(8),
                        border: Border.all(color: G20Colors.cardDark),
                      ),
                      child: _buildSoiInfoVertical(),
                    ),
                  if (_loadedHeader != null) const SizedBox(height: 8),
                  
                  // Class name + Train button
                  Container(
                    padding: const EdgeInsets.all(12),
                    decoration: BoxDecoration(
                      color: G20Colors.surfaceDark,
                      borderRadius: BorderRadius.circular(8),
                      border: Border.all(color: G20Colors.cardDark),
                    ),
                    child: _buildControlsVertical(),
                  ),
                  const SizedBox(height: 8),
                  
                  // Labels table
                  Expanded(
                    child: Container(
                      decoration: BoxDecoration(
                        color: G20Colors.surfaceDark,
                        borderRadius: BorderRadius.circular(8),
                        border: Border.all(color: G20Colors.cardDark),
                      ),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          _buildHeader('Labels (${_labelBoxes.length})', Icons.label_outline),
                          Expanded(child: _buildLabelsTableCompact()),
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
    );
  }

  Widget _buildHeader(String title, IconData icon) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
      decoration: const BoxDecoration(
        border: Border(bottom: BorderSide(color: G20Colors.cardDark)),
      ),
      child: Row(
        children: [
          Icon(icon, size: 16, color: G20Colors.primary),
          const SizedBox(width: 8),
          Text(title, style: const TextStyle(color: G20Colors.textPrimaryDark, fontWeight: FontWeight.w600, fontSize: 13)),
        ],
      ),
    );
  }

  Widget _buildFileSelector() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      mainAxisSize: MainAxisSize.min,
      children: [
        const Text('File', style: TextStyle(color: G20Colors.textSecondaryDark, fontSize: 11)),
        const SizedBox(height: 4),
        _isLoadingFiles
            ? const Text('Loading...', style: TextStyle(color: G20Colors.textSecondaryDark))
            : _availableFiles.isEmpty
                ? const Text('No .rfcap files', style: TextStyle(color: G20Colors.textSecondaryDark))
                : Row(
                    children: [
                      Expanded(
                        child: Container(
                          padding: const EdgeInsets.symmetric(horizontal: 8),
                          decoration: BoxDecoration(
                            color: G20Colors.backgroundDark,
                            borderRadius: BorderRadius.circular(4),
                            border: Border.all(color: G20Colors.primary.withOpacity(0.5)),
                          ),
                          child: DropdownButton<String>(
                            value: _selectedFile,
                            isExpanded: true,
                            dropdownColor: G20Colors.surfaceDark,
                            underline: const SizedBox(),
                            icon: const Icon(Icons.arrow_drop_down, color: G20Colors.primary),
                            selectedItemBuilder: (context) => _availableFiles.map((f) {
                              final name = path.basename(f);
                              final signalName = _extractSignalName(name);
                              final soiColor = getSOIColor(signalName);
                              return Align(
                                alignment: Alignment.centerLeft,
                                child: Text(name, 
                                  style: TextStyle(color: soiColor, fontSize: 12, fontWeight: FontWeight.w500),
                                  overflow: TextOverflow.ellipsis,
                                ),
                              );
                            }).toList(),
                            items: _availableFiles.map((f) {
                              final name = path.basename(f);
                              final signalName = _extractSignalName(name);
                              final soiColor = getSOIColor(signalName);
                              return DropdownMenuItem(
                                value: f, 
                                child: Row(
                                  children: [
                                    Container(
                                      width: 8, height: 8,
                                      margin: const EdgeInsets.only(right: 8),
                                      decoration: BoxDecoration(color: soiColor, shape: BoxShape.circle),
                                    ),
                                    Expanded(
                                      child: Text(name, 
                                        style: TextStyle(color: soiColor, fontSize: 12, fontWeight: FontWeight.w500),
                                        overflow: TextOverflow.ellipsis,
                                      ),
                                    ),
                                  ],
                                ),
                              );
                            }).toList(),
                            onChanged: _onFileSelected,
                          ),
                        ),
                      ),
                      const SizedBox(width: 8),
                      // Delete file button
                      IconButton(
                        icon: const Icon(Icons.delete_outline, size: 20, color: G20Colors.error),
                        tooltip: 'Delete file',
                        onPressed: _selectedFile != null ? () => _deleteFile(_selectedFile!) : null,
                        padding: const EdgeInsets.all(8),
                        constraints: const BoxConstraints(minWidth: 36, minHeight: 36),
                        style: IconButton.styleFrom(
                          backgroundColor: G20Colors.backgroundDark,
                          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(4)),
                        ),
                      ),
                    ],
                  ),
      ],
    );
  }

  Widget _buildSoiInfoVertical() {
    if (_loadedHeader == null) return const SizedBox();
    
    // Format sample rate nicely
    final srMHz = _loadedHeader!.sampleRate / 1e6;
    final srStr = srMHz >= 1 ? '${srMHz.toStringAsFixed(1)} Msps' : '${(_loadedHeader!.sampleRate / 1e3).toStringAsFixed(0)} ksps';
    
    // Get SOI color for the signal
    final soiColor = getSOIColor(_loadedHeader!.signalName);
    
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      mainAxisSize: MainAxisSize.min,
      children: [
        // Signal name with color indicator
        Row(
          children: [
            Container(
              width: 12, height: 12,
              margin: const EdgeInsets.only(right: 8),
              decoration: BoxDecoration(color: soiColor, shape: BoxShape.circle),
            ),
            Expanded(
              child: Text(
                _loadedHeader!.signalName.toUpperCase(),
                style: TextStyle(color: soiColor, fontWeight: FontWeight.bold, fontSize: 14),
                overflow: TextOverflow.ellipsis,
              ),
            ),
          ],
        ),
        const SizedBox(height: 8),
        // Info rows
        _buildInfoRow('Frequency', '${_loadedHeader!.centerFreqMHz.toStringAsFixed(2)} MHz'),
        _buildInfoRow('Bandwidth', '${_loadedHeader!.bandwidthMHz.toStringAsFixed(1)} MHz'),
        _buildInfoRow('Sample Rate', srStr),
        _buildInfoRow('Duration', '${_loadedHeader!.durationSec.toStringAsFixed(1)}s'),
        _buildInfoRow('Samples', '${_loadedHeader!.numSamples ~/ 1000}k'),
      ],
    );
  }

  Widget _buildInfoRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 2),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(label, style: const TextStyle(color: G20Colors.textSecondaryDark, fontSize: 11)),
          Text(value, style: const TextStyle(color: G20Colors.textPrimaryDark, fontSize: 11, fontWeight: FontWeight.w500)),
        ],
      ),
    );
  }

  Widget _buildControlsVertical() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      mainAxisSize: MainAxisSize.min,
      children: [
        // Class name field
        TextField(
          controller: _classNameController,
          style: const TextStyle(color: G20Colors.textPrimaryDark, fontSize: 13),
          decoration: InputDecoration(
            labelText: 'Class Name',
            labelStyle: const TextStyle(color: G20Colors.textSecondaryDark, fontSize: 11),
            filled: true,
            fillColor: G20Colors.backgroundDark,
            contentPadding: const EdgeInsets.symmetric(horizontal: 10, vertical: 8),
            border: OutlineInputBorder(borderRadius: BorderRadius.circular(4), borderSide: BorderSide.none),
          ),
          onChanged: _updateAllBoxClasses,
        ),
        const SizedBox(height: 12),
        // Training button + progress
        if (_isTraining) ...[
          LinearProgressIndicator(
            value: _trainingProgress,
            backgroundColor: G20Colors.cardDark,
            valueColor: const AlwaysStoppedAnimation(G20Colors.primary),
          ),
          const SizedBox(height: 4),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text('${(_trainingProgress * 100).toInt()}%', 
                style: const TextStyle(color: G20Colors.textSecondaryDark, fontSize: 11)),
              ElevatedButton(
                onPressed: _stopTraining,
                style: ElevatedButton.styleFrom(
                  backgroundColor: G20Colors.error, 
                  padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                ),
                child: const Text('Stop', style: TextStyle(fontSize: 12)),
              ),
            ],
          ),
        ] else
          ElevatedButton.icon(
            onPressed: _startTraining,
            icon: const Icon(Icons.rocket_launch, size: 18),
            label: const Text('Train Model', style: TextStyle(fontSize: 13)),
            style: ElevatedButton.styleFrom(
              backgroundColor: G20Colors.success, 
              padding: const EdgeInsets.symmetric(vertical: 12),
            ),
          ),
      ],
    );
  }

  Widget _buildLabelsTableCompact() {
    if (_labelBoxes.isEmpty) {
      return const Center(
        child: Padding(
          padding: EdgeInsets.all(16),
          child: Text(
            'Click on spectrogram to auto-detect signal, or drag to draw a box',
            style: TextStyle(color: G20Colors.textSecondaryDark, fontSize: 11),
            textAlign: TextAlign.center,
          ),
        ),
      );
    }
    
    return ListView.builder(
      padding: const EdgeInsets.all(4),
      itemCount: _labelBoxes.length,
      itemBuilder: (context, index) {
        final box = _labelBoxes[index];
        final isSelected = box.id == _selectedBoxId;
        return GestureDetector(
          onTap: () => _onBoxSelected(box.id),
          child: Container(
            margin: const EdgeInsets.symmetric(vertical: 2),
            padding: const EdgeInsets.all(8),
            decoration: BoxDecoration(
              color: isSelected ? G20Colors.primary.withOpacity(0.2) : Colors.transparent,
              borderRadius: BorderRadius.circular(4),
              border: Border.all(color: isSelected ? G20Colors.primary : G20Colors.cardDark),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    Text('#${index + 1} ${box.className}', 
                      style: const TextStyle(color: G20Colors.textPrimaryDark, fontSize: 12, fontWeight: FontWeight.w500)),
                    GestureDetector(
                      onTap: () => _onBoxDeleted(box.id),
                      child: const Icon(Icons.close, size: 14, color: G20Colors.error),
                    ),
                  ],
                ),
                const SizedBox(height: 2),
                Text(
                  '${box.freqStartMHz?.toStringAsFixed(2) ?? '?'} - ${box.freqEndMHz?.toStringAsFixed(2) ?? '?'} MHz',
                  style: const TextStyle(color: G20Colors.textSecondaryDark, fontSize: 10),
                ),
              ],
            ),
          ),
        );
      },
    );
  }

  Widget _buildControlsRow() {
    return Row(
      children: [
        // Class name field
        Expanded(
          child: TextField(
            controller: _classNameController,
            style: const TextStyle(color: G20Colors.textPrimaryDark, fontSize: 13),
            decoration: InputDecoration(
              labelText: 'Class Name',
              labelStyle: const TextStyle(color: G20Colors.textSecondaryDark, fontSize: 11),
              filled: true,
              fillColor: G20Colors.backgroundDark,
              contentPadding: const EdgeInsets.symmetric(horizontal: 10, vertical: 8),
              border: OutlineInputBorder(borderRadius: BorderRadius.circular(4), borderSide: BorderSide.none),
            ),
            onChanged: _updateAllBoxClasses,
          ),
        ),
        const SizedBox(width: 8),
        // Training button
        if (_isTraining) ...[
          SizedBox(
            width: 80,
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                LinearProgressIndicator(
                  value: _trainingProgress,
                  backgroundColor: G20Colors.cardDark,
                  valueColor: const AlwaysStoppedAnimation(G20Colors.primary),
                ),
                Text('${(_trainingProgress * 100).toInt()}%', style: const TextStyle(color: G20Colors.textSecondaryDark, fontSize: 10)),
              ],
            ),
          ),
          const SizedBox(width: 8),
          ElevatedButton(
            onPressed: _stopTraining,
            style: ElevatedButton.styleFrom(backgroundColor: G20Colors.error, padding: const EdgeInsets.symmetric(horizontal: 12)),
            child: const Text('Stop', style: TextStyle(fontSize: 12)),
          ),
        ] else
          ElevatedButton.icon(
            onPressed: _startTraining,
            icon: const Icon(Icons.rocket_launch, size: 16),
            label: const Text('Train', style: TextStyle(fontSize: 12)),
            style: ElevatedButton.styleFrom(backgroundColor: G20Colors.success, padding: const EdgeInsets.symmetric(horizontal: 12)),
          ),
      ],
    );
  }

  Widget _buildLabelsTable() {
    if (_labelBoxes.isEmpty) {
      return const Center(
        child: Text(
          'Click on spectrogram to auto-detect signal, or drag to draw a box',
          style: TextStyle(color: G20Colors.textSecondaryDark, fontSize: 12),
        ),
      );
    }
    
    return ListView.builder(
      padding: const EdgeInsets.all(4),
      itemCount: _labelBoxes.length,
      itemBuilder: (context, index) {
        final box = _labelBoxes[index];
        final isSelected = box.id == _selectedBoxId;
        return GestureDetector(
          onTap: () => _onBoxSelected(box.id),
          child: Container(
            margin: const EdgeInsets.symmetric(vertical: 2),
            padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 6),
            decoration: BoxDecoration(
              color: isSelected ? G20Colors.primary.withOpacity(0.2) : Colors.transparent,
              borderRadius: BorderRadius.circular(4),
              border: Border.all(color: isSelected ? G20Colors.primary : Colors.transparent),
            ),
            child: Row(
              children: [
                // Index
                SizedBox(
                  width: 30,
                  child: Text('#${index + 1}', style: const TextStyle(color: G20Colors.textSecondaryDark, fontSize: 11)),
                ),
                // Class name
                Expanded(
                  flex: 2,
                  child: Text(box.className, style: const TextStyle(color: G20Colors.textPrimaryDark, fontSize: 12, fontWeight: FontWeight.w500)),
                ),
                // Freq range
                Expanded(
                  flex: 2,
                  child: Text(
                    '${box.freqStartMHz?.toStringAsFixed(2) ?? '?'} - ${box.freqEndMHz?.toStringAsFixed(2) ?? '?'} MHz',
                    style: const TextStyle(color: G20Colors.textSecondaryDark, fontSize: 11),
                  ),
                ),
                // Time range
                Expanded(
                  flex: 2,
                  child: Text(
                    '${box.timeStartSec?.toStringAsFixed(1) ?? '?'}s - ${box.timeEndSec?.toStringAsFixed(1) ?? '?'}s',
                    style: const TextStyle(color: G20Colors.textSecondaryDark, fontSize: 11),
                  ),
                ),
                // Delete button
                IconButton(
                  icon: const Icon(Icons.delete_outline, size: 16, color: G20Colors.error),
                  onPressed: () => _onBoxDeleted(box.id),
                  padding: EdgeInsets.zero,
                  constraints: const BoxConstraints(minWidth: 24, minHeight: 24),
                ),
              ],
            ),
          ),
        );
      },
    );
  }
}
