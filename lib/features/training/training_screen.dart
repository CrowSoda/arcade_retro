import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:path/path.dart' as path;
import 'package:shared_preferences/shared_preferences.dart';
import '../../core/config/theme.dart';
import '../../core/database/signal_database.dart';
import '../../core/services/rfcap_service.dart';
import '../live_detection/providers/map_provider.dart' show getSOIColor;
import 'providers/training_provider.dart' as tp;
import 'providers/crop_classifier_provider.dart';
import 'widgets/training_spectrogram.dart';
import 'widgets/model_selection_dialog.dart';
import 'widgets/crop_review_dialog.dart';

/// Key for persisting last loaded file
const _kLastLoadedFileKey = 'training_last_loaded_file';

/// Training Screen - Label spectrogram data + Train
class TrainingScreen extends ConsumerStatefulWidget {
  const TrainingScreen({super.key});

  @override
  ConsumerState<TrainingScreen> createState() => _TrainingScreenState();
}

class _TrainingScreenState extends ConsumerState<TrainingScreen> {
  String? _selectedFile;
  RfcapHeader? _loadedHeader;
  // _isTraining is now derived from the provider state (persists across navigation)
  bool _isLoadingFiles = true;

  // Training preset (research-based)
  tp.TrainingPreset _selectedPreset = tp.TrainingPreset.balanced;

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
    _loadLastLoadedFile();
    _loadAvailableFiles();
    // Load available crop classifier models from disk
    ref.read(cropClassifierProvider.notifier).loadAvailableModelsFromDisk();
    // Auto-refresh file list every 5 seconds (will pick up new captures)
    _refreshTimer = Timer.periodic(const Duration(seconds: 5), (_) => _loadAvailableFiles());
  }

  /// Load last loaded file from SharedPreferences
  Future<void> _loadLastLoadedFile() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      final lastFile = prefs.getString(_kLastLoadedFileKey);
      if (lastFile != null && await File(lastFile).exists()) {
        debugPrint('[Training] 📂 Restoring last loaded file: $lastFile');
        setState(() => _selectedFile = lastFile);
        _loadFileHeader(lastFile);
      }
    } catch (e) {
      debugPrint('[Training] Error loading last file: $e');
    }
  }

  /// Save last loaded file to SharedPreferences
  Future<void> _saveLastLoadedFile(String filepath) async {
    try {
      final prefs = await SharedPreferences.getInstance();
      await prefs.setString(_kLastLoadedFileKey, filepath);
      debugPrint('[Training] 💾 Saved last loaded file: $filepath');
    } catch (e) {
      debugPrint('[Training] Error saving last file: $e');
    }
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

  /// Get the .labels.json sidecar path for an rfcap file
  String _getLabelsPath(String rfcapPath) {
    return rfcapPath.replaceAll('.rfcap', '.labels.json');
  }

  /// Save labels to sidecar JSON file
  Future<void> _saveLabels() async {
    if (_selectedFile == null || _labelBoxes.isEmpty) return;

    final labelsPath = _getLabelsPath(_selectedFile!);
    final labelData = {
      'version': 1,
      'source_file': path.basename(_selectedFile!),
      'saved_at': DateTime.now().toIso8601String(),
      'class_name': _classNameController.text.trim(),
      'boxes': _labelBoxes.map((box) => {
        'id': box.id,
        'x1': box.x1,
        'y1': box.y1,
        'x2': box.x2,
        'y2': box.y2,
        'class_name': box.className,
        'freq_start_mhz': box.freqStartMHz,
        'freq_end_mhz': box.freqEndMHz,
        'time_start_sec': box.timeStartSec,
        'time_end_sec': box.timeEndSec,
      }).toList(),
    };

    try {
      final file = File(labelsPath);
      await file.writeAsString(const JsonEncoder.withIndent('  ').convert(labelData));
      debugPrint('💾 Saved ${_labelBoxes.length} labels to ${path.basename(labelsPath)}');
    } catch (e) {
      debugPrint('Error saving labels: $e');
    }
  }

  /// Load labels from sidecar JSON file
  Future<void> _loadLabels(String rfcapPath) async {
    final labelsPath = _getLabelsPath(rfcapPath);
    final file = File(labelsPath);

    if (!await file.exists()) {
      debugPrint('No labels file found for ${path.basename(rfcapPath)}');
      return;
    }

    try {
      final jsonStr = await file.readAsString();
      final data = jsonDecode(jsonStr) as Map<String, dynamic>;

      final boxes = <LabelBox>[];
      final boxList = data['boxes'] as List<dynamic>? ?? [];

      for (final boxData in boxList) {
        boxes.add(LabelBox(
          id: boxData['id'] ?? DateTime.now().millisecondsSinceEpoch,
          x1: (boxData['x1'] as num).toDouble(),
          y1: (boxData['y1'] as num).toDouble(),
          x2: (boxData['x2'] as num).toDouble(),
          y2: (boxData['y2'] as num).toDouble(),
          className: boxData['class_name'] ?? 'unknown',
          freqStartMHz: boxData['freq_start_mhz']?.toDouble(),
          freqEndMHz: boxData['freq_end_mhz']?.toDouble(),
          timeStartSec: boxData['time_start_sec']?.toDouble(),
          timeEndSec: boxData['time_end_sec']?.toDouble(),
        ));
      }

      setState(() {
        _boxesByFile[rfcapPath] = boxes;
        // Also restore class name if present
        final savedClassName = data['class_name'] as String?;
        if (savedClassName != null && savedClassName.isNotEmpty) {
          _classNameController.text = savedClassName;
        }
      });

      debugPrint('📂 Loaded ${boxes.length} labels from ${path.basename(labelsPath)}');
    } catch (e) {
      debugPrint('Error loading labels: $e');
    }
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

    // Sort by most recent (modification time) - fetch times async first
    final fileTimes = <String, DateTime>{};
    for (final file in files) {
      try {
        final stat = await File(file).stat();
        fileTimes[file] = stat.modified;
      } catch (_) {
        fileTimes[file] = DateTime.fromMillisecondsSinceEpoch(0);
      }
    }

    files.sort((a, b) {
      final aTime = fileTimes[a] ?? DateTime.fromMillisecondsSinceEpoch(0);
      final bTime = fileTimes[b] ?? DateTime.fromMillisecondsSinceEpoch(0);
      return bTime.compareTo(aTime);  // Most recent first
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

      // Load any saved labels for this file
      await _loadLabels(filepath);
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
      _saveLastLoadedFile(filepath);  // Persist for next app launch
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

    // Auto-save labels to file
    _saveLabels();
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

    // Auto-save labels to file (even if now empty - removes the file content)
    _saveLabels();
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

  void _startTraining() async {
    if (_labelBoxes.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Draw at least one bounding box first'),
          backgroundColor: G20Colors.warning,
        ),
      );
      return;
    }

    final className = _classNameController.text.trim();
    if (className.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Enter a class name first'),
          backgroundColor: G20Colors.warning,
        ),
      );
      return;
    }

    // Step 1: Show model selection dialog with duration options
    final cropState = ref.read(cropClassifierProvider);
    final fileDuration = _loadedHeader?.durationSec ?? 60.0;
    final modelResult = await showModelSelectionDialog(
      context: context,
      currentClassName: className,
      existingModels: cropState.availableModels,
      fileDurationSec: fileDuration,
    );

    if (modelResult == null || !mounted) return; // User cancelled

    // Step 2: Update class name if user picked a new model name
    if (modelResult.isNew && modelResult.modelName != className) {
      _classNameController.text = modelResult.modelName;
      _updateAllBoxClasses(modelResult.modelName);
    }

    // ─────────────────────────────────────────────────────────────────
    // BOOTSTRAP FLOW: Use user's drawn boxes as seeds for template matching
    // ─────────────────────────────────────────────────────────────────
    final useBootstrap = _labelBoxes.length >= 3; // Need at least 3 seeds

    if (useBootstrap) {
      await _runBootstrapFlow(modelResult);
    } else {
      // Fallback: use original blob detection flow
      await _runLegacyBlobFlow(modelResult);
    }
  }

  /// BOOTSTRAP FLOW: Use user's drawn boxes as seed templates
  Future<void> _runBootstrapFlow(ModelSelectionResult modelResult) async {
    debugPrint('[Training] 🌱 BOOTSTRAP MODE: Using ${_labelBoxes.length} boxes as seed templates');

    bool dialogOpen = false;

    try {
      final cropNotifier = ref.read(cropClassifierProvider.notifier);

      // Convert LabelBox to seed box format (normalized 0-1 coords)
      // Backend will convert to pixel coords based on its spectrogram dimensions
      final seedBoxes = _labelBoxes.map((box) => <String, double>{
        'x1': box.x1,
        'y1': box.y1,
        'x2': box.x2,
        'y2': box.y2,
      }).toList();

      // Get current view window from spectrogram
      // Default: 0s start, 0.5s duration (matches backend default)
      final timeStartSec = 0.0; // TODO: Get from TrainingSpectrogram._windowStartSec
      final timeDurationSec = 0.5; // TODO: Get from TrainingSpectrogram._windowLengthSec

      // Show progress dialog
      if (mounted) {
        dialogOpen = true;
        unawaited(showDialog(
          context: context,
          barrierDismissible: false,
          builder: (ctx) => AlertDialog(
            backgroundColor: G20Colors.surfaceDark,
            title: const Text('🌱 Finding similar signals...',
              style: TextStyle(color: G20Colors.textPrimaryDark, fontSize: 16)),
            content: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                const LinearProgressIndicator(
                  backgroundColor: G20Colors.cardDark,
                  valueColor: AlwaysStoppedAnimation(G20Colors.primary),
                ),
                const SizedBox(height: 12),
                Text(
                  'Using ${_labelBoxes.length} seed boxes as templates',
                  style: const TextStyle(color: G20Colors.textSecondaryDark, fontSize: 12),
                ),
              ],
            ),
          ),
        ));
      }

      // Call bootstrap_file with seed boxes (normalized 0-1 coords)
      final bootstrapResult = await cropNotifier.bootstrapFromFile(
        rfcapPath: _selectedFile!,
        seedBoxes: seedBoxes,
        timeStartSec: timeStartSec,
        timeDurationSec: timeDurationSec,
        topK: 50,
      );

      // Close progress dialog
      if (dialogOpen && mounted && Navigator.of(context).canPop()) {
        Navigator.of(context).pop();
        dialogOpen = false;
      }

      debugPrint('[Training] 📋 Bootstrap found ${bootstrapResult.candidates.length} candidates');

      if (bootstrapResult.candidates.isEmpty) {
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(
              content: Text('No similar signals found. Try drawing more boxes or different positions.'),
              backgroundColor: G20Colors.warning,
            ),
          );
        }
        return;
      }

      // Show swipe dialog with BOOTSTRAP candidates (pre-sorted by similarity!)
      final reviewResult = await showCropReviewDialog(
        context: context,
        crops: bootstrapResult.toCropReviewDataList(),
      );

      if (reviewResult == null || !mounted) {
        debugPrint('[Training] ❌ User cancelled crop review');
        return;
      }

      // Get confirmed/rejected indices from reviewResult
      final confirmedIndices = <int>[];
      final rejectedIndices = <int>[];

      for (final entry in reviewResult.labels.entries) {
        // Extract index from crop ID (format: "bootstrap_N")
        final idParts = entry.key.split('_');
        if (idParts.length >= 2) {
          final idx = int.tryParse(idParts.last);
          if (idx != null) {
            if (entry.value) {
              confirmedIndices.add(idx);
            } else {
              rejectedIndices.add(idx);
            }
          }
        }
      }

      debugPrint('[Training] ✅ Swipe complete: ${confirmedIndices.length} confirmed, '
          '${rejectedIndices.length} rejected');

      // Record confirmations with backend
      final stats = await cropNotifier.confirmLabels(
        confirmed: confirmedIndices,
        rejected: rejectedIndices,
      );

      debugPrint('[Training] 📊 Stats after confirm: positives=${stats.positives}, '
          'negatives=${stats.negatives}, ready=${stats.readyToTrain}');

      // Proceed to training if ready
      if (stats.readyToTrain || (stats.positives >= 5 && stats.negatives >= 5)) {
        debugPrint('[Training] 🚀 Ready to train!');
        _doRealTraining();
      } else {
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Text('Need more labels: ${stats.positives} positives, ${stats.negatives} negatives. Draw more boxes or confirm more candidates.'),
              backgroundColor: G20Colors.warning,
            ),
          );
        }
      }

    } catch (e) {
      if (dialogOpen && mounted) {
        try {
          if (Navigator.of(context).canPop()) {
            Navigator.of(context).pop();
          }
        } catch (_) {}
        dialogOpen = false;
      }

      debugPrint('[Training] ❌ Bootstrap error: $e');

      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Bootstrap failed: $e'),
            backgroundColor: G20Colors.error,
          ),
        );
      }
    }
  }

  /// Legacy flow: Scan entire file for blobs (slower, less targeted)
  Future<void> _runLegacyBlobFlow(ModelSelectionResult modelResult) async {
    debugPrint('[Training] 🔍 LEGACY MODE: Scanning ${modelResult.scanDurationSec}s for blob detection...');

    bool dialogOpen = false;
    double currentProgress = 0.0;
    int cropsFoundSoFar = 0;
    StateSetter? dialogSetState;

    try {
      final cropNotifier = ref.read(cropClassifierProvider.notifier);
      final totalChunks = (modelResult.scanDurationSec / 0.5).ceil();

      if (mounted) {
        dialogOpen = true;
        unawaited(showDialog(
          context: context,
          barrierDismissible: false,
          builder: (ctx) => StatefulBuilder(
            builder: (ctx, setState) {
              dialogSetState = setState;
              final processedChunks = (currentProgress * totalChunks).round();
              return AlertDialog(
                backgroundColor: G20Colors.surfaceDark,
                title: const Text('🔍 Scanning for signals...',
                  style: TextStyle(color: G20Colors.textPrimaryDark, fontSize: 16)),
                content: Column(
                  mainAxisSize: MainAxisSize.min,
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    LinearProgressIndicator(
                      value: currentProgress,
                      backgroundColor: G20Colors.cardDark,
                      valueColor: const AlwaysStoppedAnimation(G20Colors.primary),
                    ),
                    const SizedBox(height: 12),
                    Text('Chunk $processedChunks/$totalChunks',
                      style: const TextStyle(color: G20Colors.textSecondaryDark, fontSize: 12)),
                    const SizedBox(height: 4),
                    Text('${(currentProgress * 100).toInt()}% • $cropsFoundSoFar signals found',
                      style: const TextStyle(color: G20Colors.textPrimaryDark, fontSize: 14, fontWeight: FontWeight.w500)),
                  ],
                ),
              );
            },
          ),
        ));
      }

      final crops = await cropNotifier.detectCropsFromFile(
        rfcapPath: _selectedFile!,
        scanDurationSec: modelResult.scanDurationSec,
        progressCallback: (progress, cropsFound) {
          currentProgress = progress;
          cropsFoundSoFar = cropsFound;
          if (dialogSetState != null) {
            try { dialogSetState!(() {}); } catch (_) {}
          }
        },
      );

      if (dialogOpen && mounted && Navigator.of(context).canPop()) {
        Navigator.of(context).pop();
        dialogOpen = false;
      }

      if (crops.isEmpty) {
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(
              content: Text('No signals detected. Try drawing boxes for bootstrap mode.'),
              backgroundColor: G20Colors.warning,
            ),
          );
        }
        return;
      }

      final reviewResult = await showCropReviewDialog(context: context, crops: crops);

      if (reviewResult == null || !mounted) return;

      if (reviewResult.labels.isEmpty) {
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(
              content: Text('No crops labeled.'),
              backgroundColor: G20Colors.warning,
            ),
          );
        }
        return;
      }

      _doRealTraining();

    } catch (e) {
      if (dialogOpen && mounted) {
        try {
          if (Navigator.of(context).canPop()) Navigator.of(context).pop();
        } catch (_) {}
      }

      debugPrint('[Training] ❌ Error: $e');

      if (mounted) {
        final errorStr = e.toString();
        final isConnectionError = errorStr.contains('refused') || errorStr.contains('SocketException');

        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text(isConnectionError
              ? 'Backend not running! Run: python backend/server.py'
              : 'Detection failed: $e'),
            backgroundColor: G20Colors.error,
          ),
        );
      }
    }
  }

  void _doRealTraining() async {
    if (_selectedFile == null || _loadedHeader == null) return;

    final className = _classNameController.text.trim();
    final notifier = ref.read(tp.trainingProvider.notifier);
    final header = _loadedHeader!;

    // Convert LabelBox to the format expected by training provider
    // For OLD boxes with null timeStartSec: we can't recover the original time,
    // so log a warning and skip them (user needs to redraw).
    final validBoxes = <Map<String, dynamic>>[];

    for (final box in _labelBoxes) {
      // Check if this box has valid time coordinates
      if (box.timeStartSec == null || box.timeEndSec == null) {
        debugPrint('[Training] ⚠️ SKIPPING box with null time coordinates - user must redraw!');
        debugPrint('           Box normalized coords: x1=${box.x1}, x2=${box.x2}');
        continue;  // Skip this box - can't recover time without knowing original window
      }

      // Time coordinates are already absolute - use them directly
      final timeStart = box.timeStartSec!;
      final timeEnd = box.timeEndSec!;

      // Python reads a SMALL window (0.1s) centered on the box center, regardless of box size.
      // So even if the user drew a large box, we only need to send the center time.
      // Let's clamp to reasonable values.
      final boxDuration = timeEnd - timeStart;
      if (boxDuration > 1.0) {
        debugPrint('[Training] ⚠️ Box duration ${boxDuration.toStringAsFixed(2)}s is very large - Python will only use 0.1s window');
      }

      // Frequency: use box values or calculate from normalized coords
      final bandwidth = header.sampleRate;  // FFT Nyquist = sample_rate
      final freqStart = box.freqStartMHz ??
          (header.centerFreqHz - bandwidth/2 + (1 - box.y2.clamp(0, 1)) * bandwidth) / 1e6;
      final freqEnd = box.freqEndMHz ??
          (header.centerFreqHz - bandwidth/2 + (1 - box.y1.clamp(0, 1)) * bandwidth) / 1e6;

      debugPrint('[Training] Box: time=${timeStart.toStringAsFixed(3)}s-${timeEnd.toStringAsFixed(3)}s, '
          'freq=${freqStart.toStringAsFixed(2)}-${freqEnd.toStringAsFixed(2)}MHz');

      validBoxes.add({
        'x1': box.x1,
        'y1': box.y1,
        'x2': box.x2,
        'y2': box.y2,
        'time_start_sec': timeStart,
        'time_end_sec': timeEnd,
        'freq_start_mhz': freqStart,
        'freq_end_mhz': freqEnd,
      });
    }

    final boxes = validBoxes;

    if (boxes.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('No valid boxes found. Please redraw your labels.'),
          backgroundColor: G20Colors.warning,
        ),
      );
      return;
    }

    try {
      final result = await notifier.trainFromFile(
        rfcapPath: _selectedFile!,
        signalName: className,
        boxes: boxes,
        preset: _selectedPreset,  // Use selected preset
        header: _loadedHeader,
      );

      if (mounted && result != null) {
        // Save to local database too
        _saveTrainingResultsFromBackend(result);

        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Training complete! F1: ${result.f1Score.toStringAsFixed(3)}'),
            backgroundColor: G20Colors.success,
            duration: const Duration(seconds: 3),
          ),
        );
      } else if (mounted) {
        final error = ref.read(tp.trainingProvider).error;
        if (error != null) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Text('Training failed: $error'),
              backgroundColor: G20Colors.error,
            ),
          );
        }
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Training error: $e'),
            backgroundColor: G20Colors.error,
          ),
        );
      }
    }
  }

  void _stopTraining() {
    ref.read(tp.trainingProvider.notifier).cancelTraining();
  }

  /// Save training results from backend to local signal database
  void _saveTrainingResultsFromBackend(tp.TrainingResult result) {
    final dbResult = TrainingResult(
      timestamp: DateTime.now(),
      dataLabels: result.sampleCount,
      f1Score: result.f1Score,
      precision: result.precision,
      recall: result.recall,
      epochs: result.epochsTrained,
      loss: 0.0, // Backend doesn't return final loss
      modelPath: 'models/heads/${result.signalName}/active.pth',
    );

    ref.read(signalDatabaseProvider.notifier).addTrainingResult(result.signalName, dbResult);
    debugPrint('[Training] Saved backend result to DB: ${result.signalName}');
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
    // Watch training state from provider (persists across navigation)
    final trainingState = ref.watch(tp.trainingProvider);
    final isTraining = trainingState.isTraining || trainingState.isSavingSamples;

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
        const SizedBox(height: 8),

        // Training preset selector (4 presets: Fast, Balanced, Quality, Extreme)
        const Text('Preset', style: TextStyle(color: G20Colors.textSecondaryDark, fontSize: 11)),
        const SizedBox(height: 4),
        Row(
          children: tp.TrainingPreset.values.map((preset) {
            final isSelected = preset == _selectedPreset;
            final isFirst = preset == tp.TrainingPreset.fast;
            final isLast = preset == tp.TrainingPreset.extreme;
            return Expanded(
              child: GestureDetector(
                onTap: isTraining ? null : () => setState(() => _selectedPreset = preset),
                child: Container(
                  padding: const EdgeInsets.symmetric(vertical: 6),
                  decoration: BoxDecoration(
                    color: isSelected ? G20Colors.primary.withOpacity(0.3) : G20Colors.backgroundDark,
                    border: Border.all(color: isSelected ? G20Colors.primary : G20Colors.cardDark),
                    borderRadius: BorderRadius.horizontal(
                      left: isFirst ? const Radius.circular(4) : Radius.zero,
                      right: isLast ? const Radius.circular(4) : Radius.zero,
                    ),
                  ),
                  child: Text(
                    preset.label,
                    textAlign: TextAlign.center,
                    style: TextStyle(
                      fontSize: 9,  // Smaller font to fit 4 presets
                      color: isSelected ? G20Colors.primary : G20Colors.textSecondaryDark,
                      fontWeight: isSelected ? FontWeight.w600 : FontWeight.normal,
                    ),
                  ),
                ),
              ),
            );
          }).toList(),
        ),
        const SizedBox(height: 12),
        // Training button + progress
        if (isTraining) ...[
          Consumer(
            builder: (context, ref, _) {
              final state = ref.watch(tp.trainingProvider);
              final progress = state.overallProgress;
              final epochProgress = state.progress;
              final currentEpoch = epochProgress?.epoch ?? 0;
              final totalEpochs = epochProgress?.totalEpochs ?? _selectedPreset.epochs;
              final currentF1 = epochProgress?.f1Score ?? 0.0;

              // Check if training is complete (all epochs done)
              final isComplete = currentEpoch >= totalEpochs && totalEpochs > 0 && !state.isSavingSamples;

              return Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  // Progress bar
                  LinearProgressIndicator(
                    value: progress.isNaN ? null : progress,
                    backgroundColor: G20Colors.cardDark,
                    valueColor: AlwaysStoppedAnimation(isComplete ? G20Colors.success : G20Colors.primary),
                  ),
                  const SizedBox(height: 6),
                  // Epoch X/N - F1: 0.XXX
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      Text(
                        state.isSavingSamples
                          ? 'Saving ${state.samplesSaved}/${state.totalSamplesToSave}'
                          : isComplete
                            ? 'Complete! $totalEpochs epochs'
                            : 'Epoch $currentEpoch/$totalEpochs',
                        style: TextStyle(
                          color: isComplete ? G20Colors.success : G20Colors.textPrimaryDark,
                          fontSize: 12,
                          fontWeight: FontWeight.w500,
                        ),
                      ),
                      Text(
                        'F1: ${currentF1.toStringAsFixed(3)}',
                        style: TextStyle(
                          color: currentF1 > 0.7 ? G20Colors.success : (currentF1 > 0.4 ? G20Colors.warning : G20Colors.textSecondaryDark),
                          fontSize: 12,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 6),
                      // Best F1 indicator + Stop button (only during training)
                      Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          // Best F1 indicator (always show best so far)
                          Container(
                            padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                            decoration: BoxDecoration(
                              color: state.bestF1 > 0 ? G20Colors.success.withOpacity(0.2) : G20Colors.cardDark,
                              borderRadius: BorderRadius.circular(4),
                            ),
                            child: Row(
                              mainAxisSize: MainAxisSize.min,
                              children: [
                                if (epochProgress?.isBest == true || isComplete)
                                  const Text('★ ', style: TextStyle(color: G20Colors.success, fontSize: 10)),
                                Text(
                                  'Best: ${state.bestF1.toStringAsFixed(3)}',
                                  style: TextStyle(
                                    color: state.bestF1 > 0.7 ? G20Colors.success : G20Colors.textSecondaryDark,
                                    fontSize: 10,
                                    fontWeight: FontWeight.w600,
                                  ),
                                ),
                              ],
                            ),
                          ),
                          // Only show Stop button during training (not when complete)
                          if (!isComplete)
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
                ],
              );
            },
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
