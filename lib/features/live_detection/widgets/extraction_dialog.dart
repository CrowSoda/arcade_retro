import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../providers/subband_extraction_provider.dart';

/// Dialog for configuring and showing progress of sub-band extraction
class ExtractionDialog extends ConsumerStatefulWidget {
  final String sourceFile;
  final double? boxX1;
  final double? boxX2;
  final double sourceBandwidthHz;
  final int? durationMinutes;

  const ExtractionDialog({
    super.key,
    required this.sourceFile,
    this.boxX1,
    this.boxX2,
    this.sourceBandwidthHz = 20e6,
    this.durationMinutes,
  });

  @override
  ConsumerState<ExtractionDialog> createState() => _ExtractionDialogState();
}

class _ExtractionDialogState extends ConsumerState<ExtractionDialog> {
  bool extractNarrowband = true;
  double stopbandDb = 60.0;

  @override
  Widget build(BuildContext context) {
    final extractionState = ref.watch(subbandExtractionProvider);
    
    // Calculate extraction parameters from box
    ExtractionRequest? request;
    if (widget.boxX1 != null && widget.boxX2 != null) {
      request = ExtractionRequest.fromBox(
        sourceFile: widget.sourceFile,
        boxX1: widget.boxX1!,
        boxX2: widget.boxX2!,
        sourceBandwidthHz: widget.sourceBandwidthHz,
        durationSec: widget.durationMinutes != null 
            ? widget.durationMinutes! * 60.0 
            : null,
        stopbandDb: stopbandDb,
      );
    }

    // Get estimates
    final estimates = ref.watch(extractionEstimateProvider(request));

    return AlertDialog(
      title: Row(
        children: [
          const Icon(Icons.content_cut, color: Colors.amber),
          const SizedBox(width: 8),
          const Text('Sub-Band Extraction'),
        ],
      ),
      content: SizedBox(
        width: 400,
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Extraction toggle
            CheckboxListTile(
              title: const Text('Extract narrowband for training'),
              subtitle: Text(
                extractNarrowband && request != null
                    ? 'Reduces to ${(request.bandwidthHz / 1e6).toStringAsFixed(1)} MHz '
                      '(${estimates['output_rate_mhz']?.toStringAsFixed(2)} MHz sample rate)'
                    : 'Keep full 20 MHz capture',
              ),
              value: extractNarrowband,
              onChanged: (v) => setState(() => extractNarrowband = v ?? true),
            ),
            
            const Divider(),
            
            // Extraction parameters
            if (extractNarrowband && request != null) ...[
              _buildInfoRow(
                'Signal Bandwidth',
                '${(request.bandwidthHz / 1e6).toStringAsFixed(2)} MHz',
              ),
              _buildInfoRow(
                'Center Offset',
                '${(request.centerOffsetHz / 1e6).toStringAsFixed(2)} MHz from DC',
              ),
              _buildInfoRow(
                'Output Sample Rate',
                '${estimates['output_rate_mhz']?.toStringAsFixed(2)} MHz',
              ),
              _buildInfoRow(
                'Decimation Ratio',
                '${estimates['decimation_ratio']?.toStringAsFixed(1)}:1',
              ),
              _buildInfoRow(
                'Estimated Output Size',
                '${estimates['output_size_mb']?.toStringAsFixed(1)} MB',
              ),
              
              const SizedBox(height: 12),
              
              // Stopband slider
              Row(
                children: [
                  const Text('Stopband Attenuation:'),
                  Expanded(
                    child: Slider(
                      value: stopbandDb,
                      min: 40,
                      max: 80,
                      divisions: 4,
                      label: '${stopbandDb.toInt()} dB',
                      onChanged: (v) => setState(() => stopbandDb = v),
                    ),
                  ),
                  Text('${stopbandDb.toInt()} dB'),
                ],
              ),
              
              // Quality indicator
              Container(
                padding: const EdgeInsets.all(8),
                decoration: BoxDecoration(
                  color: stopbandDb >= 60 ? Colors.green.shade50 : Colors.orange.shade50,
                  borderRadius: BorderRadius.circular(4),
                ),
                child: Row(
                  children: [
                    Icon(
                      stopbandDb >= 60 ? Icons.check_circle : Icons.warning,
                      color: stopbandDb >= 60 ? Colors.green : Colors.orange,
                      size: 16,
                    ),
                    const SizedBox(width: 8),
                    Expanded(
                      child: Text(
                        stopbandDb >= 60
                            ? 'Good for CNN training (60+ dB recommended)'
                            : 'May cause aliasing artifacts in training data',
                        style: TextStyle(
                          fontSize: 12,
                          color: stopbandDb >= 60 ? Colors.green.shade700 : Colors.orange.shade700,
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ],
            
            // Progress indicator
            if (extractionState.isExtracting) ...[
              const SizedBox(height: 16),
              LinearProgressIndicator(value: extractionState.progress),
              const SizedBox(height: 8),
              Text(
                'Extracting sub-band: ${(extractionState.progress * 100).toInt()}%',
                style: Theme.of(context).textTheme.bodySmall,
              ),
            ],
            
            // Error display
            if (extractionState.error != null) ...[
              const SizedBox(height: 16),
              Container(
                padding: const EdgeInsets.all(8),
                decoration: BoxDecoration(
                  color: Colors.red.shade50,
                  borderRadius: BorderRadius.circular(4),
                ),
                child: Row(
                  children: [
                    const Icon(Icons.error, color: Colors.red, size: 16),
                    const SizedBox(width: 8),
                    Expanded(
                      child: Text(
                        extractionState.error!,
                        style: const TextStyle(color: Colors.red, fontSize: 12),
                      ),
                    ),
                  ],
                ),
              ),
            ],
            
            // Last result
            if (extractionState.lastResult != null && !extractionState.isExtracting) ...[
              const SizedBox(height: 16),
              Container(
                padding: const EdgeInsets.all(8),
                decoration: BoxDecoration(
                  color: Colors.blue.shade50,
                  borderRadius: BorderRadius.circular(4),
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text(
                      'Last Extraction:',
                      style: TextStyle(fontWeight: FontWeight.bold, fontSize: 12),
                    ),
                    Text(
                      'Output: ${extractionState.lastResult!.outputFile}',
                      style: const TextStyle(fontSize: 11),
                    ),
                    Text(
                      'Samples: ${extractionState.lastResult!.outputSamples} '
                      '(${extractionState.lastResult!.durationSec.toStringAsFixed(2)}s)',
                      style: const TextStyle(fontSize: 11),
                    ),
                    Text(
                      'Time: ${extractionState.lastResult!.processingTimeSec.toStringAsFixed(2)}s',
                      style: const TextStyle(fontSize: 11),
                    ),
                  ],
                ),
              ),
            ],
          ],
        ),
      ),
      actions: [
        TextButton(
          onPressed: () => Navigator.of(context).pop(null),
          child: const Text('Cancel'),
        ),
        ElevatedButton(
          onPressed: extractionState.isExtracting
              ? null
              : () {
                  Navigator.of(context).pop(
                    extractNarrowband && request != null
                        ? request.copyWith(stopbandDb: stopbandDb)
                        : null,
                  );
                },
          child: Text(extractNarrowband ? 'Extract & Capture' : 'Capture Only'),
        ),
      ],
    );
  }

  Widget _buildInfoRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(label, style: const TextStyle(fontSize: 13)),
          Text(value, style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 13)),
        ],
      ),
    );
  }
}

/// Extension to allow copyWith on ExtractionRequest
extension ExtractionRequestCopyWith on ExtractionRequest {
  ExtractionRequest copyWith({
    String? sourceFile,
    String? outputFile,
    double? centerOffsetHz,
    double? bandwidthHz,
    double? startSec,
    double? durationSec,
    double? stopbandDb,
  }) {
    return ExtractionRequest(
      sourceFile: sourceFile ?? this.sourceFile,
      outputFile: outputFile ?? this.outputFile,
      centerOffsetHz: centerOffsetHz ?? this.centerOffsetHz,
      bandwidthHz: bandwidthHz ?? this.bandwidthHz,
      startSec: startSec ?? this.startSec,
      durationSec: durationSec ?? this.durationSec,
      stopbandDb: stopbandDb ?? this.stopbandDb,
    );
  }
}

/// Show extraction dialog and return the request if confirmed
Future<ExtractionRequest?> showExtractionDialog({
  required BuildContext context,
  required String sourceFile,
  double? boxX1,
  double? boxX2,
  double sourceBandwidthHz = 20e6,
  int? durationMinutes,
}) {
  return showDialog<ExtractionRequest?>(
    context: context,
    builder: (context) => ExtractionDialog(
      sourceFile: sourceFile,
      boxX1: boxX1,
      boxX2: boxX2,
      sourceBandwidthHz: sourceBandwidthHz,
      durationMinutes: durationMinutes,
    ),
  );
}
