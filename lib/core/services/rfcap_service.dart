import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

/// G20 RFCAP File Format Header
/// 
/// Total: 512 bytes (fixed)
/// 
/// | Offset | Size | Type     | Field              |
/// |--------|------|----------|--------------------|
/// | 0      | 4    | char[4]  | Magic ("G20\0")    |
/// | 4      | 4    | uint32   | Version            |
/// | 8      | 8    | float64  | Sample rate (Hz)   |
/// | 16     | 8    | float64  | Center freq (Hz)   |
/// | 24     | 8    | float64  | Bandwidth (Hz)     |
/// | 32     | 8    | uint64   | Number of samples  |
/// | 40     | 8    | float64  | Start time (epoch) |
/// | 48     | 32   | char[32] | Signal name        |
/// | 80     | 8    | float64  | Latitude           |
/// | 88     | 8    | float64  | Longitude          |
/// | 96     | 416  | reserved | (zeros)            |
/// 
/// After header: complex64 IQ data (float32 I, float32 Q pairs)

const int rfcapHeaderSize = 512;
const String rfcapMagic = 'G20\x00';

/// RFCAP file header metadata
class RfcapHeader {
  final int version;
  final double sampleRate;
  final double centerFreqHz;
  final double bandwidthHz;
  final int numSamples;
  final double startTimeEpoch;
  final String signalName;
  final double latitude;
  final double longitude;

  const RfcapHeader({
    required this.version,
    required this.sampleRate,
    required this.centerFreqHz,
    required this.bandwidthHz,
    required this.numSamples,
    required this.startTimeEpoch,
    required this.signalName,
    required this.latitude,
    required this.longitude,
  });

  double get centerFreqMHz => centerFreqHz / 1e6;
  double get bandwidthMHz => bandwidthHz / 1e6;
  double get sampleRateMHz => sampleRate / 1e6;
  double get durationSec => numSamples / sampleRate;
  DateTime get startTime => DateTime.fromMillisecondsSinceEpoch((startTimeEpoch * 1000).toInt());
  int get dataOffset => rfcapHeaderSize;
  int get fileSizeBytes => rfcapHeaderSize + (numSamples * 8);  // 8 bytes per complex64 sample

  @override
  String toString() => 'RfcapHeader(signal=$signalName, cf=${centerFreqMHz.toStringAsFixed(2)}MHz, '
      'bw=${bandwidthMHz.toStringAsFixed(2)}MHz, sr=${sampleRateMHz.toStringAsFixed(2)}MHz, '
      'duration=${durationSec.toStringAsFixed(1)}s)';
}

/// Service for reading and writing G20 RFCAP files
class RfcapService {
  /// Read header from RFCAP file
  static Future<RfcapHeader?> readHeader(String filepath) async {
    try {
      final file = File(filepath);
      if (!await file.exists()) return null;

      final raf = await file.open(mode: FileMode.read);
      final headerBytes = await raf.read(rfcapHeaderSize);
      await raf.close();

      return parseHeader(headerBytes);
    } catch (e) {
      print('Error reading RFCAP header: $e');
      return null;
    }
  }

  /// Parse header bytes into RfcapHeader
  static RfcapHeader? parseHeader(Uint8List bytes) {
    if (bytes.length < rfcapHeaderSize) return null;

    final data = ByteData.sublistView(bytes);

    // Check magic
    final magic = String.fromCharCodes(bytes.sublist(0, 4));
    if (magic != rfcapMagic) {
      print('Invalid RFCAP magic: $magic');
      return null;
    }

    // Parse fields (little-endian)
    final version = data.getUint32(4, Endian.little);
    final sampleRate = data.getFloat64(8, Endian.little);
    final centerFreq = data.getFloat64(16, Endian.little);
    final bandwidth = data.getFloat64(24, Endian.little);
    final numSamples = data.getUint64(32, Endian.little);
    final startTime = data.getFloat64(40, Endian.little);

    // Signal name (32 bytes, null-terminated)
    final nameBytes = bytes.sublist(48, 80);
    final nullIdx = nameBytes.indexOf(0);
    final signalName = String.fromCharCodes(
      nullIdx >= 0 ? nameBytes.sublist(0, nullIdx) : nameBytes,
    );

    final latitude = data.getFloat64(80, Endian.little);
    final longitude = data.getFloat64(88, Endian.little);

    return RfcapHeader(
      version: version,
      sampleRate: sampleRate,
      centerFreqHz: centerFreq,
      bandwidthHz: bandwidth,
      numSamples: numSamples,
      startTimeEpoch: startTime,
      signalName: signalName,
      latitude: latitude,
      longitude: longitude,
    );
  }

  /// Write RFCAP header to file
  static Uint8List createHeader({
    required double sampleRate,
    required double centerFreqHz,
    required double bandwidthHz,
    required int numSamples,
    required String signalName,
    double latitude = 0.0,
    double longitude = 0.0,
    DateTime? startTime,
  }) {
    final header = Uint8List(rfcapHeaderSize);
    final data = ByteData.sublistView(header);

    // Magic
    header[0] = 0x47; // G
    header[1] = 0x32; // 2
    header[2] = 0x30; // 0
    header[3] = 0x00; // null

    // Version
    data.setUint32(4, 1, Endian.little);

    // Sample rate
    data.setFloat64(8, sampleRate, Endian.little);

    // Center frequency
    data.setFloat64(16, centerFreqHz, Endian.little);

    // Bandwidth
    data.setFloat64(24, bandwidthHz, Endian.little);

    // Number of samples
    data.setUint64(32, numSamples, Endian.little);

    // Start time
    final epoch = (startTime ?? DateTime.now()).millisecondsSinceEpoch / 1000.0;
    data.setFloat64(40, epoch, Endian.little);

    // Signal name (32 bytes, null-padded)
    final nameBytes = signalName.codeUnits;
    for (int i = 0; i < 32 && i < nameBytes.length; i++) {
      header[48 + i] = nameBytes[i];
    }

    // Latitude
    data.setFloat64(80, latitude, Endian.little);

    // Longitude
    data.setFloat64(88, longitude, Endian.little);

    return header;
  }

  /// Write complete RFCAP file (header + IQ data)
  static Future<void> writeFile({
    required String filepath,
    required double sampleRate,
    required double centerFreqHz,
    required double bandwidthHz,
    required Uint8List iqData,
    required String signalName,
    double latitude = 0.0,
    double longitude = 0.0,
    DateTime? startTime,
  }) async {
    final numSamples = iqData.length ~/ 8;  // complex64 = 8 bytes

    final header = createHeader(
      sampleRate: sampleRate,
      centerFreqHz: centerFreqHz,
      bandwidthHz: bandwidthHz,
      numSamples: numSamples,
      signalName: signalName,
      latitude: latitude,
      longitude: longitude,
      startTime: startTime,
    );

    final file = File(filepath);
    await file.writeAsBytes(header, flush: false);
    await file.writeAsBytes(iqData, mode: FileMode.append, flush: true);
  }

  /// Read IQ samples from RFCAP file
  /// Returns Float32List of interleaved I,Q,I,Q... samples
  /// 
  /// NOTE: This validates actual file size vs header claims to prevent crashes.
  /// The returned data may be shorter than requested if file is truncated.
  static Future<Float32List?> readIqData(String filepath, {int? offsetSamples, int? numSamples}) async {
    try {
      final header = await readHeader(filepath);
      if (header == null) return null;

      final file = File(filepath);
      
      // Get actual file size to validate against header claims
      final actualFileSize = await file.length();
      final actualDataBytes = actualFileSize - rfcapHeaderSize;
      final actualSamplesInFile = actualDataBytes ~/ 8;  // 8 bytes per complex sample
      
      // Warn if header claims more data than exists
      if (header.numSamples > actualSamplesInFile) {
        print('WARNING: RFCAP header claims ${header.numSamples} samples, but file only contains $actualSamplesInFile');
      }
      
      final raf = await file.open(mode: FileMode.read);

      // Calculate offset with bounds checking
      final offset = offsetSamples ?? 0;
      if (offset >= actualSamplesInFile) {
        print('ERROR: Offset $offset exceeds available samples $actualSamplesInFile');
        await raf.close();
        return Float32List(0);
      }
      
      // Seek to data start
      final dataStart = rfcapHeaderSize + offset * 8;
      await raf.setPosition(dataStart);

      // Calculate how many samples we CAN read (bounded by actual file)
      final requestedSamples = numSamples ?? header.numSamples;
      final availableSamples = actualSamplesInFile - offset;
      final samplesToRead = requestedSamples < availableSamples ? requestedSamples : availableSamples;
      
      if (samplesToRead <= 0) {
        print('ERROR: No samples available to read');
        await raf.close();
        return Float32List(0);
      }
      
      final bytesToRead = samplesToRead * 8;
      final bytes = await raf.read(bytesToRead);
      await raf.close();

      // Convert to Float32List
      return bytes.buffer.asFloat32List();
    } catch (e) {
      print('Error reading RFCAP IQ data: $e');
      return null;
    }
  }

  /// Read raw IQ bytes from RFCAP file (for sending to backend)
  /// Returns Uint8List of raw complex64 bytes (ready for base64 encoding)
  static Future<Uint8List?> readIqDataRaw(String filepath, {int? offsetSamples, int? numSamples}) async {
    try {
      final header = await readHeader(filepath);
      if (header == null) return null;

      final file = File(filepath);
      
      // Get actual file size to validate
      final actualFileSize = await file.length();
      final actualDataBytes = actualFileSize - rfcapHeaderSize;
      final actualSamplesInFile = actualDataBytes ~/ 8;
      
      final raf = await file.open(mode: FileMode.read);

      // Calculate offset with bounds checking
      final offset = offsetSamples ?? 0;
      if (offset >= actualSamplesInFile) {
        await raf.close();
        return Uint8List(0);
      }
      
      // Seek to data start
      final dataStart = rfcapHeaderSize + offset * 8;
      await raf.setPosition(dataStart);

      // Calculate how many samples we CAN read
      final requestedSamples = numSamples ?? header.numSamples;
      final availableSamples = actualSamplesInFile - offset;
      final samplesToRead = requestedSamples < availableSamples ? requestedSamples : availableSamples;
      
      if (samplesToRead <= 0) {
        await raf.close();
        return Uint8List(0);
      }
      
      final bytesToRead = samplesToRead * 8;
      final bytes = await raf.read(bytesToRead);
      await raf.close();

      return bytes;
    } catch (e) {
      print('Error reading RFCAP IQ data raw: $e');
      return null;
    }
  }

  /// List all .rfcap files in a directory
  static Future<List<String>> listRfcapFiles(String dirPath) async {
    final dir = Directory(dirPath);
    if (!await dir.exists()) return [];

    final files = await dir.list().where((e) => e.path.endsWith('.rfcap')).toList();
    return files.map((e) => e.path).toList();
  }

  /// Generate mock IQ data with simulated signal
  /// 
  /// Creates realistic-looking IQ data with:
  /// - Background noise
  /// - A signal burst in the specified frequency/time range
  /// 
  /// [durationSec] - How many seconds of data
  /// [sampleRate] - Sample rate in Hz
  /// [signalFreqOffset] - Frequency offset from center in Hz (0 = center)
  /// [signalBandwidthHz] - Bandwidth of the simulated signal
  static Uint8List generateMockIqData({
    required double durationSec,
    required double sampleRate,
    double signalFreqOffset = 0,
    double signalBandwidthHz = 100000,
    double snrDb = 20.0,
  }) {
    final rng = math.Random();
    final numSamples = (durationSec * sampleRate).toInt();
    
    // Create Float32 buffer for I/Q pairs
    final floatData = Float32List(numSamples * 2);
    
    // Signal parameters
    final signalPower = math.pow(10, snrDb / 20);  // Linear amplitude
    final noiseLevel = 0.1;  // Baseline noise
    
    // Simulate signal with some modulation
    final signalFreqRad = 2 * math.pi * signalFreqOffset / sampleRate;
    
    for (int i = 0; i < numSamples; i++) {
      // Noise (complex Gaussian)
      final noiseI = (rng.nextDouble() - 0.5) * 2 * noiseLevel;
      final noiseQ = (rng.nextDouble() - 0.5) * 2 * noiseLevel;
      
      // Simulated signal - carrier with some random modulation
      final t = i / sampleRate;
      final phase = signalFreqRad * i;
      
      // Add some bursty behavior (signal fades in/out)
      final burst = (math.sin(2 * math.pi * t * 0.5) + 1) / 2;  // Slow fade
      final mod = 0.5 + 0.5 * math.sin(2 * math.pi * t * 1000);  // AM modulation
      
      final sigAmp = signalPower * burst * mod * noiseLevel;
      final sigI = sigAmp * math.cos(phase);
      final sigQ = sigAmp * math.sin(phase);
      
      // Combine
      floatData[i * 2] = (noiseI + sigI).toDouble();
      floatData[i * 2 + 1] = (noiseQ + sigQ).toDouble();
    }
    
    // Convert to bytes
    return floatData.buffer.asUint8List();
  }

  /// Generate DTG (Date Time Group) string for filenames
  /// Format: ddmmyyyy_hhmmss (e.g., "26012026_043243")
  static String generateDTG([DateTime? time]) {
    final t = time ?? DateTime.now().toUtc();
    final dd = t.day.toString().padLeft(2, '0');
    final mm = t.month.toString().padLeft(2, '0');
    final yyyy = t.year.toString();
    final hh = t.hour.toString().padLeft(2, '0');
    final min = t.minute.toString().padLeft(2, '0');
    final ss = t.second.toString().padLeft(2, '0');
    return '${dd}${mm}${yyyy}_$hh$min$ss';
  }

  /// Generate filename for manual capture
  /// Format: MAN_{ddmmyyyy_hhmmss}_{FREQ}MHz.rfcap
  /// Example: MAN_26012026_043243_825MHz.rfcap
  static String generateFilename(String signalName, [DateTime? time, double? freqMHz]) {
    final dtg = generateDTG(time);
    
    // Format frequency - round to nearest integer for clean filenames
    final freqStr = freqMHz != null ? '${freqMHz.round()}MHz' : '';
    
    // Use MAN_ prefix for manual captures, otherwise use signal name
    if (signalName.toUpperCase() == 'MAN' || signalName.toUpperCase().startsWith('MAN_')) {
      // Manual capture: MAN_ddmmyyyy_hhmmss_FREQMHz.rfcap
      return freqStr.isNotEmpty 
          ? 'MAN_${dtg}_$freqStr.rfcap'
          : 'MAN_$dtg.rfcap';
    } else {
      // Named signal: SIGNALNAME_ddmmyyyy_hhmmss_FREQMHz.rfcap
      final safeName = signalName.toUpperCase().replaceAll(RegExp(r'[^\w]'), '_');
      return freqStr.isNotEmpty
          ? '${safeName}_${dtg}_$freqStr.rfcap'
          : '${safeName}_$dtg.rfcap';
    }
  }
}
