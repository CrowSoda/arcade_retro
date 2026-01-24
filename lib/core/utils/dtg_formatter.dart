/// Date Time Group (DTG) formatting utilities
/// Military standard format: DDHHMMZ MMM YY

const _months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 
                 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'];

/// Format DateTime to DTG: DDHHMMZ MMM YY
/// Example: 211835Z JAN 26
String formatDTG(DateTime dt) {
  final day = dt.day.toString().padLeft(2, '0');
  final hour = dt.hour.toString().padLeft(2, '0');
  final min = dt.minute.toString().padLeft(2, '0');
  final month = _months[dt.month - 1];
  final year = (dt.year % 100).toString().padLeft(2, '0');
  return '$day$hour${min}Z $month $year';
}

/// Format DateTime to compact DTG for filenames: DDHHMMZMMYY
/// Example: 211835ZJAN26
String formatDTGCompact(DateTime dt) {
  final day = dt.day.toString().padLeft(2, '0');
  final hour = dt.hour.toString().padLeft(2, '0');
  final min = dt.minute.toString().padLeft(2, '0');
  final month = _months[dt.month - 1];
  final year = (dt.year % 100).toString().padLeft(2, '0');
  return '$day$hour${min}Z$month$year';
}

/// Generate UNK filename: unk_[DTG]_[freq]MHz
/// Example: unk_211835ZJAN26_825.50MHz
String generateUnkFilename(DateTime dt, String freqMHz) {
  final dtg = formatDTGCompact(dt);
  // Clean freq (remove trailing zeros, keep decimal if needed)
  final freqClean = freqMHz.replaceAll(RegExp(r'\.?0+$'), '');
  return 'unk_${dtg}_${freqClean}MHz';
}
