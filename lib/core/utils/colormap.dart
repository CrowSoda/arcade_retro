/// Shared colormap utilities for waterfall and PSD displays

/// Viridis-style colormap LUT (256 entries)
/// Boosted for RF visualization - good contrast on dark backgrounds
final List<List<int>> viridisLut = _buildViridisLut();

List<List<int>> _buildViridisLut() {
  const colors = [
    [52, 94, 141],    // 0.0 - blue (boosted start)
    [41, 120, 142],   // 0.1 - teal-blue
    [32, 144, 140],   // 0.2 - teal
    [34, 167, 132],   // 0.3 - teal-green
    [53, 183, 121],   // 0.4 - green-teal
    [68, 190, 112],   // 0.5 - green
    [94, 201, 98],    // 0.6 - bright green
    [121, 209, 81],   // 0.7 - lime green
    [165, 219, 54],   // 0.8 - yellow-lime
    [210, 226, 42],   // 0.9 - yellow-green
    [253, 231, 37],   // 1.0 - yellow
  ];
  
  return List.generate(256, (i) {
    final n = i / 255.0;
    final idx = (n * 10).clamp(0.0, 9.99);
    final ii = idx.floor();
    final t = idx - ii;
    
    final c0 = colors[ii];
    final c1 = colors[ii + 1];
    
    return [
      (c0[0] + t * (c1[0] - c0[0])).round(),
      (c0[1] + t * (c1[1] - c0[1])).round(),
      (c0[2] + t * (c1[2] - c0[2])).round(),
    ];
  });
}

/// Get RGB color for normalized value (0-1)
List<int> getViridisColor(double normalized) {
  final idx = (normalized.clamp(0.0, 1.0) * 255).round();
  return viridisLut[idx];
}
