/// WaterfallFrameHandler - Handles spectrogram frame lifecycle with aggressive disposal.
/// 
/// CRITICAL: ui.Image objects must be disposed aggressively to prevent memory growth.
/// Do NOT wait for widget dispose() - dispose previous image BEFORE setting new one.

import 'dart:async';
import 'dart:typed_data';
import 'dart:ui' as ui;
import 'package:flutter/material.dart';

/// Manages spectrogram frame images with proper lifecycle.
/// 
/// MEMORY SAFETY:
/// - Disposes previous image BEFORE setting new one (not in dispose())
/// - Tracks disposal state to prevent double-dispose
/// - Safe for rapid frame updates (30+ fps)
class WaterfallFrameHandler {
  ui.Image? _currentImage;
  bool _isDisposed = false;
  int _frameId = -1;
  
  /// Current frame ID
  int get frameId => _frameId;
  
  /// Current image (may be null during updates)
  ui.Image? get currentImage => _currentImage;
  
  /// Whether handler has been disposed
  bool get isDisposed => _isDisposed;
  
  /// Update with new frame data.
  /// 
  /// CRITICAL: Disposes previous image BEFORE decoding new one.
  /// This prevents memory accumulation during rapid updates.
  Future<void> updateFrame({
    required Uint8List rgbaBytes,
    required int width,
    required int height,
    required int frameId,
  }) async {
    if (_isDisposed) return;
    
    // CRITICAL: Dispose previous image BEFORE creating new one
    // Do NOT wait for widget dispose() - that's too late!
    _disposeCurrentImage();
    
    try {
      // Decode new image
      final completer = Completer<ui.Image>();
      
      ui.decodeImageFromPixels(
        rgbaBytes,
        width,
        height,
        ui.PixelFormat.rgba8888,
        (image) {
          if (!_isDisposed) {
            completer.complete(image);
          } else {
            // Handler was disposed during decode - dispose the new image
            image.dispose();
            completer.completeError('Handler disposed during decode');
          }
        },
      );
      
      final newImage = await completer.future;
      
      if (!_isDisposed) {
        _currentImage = newImage;
        _frameId = frameId;
      }
    } catch (e) {
      // Decode failed or handler disposed - that's ok
    }
  }
  
  /// Dispose current image immediately.
  void _disposeCurrentImage() {
    if (_currentImage != null) {
      _currentImage!.dispose();
      _currentImage = null;
    }
  }
  
  /// Full disposal - call when widget is disposed.
  void dispose() {
    _isDisposed = true;
    _disposeCurrentImage();
  }
}


/// Widget that renders waterfall frames with proper image lifecycle.
/// 
/// Usage:
///   WaterfallFrameWidget(
///     frameStream: provider.frameStream,
///     width: 4096,
///     displayRows: 256,
///   )
class WaterfallFrameWidget extends StatefulWidget {
  final Stream<WaterfallFrameData> frameStream;
  final int displayRows;
  
  const WaterfallFrameWidget({
    super.key,
    required this.frameStream,
    this.displayRows = 256,
  });
  
  @override
  State<WaterfallFrameWidget> createState() => _WaterfallFrameWidgetState();
}

class _WaterfallFrameWidgetState extends State<WaterfallFrameWidget> {
  final WaterfallFrameHandler _frameHandler = WaterfallFrameHandler();
  StreamSubscription<WaterfallFrameData>? _subscription;
  
  @override
  void initState() {
    super.initState();
    _subscription = widget.frameStream.listen(_onFrame);
  }
  
  @override
  void didUpdateWidget(WaterfallFrameWidget oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (widget.frameStream != oldWidget.frameStream) {
      _subscription?.cancel();
      _subscription = widget.frameStream.listen(_onFrame);
    }
  }
  
  Future<void> _onFrame(WaterfallFrameData frame) async {
    // CRITICAL: updateFrame disposes previous image BEFORE decoding new
    await _frameHandler.updateFrame(
      rgbaBytes: frame.rgbaBytes,
      width: frame.width,
      height: frame.height,
      frameId: frame.frameId,
    );
    
    // Trigger repaint
    if (mounted) {
      setState(() {});
    }
  }
  
  @override
  void dispose() {
    _subscription?.cancel();
    _frameHandler.dispose();  // Disposes any remaining image
    super.dispose();
  }
  
  @override
  Widget build(BuildContext context) {
    final image = _frameHandler.currentImage;
    
    if (image == null) {
      return const Center(child: CircularProgressIndicator());
    }
    
    return CustomPaint(
      painter: _WaterfallPainter(image: image),
      size: Size.infinite,
    );
  }
}


class _WaterfallPainter extends CustomPainter {
  final ui.Image image;
  
  _WaterfallPainter({required this.image});
  
  @override
  void paint(Canvas canvas, Size size) {
    // Scale image to fill canvas
    final srcRect = Rect.fromLTWH(
      0, 0,
      image.width.toDouble(),
      image.height.toDouble(),
    );
    final dstRect = Rect.fromLTWH(0, 0, size.width, size.height);
    
    canvas.drawImageRect(image, srcRect, dstRect, Paint());
  }
  
  @override
  bool shouldRepaint(_WaterfallPainter oldDelegate) {
    // Always repaint when image changes (different reference)
    return image != oldDelegate.image;
  }
}


/// Data class for waterfall frames.
class WaterfallFrameData {
  final int frameId;
  final Uint8List rgbaBytes;
  final int width;
  final int height;
  final double freqStartHz;
  final double freqEndHz;
  
  const WaterfallFrameData({
    required this.frameId,
    required this.rgbaBytes,
    required this.width,
    required this.height,
    required this.freqStartHz,
    required this.freqEndHz,
  });
}


/// Memory safety guidelines for ui.Image:
/// 
/// DO:
/// ✓ Dispose previous image BEFORE setting new one
/// ✓ Check isDisposed before using image
/// ✓ Dispose in widget dispose() as final cleanup
/// ✓ Track frame_id to correlate with tracks
/// 
/// DON'T:
/// ✗ Wait until dispose() to clean up all images
/// ✗ Hold references to multiple decoded images
/// ✗ Assume garbage collection will handle ui.Image
/// ✗ Use image after dispose() is called
/// 
/// Memory growth symptoms:
/// - Gradual increase in memory over time
/// - Eventually leads to OOM on embedded devices
/// - Worse with larger frames or higher frame rates
