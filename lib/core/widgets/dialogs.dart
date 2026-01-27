/// G20 Dialogs and Toasts - Reusable UI components
///
/// Provides consistent styling for toasts, confirmations, and error dialogs
library;

import 'dart:async';
import 'package:flutter/material.dart';
import '../config/theme.dart';

// =============================================================================
// FADING TOAST
// =============================================================================

/// Show a fading toast overlay that disappears after duration
///
/// Example:
/// ```dart
/// showG20Toast(context, 'Saved!', icon: Icons.check_circle);
/// showG20Toast(context, 'Error!', icon: Icons.error, color: Colors.red);
/// ```
void showG20Toast(
  BuildContext context,
  String message, {
  IconData icon = Icons.check_circle,
  Color color = Colors.green,
  Duration duration = const Duration(seconds: 2),
}) {
  final overlay = Overlay.of(context);
  late OverlayEntry entry;
  bool isRemoved = false;

  void safeRemove() {
    if (!isRemoved) {
      isRemoved = true;
      entry.remove();
    }
  }

  entry = OverlayEntry(
    builder: (context) => Positioned(
      top: MediaQuery.of(context).size.height * 0.15,
      left: 0,
      right: 0,
      child: Center(
        child: Material(
          color: Colors.transparent,
          child: _FadingToast(
            message: message,
            icon: icon,
            color: color.withOpacity(0.9),
            duration: duration,
            onComplete: safeRemove,
          ),
        ),
      ),
    ),
  );

  overlay.insert(entry);

  // Safety fallback: ensure removal after max duration even if animation fails
  Future.delayed(duration + const Duration(seconds: 3), safeRemove);
}

/// Internal fading toast widget
class _FadingToast extends StatefulWidget {
  final String message;
  final IconData icon;
  final Color color;
  final Duration duration;
  final VoidCallback onComplete;

  const _FadingToast({
    required this.message,
    required this.icon,
    required this.color,
    required this.duration,
    required this.onComplete,
  });

  @override
  State<_FadingToast> createState() => _FadingToastState();
}

class _FadingToastState extends State<_FadingToast> with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<double> _fadeAnimation;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(vsync: this, duration: const Duration(milliseconds: 300));
    _fadeAnimation = Tween<double>(begin: 0, end: 1).animate(
      CurvedAnimation(parent: _controller, curve: Curves.easeOut),
    );

    // Fade in
    _controller.forward();

    // Wait then fade out
    Future.delayed(widget.duration - const Duration(milliseconds: 300), () {
      if (mounted) {
        _controller.reverse().then((_) => widget.onComplete());
      }
    });
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return FadeTransition(
      opacity: _fadeAnimation,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
        decoration: BoxDecoration(
          color: widget.color,
          borderRadius: BorderRadius.circular(8),
          boxShadow: const [BoxShadow(color: Colors.black38, blurRadius: 8, offset: Offset(0, 2))],
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(widget.icon, color: Colors.white, size: 20),
            const SizedBox(width: 8),
            Text(widget.message, style: const TextStyle(color: Colors.white, fontWeight: FontWeight.w600)),
          ],
        ),
      ),
    );
  }
}

// =============================================================================
// CONFIRMATION DIALOG
// =============================================================================

/// Show a confirmation dialog with warning styling
///
/// Returns true if user confirms, false if cancelled
///
/// Example:
/// ```dart
/// final confirmed = await showG20ConfirmDialog(
///   context,
///   title: 'Delete Item?',
///   message: 'This cannot be undone.',
///   confirmText: 'Delete',
///   confirmColor: Colors.red,
/// );
/// ```
Future<bool> showG20ConfirmDialog(
  BuildContext context, {
  required String title,
  required String message,
  String confirmText = 'Confirm',
  String cancelText = 'Cancel',
  Color? confirmColor,
  IconData icon = Icons.warning_amber,
  Color iconColor = G20Colors.warning,
}) async {
  final result = await showDialog<bool>(
    context: context,
    builder: (ctx) => AlertDialog(
      backgroundColor: G20Colors.surfaceDark,
      title: Row(
        children: [
          Icon(icon, color: iconColor, size: 24),
          const SizedBox(width: 8),
          Expanded(
            child: Text(
              title,
              style: const TextStyle(color: G20Colors.textPrimaryDark, fontSize: 16),
            ),
          ),
        ],
      ),
      content: Text(
        message,
        style: const TextStyle(color: G20Colors.textSecondaryDark),
      ),
      actions: [
        TextButton(
          onPressed: () => Navigator.pop(ctx, false),
          child: Text(cancelText, style: const TextStyle(color: G20Colors.textSecondaryDark)),
        ),
        ElevatedButton(
          onPressed: () => Navigator.pop(ctx, true),
          style: ElevatedButton.styleFrom(
            backgroundColor: confirmColor ?? G20Colors.primary,
          ),
          child: Text(confirmText),
        ),
      ],
    ),
  );
  return result == true;
}

// =============================================================================
// ERROR DIALOG
// =============================================================================

/// Show an error dialog with red styling
///
/// Example:
/// ```dart
/// showG20ErrorDialog(
///   context,
///   title: 'Connection Failed',
///   message: 'Could not connect to backend.',
/// );
/// ```
Future<void> showG20ErrorDialog(
  BuildContext context, {
  required String title,
  required String message,
  List<String>? details,
  String dismissText = 'OK',
}) async {
  await showDialog(
    context: context,
    builder: (ctx) => AlertDialog(
      backgroundColor: G20Colors.surfaceDark,
      title: Row(
        children: [
          const Icon(Icons.error_outline, color: G20Colors.error, size: 24),
          const SizedBox(width: 8),
          Expanded(
            child: Text(
              title,
              style: const TextStyle(color: G20Colors.textPrimaryDark, fontSize: 16),
            ),
          ),
        ],
      ),
      content: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            message,
            style: const TextStyle(color: G20Colors.textSecondaryDark),
          ),
          if (details != null && details.isNotEmpty) ...[
            const SizedBox(height: 12),
            ...details.map((d) => Padding(
              padding: const EdgeInsets.only(left: 8, top: 4),
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text('• ', style: TextStyle(color: G20Colors.textSecondaryDark, fontSize: 12)),
                  Expanded(
                    child: Text(
                      d,
                      style: const TextStyle(color: G20Colors.textSecondaryDark, fontSize: 12),
                    ),
                  ),
                ],
              ),
            )),
          ],
        ],
      ),
      actions: [
        ElevatedButton(
          onPressed: () => Navigator.pop(ctx),
          style: ElevatedButton.styleFrom(backgroundColor: G20Colors.primary),
          child: Text(dismissText),
        ),
      ],
    ),
  );
}

// =============================================================================
// GENERIC SEGMENTED OPTION BUTTON
// =============================================================================

/// A single option button for segmented selectors
///
/// Used for FFT size, FPS, threshold, etc. selectors
///
/// Example:
/// ```dart
/// SegmentedOption<int>(
///   label: '30',
///   sublabel: 'fps',
///   value: 30,
///   selected: currentFps == 30,
///   onTap: () => setFps(30),
/// )
/// ```
class SegmentedOption<T> extends StatelessWidget {
  final String label;
  final String? sublabel;
  final T value;
  final bool selected;
  final VoidCallback onTap;
  final Color? activeColor;

  const SegmentedOption({
    super.key,
    required this.label,
    this.sublabel,
    required this.value,
    required this.selected,
    required this.onTap,
    this.activeColor,
  });

  @override
  Widget build(BuildContext context) {
    final color = activeColor ?? G20Colors.primary;

    return Expanded(
      child: GestureDetector(
        onTap: onTap,
        child: Container(
          padding: const EdgeInsets.symmetric(vertical: 10),
          decoration: BoxDecoration(
            color: selected ? color.withOpacity(0.2) : G20Colors.cardDark,
            borderRadius: BorderRadius.circular(6),
            border: Border.all(
              color: selected ? color : G20Colors.cardDark,
              width: selected ? 2 : 1,
            ),
          ),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Text(
                label,
                style: TextStyle(
                  color: selected ? color : G20Colors.textSecondaryDark,
                  fontWeight: selected ? FontWeight.bold : FontWeight.normal,
                  fontSize: 14,
                ),
              ),
              if (sublabel != null)
                Text(
                  sublabel!,
                  style: TextStyle(
                    color: selected ? color.withOpacity(0.7) : G20Colors.textSecondaryDark,
                    fontSize: 10,
                  ),
                ),
            ],
          ),
        ),
      ),
    );
  }
}

/// A row of segmented options
///
/// Example:
/// ```dart
/// SegmentedSelector<int>(
///   options: [
///     (label: '8K', sublabel: 'Fast', value: 8192),
///     (label: '16K', sublabel: 'Medium', value: 16384),
///     (label: '32K', sublabel: 'Detailed', value: 32768),
///   ],
///   selectedValue: currentFftSize,
///   onChanged: (v) => setFftSize(v),
/// )
/// ```
class SegmentedSelector<T> extends StatelessWidget {
  final List<({String label, String? sublabel, T value})> options;
  final T selectedValue;
  final ValueChanged<T> onChanged;
  final Color? activeColor;
  final double spacing;

  const SegmentedSelector({
    super.key,
    required this.options,
    required this.selectedValue,
    required this.onChanged,
    this.activeColor,
    this.spacing = 6,
  });

  @override
  Widget build(BuildContext context) {
    return Row(
      children: options.asMap().entries.map((entry) {
        final index = entry.key;
        final opt = entry.value;
        return Expanded(
          child: Padding(
            padding: EdgeInsets.only(right: index < options.length - 1 ? spacing : 0),
            child: SegmentedOption<T>(
              label: opt.label,
              sublabel: opt.sublabel,
              value: opt.value,
              selected: selectedValue == opt.value,
              onTap: () => onChanged(opt.value),
              activeColor: activeColor,
            ),
          ),
        );
      }).toList(),
    );
  }
}
