/// Error Boundary Widget - Catches and displays widget tree errors gracefully
///
/// Features:
/// - Catches errors in child widget tree
/// - Logs errors with stack traces
/// - Shows fallback UI with retry option
/// - Reports errors to logging system
///
/// Usage:
///   ErrorBoundary(
///     module: 'VideoStream',
///     child: VideoStreamWidget(),
///     fallback: (error, retry) => ErrorCard(error: error, onRetry: retry),
///   )
library;

import 'package:flutter/material.dart';
import '../logging/g20_logger.dart';
import '../config/theme.dart';

/// Error boundary widget that catches errors in its child tree
class ErrorBoundary extends StatefulWidget {
  final Widget child;
  final String module;
  final Widget Function(Object error, VoidCallback retry)? fallback;
  final void Function(Object error, StackTrace stack)? onError;

  const ErrorBoundary({
    super.key,
    required this.child,
    required this.module,
    this.fallback,
    this.onError,
  });

  @override
  State<ErrorBoundary> createState() => _ErrorBoundaryState();
}

class _ErrorBoundaryState extends State<ErrorBoundary> {
  Object? _error;
  StackTrace? _stackTrace;

  @override
  void initState() {
    super.initState();
  }

  void _handleError(Object error, StackTrace stack) {
    setState(() {
      _error = error;
      _stackTrace = stack;
    });

    // Log the error
    final log = G20Logger.of(widget.module);
    log.error('Widget error caught', error: error, stackTrace: stack);

    // Call custom error handler
    widget.onError?.call(error, stack);
  }

  void _retry() {
    setState(() {
      _error = null;
      _stackTrace = null;
    });
  }

  @override
  Widget build(BuildContext context) {
    if (_error != null) {
      // Show fallback UI
      if (widget.fallback != null) {
        return widget.fallback!(_error!, _retry);
      }
      return DefaultErrorWidget(
        error: _error!,
        stackTrace: _stackTrace,
        module: widget.module,
        onRetry: _retry,
      );
    }

    // Wrap child with error catcher
    return _ErrorCatcher(
      onError: _handleError,
      child: widget.child,
    );
  }
}

/// Internal widget that catches errors in build phase
class _ErrorCatcher extends StatelessWidget {
  final Widget child;
  final void Function(Object error, StackTrace stack) onError;

  const _ErrorCatcher({required this.child, required this.onError});

  @override
  Widget build(BuildContext context) {
    // Use ErrorWidget.builder for catching build errors
    ErrorWidget.builder = (FlutterErrorDetails details) {
      // Schedule error handling after build
      WidgetsBinding.instance.addPostFrameCallback((_) {
        onError(details.exception, details.stack ?? StackTrace.current);
      });
      return const SizedBox.shrink();
    };

    return child;
  }
}

/// Default error display widget
class DefaultErrorWidget extends StatelessWidget {
  final Object error;
  final StackTrace? stackTrace;
  final String module;
  final VoidCallback onRetry;
  final bool showDetails;

  const DefaultErrorWidget({
    super.key,
    required this.error,
    this.stackTrace,
    required this.module,
    required this.onRetry,
    this.showDetails = false,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: G20Colors.error.withOpacity(0.1),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: G20Colors.error.withOpacity(0.3)),
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Row(
            children: [
              Icon(Icons.error_outline, color: G20Colors.error, size: 24),
              const SizedBox(width: 12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Error in $module',
                      style: const TextStyle(
                        color: G20Colors.error,
                        fontWeight: FontWeight.bold,
                        fontSize: 14,
                      ),
                    ),
                    const SizedBox(height: 4),
                    Text(
                      error.toString(),
                      style: const TextStyle(
                        color: G20Colors.textSecondaryDark,
                        fontSize: 12,
                      ),
                      maxLines: 3,
                      overflow: TextOverflow.ellipsis,
                    ),
                  ],
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          Row(
            mainAxisAlignment: MainAxisAlignment.end,
            children: [
              if (showDetails && stackTrace != null)
                TextButton.icon(
                  onPressed: () => _showStackTrace(context),
                  icon: const Icon(Icons.bug_report, size: 16),
                  label: const Text('Details'),
                  style: TextButton.styleFrom(foregroundColor: G20Colors.textSecondaryDark),
                ),
              const SizedBox(width: 8),
              ElevatedButton.icon(
                onPressed: onRetry,
                icon: const Icon(Icons.refresh, size: 16),
                label: const Text('Retry'),
                style: ElevatedButton.styleFrom(
                  backgroundColor: G20Colors.primary,
                  foregroundColor: Colors.white,
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  void _showStackTrace(BuildContext context) {
    showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        backgroundColor: G20Colors.surfaceDark,
        title: const Text('Error Details', style: TextStyle(color: G20Colors.textPrimaryDark)),
        content: SingleChildScrollView(
          child: Text(
            stackTrace.toString(),
            style: const TextStyle(
              fontFamily: 'monospace',
              fontSize: 11,
              color: G20Colors.textSecondaryDark,
            ),
          ),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx),
            child: const Text('Close'),
          ),
        ],
      ),
    );
  }
}

/// Compact inline error indicator
class ErrorIndicator extends StatelessWidget {
  final String message;
  final VoidCallback? onRetry;

  const ErrorIndicator({
    super.key,
    required this.message,
    this.onRetry,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        color: G20Colors.error.withOpacity(0.1),
        borderRadius: BorderRadius.circular(4),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          const Icon(Icons.error_outline, color: G20Colors.error, size: 16),
          const SizedBox(width: 8),
          Text(
            message,
            style: const TextStyle(color: G20Colors.error, fontSize: 12),
          ),
          if (onRetry != null) ...[
            const SizedBox(width: 8),
            GestureDetector(
              onTap: onRetry,
              child: const Icon(Icons.refresh, color: G20Colors.error, size: 16),
            ),
          ],
        ],
      ),
    );
  }
}

/// Global error handler setup for uncaught errors
void setupGlobalErrorHandling() {
  // Catch Flutter framework errors
  FlutterError.onError = (FlutterErrorDetails details) {
    final log = G20Logger.of('Flutter');
    log.fatal(
      'Uncaught Flutter error',
      error: details.exception,
      stackTrace: details.stack,
      data: {
        'library': details.library,
        'context': details.context?.toString(),
      },
    );

    // Still print to console in debug mode
    FlutterError.presentError(details);
  };
}
