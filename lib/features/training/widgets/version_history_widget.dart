import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:intl/intl.dart';
import '../../../core/config/theme.dart';
import '../providers/signal_versions_provider.dart';

/// Widget displaying version history for a signal
class VersionHistoryWidget extends ConsumerStatefulWidget {
  final String signalName;
  final VoidCallback? onVersionPromoted;

  const VersionHistoryWidget({
    super.key,
    required this.signalName,
    this.onVersionPromoted,
  });

  @override
  ConsumerState<VersionHistoryWidget> createState() => _VersionHistoryWidgetState();
}

class _VersionHistoryWidgetState extends ConsumerState<VersionHistoryWidget> {
  @override
  void initState() {
    super.initState();
    // Load version history when widget is created
    Future.microtask(() {
      ref.read(signalVersionsProvider.notifier).loadVersionHistory(widget.signalName);
    });
  }

  @override
  Widget build(BuildContext context) {
    final state = ref.watch(signalVersionsProvider);
    final signal = state.signals[widget.signalName];
    final versions = signal?.versions ?? [];

    if (state.isLoading && versions.isEmpty) {
      return const Center(child: CircularProgressIndicator());
    }

    if (versions.isEmpty) {
      return Center(
        child: Text(
          'No versions yet',
          style: TextStyle(color: G20Colors.textSecondaryDark),
        ),
      );
    }

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        // Header
        Padding(
          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
          child: Row(
            children: [
              const Icon(Icons.history, size: 16, color: G20Colors.primary),
              const SizedBox(width: 8),
              Text(
                'Version History',
                style: const TextStyle(
                  color: G20Colors.textPrimaryDark,
                  fontWeight: FontWeight.w600,
                  fontSize: 13,
                ),
              ),
              const Spacer(),
              if (signal != null)
                Text(
                  'Active: v${signal.activeVersion}',
                  style: TextStyle(
                    color: G20Colors.success,
                    fontSize: 11,
                    fontWeight: FontWeight.w500,
                  ),
                ),
            ],
          ),
        ),
        const Divider(color: G20Colors.cardDark, height: 1),

        // Version list
        Expanded(
          child: ListView.builder(
            padding: const EdgeInsets.all(8),
            itemCount: versions.length,
            itemBuilder: (context, index) {
              // Show most recent first
              final version = versions[versions.length - 1 - index];
              return _VersionCard(
                version: version,
                isActive: version.version == signal?.activeVersion,
                onPromote: version.isActive
                    ? null
                    : () => _promoteVersion(version.version),
              );
            },
          ),
        ),

        // Rollback button
        if (signal != null && signal.activeVersion > 1)
          Padding(
            padding: const EdgeInsets.all(8),
            child: OutlinedButton.icon(
              onPressed: () => _rollback(),
              icon: const Icon(Icons.undo, size: 16),
              label: const Text('Rollback to Previous'),
              style: OutlinedButton.styleFrom(
                foregroundColor: G20Colors.warning,
                side: const BorderSide(color: G20Colors.warning),
              ),
            ),
          ),
      ],
    );
  }

  void _promoteVersion(int version) async {
    final confirm = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        backgroundColor: G20Colors.surfaceDark,
        title: const Text('Promote Version?', style: TextStyle(color: G20Colors.textPrimaryDark)),
        content: Text(
          'Make v$version the active version for ${widget.signalName}?',
          style: const TextStyle(color: G20Colors.textSecondaryDark),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx, false),
            child: const Text('Cancel'),
          ),
          ElevatedButton(
            onPressed: () => Navigator.pop(ctx, true),
            style: ElevatedButton.styleFrom(backgroundColor: G20Colors.primary),
            child: const Text('Promote'),
          ),
        ],
      ),
    );

    if (confirm == true) {
      ref.read(signalVersionsProvider.notifier).promoteVersion(widget.signalName, version);
      widget.onVersionPromoted?.call();
    }
  }

  void _rollback() async {
    final confirm = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        backgroundColor: G20Colors.surfaceDark,
        title: const Text('Rollback?', style: TextStyle(color: G20Colors.textPrimaryDark)),
        content: Text(
          'Rollback ${widget.signalName} to the previous version?',
          style: const TextStyle(color: G20Colors.textSecondaryDark),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx, false),
            child: const Text('Cancel'),
          ),
          ElevatedButton(
            onPressed: () => Navigator.pop(ctx, true),
            style: ElevatedButton.styleFrom(backgroundColor: G20Colors.warning),
            child: const Text('Rollback'),
          ),
        ],
      ),
    );

    if (confirm == true) {
      ref.read(signalVersionsProvider.notifier).rollback(widget.signalName);
      widget.onVersionPromoted?.call();
    }
  }
}

/// Card for a single version
class _VersionCard extends StatelessWidget {
  final VersionInfo version;
  final bool isActive;
  final VoidCallback? onPromote;

  const _VersionCard({
    required this.version,
    required this.isActive,
    this.onPromote,
  });

  @override
  Widget build(BuildContext context) {
    final dateFormat = DateFormat('MMM d, HH:mm');

    return Container(
      margin: const EdgeInsets.only(bottom: 6),
      padding: const EdgeInsets.all(10),
      decoration: BoxDecoration(
        color: isActive ? G20Colors.primary.withOpacity(0.1) : G20Colors.backgroundDark,
        borderRadius: BorderRadius.circular(6),
        border: Border.all(
          color: isActive ? G20Colors.primary : G20Colors.cardDark,
          width: isActive ? 2 : 1,
        ),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Header row
          Row(
            children: [
              // Version badge
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
                decoration: BoxDecoration(
                  color: isActive ? G20Colors.primary : G20Colors.cardDark,
                  borderRadius: BorderRadius.circular(4),
                ),
                child: Text(
                  'v${version.version}',
                  style: TextStyle(
                    color: isActive ? Colors.white : G20Colors.textPrimaryDark,
                    fontWeight: FontWeight.bold,
                    fontSize: 12,
                  ),
                ),
              ),
              const SizedBox(width: 8),

              // F1 score
              if (version.f1Score != null)
                Text(
                  'F1: ${(version.f1Score! * 100).toStringAsFixed(1)}%',
                  style: TextStyle(
                    color: _getF1Color(version.f1Score!),
                    fontWeight: FontWeight.w600,
                    fontSize: 12,
                  ),
                ),

              const Spacer(),

              // Active badge or Promote button
              if (isActive)
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
                  decoration: BoxDecoration(
                    color: G20Colors.success.withOpacity(0.2),
                    borderRadius: BorderRadius.circular(4),
                  ),
                  child: const Text(
                    'Active',
                    style: TextStyle(color: G20Colors.success, fontSize: 10, fontWeight: FontWeight.w600),
                  ),
                )
              else if (onPromote != null)
                TextButton(
                  onPressed: onPromote,
                  style: TextButton.styleFrom(
                    padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                    minimumSize: Size.zero,
                  ),
                  child: const Text('Promote', style: TextStyle(fontSize: 11)),
                ),
            ],
          ),
          const SizedBox(height: 6),

          // Info row
          Row(
            children: [
              _InfoChip(Icons.storage, '${version.sampleCount} samples'),
              const SizedBox(width: 12),
              _InfoChip(Icons.access_time, dateFormat.format(version.createdAt)),
            ],
          ),

          // Notes
          if (version.notes != null && version.notes!.isNotEmpty) ...[
            const SizedBox(height: 4),
            Text(
              version.notes!,
              style: const TextStyle(
                color: G20Colors.textSecondaryDark,
                fontSize: 10,
                fontStyle: FontStyle.italic,
              ),
              maxLines: 2,
              overflow: TextOverflow.ellipsis,
            ),
          ],

          // Promotion reason
          if (version.promotionReason != null) ...[
            const SizedBox(height: 4),
            Text(
              '↑ ${version.promotionReason}',
              style: const TextStyle(
                color: G20Colors.success,
                fontSize: 10,
              ),
            ),
          ],
        ],
      ),
    );
  }

  Color _getF1Color(double f1) {
    if (f1 >= 0.9) return G20Colors.success;
    if (f1 >= 0.8) return G20Colors.primary;
    if (f1 >= 0.7) return G20Colors.warning;
    return G20Colors.error;
  }
}

/// Small info chip
class _InfoChip extends StatelessWidget {
  final IconData icon;
  final String text;

  const _InfoChip(this.icon, this.text);

  @override
  Widget build(BuildContext context) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Icon(icon, size: 12, color: G20Colors.textSecondaryDark),
        const SizedBox(width: 4),
        Text(
          text,
          style: const TextStyle(
            color: G20Colors.textSecondaryDark,
            fontSize: 10,
          ),
        ),
      ],
    );
  }
}
