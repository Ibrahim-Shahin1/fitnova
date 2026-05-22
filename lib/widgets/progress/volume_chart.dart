import 'package:fl_chart/fl_chart.dart';
import 'package:flutter/material.dart';

/// Weekly training-volume bar chart (Σ weight×reps per rolling 7-day bucket).
class VolumeChart extends StatelessWidget {
  const VolumeChart({super.key, required this.weeks});

  /// Oldest → newest buckets.
  final List<({DateTime weekStart, double volume})> weeks;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final cs = theme.colorScheme;

    final maxV = weeks.fold<double>(0, (m, w) => w.volume > m ? w.volume : m);
    if (maxV <= 0) {
      return SizedBox(
        height: 180,
        child: Center(
          child: Text('No volume logged yet.',
              style:
                  theme.textTheme.bodySmall?.copyWith(color: cs.onSurfaceVariant)),
        ),
      );
    }
    final top = maxV * 1.2;

    return SizedBox(
      height: 180,
      child: BarChart(
        BarChartData(
          alignment: BarChartAlignment.spaceAround,
          maxY: top,
          gridData: FlGridData(
            show: true,
            drawVerticalLine: false,
            horizontalInterval: top / 3,
            getDrawingHorizontalLine: (_) =>
                FlLine(color: cs.outlineVariant.withValues(alpha: 0.4), strokeWidth: 1),
          ),
          borderData: FlBorderData(show: false),
          titlesData: FlTitlesData(
            topTitles:
                const AxisTitles(sideTitles: SideTitles(showTitles: false)),
            rightTitles:
                const AxisTitles(sideTitles: SideTitles(showTitles: false)),
            leftTitles: AxisTitles(
              sideTitles: SideTitles(
                showTitles: true,
                reservedSize: 40,
                interval: top / 3,
                getTitlesWidget: (v, _) {
                  if (v <= 0) return const SizedBox.shrink();
                  final k = v / 1000.0;
                  return Text(k >= 1 ? '${k.toStringAsFixed(0)}k' : v.toInt().toString(),
                      style: theme.textTheme.labelSmall
                          ?.copyWith(color: cs.onSurfaceVariant));
                },
              ),
            ),
            bottomTitles: AxisTitles(
              sideTitles: SideTitles(
                showTitles: true,
                reservedSize: 22,
                getTitlesWidget: (v, _) {
                  final i = v.toInt();
                  if (i < 0 || i >= weeks.length) return const SizedBox.shrink();
                  final d = weeks[i].weekStart;
                  return Padding(
                    padding: const EdgeInsets.only(top: 6),
                    child: Text('${d.month}/${d.day}',
                        style: theme.textTheme.labelSmall
                            ?.copyWith(color: cs.onSurfaceVariant)),
                  );
                },
              ),
            ),
          ),
          barGroups: [
            for (var i = 0; i < weeks.length; i++)
              BarChartGroupData(
                x: i,
                barRods: [
                  BarChartRodData(
                    toY: weeks[i].volume,
                    color: cs.primary,
                    width: 14,
                    borderRadius: const BorderRadius.vertical(top: Radius.circular(4)),
                  ),
                ],
              ),
          ],
        ),
      ),
    );
  }
}
