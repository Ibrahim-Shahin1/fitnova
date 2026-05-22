import 'package:fl_chart/fl_chart.dart';
import 'package:flutter/material.dart';

/// Line chart of estimated-1RM over time for one lift.
class E1rmChart extends StatelessWidget {
  const E1rmChart({super.key, required this.points});

  final List<({DateTime date, double e1rm})> points;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final cs = theme.colorScheme;

    if (points.length < 2) {
      return SizedBox(
        height: 200,
        child: Center(
          child: Text(
            'Log this lift on 2+ days to see your strength trend.',
            textAlign: TextAlign.center,
            style: theme.textTheme.bodySmall?.copyWith(color: cs.onSurfaceVariant),
          ),
        ),
      );
    }

    final spots = <FlSpot>[
      for (var i = 0; i < points.length; i++) FlSpot(i.toDouble(), points[i].e1rm),
    ];
    final ys = points.map((p) => p.e1rm).toList();
    final minY = ys.reduce((a, b) => a < b ? a : b);
    final maxY = ys.reduce((a, b) => a > b ? a : b);
    final pad = ((maxY - minY) * 0.18).clamp(2.0, 60.0);
    final lo = (minY - pad).clamp(0.0, double.infinity);
    final hi = maxY + pad;
    final yStep = ((hi - lo) / 4).clamp(1.0, double.infinity);
    final xStep = (points.length / 4).ceilToDouble().clamp(1.0, double.infinity);

    return SizedBox(
      height: 200,
      child: LineChart(
        LineChartData(
          minX: 0,
          maxX: (points.length - 1).toDouble(),
          minY: lo,
          maxY: hi,
          gridData: FlGridData(
            show: true,
            drawVerticalLine: false,
            horizontalInterval: yStep,
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
                reservedSize: 38,
                interval: yStep,
                getTitlesWidget: (v, _) => Text(
                  v.toInt().toString(),
                  style: theme.textTheme.labelSmall
                      ?.copyWith(color: cs.onSurfaceVariant),
                ),
              ),
            ),
            bottomTitles: AxisTitles(
              sideTitles: SideTitles(
                showTitles: true,
                reservedSize: 24,
                interval: xStep,
                getTitlesWidget: (v, _) {
                  final i = v.round();
                  if (i < 0 || i >= points.length) return const SizedBox.shrink();
                  final d = points[i].date;
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
          lineBarsData: [
            LineChartBarData(
              spots: spots,
              isCurved: true,
              preventCurveOverShooting: true,
              color: cs.primary,
              barWidth: 3,
              dotData: FlDotData(show: points.length <= 14),
              belowBarData: BarAreaData(
                show: true,
                color: cs.primary.withValues(alpha: 0.12),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
