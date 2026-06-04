import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../models/benchmark_models.dart';
import '../providers/benchmark_provider.dart';
import '../theme/app_colors.dart';
import '../theme/app_spacing.dart';
import '../widgets/benchmark/benchmark_clip_row.dart';
import '../widgets/ui/app_empty_state.dart';
import '../widgets/ui/app_error_state.dart';
import '../widgets/ui/app_loader.dart';

class BenchmarkBrowseScreen extends StatefulWidget {
  const BenchmarkBrowseScreen({super.key});

  @override
  State<BenchmarkBrowseScreen> createState() => _BenchmarkBrowseScreenState();
}

class _BenchmarkBrowseScreenState extends State<BenchmarkBrowseScreen>
    with SingleTickerProviderStateMixin {
  static const _exercises = ['squat', 'ohp', 'shallow'];

  static const _filterChips = {
    'squat':   ['All', 'Correct', 'Wrong', 'KIE', 'KFE'],
    'ohp':     ['All', 'Correct', 'Wrong', 'ELBOWS', 'KNEES'],
    'shallow': ['All', 'Correct', 'Wrong', 'DEPTH'],
  };

  late final TabController _tabController;

  @override
  void initState() {
    super.initState();
    _tabController = TabController(length: 3, vsync: this);
    _tabController.addListener(_onTabChanged);
    // Load the first tab immediately.
    WidgetsBinding.instance.addPostFrameCallback((_) {
      context.read<BenchmarkProvider>().loadCatalog('squat');
    });
  }

  void _onTabChanged() {
    if (_tabController.indexIsChanging) return;
    final exercise = _exercises[_tabController.index];
    context.read<BenchmarkProvider>().loadCatalog(exercise);
  }

  @override
  void dispose() {
    _tabController
      ..removeListener(_onTabChanged)
      ..dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final cs = theme.colorScheme;
    final ext = theme.extension<AppColors>()!;
    final provider = context.watch<BenchmarkProvider>();

    return Scaffold(
      backgroundColor: cs.surface,
      appBar: AppBar(
        title: const Text('Benchmark Inspector'),
        backgroundColor: cs.surface,
        bottom: TabBar(
          controller: _tabController,
          indicatorColor: cs.primary,
          labelColor: cs.primary,
          unselectedLabelColor: ext.mutedText,
          labelStyle: theme.textTheme.titleMedium?.copyWith(
            fontWeight: FontWeight.w700,
          ),
          tabs: const [
            Tab(text: 'Squat (244)'),
            Tab(text: 'OHP (339)'),
            Tab(text: 'Shallow (540)'),
          ],
        ),
      ),
      body: TabBarView(
        controller: _tabController,
        children: _exercises.map((exercise) {
          return _ExerciseTabBody(
            exercise: exercise,
            provider: provider,
            filterChips: _filterChips[exercise]!,
            ext: ext,
            theme: theme,
            cs: cs,
          );
        }).toList(),
      ),
    );
  }
}

class _ExerciseTabBody extends StatelessWidget {
  final String exercise;
  final BenchmarkProvider provider;
  final List<String> filterChips;
  final AppColors ext;
  final ThemeData theme;
  final ColorScheme cs;

  const _ExerciseTabBody({
    required this.exercise,
    required this.provider,
    required this.filterChips,
    required this.ext,
    required this.theme,
    required this.cs,
  });

  @override
  Widget build(BuildContext context) {
    final state = provider.loadState(exercise);

    if (state == BenchmarkLoadState.loading ||
        state == BenchmarkLoadState.idle) {
      return const Center(
        child: AppLoader.large(label: 'Loading clips…'),
      );
    }

    if (state == BenchmarkLoadState.error) {
      return AppErrorState(
        icon: Icons.cloud_off,
        title: 'Could not load clips',
        message: 'Check the backend is running on port 8000.',
        actionLabel: 'Retry',
        onAction: () => provider.loadCatalog(exercise),
      );
    }

    // Compute filtered list in build — pure client-side, no server round-trip.
    // browseClips collapses Shallow's per-frame crops to one representative per rep.
    final allClips = provider.browseClips(exercise);
    final filtered = _applyFilters(allClips, exercise, provider.activeFilters);

    return Column(
      children: [
        _FilterChipRow(
          chips: filterChips,
          activeFilters: provider.activeFilters,
          ext: ext,
          theme: theme,
          cs: cs,
          onToggle: (chip) => _toggleFilter(context, chip),
        ),
        if (filtered.isEmpty)
          Expanded(
            child: AppEmptyState(
              icon: Icons.filter_list_off,
              title: 'No clips match this filter',
              message: 'Try removing a filter.',
              actionLabel: 'Clear filters',
              onAction: () => provider.clearFilters(),
            ),
          )
        else
          Expanded(
            child: ListView.builder(
              padding: const EdgeInsets.all(AppSpacing.md),
              itemCount: filtered.length,
              itemBuilder: (context, i) {
                final clip = filtered[i];
                final isCorrect = provider.activeFilters.contains('All')
                    ? null
                    : benchmarkClipIsCorrect(exercise, clip);
                return Padding(
                  padding: const EdgeInsets.only(bottom: AppSpacing.sm),
                  child: BenchmarkClipRow(
                    clip: clip,
                    exercise: exercise,
                    isCorrect: isCorrect,
                    onTap: () => _onTapClip(context, exercise, clip),
                  ),
                );
              },
            ),
          ),
      ],
    );
  }

  List<BenchmarkClip> _applyFilters(
    List<BenchmarkClip> clips,
    String exercise,
    Set<String> activeFilters,
  ) {
    if (activeFilters.contains('All')) return clips;
    return clips.where((clip) {
      for (final f in activeFilters) {
        if (f == 'Correct') {
          if (!benchmarkClipIsCorrect(exercise, clip)) return false;
        } else if (f == 'Wrong') {
          if (benchmarkClipIsCorrect(exercise, clip)) return false;
        } else {
          // Per-error filter: show clips that have this error key in their score map.
          if (!clip.score.containsKey(f)) return false;
        }
      }
      return true;
    }).toList();
  }

  void _toggleFilter(BuildContext context, String chip) {
    final provider = context.read<BenchmarkProvider>();
    final current = Set<String>.from(provider.activeFilters);

    if (chip == 'All') {
      provider.setActiveFilters({'All'});
      return;
    }

    current.remove('All');
    if (current.contains(chip)) {
      current.remove(chip);
      if (current.isEmpty) current.add('All');
    } else {
      current.add(chip);
    }
    provider.setActiveFilters(current);
  }

  Future<void> _onTapClip(
    BuildContext context,
    String exercise,
    BenchmarkClip clip,
  ) async {
    final provider = context.read<BenchmarkProvider>();
    if (exercise == 'shallow') {
      provider.selectShallowRep(clip);
      Navigator.of(context).pushNamed('/benchmark/filmstrip');
      return;
    }
    await provider.startAnalysis(exercise, clip);
    if (!context.mounted) return;
    Navigator.of(context).pushNamed('/benchmark/results');
  }
}

class _FilterChipRow extends StatelessWidget {
  final List<String> chips;
  final Set<String> activeFilters;
  final AppColors ext;
  final ThemeData theme;
  final ColorScheme cs;
  final void Function(String) onToggle;

  const _FilterChipRow({
    required this.chips,
    required this.activeFilters,
    required this.ext,
    required this.theme,
    required this.cs,
    required this.onToggle,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      color: ext.surfaceContainer,
      padding: const EdgeInsets.symmetric(
        horizontal: AppSpacing.md,
        vertical: AppSpacing.sm,
      ),
      child: SingleChildScrollView(
        scrollDirection: Axis.horizontal,
        child: Row(
          children: chips.map((chip) {
            final selected = activeFilters.contains(chip);
            final (bgColor, borderColor, labelColor) = _chipColors(chip, selected);
            return Padding(
              padding: const EdgeInsets.only(right: AppSpacing.sm),
              child: FilterChip(
                label: Text(
                  _chipLabel(chip),
                  style: theme.textTheme.labelMedium?.copyWith(
                    color: selected ? labelColor : ext.mutedText,
                    fontWeight:
                        selected ? FontWeight.w700 : FontWeight.w400,
                  ),
                ),
                selected: selected,
                onSelected: (_) => onToggle(chip),
                backgroundColor: ext.surfaceContainer,
                selectedColor: bgColor,
                side: BorderSide(color: selected ? borderColor : ext.outline),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(AppRadius.pill),
                ),
                showCheckmark: false,
                padding: const EdgeInsets.symmetric(
                  horizontal: 4,
                  vertical: AppSpacing.sm,
                ),
              ),
            );
          }).toList(),
        ),
      ),
    );
  }

  String _chipLabel(String v) => switch (v) {
        'All' => 'All',
        'Correct' => 'Model correct',
        'Wrong' => 'Model wrong',
        _ => benchmarkErrorChip(v),
      };

  (Color, Color, Color) _chipColors(String chip, bool selected) {
    if (!selected) {
      return (ext.surfaceContainer, ext.outline, ext.mutedText);
    }
    switch (chip) {
      case 'All':
        return (
          cs.primary.withValues(alpha: 0.15),
          cs.primary,
          cs.primary,
        );
      case 'Correct':
        return (
          ext.success.withValues(alpha: 0.15),
          ext.success,
          ext.success,
        );
      case 'Wrong':
        return (
          ext.error.withValues(alpha: 0.15),
          ext.error,
          ext.error,
        );
      default:
        // Per-error chips use primary color.
        return (
          cs.primary.withValues(alpha: 0.15),
          cs.primary,
          cs.primary,
        );
    }
  }
}
