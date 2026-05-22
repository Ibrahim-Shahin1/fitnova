import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../../services/log_service.dart';
import '../../theme/app_spacing.dart';
import '../../widgets/ui/app_button.dart';

/// Bottom sheet to log one set for an exercise (weight × reps, optional RPE +
/// notes). Returns true via Navigator.pop if a set was logged.
class LogSetSheet extends StatefulWidget {
  const LogSetSheet({
    super.key,
    required this.exerciseName,
    this.planExerciseId,
    this.suggestedSetNumber = 1,
  });

  final String exerciseName;
  final String? planExerciseId;
  final int suggestedSetNumber;

  static Future<bool?> show(
    BuildContext context, {
    required String exerciseName,
    String? planExerciseId,
    int suggestedSetNumber = 1,
  }) {
    return showModalBottomSheet<bool>(
      context: context,
      isScrollControlled: true,
      showDragHandle: true,
      builder: (_) => LogSetSheet(
        exerciseName: exerciseName,
        planExerciseId: planExerciseId,
        suggestedSetNumber: suggestedSetNumber,
      ),
    );
  }

  @override
  State<LogSetSheet> createState() => _LogSetSheetState();
}

class _LogSetSheetState extends State<LogSetSheet> {
  late final TextEditingController _set =
      TextEditingController(text: '${widget.suggestedSetNumber}');
  final _weight = TextEditingController();
  final _reps = TextEditingController();
  final _rpe = TextEditingController();
  final _notes = TextEditingController();
  bool _saving = false;
  String? _err;

  @override
  void dispose() {
    _set.dispose();
    _weight.dispose();
    _reps.dispose();
    _rpe.dispose();
    _notes.dispose();
    super.dispose();
  }

  Future<void> _save() async {
    final setNum = int.tryParse(_set.text.trim()) ?? widget.suggestedSetNumber;
    final reps = int.tryParse(_reps.text.trim());
    final weight = double.tryParse(_weight.text.trim().replaceAll(',', '.'));
    final rpe = double.tryParse(_rpe.text.trim().replaceAll(',', '.'));

    if (reps == null && weight == null) {
      setState(() => _err = 'Enter at least reps or weight.');
      return;
    }
    if (rpe != null && (rpe < 1 || rpe > 10)) {
      setState(() => _err = 'RPE must be between 1 and 10.');
      return;
    }

    setState(() {
      _saving = true;
      _err = null;
    });
    try {
      await LogService.createLog(
        exerciseName: widget.exerciseName,
        setNumber: setNum,
        repsCompleted: reps,
        weightKg: weight,
        rpe: rpe,
        notes: _notes.text.trim(),
        planExerciseId: widget.planExerciseId,
      );
      if (mounted) Navigator.of(context).pop(true);
    } catch (e) {
      setState(() {
        _saving = false;
        _err = "Couldn't save — check your connection and try again.";
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final cs = theme.colorScheme;
    return Padding(
      padding: EdgeInsets.fromLTRB(
        AppSpacing.lg,
        0,
        AppSpacing.lg,
        MediaQuery.of(context).viewInsets.bottom + AppSpacing.lg,
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Text('Log a set',
              style:
                  theme.textTheme.titleLarge?.copyWith(fontWeight: FontWeight.w800)),
          const SizedBox(height: 2),
          Text(widget.exerciseName,
              style:
                  theme.textTheme.bodyMedium?.copyWith(color: cs.onSurfaceVariant)),
          const SizedBox(height: AppSpacing.lg),
          Row(
            children: [
              Expanded(
                child: _numField(_weight, 'Weight (kg)', decimal: true),
              ),
              const SizedBox(width: AppSpacing.sm),
              Expanded(child: _numField(_reps, 'Reps')),
            ],
          ),
          const SizedBox(height: AppSpacing.sm),
          Row(
            children: [
              Expanded(child: _numField(_set, 'Set #')),
              const SizedBox(width: AppSpacing.sm),
              Expanded(child: _numField(_rpe, 'RPE (1-10)', decimal: true)),
            ],
          ),
          const SizedBox(height: AppSpacing.sm),
          TextField(
            controller: _notes,
            textCapitalization: TextCapitalization.sentences,
            decoration: const InputDecoration(
              labelText: 'Notes (optional)',
              border: OutlineInputBorder(),
            ),
          ),
          if (_err != null) ...[
            const SizedBox(height: AppSpacing.sm),
            Text(_err!, style: TextStyle(color: cs.error)),
          ],
          const SizedBox(height: AppSpacing.lg),
          AppButton(
            label: _saving ? 'Saving…' : 'Log set',
            icon: Icons.check,
            expand: true,
            onPressed: _saving ? null : _save,
          ),
        ],
      ),
    );
  }

  Widget _numField(TextEditingController c, String label, {bool decimal = false}) {
    return TextField(
      controller: c,
      keyboardType: TextInputType.numberWithOptions(decimal: decimal),
      inputFormatters: [
        FilteringTextInputFormatter.allow(
            RegExp(decimal ? r'[0-9.,]' : r'[0-9]')),
      ],
      decoration: InputDecoration(
        labelText: label,
        border: const OutlineInputBorder(),
      ),
    );
  }
}
