import 'package:flutter/material.dart';

import '../../widgets/ui/app_empty_state.dart';

/// Form Correction tab. Placeholder for now — the next unit wires this to the
/// existing camera / video-upload form-analysis flow (unchanged behaviour).
class FormCorrectionTab extends StatelessWidget {
  const FormCorrectionTab({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Form Correction')),
      body: const AppEmptyState(
        icon: Icons.videocam_outlined,
        title: 'Form correction',
        message: 'Record or upload a lift to get feedback on your form.',
      ),
    );
  }
}
