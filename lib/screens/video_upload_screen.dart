import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:provider/provider.dart';
import 'dart:io';

import '../models/exercise_meta.dart';
import '../providers/form_session_provider.dart';
import '../services/api_service.dart';
import '../theme/app_colors.dart';
import '../theme/app_spacing.dart';

class VideoUploadScreen extends StatefulWidget {
  final ExerciseMeta meta;

  const VideoUploadScreen({super.key, required this.meta});

  @override
  State<VideoUploadScreen> createState() => _VideoUploadScreenState();
}

class _VideoUploadScreenState extends State<VideoUploadScreen> {
  String? _videoPath;
  int? _videoSizeBytes;
  bool _uploading = false;

  Future<void> _pickVideo() async {
    final picker = ImagePicker();
    final xfile = await picker.pickVideo(source: ImageSource.gallery);
    if (xfile != null) {
      final file = File(xfile.path);
      final size = await file.length();
      setState(() {
        _videoPath = xfile.path;
        _videoSizeBytes = size;
      });
    }
  }

  Future<void> _analyze() async {
    if (_videoPath == null) return;

    final provider = context.read<FormSessionProvider>();
    provider.reset();
    provider.startSession();
    setState(() => _uploading = true);

    try {
      final report = await ApiService.uploadFormVideo(
        filePath: _videoPath!,
        exerciseName: widget.meta.name,
      );
      provider.setReport(report);
      if (!mounted) return;
      // Single-rep replay: play the clip back with the model's view + exact KIE/KFE.
      Navigator.of(context).pushReplacementNamed(
        '/form-replay',
        arguments: {'videoPath': _videoPath!, 'report': report},
      );
    } catch (e) {
      provider.setError(e.toString());
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Upload failed: $e')),
      );
    } finally {
      if (mounted) setState(() => _uploading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final cs = theme.colorScheme;
    final ext = theme.extension<AppColors>()!;

    return Scaffold(
      appBar: AppBar(
        title: Text('Upload ${widget.meta.displayName}'),
      ),
      body: Center(
        child: Padding(
          padding: const EdgeInsets.all(AppSpacing.lg),
          child: _uploading
              ? _UploadingView(theme: theme)
              : _videoPath == null
                  ? _PickerView(onPick: _pickVideo, theme: theme, cs: cs)
                  : _SelectedView(
                      videoPath: _videoPath!,
                      videoSizeBytes: _videoSizeBytes!,
                      onAnalyze: _analyze,
                      onReset: () {
                        setState(() {
                          _videoPath = null;
                          _videoSizeBytes = null;
                        });
                      },
                      theme: theme,
                      success: ext.success,
                    ),
        ),
      ),
    );
  }
}

class _UploadingView extends StatelessWidget {
  final ThemeData theme;
  const _UploadingView({required this.theme});

  @override
  Widget build(BuildContext context) {
    final cs = theme.colorScheme;
    return Column(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        CircularProgressIndicator(
          valueColor: AlwaysStoppedAnimation<Color>(cs.primary),
        ),
        const SizedBox(height: AppSpacing.lg),
        Text(
          'Uploading and analyzing',
          style: theme.textTheme.bodyLarge,
        ),
        const SizedBox(height: AppSpacing.sm),
        Text(
          'This may take a minute',
          style: theme.textTheme.bodyMedium?.copyWith(
            color: cs.onSurfaceVariant,
          ),
        ),
      ],
    );
  }
}

class _PickerView extends StatelessWidget {
  final VoidCallback onPick;
  final ThemeData theme;
  final ColorScheme cs;
  const _PickerView({required this.onPick, required this.theme, required this.cs});

  @override
  Widget build(BuildContext context) {
    return Column(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        Icon(
          Icons.video_library_outlined,
          size: 80,
          color: cs.onSurfaceVariant,
        ),
        const SizedBox(height: AppSpacing.lg),
        Text(
          'Pick a video from your gallery',
          textAlign: TextAlign.center,
          style: theme.textTheme.bodyLarge,
        ),
        const SizedBox(height: AppSpacing.xl),
        FilledButton.icon(
          icon: const Icon(Icons.add),
          label: const Text('Select Video'),
          onPressed: onPick,
        ),
      ],
    );
  }
}

class _SelectedView extends StatelessWidget {
  final String videoPath;
  final int videoSizeBytes;
  final VoidCallback onAnalyze;
  final VoidCallback onReset;
  final ThemeData theme;
  final Color success;

  const _SelectedView({
    required this.videoPath,
    required this.videoSizeBytes,
    required this.onAnalyze,
    required this.onReset,
    required this.theme,
    required this.success,
  });

  @override
  Widget build(BuildContext context) {
    final cs = theme.colorScheme;
    return Column(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        Icon(Icons.check_circle, size: 64, color: success),
        const SizedBox(height: AppSpacing.md),
        Text(
          videoPath.split(Platform.pathSeparator).last,
          textAlign: TextAlign.center,
          style: theme.textTheme.bodyLarge,
        ),
        const SizedBox(height: AppSpacing.sm),
        Text(
          '${(videoSizeBytes / (1024 * 1024)).toStringAsFixed(1)} MB',
          style: theme.textTheme.bodySmall?.copyWith(
            color: cs.onSurfaceVariant,
          ),
        ),
        const SizedBox(height: AppSpacing.xl),
        FilledButton.icon(
          icon: const Icon(Icons.cloud_upload),
          label: const Text('Analyze Form'),
          onPressed: onAnalyze,
        ),
        const SizedBox(height: AppSpacing.sm),
        TextButton(
          onPressed: onReset,
          child: const Text('Choose different video'),
        ),
      ],
    );
  }
}
