import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:provider/provider.dart';
import 'dart:io';
import '../models/exercise_meta.dart';
import '../providers/form_session_provider.dart';
import '../services/api_service.dart';

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
      final summary = await ApiService.uploadFormVideo(
        filePath: _videoPath!,
        exerciseName: widget.meta.name,
      );
      provider.setSummary(summary);
      if (!mounted) return;
      Navigator.of(context).pushReplacementNamed('/form-results');
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
    return Scaffold(
      appBar: AppBar(
        title: Text('Upload ${widget.meta.displayName}'),
        backgroundColor: const Color(0xFF1A1A2E),
        foregroundColor: Colors.white,
      ),
      backgroundColor: const Color(0xFF0F0F1E),
      body: Center(
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: _uploading
              ? Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    const CircularProgressIndicator(
                      valueColor: AlwaysStoppedAnimation<Color>(
                        Color(0xFF6C63FF),
                      ),
                    ),
                    const SizedBox(height: 24),
                    Text(
                      'Uploading and analyzing',
                      style: Theme.of(context).textTheme.bodyLarge?.copyWith(
                            color: Colors.white,
                          ),
                    ),
                    const SizedBox(height: 8),
                    Text(
                      'This may take a minute',
                      style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                            color: Colors.grey[400],
                          ),
                    ),
                  ],
                )
              : _videoPath == null
                  ? Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        Icon(
                          Icons.video_library_outlined,
                          size: 80,
                          color: Colors.grey[600],
                        ),
                        const SizedBox(height: 24),
                        Text(
                          'Pick a video from your gallery',
                          textAlign: TextAlign.center,
                          style: Theme.of(context).textTheme.bodyLarge?.copyWith(
                                color: Colors.white,
                              ),
                        ),
                        const SizedBox(height: 32),
                        ElevatedButton.icon(
                          icon: const Icon(Icons.add),
                          label: const Text('Select Video'),
                          onPressed: _pickVideo,
                          style: ElevatedButton.styleFrom(
                            backgroundColor: const Color(0xFF6C63FF),
                          ),
                        ),
                      ],
                    )
                  : Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        Icon(
                          Icons.check_circle,
                          size: 64,
                          color: Colors.green[400],
                        ),
                        const SizedBox(height: 16),
                        Text(
                          _videoPath!.split('/').last,
                          textAlign: TextAlign.center,
                          style: Theme.of(context).textTheme.bodyLarge?.copyWith(
                                color: Colors.white,
                              ),
                        ),
                        const SizedBox(height: 8),
                        Text(
                          '${(_videoSizeBytes! / (1024 * 1024)).toStringAsFixed(1)} MB',
                          style: Theme.of(context).textTheme.bodySmall?.copyWith(
                                color: Colors.grey[400],
                              ),
                        ),
                        const SizedBox(height: 32),
                        ElevatedButton.icon(
                          icon: const Icon(Icons.cloud_upload),
                          label: const Text('Analyze Form'),
                          onPressed: _analyze,
                          style: ElevatedButton.styleFrom(
                            backgroundColor: const Color(0xFF6C63FF),
                          ),
                        ),
                        const SizedBox(height: 12),
                        TextButton(
                          onPressed: () {
                            setState(() {
                              _videoPath = null;
                              _videoSizeBytes = null;
                            });
                          },
                          child: const Text('Choose different video'),
                        ),
                      ],
                    ),
        ),
      ),
    );
  }
}
