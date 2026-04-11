import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../models/chat_models.dart';
import '../providers/user_provider.dart';
import '../services/api_service.dart';
import '../widgets/chat_bubble.dart';

class ChatScreen extends StatefulWidget {
  const ChatScreen({super.key});

  @override
  State<ChatScreen> createState() => _ChatScreenState();
}

class _ChatScreenState extends State<ChatScreen> {
  final _textCtrl = TextEditingController();
  final _scrollCtrl = ScrollController();
  final List<ChatMessage> _messages = [];
  Map<String, dynamic> _accumulated = {};
  bool _isLoading = false;
  bool _isGenerating = false;
  int _turnCount = 0;

  @override
  void initState() {
    super.initState();
    _fetchGreeting();
  }

  @override
  void dispose() {
    _textCtrl.dispose();
    _scrollCtrl.dispose();
    super.dispose();
  }

  Future<void> _fetchGreeting() async {
    setState(() => _isLoading = true);
    try {
      final user = context.read<UserProvider>();
      final resp = await ApiService.sendChat(
        conversation: [],
        userContext: user.chatUserContext,
      );
      setState(() {
        _messages.add(ChatMessage(role: 'assistant', content: resp.message));
      });
    } catch (e) {
      _showError('Could not connect to server. Is the backend running?');
    } finally {
      setState(() => _isLoading = false);
    }
  }

  Future<void> _sendMessage() async {
    final text = _textCtrl.text.trim();
    if (text.isEmpty || _isLoading) return;

    _textCtrl.clear();
    setState(() {
      _messages.add(ChatMessage(role: 'user', content: text));
      _isLoading = true;
      _turnCount++;
    });
    _scrollToBottom();

    try {
      final user = context.read<UserProvider>();
      final resp = await ApiService.sendChat(
        conversation: _messages,
        userContext: user.chatUserContext,
      );

      // Merge extracted fields
      _accumulated = {..._accumulated, ...resp.extracted};
      user.updateExtracted(_accumulated);

      setState(() {
        _messages.add(ChatMessage(role: 'assistant', content: resp.message));
        _isLoading = false;
      });
      _scrollToBottom();

      if (resp.isReady) {
        await _generatePlan();
      }
    } catch (e) {
      setState(() => _isLoading = false);
      _showError('Something went wrong. Please try again.');
    }
  }

  Future<void> _generatePlan() async {
    setState(() => _isGenerating = true);
    try {
      final user = context.read<UserProvider>();
      final plan = await ApiService.generatePlan(user.generatePlanPayload);
      user.setPlan(plan);
      if (mounted) {
        Navigator.pushReplacementNamed(context, '/plan');
      }
    } catch (e) {
      setState(() => _isGenerating = false);
      _showError('Plan generation failed. Please try again.');
    }
  }

  void _forceProceed() {
    final user = context.read<UserProvider>();
    // Fill defaults for missing fields
    if (user.experienceLevel == null) {
      user.updateExtracted({'experience_level': 1});
    }
    if (user.sessionDurationHours == null) {
      user.updateExtracted({'session_duration_hours': 1.0});
    }
    if (user.workoutFrequency == null) {
      user.updateExtracted({'workout_frequency': 3});
    }
    _generatePlan();
  }

  void _scrollToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scrollCtrl.hasClients) {
        _scrollCtrl.animateTo(
          _scrollCtrl.position.maxScrollExtent,
          duration: const Duration(milliseconds: 300),
          curve: Curves.easeOut,
        );
      }
    });
  }

  void _showError(String msg) {
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(msg), behavior: SnackBarBehavior.floating),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Stack(
      children: [
        Scaffold(
          appBar: AppBar(
            title: const Text('FitNova AI'),
            leading: IconButton(
              icon: const Icon(Icons.arrow_back),
              onPressed: () => Navigator.pop(context),
            ),
          ),
          body: Column(
            children: [
              Expanded(
                child: ListView.builder(
                  controller: _scrollCtrl,
                  padding: const EdgeInsets.all(16),
                  itemCount: _messages.length + (_isLoading ? 1 : 0),
                  itemBuilder: (context, i) {
                    if (i == _messages.length) {
                      return const Align(
                        alignment: Alignment.centerLeft,
                        child: Padding(
                          padding: EdgeInsets.all(8),
                          child: SizedBox(
                            width: 24,
                            height: 24,
                            child: CircularProgressIndicator(strokeWidth: 2),
                          ),
                        ),
                      );
                    }
                    final msg = _messages[i];
                    return ChatBubble(
                      text: msg.content,
                      isUser: msg.role == 'user',
                    );
                  },
                ),
              ),
              if (_turnCount >= 8 && !_isGenerating)
                Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 16),
                  child: TextButton(
                    onPressed: _forceProceed,
                    child: const Text("Let's proceed with what we have"),
                  ),
                ),
              Padding(
                padding: const EdgeInsets.all(12),
                child: Row(
                  children: [
                    Expanded(
                      child: TextField(
                        controller: _textCtrl,
                        decoration: InputDecoration(
                          hintText: 'Type your message...',
                          border: OutlineInputBorder(
                            borderRadius: BorderRadius.circular(24),
                          ),
                          contentPadding: const EdgeInsets.symmetric(
                            horizontal: 20,
                            vertical: 12,
                          ),
                        ),
                        onSubmitted: (_) => _sendMessage(),
                        enabled: !_isLoading && !_isGenerating,
                      ),
                    ),
                    const SizedBox(width: 8),
                    IconButton.filled(
                      onPressed:
                          (!_isLoading && !_isGenerating) ? _sendMessage : null,
                      icon: const Icon(Icons.send),
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
        if (_isGenerating)
          Container(
            color: Colors.black54,
            child: const Center(
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  CircularProgressIndicator(color: Colors.white),
                  SizedBox(height: 24),
                  Text(
                    'Generating your personalized plan...',
                    style: TextStyle(color: Colors.white, fontSize: 18),
                  ),
                ],
              ),
            ),
          ),
      ],
    );
  }
}
