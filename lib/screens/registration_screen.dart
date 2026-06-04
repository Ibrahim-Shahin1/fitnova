import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/user_provider.dart';

class RegistrationScreen extends StatefulWidget {
  const RegistrationScreen({super.key});

  @override
  State<RegistrationScreen> createState() => _RegistrationScreenState();
}

class _RegistrationScreenState extends State<RegistrationScreen> {
  final _formKey = GlobalKey<FormState>();
  final _nameCtrl = TextEditingController();
  final _ageCtrl = TextEditingController(text: '25');
  final _heightCtrl = TextEditingController(text: '170');
  final _weightCtrl = TextEditingController(text: '70');
  String _gender = 'Male';
  final List<String> _selectedInjuries = [];
  double _yearsTraining = 1;
  bool _yearsSet = false;
  final List<String> _selectedEquipment = [];

  static const List<String> _equipmentOptions = [
    'Barbell',
    'Dumbbells',
    'Machines',
    'Cables',
    'Bands',
  ];

  static const Map<String, String> _injuryOptions = {
    'lower_back': 'Lower back',
    'knees': 'Knees',
    'shoulders': 'Shoulders',
    'wrists': 'Wrists / elbows',
    'neck': 'Neck',
    'hips': 'Hips',
    'ankles': 'Ankles',
  };

  double get _bmi {
    final h = double.tryParse(_heightCtrl.text) ?? 170;
    final w = double.tryParse(_weightCtrl.text) ?? 70;
    if (h <= 0) return 0;
    return w / ((h / 100) * (h / 100));
  }

  @override
  void dispose() {
    _nameCtrl.dispose();
    _ageCtrl.dispose();
    _heightCtrl.dispose();
    _weightCtrl.dispose();
    super.dispose();
  }

  void _submit() {
    if (!_formKey.currentState!.validate()) return;

    final provider = context.read<UserProvider>();
    provider.setProfile(
          name: _nameCtrl.text.trim(),
          age: int.parse(_ageCtrl.text),
          gender: _gender,
          heightCm: double.parse(_heightCtrl.text),
          weightKg: double.parse(_weightCtrl.text),
        );
    provider.setInjuries(_selectedInjuries);
    if (_yearsSet) {
      provider.yearsTraining = _yearsTraining.round();
    }
    if (_selectedEquipment.isNotEmpty) {
      provider.equipment = _selectedEquipment;
    }

    Navigator.pushReplacementNamed(context, '/mode-select');
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Create Your Profile')),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(24),
        child: Form(
          key: _formKey,
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              TextFormField(
                controller: _nameCtrl,
                decoration: const InputDecoration(
                  labelText: 'Name',
                  prefixIcon: Icon(Icons.person),
                ),
                validator: (v) =>
                    (v == null || v.trim().isEmpty) ? 'Required' : null,
              ),
              const SizedBox(height: 16),
              TextFormField(
                controller: _ageCtrl,
                decoration: const InputDecoration(
                  labelText: 'Age',
                  prefixIcon: Icon(Icons.cake),
                ),
                keyboardType: TextInputType.number,
                validator: (v) {
                  final n = int.tryParse(v ?? '');
                  if (n == null || n < 10 || n > 100) return '10-100';
                  return null;
                },
              ),
              const SizedBox(height: 16),
              Text('Gender', style: Theme.of(context).textTheme.bodyLarge),
              const SizedBox(height: 8),
              SegmentedButton<String>(
                segments: const [
                  ButtonSegment(value: 'Male', label: Text('Male')),
                  ButtonSegment(value: 'Female', label: Text('Female')),
                ],
                selected: {_gender},
                onSelectionChanged: (v) => setState(() => _gender = v.first),
              ),
              const SizedBox(height: 16),
              TextFormField(
                controller: _heightCtrl,
                decoration: const InputDecoration(
                  labelText: 'Height (cm)',
                  prefixIcon: Icon(Icons.height),
                ),
                keyboardType: TextInputType.number,
                onChanged: (_) => setState(() {}),
                validator: (v) {
                  final n = double.tryParse(v ?? '');
                  if (n == null || n < 100 || n > 250) return '100-250 cm';
                  return null;
                },
              ),
              const SizedBox(height: 16),
              TextFormField(
                controller: _weightCtrl,
                decoration: const InputDecoration(
                  labelText: 'Weight (kg)',
                  prefixIcon: Icon(Icons.monitor_weight),
                ),
                keyboardType: TextInputType.number,
                onChanged: (_) => setState(() {}),
                validator: (v) {
                  final n = double.tryParse(v ?? '');
                  if (n == null || n < 30 || n > 300) return '30-300 kg';
                  return null;
                },
              ),
              const SizedBox(height: 24),
              Card(
                child: Padding(
                  padding: const EdgeInsets.all(16),
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      const Icon(Icons.analytics, size: 28),
                      const SizedBox(width: 12),
                      Text(
                        'BMI: ${_bmi.toStringAsFixed(1)}',
                        style: Theme.of(context).textTheme.titleLarge,
                      ),
                    ],
                  ),
                ),
              ),
              const SizedBox(height: 24),
              Text(
                'Any injuries we should know about?',
                style: Theme.of(context).textTheme.titleMedium,
              ),
              const SizedBox(height: 4),
              Text(
                'Your plan will avoid triggering movements for selected areas.',
                style: Theme.of(context).textTheme.bodySmall?.copyWith(
                      color: Theme.of(context).colorScheme.onSurfaceVariant,
                    ),
              ),
              const SizedBox(height: 12),
              Wrap(
                spacing: 8,
                runSpacing: 4,
                children: _injuryOptions.entries.map((entry) {
                  final selected = _selectedInjuries.contains(entry.key);
                  return FilterChip(
                    label: Text(entry.value),
                    selected: selected,
                    onSelected: (bool value) {
                      setState(() {
                        if (value) {
                          _selectedInjuries.add(entry.key);
                        } else {
                          _selectedInjuries.remove(entry.key);
                        }
                      });
                    },
                  );
                }).toList(),
              ),
              const SizedBox(height: 24),
              ExpansionTile(
                title: const Text('Optional details'),
                subtitle: const Text('Improves your plan'),
                tilePadding: EdgeInsets.zero,
                children: [
                  const SizedBox(height: 8),
                  Row(
                    children: [
                      Text(
                        'Years training: ${_yearsTraining.round()}',
                        style: Theme.of(context).textTheme.bodyMedium,
                      ),
                    ],
                  ),
                  Slider(
                    value: _yearsTraining,
                    min: 0,
                    max: 15,
                    divisions: 15,
                    label: '${_yearsTraining.round()} years',
                    onChanged: (v) => setState(() {
                      _yearsTraining = v;
                      _yearsSet = true;
                    }),
                  ),
                  const SizedBox(height: 12),
                  Align(
                    alignment: Alignment.centerLeft,
                    child: Text(
                      'Equipment available:',
                      style: Theme.of(context).textTheme.bodyMedium,
                    ),
                  ),
                  const SizedBox(height: 8),
                  Wrap(
                    spacing: 8,
                    runSpacing: 4,
                    children: _equipmentOptions.map((eq) {
                      final selected = _selectedEquipment.contains(eq.toLowerCase());
                      return FilterChip(
                        label: Text(eq),
                        selected: selected,
                        onSelected: (bool value) {
                          setState(() {
                            if (value) {
                              _selectedEquipment.add(eq.toLowerCase());
                            } else {
                              _selectedEquipment.remove(eq.toLowerCase());
                            }
                          });
                        },
                      );
                    }).toList(),
                  ),
                  const SizedBox(height: 8),
                ],
              ),
              const SizedBox(height: 24),
              FilledButton.icon(
                onPressed: _submit,
                icon: const Icon(Icons.arrow_forward),
                label: const Text('Get Started'),
                style: FilledButton.styleFrom(
                  padding: const EdgeInsets.symmetric(vertical: 16),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
