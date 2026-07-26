import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

const String apiBaseUrl = 'https://linear-regression-model-4z71.onrender.com';

void main() {
  runApp(const OrangeEconomyApp());
}

class OrangeEconomyApp extends StatelessWidget {
  const OrangeEconomyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Streaming Success Predictor',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xFFFF8A00)),
        useMaterial3: true,
        inputDecorationTheme: const InputDecorationTheme(
          border: OutlineInputBorder(),
          isDense: true,
        ),
      ),
      home: const PredictionPage(),
    );
  }
}

class FieldSpec {
  final String key;
  final String label;
  final String hint;
  final String helperText;
  final String? sectionHeader;
  final TextEditingController controller;
  final String? Function(String?) validator;

  FieldSpec({
    required this.key,
    required this.label,
    required this.hint,
    required this.helperText,
    required this.validator,
    this.sectionHeader,
  }) : controller = TextEditingController();
}

class PredictionPage extends StatefulWidget {
  const PredictionPage({super.key});

  @override
  State<PredictionPage> createState() => _PredictionPageState();
}

class _PredictionPageState extends State<PredictionPage> {
  final _formKey = GlobalKey<FormState>();
  bool _isLoading = false;
  String? _resultText;
  String? _errorText;

  static String? _numberValidator(
    String? v, {
    required num min,
    required num max,
    bool isInt = false,
  }) {
    if (v == null || v.trim().isEmpty) return 'Required';
    final parsed = isInt ? int.tryParse(v.trim()) : double.tryParse(v.trim());
    if (parsed == null) {
      return isInt ? 'Enter a whole number' : 'Enter a number';
    }
    if (parsed < min || parsed > max) return 'Must be between $min and $max';
    return null;
  }

  static String? _boolValidator(String? v) {
    if (v == null || v.trim().isEmpty) return 'Required';
    final lower = v.trim().toLowerCase();
    if (lower != 'true' && lower != 'false') return 'Enter true or false';
    return null;
  }

  static String? _albumTypeValidator(String? v) {
    if (v == null || v.trim().isEmpty) return 'Required';
    final lower = v.trim().toLowerCase();
    if (!['album', 'single', 'compilation'].contains(lower)) {
      return 'Must be album, single, or compilation';
    }
    return null;
  }

  late final List<FieldSpec> _fields = [
    FieldSpec(
      key: 'Danceability',
      label: 'Danceability',
      hint: '0.0 - 1.0',
      helperText:
          'How suitable for dancing. Look this up on a free Spotify '
          'audio-features tool using your track\'s Spotify link.',
      sectionHeader: '🎧 Audio characteristics (from Spotify)',
      validator: (v) => _numberValidator(v, min: 0, max: 1),
    ),
    FieldSpec(
      key: 'Energy',
      label: 'Energy',
      hint: '0.0 - 1.0',
      helperText: 'Intensity and activity level of the track.',
      validator: (v) => _numberValidator(v, min: 0, max: 1),
    ),
    FieldSpec(
      key: 'Loudness',
      label: 'Loudness (dB)',
      hint: '-60.0 - 5.0',
      helperText: 'Overall loudness in decibels (quiet ≈ -60, very loud ≈ 5).',
      validator: (v) => _numberValidator(v, min: -60, max: 5),
    ),
    FieldSpec(
      key: 'Speechiness',
      label: 'Speechiness',
      hint: '0.0 - 1.0',
      helperText: 'How much spoken word vs music (higher = more rap/spoken).',
      validator: (v) => _numberValidator(v, min: 0, max: 1),
    ),
    FieldSpec(
      key: 'Acousticness',
      label: 'Acousticness',
      hint: '0.0 - 1.0',
      helperText: 'Confidence the track is acoustic (1 = fully acoustic).',
      validator: (v) => _numberValidator(v, min: 0, max: 1),
    ),
    FieldSpec(
      key: 'Instrumentalness',
      label: 'Instrumentalness',
      hint: '0.0 - 1.0',
      helperText: 'Likelihood the track has no vocals.',
      validator: (v) => _numberValidator(v, min: 0, max: 1),
    ),
    FieldSpec(
      key: 'Liveness',
      label: 'Liveness',
      hint: '0.0 - 1.0',
      helperText:
          'Likelihood the track was recorded live in front of an audience.',
      validator: (v) => _numberValidator(v, min: 0, max: 1),
    ),
    FieldSpec(
      key: 'Valence',
      label: 'Valence',
      hint: '0.0 - 1.0',
      helperText: 'Musical positivity (0 = sad/negative, 1 = happy/upbeat).',
      validator: (v) => _numberValidator(v, min: 0, max: 1),
    ),
    FieldSpec(
      key: 'Tempo',
      label: 'Tempo (BPM)',
      hint: '0 - 250',
      helperText: 'Speed of the track in beats per minute.',
      validator: (v) => _numberValidator(v, min: 0, max: 250),
    ),
    FieldSpec(
      key: 'Duration_ms',
      label: 'Duration (ms)',
      hint: '1,000 - 6,000,000',
      helperText: 'Track length in milliseconds (e.g. 3 minutes ≈ 180000).',
      validator: (v) =>
          _numberValidator(v, min: 1000, max: 6000000, isInt: true),
    ),
    FieldSpec(
      key: 'Key',
      label: 'Musical Key',
      hint: '0 - 11',
      helperText: 'Pitch class as a number: 0=C, 1=C♯/D♭, 2=D, ... 11=B.',
      validator: (v) => _numberValidator(v, min: 0, max: 11, isInt: true),
    ),
    FieldSpec(
      key: 'Views',
      label: 'YouTube Views',
      hint: '0 - 10,000,000,000',
      helperText:
          "Current views on the video, or a target you're planning "
          'toward before/after release.',
      sectionHeader: '📈 YouTube engagement (real or target numbers)',
      validator: (v) =>
          _numberValidator(v, min: 0, max: 10000000000, isInt: true),
    ),
    FieldSpec(
      key: 'Likes',
      label: 'YouTube Likes',
      hint: '0 - 100,000,000',
      helperText:
          'Current or target likes — the single strongest signal '
          'of streaming success in our data.',
      validator: (v) =>
          _numberValidator(v, min: 0, max: 100000000, isInt: true),
    ),
    FieldSpec(
      key: 'Comments',
      label: 'YouTube Comments',
      hint: '0 - 50,000,000',
      helperText: 'Current or target comments on the video.',
      validator: (v) => _numberValidator(v, min: 0, max: 50000000, isInt: true),
    ),
    FieldSpec(
      key: 'Licensed',
      label: 'Licensed',
      hint: 'true or false',
      helperText:
          'Is this track registered as licensed content on '
          'streaming platforms?',
      sectionHeader: '💿 Release details',
      validator: _boolValidator,
    ),
    FieldSpec(
      key: 'official_video',
      label: 'Official Video',
      hint: 'true or false',
      helperText: 'Does the track have an official music video released?',
      validator: _boolValidator,
    ),
    FieldSpec(
      key: 'Album_type',
      label: 'Album Type',
      hint: 'album, single, or compilation',
      helperText: 'How the track was released.',
      validator: _albumTypeValidator,
    ),
  ];

  @override
  void dispose() {
    for (final f in _fields) {
      f.controller.dispose();
    }
    super.dispose();
  }

  Future<void> _predict() async {
    setState(() {
      _resultText = null;
      _errorText = null;
    });

    if (!_formKey.currentState!.validate()) {
      setState(() => _errorText = 'Please fix the highlighted fields above.');
      return;
    }

    setState(() => _isLoading = true);

    final body = <String, dynamic>{};
    for (final f in _fields) {
      final raw = f.controller.text.trim();
      switch (f.key) {
        case 'Duration_ms':
        case 'Views':
        case 'Likes':
        case 'Comments':
        case 'Key':
          body[f.key] = int.parse(raw);
          break;
        case 'Licensed':
        case 'official_video':
          body[f.key] = raw.toLowerCase() == 'true';
          break;
        case 'Album_type':
          body[f.key] = raw.toLowerCase();
          break;
        default:
          body[f.key] = double.parse(raw);
      }
    }

    try {
      final response = await http
          .post(
            Uri.parse('$apiBaseUrl/predict'),
            headers: {'Content-Type': 'application/json'},
            body: jsonEncode(body),
          )
          .timeout(const Duration(seconds: 60));

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body) as Map<String, dynamic>;
        final streams = (data['predicted_stream'] as num).round();
        setState(() {
          _resultText =
              'Predicted Spotify streams: ${_formatNumber(streams)}\n'
              '${_interpretTier(streams)}';
        });
      } else {
        setState(() {
          _errorText =
              'API error (${response.statusCode}): ${_extractErrorDetail(response.body)}';
        });
      }
    } catch (e) {
      setState(() {
        _errorText = 'Network error: could not reach the API. ($e)';
      });
    } finally {
      setState(() => _isLoading = false);
    }
  }

  String _formatNumber(int n) {
    final s = n.toString();
    final buf = StringBuffer();
    for (int i = 0; i < s.length; i++) {
      if (i > 0 && (s.length - i) % 3 == 0) buf.write(',');
      buf.write(s[i]);
    }
    return buf.toString();
  }

  // Thresholds are the actual 10/25/50/75/90/99th percentiles of Stream
  // across the 20,718-track training dataset — so this places the
  // prediction in context rather than showing a bare, hard-to-judge number.
  String _interpretTier(int streams) {
    if (streams < 6025145) {
      return 'Niche audience — bottom 10% of tracks in our data. YouTube '
          'likes/comments are the strongest lever for growth.';
    } else if (streams < 17674864) {
      return 'Growing local following (10th-25th percentile).';
    } else if (streams < 49682982) {
      return 'Solid regional traction — approaching the median track.';
    } else if (streams < 138358065) {
      return 'Above-average performance — top half of tracks in our data.';
    } else if (streams < 347337798) {
      return 'Strong performance — top 25% of tracks in our data.';
    } else if (streams < 1279737819) {
      return 'Breakout hit — top 10% of tracks in our data.';
    }
    return 'Viral, global-phenomenon territory — top 1% of tracks in our data.';
  }

  String _extractErrorDetail(String responseBody) {
    try {
      final decoded = jsonDecode(responseBody);
      final detail = decoded['detail'];
      if (detail is String) return detail;
      if (detail is List) {
        return detail.map((d) => '${d['loc']?.last}: ${d['msg']}').join('; ');
      }
      return responseBody;
    } catch (_) {
      return responseBody;
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Streaming Success Predictor'),
        centerTitle: true,
      ),
      body: SafeArea(
        child: Form(
          key: _formKey,
          child: SingleChildScrollView(
            padding: const EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                Card(
                  color: Theme.of(context).colorScheme.secondaryContainer,
                  child: const Padding(
                    padding: EdgeInsets.all(12),
                    child: Text(
                      'Orange Economy Ghana\n'
                      'This tool helps Ghanaian musicians and producers gauge a '
                      "track's likely Spotify streaming success from its audio "
                      "style and its YouTube video's engagement. Use real "
                      'numbers once a song is live, or target numbers to plan '
                      'how much promotion a streaming goal will take.',
                      style: TextStyle(fontSize: 13, height: 1.4),
                    ),
                  ),
                ),
                const SizedBox(height: 20),
                for (final field in _fields) ...[
                  if (field.sectionHeader != null) ...[
                    Padding(
                      padding: const EdgeInsets.only(bottom: 8),
                      child: Text(
                        field.sectionHeader!,
                        style: Theme.of(context).textTheme.titleMedium
                            ?.copyWith(fontWeight: FontWeight.bold),
                      ),
                    ),
                  ],
                  TextFormField(
                    controller: field.controller,
                    decoration: InputDecoration(
                      labelText: field.label,
                      hintText: field.hint,
                      helperText: field.helperText,
                      helperMaxLines: 2,
                    ),
                    validator: field.validator,
                    keyboardType: TextInputType.text,
                  ),
                  const SizedBox(height: 16),
                ],
                const SizedBox(height: 8),
                SizedBox(
                  width: double.infinity,
                  child: FilledButton(
                    onPressed: _isLoading ? null : _predict,
                    child: _isLoading
                        ? const SizedBox(
                            height: 20,
                            width: 20,
                            child: CircularProgressIndicator(strokeWidth: 2),
                          )
                        : const Text('Predict'),
                  ),
                ),
                const SizedBox(height: 20),
                if (_resultText != null)
                  Card(
                    color: Colors.green.shade50,
                    child: Padding(
                      padding: const EdgeInsets.all(16),
                      child: Text(
                        _resultText!,
                        style: const TextStyle(
                          fontSize: 16,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ),
                  ),
                if (_errorText != null)
                  Card(
                    color: Colors.red.shade50,
                    child: Padding(
                      padding: const EdgeInsets.all(16),
                      child: Text(
                        _errorText!,
                        style: TextStyle(
                          fontSize: 14,
                          color: Colors.red.shade900,
                        ),
                      ),
                    ),
                  ),
                const SizedBox(height: 24),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
