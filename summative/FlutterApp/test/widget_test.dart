import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:streaming_predictor/main.dart';

void main() {
  testWidgets('Prediction page renders all input fields and a Predict button', (
    WidgetTester tester,
  ) async {
    await tester.pumpWidget(const OrangeEconomyApp());

    expect(find.text('Streaming Success Predictor'), findsWidgets);
    expect(find.byType(TextFormField), findsNWidgets(17));
    expect(find.widgetWithText(FilledButton, 'Predict'), findsOneWidget);
  });

  testWidgets(
    'Filling in valid values and tapping Predict shows a result or error '
    'without crashing (flutter test fakes all HTTP responses as 400, so '
    'this exercises the request-building and error-display path; the '
    'success path is verified separately against the live API)',
    (WidgetTester tester) async {
      await tester.pumpWidget(const OrangeEconomyApp());

      final sampleValues = [
        '0.68', // Danceability
        '0.75', // Energy
        '-5.5', // Loudness
        '0.05', // Speechiness
        '0.1', // Acousticness
        '0.0', // Instrumentalness
        '0.12', // Liveness
        '0.55', // Valence
        '120', // Tempo
        '210000', // Duration_ms
        '5', // Key
        '5000000', // Views
        '250000', // Likes
        '8000', // Comments
        'true', // Licensed
        'true', // official_video
        'single', // Album_type
      ];

      final fieldFinder = find.byType(TextFormField);
      expect(fieldFinder, findsNWidgets(sampleValues.length));

      for (var i = 0; i < sampleValues.length; i++) {
        await tester.enterText(fieldFinder.at(i), sampleValues[i]);
      }
      await tester.pump();

      final predictButton = find.widgetWithText(FilledButton, 'Predict');
      await tester.ensureVisible(predictButton);
      await tester.pump();
      await tester.tap(predictButton);
      await tester.pump(); // show loading spinner

      // flutter test's fake HttpClient returns 400 for every request, so this
      // verifies the app surfaces that as a readable error instead of crashing.
      await tester.pumpAndSettle(const Duration(seconds: 2));

      expect(find.textContaining('API error'), findsOneWidget);
    },
  );
}
