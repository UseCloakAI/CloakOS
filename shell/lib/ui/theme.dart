import 'package:flutter/material.dart';

const ink = Color(0xFFF4F6FB);
const inkDim = Color(0x99F4F6FB);
const accent = Color(0xFF8AB4FF);

final cloakTheme = ThemeData(
  brightness: Brightness.dark,
  useMaterial3: true,
  fontFamily: '.AppleSystemUIFont',
  colorScheme: ColorScheme.fromSeed(seedColor: accent, brightness: Brightness.dark),
  splashFactory: NoSplash.splashFactory,
  textTheme: const TextTheme(
    bodyMedium: TextStyle(color: ink, fontSize: 13, letterSpacing: -0.1),
  ),
);
