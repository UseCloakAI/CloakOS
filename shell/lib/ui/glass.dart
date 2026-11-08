import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:liquid_glass_renderer/liquid_glass_renderer.dart';

// Real refraction needs Impeller (macOS, iOS, Android). Linux falls back to FakeGlass.
final bool realGlass = !kIsWeb &&
    (defaultTargetPlatform == TargetPlatform.macOS ||
        defaultTargetPlatform == TargetPlatform.iOS ||
        defaultTargetPlatform == TargetPlatform.android);

const chrome = LiquidGlassSettings(
  thickness: 22,
  blur: 6,
  glassColor: Color(0x22FFFFFF),
  lightIntensity: .7,
  saturation: 1.6,
  refractiveIndex: 1.25,
  chromaticAberration: .02,
);

// Big surfaces (windows) use cheap glass: refraction over large areas costs GPU every frame they move.
const pane = LiquidGlassSettings(blur: 28, glassColor: Color(0x3314161C), saturation: 1.8);

LiquidShape squircle(double r) => LiquidRoundedSuperellipse(borderRadius: r);

/// A standalone piece of chrome glass (top bar, command bar).
class Glass extends StatelessWidget {
  const Glass({super.key, required this.child, this.radius = 20, this.settings = chrome});
  final Widget child;
  final double radius;
  final LiquidGlassSettings settings;

  @override
  Widget build(BuildContext context) => LiquidGlass.withOwnLayer(
        settings: settings,
        fake: !realGlass,
        shape: squircle(radius),
        child: child,
      );
}

/// Window glass: always the cheap path.
class Pane extends StatelessWidget {
  const Pane({super.key, required this.child, this.radius = 18});
  final Widget child;
  final double radius;

  @override
  Widget build(BuildContext context) => FakeGlass(shape: squircle(radius), settings: pane, child: child);
}
