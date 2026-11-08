#!/usr/bin/env bash
# Generates the platform runners (not checked in) and applies our macOS window tweaks.
#   ./bootstrap.sh && flutter run -d macos --release
set -euo pipefail
cd "$(dirname "$0")"
command -v flutter >/dev/null || { echo "Install Flutter first: brew install --cask flutter"; exit 1; }
flutter create --project-name cloakos_shell --org org.usecloak --platforms=macos,linux,android . >/dev/null
rm -rf test
cp macos_overrides/MainFlutterWindow.swift macos/Runner/MainFlutterWindow.swift
# liquid glass needs Impeller
/usr/libexec/PlistBuddy -c "Add :FLTEnableImpeller bool true" macos/Runner/Info.plist 2>/dev/null || true
flutter pub get
echo "Ready: flutter run -d macos --release"
