# CloakOS shell

The native CloakOS UI (Flutter, no web views). One codebase for macOS now, then the Linux distro and Android.
Glass is [liquid_glass_renderer](https://pub.dev/packages/liquid_glass_renderer): real refraction on Impeller
(macOS/Android), cheap fallback on Linux. Windows use the cheap glass so dragging stays smooth.

```bash
brew install --cask flutter
./bootstrap.sh
flutter run -d macos --release
```

⌘K opens the command bar. Click dock icons to open windows; drag by the title bar.
