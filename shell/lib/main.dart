import 'package:flutter/material.dart';

import 'ui/desktop.dart';
import 'ui/theme.dart';

void main() => runApp(const CloakOS());

class CloakOS extends StatelessWidget {
  const CloakOS({super.key});

  @override
  Widget build(BuildContext context) => MaterialApp(
        title: 'CloakOS',
        debugShowCheckedModeBanner: false,
        theme: cloakTheme,
        home: const Desktop(),
      );
}
