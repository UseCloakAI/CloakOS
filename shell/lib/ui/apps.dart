import 'package:flutter/material.dart';

import 'theme.dart';

class App {
  const App(this.id, this.name, this.icon, this.tint, this.build, {this.size = const Size(560, 380)});
  final String id, name;
  final IconData icon;
  final Color tint;
  final Size size;
  final WidgetBuilder build;
}

Widget _soon(BuildContext _) => const Center(child: Text('Coming soon', style: TextStyle(color: inkDim)));

final apps = <App>[
  App('cloak', 'Cloak', Icons.blur_on_rounded, accent, _soon),
  App('files', 'Files', Icons.folder_rounded, const Color(0xFF6CC7FF), _files),
  App('notes', 'Notes', Icons.sticky_note_2_rounded, const Color(0xFFFFD466), _notes),
  App('terminal', 'Terminal', Icons.terminal_rounded, const Color(0xFFB8F5C8), _terminal, size: const Size(620, 360)),
  App('browser', 'Browser', Icons.public_rounded, const Color(0xFF9AA8FF), _soon),
  App('vector', 'Vector', Icons.smart_toy_rounded, const Color(0xFFFF9E6C), _soon),
  App('settings', 'Settings', Icons.tune_rounded, const Color(0xFFD0D4DE), _settings, size: const Size(480, 400)),
];

Widget _files(BuildContext _) => GridView.count(
      crossAxisCount: 5,
      padding: const EdgeInsets.all(18),
      children: [
        for (final n in ['Documents', 'Downloads', 'Pictures', 'Music', 'Projects', 'Cloak', 'Desktop'])
          Column(mainAxisSize: MainAxisSize.min, children: [
            const Icon(Icons.folder_rounded, size: 44, color: Color(0xFF6CC7FF)),
            const SizedBox(height: 6),
            Text(n, style: const TextStyle(fontSize: 12)),
          ]),
      ],
    );

Widget _notes(BuildContext _) => const Padding(
      padding: EdgeInsets.all(18),
      child: TextField(
        maxLines: null,
        expands: true,
        style: TextStyle(color: ink, fontSize: 15, height: 1.5),
        decoration: InputDecoration(border: InputBorder.none, hintText: 'Start writing…', hintStyle: TextStyle(color: inkDim)),
      ),
    );

Widget _terminal(BuildContext _) => Container(
      color: const Color(0x66000000),
      padding: const EdgeInsets.all(14),
      alignment: Alignment.topLeft,
      child: const Text('cloak@cloakos ~ % uname -a\nCloakOS 0.1.0 prototype\ncloak@cloakos ~ % █',
          style: TextStyle(fontFamily: 'Menlo', fontSize: 13, color: Color(0xFFB8F5C8), height: 1.5)),
    );

Widget _settings(BuildContext _) => ListView(
      padding: const EdgeInsets.symmetric(vertical: 8),
      children: [
        for (final s in ['Liquid glass', 'Reduce motion', 'Cloak assistant', 'Dark appearance'])
          SwitchListTile(value: s != 'Reduce motion', onChanged: (_) {}, title: Text(s, style: const TextStyle(color: ink, fontSize: 14))),
      ],
    );
