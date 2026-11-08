import 'dart:async';
import 'dart:math';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:liquid_glass_renderer/liquid_glass_renderer.dart';

import 'apps.dart';
import 'glass.dart';
import 'theme.dart';

class Desktop extends StatefulWidget {
  const Desktop({super.key});
  @override
  State<Desktop> createState() => _DesktopState();
}

class _Win {
  _Win(this.app, this.pos);
  final App app;
  Offset pos;
}

class _DesktopState extends State<Desktop> {
  final _wins = <_Win>[];
  bool _command = false;

  void _open(App app) {
    if (app.id == 'cloak') return setState(() => _command = true);
    setState(() {
      final i = _wins.indexWhere((w) => w.app.id == app.id);
      if (i >= 0) {
        _wins.add(_wins.removeAt(i)); // focus = move to top
      } else {
        final n = _wins.length;
        _wins.add(_Win(app, Offset(140.0 + n * 36, 90.0 + n * 30)));
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    return CallbackShortcuts(
      bindings: {
        const SingleActivator(LogicalKeyboardKey.keyK, meta: true): () => setState(() => _command = !_command),
        const SingleActivator(LogicalKeyboardKey.escape): () => setState(() => _command = false),
      },
      child: Focus(
        autofocus: true,
        child: Scaffold(
          backgroundColor: Colors.black,
          body: Stack(children: [
            const Positioned.fill(child: RepaintBoundary(child: CustomPaint(painter: _Wallpaper()))),
            for (final w in _wins)
              Positioned(
                key: ValueKey(w.app.id),
                left: w.pos.dx,
                top: w.pos.dy,
                child: _Window(
                  app: w.app,
                  onFocus: () => _open(w.app),
                  onMove: (d) => setState(() => w.pos += d),
                  onClose: () => setState(() => _wins.remove(w)),
                ),
              ),
            const Positioned(top: 8, left: 80, right: 12, child: _TopBar()),
            Positioned(left: 0, right: 0, bottom: 14, child: Center(child: _Dock(onOpen: _open))),
            if (_command)
              Positioned.fill(
                child: GestureDetector(
                  onTap: () => setState(() => _command = false),
                  child: Container(
                    color: Colors.black26,
                    alignment: const Alignment(0, -0.45),
                    child: GestureDetector(onTap: () {}, child: _CommandBar(onOpen: (a) {
                      setState(() => _command = false);
                      _open(a);
                    })),
                  ),
                ),
              ),
          ]),
        ),
      ),
    );
  }
}

// Painted once; soft aurora blobs give the glass something to bend.
class _Wallpaper extends CustomPainter {
  const _Wallpaper();
  @override
  void paint(Canvas c, Size s) {
    c.drawRect(Offset.zero & s, Paint()..color = const Color(0xFF070912));
    void blob(double x, double y, double r, Color col) {
      final o = Offset(s.width * x, s.height * y);
      final rad = r * max(s.width, s.height);
      c.drawCircle(o, rad, Paint()..shader = RadialGradient(colors: [col, col.withValues(alpha: 0)]).createShader(Rect.fromCircle(center: o, radius: rad)));
    }
    blob(.18, .25, .55, const Color(0xCC3B5BFF));
    blob(.85, .2, .45, const Color(0xAA9B4DFF));
    blob(.7, .9, .55, const Color(0xAA00C2B8));
    blob(.25, .95, .35, const Color(0x88FF5E9C));
  }

  @override
  bool shouldRepaint(_) => false;
}

class _TopBar extends StatelessWidget {
  const _TopBar();
  @override
  Widget build(BuildContext context) => Glass(
        radius: 14,
        child: SizedBox(
          height: 30,
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 14),
            child: Row(children: const [
              Icon(Icons.blur_on_rounded, size: 16, color: ink),
              SizedBox(width: 8),
              Text('Cloak', style: TextStyle(fontWeight: FontWeight.w600)),
              Spacer(),
              _Clock(),
              Spacer(),
              Icon(Icons.wifi_rounded, size: 15, color: inkDim),
              SizedBox(width: 12),
              Icon(Icons.battery_5_bar_rounded, size: 15, color: inkDim),
              SizedBox(width: 12),
              Text('⌘K', style: TextStyle(color: inkDim, fontSize: 12)),
            ]),
          ),
        ),
      );
}

class _Clock extends StatefulWidget {
  const _Clock();
  @override
  State<_Clock> createState() => _ClockState();
}

class _ClockState extends State<_Clock> {
  late final Timer _t;
  @override
  void initState() {
    super.initState();
    _t = Timer.periodic(const Duration(seconds: 15), (_) => setState(() {}));
  }


  @override
  void dispose() {
    _t.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final n = TimeOfDay.now();
    return Text(n.format(context), style: const TextStyle(fontWeight: FontWeight.w500));
  }
}

// Icons are grouped glass, so neighbours melt into one liquid bar.
class _Dock extends StatelessWidget {
  const _Dock({required this.onOpen});
  final void Function(App) onOpen;
  @override
  Widget build(BuildContext context) => LiquidGlassLayer(
        settings: chrome,
        fake: !realGlass,
        child: LiquidGlassBlendGroup(
          blend: 26,
          child: Row(mainAxisSize: MainAxisSize.min, children: [
            for (final a in apps)
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 5),
                child: Tooltip(
                  message: a.name,
                  child: LiquidStretch(
                    child: GestureDetector(
                      onTap: () => onOpen(a),
                      child: LiquidGlass.grouped(
                        shape: squircle(18),
                        child: SizedBox.square(dimension: 54, child: Icon(a.icon, color: a.tint, size: 26)),
                      ),
                    ),
                  ),
                ),
              ),
          ]),
        ),
      );
}

class _Window extends StatelessWidget {
  const _Window({required this.app, required this.onFocus, required this.onMove, required this.onClose});
  final App app;
  final VoidCallback onFocus, onClose;
  final void Function(Offset) onMove;

  @override
  Widget build(BuildContext context) => Listener(
        onPointerDown: (_) => onFocus(),
        child: RepaintBoundary(
          child: Pane(
            child: SizedBox(
              width: app.size.width,
              height: app.size.height,
              child: Column(children: [
                GestureDetector(
                  onPanUpdate: (d) => onMove(d.delta),
                  child: Container(
                    height: 38,
                    color: Colors.transparent,
                    padding: const EdgeInsets.symmetric(horizontal: 14),
                    child: Row(children: [
                      _Dot(color: const Color(0xFFFF5F57), onTap: onClose),
                      const SizedBox(width: 12),
                      Icon(app.icon, size: 15, color: app.tint),
                      const SizedBox(width: 8),
                      Text(app.name, style: const TextStyle(fontWeight: FontWeight.w600)),
                    ]),
                  ),
                ),
                const Divider(height: 1, color: Color(0x1FFFFFFF)),
                Expanded(child: app.build(context)),
              ]),
            ),
          ),
        ),
      );
}

class _Dot extends StatelessWidget {
  const _Dot({required this.color, required this.onTap});
  final Color color;
  final VoidCallback onTap;
  @override
  Widget build(BuildContext context) => GestureDetector(
        onTap: onTap,
        child: Container(width: 12, height: 12, decoration: BoxDecoration(color: color, shape: BoxShape.circle)),
      );
}

class _CommandBar extends StatefulWidget {
  const _CommandBar({required this.onOpen});
  final void Function(App) onOpen;
  @override
  State<_CommandBar> createState() => _CommandBarState();
}

class _CommandBarState extends State<_CommandBar> {
  String _q = '';
  @override
  Widget build(BuildContext context) {
    final hits = apps.where((a) => a.id != 'cloak' && a.name.toLowerCase().contains(_q.toLowerCase())).toList();
    return TweenAnimationBuilder<double>(
      tween: Tween(begin: .94, end: 1),
      duration: const Duration(milliseconds: 220),
      curve: Curves.easeOutBack,
      builder: (_, s, child) => Transform.scale(scale: s, child: child),
      child: Glass(
        radius: 26,
        child: SizedBox(
          width: 620,
          child: Column(mainAxisSize: MainAxisSize.min, children: [
            Padding(
              padding: const EdgeInsets.fromLTRB(20, 14, 20, 10),
              child: Row(children: [
                const Icon(Icons.blur_on_rounded, color: accent),
                const SizedBox(width: 12),
                Expanded(
                  child: TextField(
                    autofocus: true,
                    onChanged: (v) => setState(() => _q = v),
                    onSubmitted: (_) => hits.isNotEmpty ? widget.onOpen(hits.first) : null,
                    style: const TextStyle(fontSize: 20, color: ink),
                    cursorColor: accent,
                    decoration: const InputDecoration(
                      border: InputBorder.none,
                      hintText: 'Ask Cloak or open anything',
                      hintStyle: TextStyle(color: inkDim, fontSize: 20),
                    ),
                  ),
                ),
              ]),
            ),
            for (final a in hits.take(5))
              ListTile(
                dense: true,
                leading: Icon(a.icon, color: a.tint),
                title: Text(a.name, style: const TextStyle(color: ink, fontSize: 14)),
                onTap: () => widget.onOpen(a),
              ),
            if (_q.isNotEmpty)
              ListTile(
                dense: true,
                leading: const Icon(Icons.auto_awesome_rounded, color: accent),
                title: Text('Ask Cloak: "$_q"', style: const TextStyle(color: ink, fontSize: 14)),
              ),
            const SizedBox(height: 8),
          ]),
        ),
      ),
    );
  }
}
