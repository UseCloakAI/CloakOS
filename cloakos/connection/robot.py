"""RobotLink: owns the robot connection. Discover via wire-pod, connect, watchdog, reconnect, clean shutdown.

Built on our own gRPC client (cloakos.vector); no Anki SDK. Streams (events, camera,
behavior control) run as tasks on the same asyncio loop as the brain.
"""
from __future__ import annotations

import asyncio
import logging
import math
import time
from typing import Any

from ..bus import Bus
from ..vector import VectorClient, discover
from ..world import CAMERA_H, CAMERA_W, Face, World

log = logging.getLogger(__name__)


class _Retarget(Exception):
    """Internal: switch between robot and sim."""


def hint_for(error: str) -> str | None:
    """Turn a connect error into something the UI can explain."""
    e = error.lower()
    if "no route to host" in e or "errno 65" in e:
        # macOS: the robot answers ping but this app isn't allowed on the local network
        return "local_network"
    if "can't reach wire-pod" in e:
        return "no_wirepod"
    if "has no bot" in e or "no cert for" in e:
        return "not_paired"
    return None


class RobotLink:
    def __init__(self, cfg: dict, bus: Bus, world: World) -> None:
        self.full_cfg = cfg
        self.cfg = cfg.get("robot", {}) or {}
        self.bus = bus
        self.world = world
        self.client: VectorClient | None = None
        self.state = "disconnected"
        self.anim_triggers: list[str] = []
        self.hint: str | None = None          # a known, fixable reason the last connect failed
        self.volume = max(0, min(100, int(self.cfg.get("volume", 80))))   # 0..100, 0 = Cloak's sounds muted
        self.control_gen = 0
        self._button = False
        self._edge = False
        self.edge_events = 0                  # bumps each time the edge reflex fires
        self.edge_until = 0.0                 # the back-off owns the wheels until then
        self.edge_v = -75.0
        self._drove_t = 0.0
        self.last_edge: dict | None = None
        self.pose: dict | None = None         # latest full pose for the mapper: pos (m), quat, head, origin                  # bumps on every control grant; Body resyncs its motor state
        self._want_control = False
        self._stop = asyncio.Event()
        self._kick = asyncio.Event()          # set_target(): drop the current link and reconnect now
        sim_cfg = cfg.get("sim", {}) or {}
        self.target = "sim" if sim_cfg.get("enabled") else "real"
        self.sim = None                       # cloakos.sim.Sim, created the first time it's used

    @property
    def connected(self) -> bool:
        return self.state == "connected" and self.client is not None

    @property
    def has_control(self) -> bool:
        return self.connected and self.client.has_control

    # ── lifecycle ──────────────────────────────────────────────

    async def run(self) -> None:
        """Connect and stay connected until stop() is called."""
        backoff = 1.0
        while not self._stop.is_set():
            hold = None
            try:
                await self._connect()
                backoff = 1.0
                hold = asyncio.create_task(self._hold_control(), name="hold-control")
                await self._watch()
            except asyncio.CancelledError:
                raise
            except _Retarget:
                backoff = 0.0
            except Exception as e:
                msg = " ".join(str(e).split()) or type(e).__name__
                self.hint = hint_for(msg)
                log.warning("robot link: %s", msg)
            finally:
                if hold is not None:
                    hold.cancel()
                await self._disconnect()
            if self._stop.is_set():
                break
            if backoff:
                self._set_state("waiting", retry_in=backoff)
                await self._nap(backoff)
            backoff = min(max(1.0, backoff * 2), float(self.cfg.get("max_backoff_s", 30)))

    async def stop(self) -> None:
        self._stop.set()
        self._kick.set()

    async def _nap(self, seconds: float) -> None:
        """Sleep, but wake early for stop() or set_target()."""
        try:
            await asyncio.wait_for(self._kick.wait(), seconds)
        except asyncio.TimeoutError:
            pass

    async def set_target(self, target: str) -> None:
        """Switch between the real Vector and the simulator; reconnects right away."""
        target = "sim" if target == "sim" else "real"
        if target != self.target:
            self.target = target
            self._kick.set()

    def get_sim(self):
        if self.sim is None:
            from ..sim.engine import Sim
            self.sim = Sim(self.full_cfg)
        return self.sim

    async def _connect(self) -> None:
        self._kick.clear()
        self._set_state("connecting")
        if self.target == "sim":
            from ..sim.client import SimClient
            client: VectorClient = SimClient(self.get_sim())
            bot = client.bot
        else:
            if self.sim is not None:              # don't burn CPU on physics nobody is watching
                await asyncio.to_thread(self.sim.stop)
            bot = await discover(self.full_cfg, cert_dir=self.cfg.get("cert_dir", "data/certs"))
            client = VectorClient(bot)
        self.client = client
        await client.connect(float(self.cfg.get("connect_timeout_s", 10)))
        client.spawn(client.control_loop(), "control")
        client.spawn(client.events(self._on_event), "events")
        client.spawn(self._camera(client), "camera")
        try:
            await client.face_detection(True)
        except Exception as e:
            log.warning("face detection: %s", e)
        await self._push_volume()
        try:
            self.anim_triggers = await client.anim_triggers()
        except Exception as e:
            log.warning("could not list animation triggers: %s", e)
        self.hint = None
        self._set_state("connected", name=bot.name, serial=bot.serial, host=bot.host, target=self.target)
        log.info("connected to %s (%d anim triggers)", bot.name, len(self.anim_triggers))
        if self._want_control:
            await self._request_control()

    async def _watch(self) -> None:
        """Heartbeat: battery read every few seconds; two misses or a dead stream = reconnect."""
        misses = 0
        interval = float(self.cfg.get("heartbeat_s", 3))
        while not self._stop.is_set():
            dead = [t.get_name() for t in self.client._tasks if t.done()]
            if dead:
                raise ConnectionError(f"stream ended: {', '.join(dead)}")
            try:
                b = await self.client.battery()
                self.world.battery_level = int(b.battery_level)
                self.world.battery_volts = round(float(b.battery_volts), 2)
                misses = 0
            except Exception as e:
                misses += 1
                log.debug("heartbeat miss %d: %s", misses, e)
                if misses >= 2:
                    raise ConnectionError("heartbeat lost") from e
            if self._want_control and not self.has_control:
                log.info("behavior control lost; asking again")
                await self._request_control()
            await self._nap(interval)
            if self._kick.is_set() and not self._stop.is_set():
                raise _Retarget()

    async def _hold_control(self) -> None:
        """The moment stock Vector takes control back, ask for it again (not on the next heartbeat)."""
        while True:
            client = self.client
            if client is None:
                return
            await client.control_changed.wait()
            if self._want_control and not client.has_control and self.client is client:
                await asyncio.sleep(0.05)
                log.info("stock Vector took over; taking control back")
                try:
                    await self._request_control()
                except Exception as e:
                    log.debug("re-request control: %s", e)

    async def _disconnect(self) -> None:
        client, self.client = self.client, None
        if client is None:
            return
        self._set_state("disconnecting")
        try:
            if client.has_control:
                await asyncio.wait_for(client.stop_all(), 2)
                await client.release_control()
                await asyncio.sleep(0.2)          # let the release go out before closing the stream
        except Exception:
            pass
        try:
            await asyncio.wait_for(client.close(), 5)
        except Exception as e:
            log.debug("close: %s", e)
        self._set_state("disconnected")

    def _set_state(self, state: str, **extra: Any) -> None:
        self.state = state
        self.world.connected = state == "connected"
        self.bus.emit("robot.connection", state=state, **extra)

    async def _camera(self, client: VectorClient) -> None:
        """Camera feed that restarts itself: a dropped feed never takes the whole link down."""
        warned = False
        while True:
            try:
                await client.camera(self._on_frame)
                err = "feed ended"
            except asyncio.CancelledError:
                raise
            except Exception as e:
                err = " ".join(str(e).split()) or type(e).__name__
            if not warned:
                log.warning("camera: %s; retrying", err)
                warned = True
            await asyncio.sleep(2)

    # ── volume ─────────────────────────────────────────────────

    async def set_volume(self, volume: int) -> None:
        self.volume = max(0, min(100, int(volume)))
        if self.client is not None:
            await self._push_volume()

    async def _push_volume(self) -> None:
        try:
            await self.client.master_volume(round(self.volume / 25))
        except Exception as e:
            log.debug("master volume: %s", e)

    # ── control ────────────────────────────────────────────────

    async def set_control(self, want: bool) -> None:
        """Hold behavior control (our brain drives) or release it (stock Vector runs)."""
        self._want_control = want
        if not self.connected:
            return
        if want:
            await self._request_control()
        else:
            await self.client.release_control()

    async def _request_control(self) -> None:
        ok = await self.client.request_control(self.cfg.get("control_priority", "override"),
                                               float(self.cfg.get("control_timeout_s", 10)))
        if not ok:
            log.warning("behavior control not granted yet")
            return
        self.control_gen += 1
        try:                                  # a dropped session can leave the wheels running
            await self.client.stop_all()
        except Exception as e:
            log.debug("stop on control grant: %s", e)

    # ── sensors ────────────────────────────────────────────────

    def poll(self) -> None:
        """Copy the latest streamed robot state into World. Cheap; called every tick."""
        w = self.world
        w.has_control = self.has_control
        c = self.client
        if c is None or c.state is None:
            return
        st = c.state
        w.picked_up = c.status("picked_up")
        w.held = c.status("held")
        w.cliff = c.status("cliff")
        w.button = c.status("button")
        w.on_charger = c.status("on_charger")
        w.charging = c.status("charging")
        w.wheels_moving = c.status("wheels_moving")
        w.animating = c.status("animating")
        w.touching = bool(st.touch_data.is_being_touched)
        prox = st.prox_data
        w.proximity_mm = None if prox.unobstructed or not prox.found_object else float(prox.distance_mm)
        w.head_angle = round(st.head_angle_rad, 4)
        w.lift_mm = round(st.lift_height_mm, 1)
        w.left_mmps = round(st.left_wheel_speed_mmps, 1)
        w.right_mmps = round(st.right_wheel_speed_mmps, 1)
        w.accel = (round(st.accel.x, 1), round(st.accel.y, 1), round(st.accel.z, 1))
        w.gyro = (round(st.gyro.x, 3), round(st.gyro.y, 3), round(st.gyro.z, 3))
        q = st.pose
        w.pose = (round(q.x, 1), round(q.y, 1), round(st.pose_angle_rad, 4))

    def _on_event(self, kind: str, msg: Any) -> None:
        if kind == "robot_state":
            q = msg.pose
            self.pose = {"pos": (q.x / 1000, q.y / 1000, q.z / 1000), "quat": (q.q0, q.q1, q.q2, q.q3),
                         "head": msg.head_angle_rad, "origin": int(q.origin_id), "t": time.time()}
            down = bool(msg.status & 0x10)
            if down and not self._button:
                self.bus.emit("raw.button_press")
            self._button = down
            # edge reflex, right where the sensor data lands (not on the next brain tick): at override
            # priority Vector's own edge stop is off, so every millisecond here is centimetres saved
            # "picked up" also flips when the nose tips over an edge, so it only cancels the edge if
            # the robot wasn't driving just now (really lifted, not tipping)
            driving = abs(msg.left_wheel_speed_mmps) + abs(msg.right_wheel_speed_mmps) > 15
            if driving:
                self._drove_t = time.time()
            lifted = bool(msg.status & 0x8) and time.time() - self._drove_t > 1.0
            edge = bool(msg.status & 0x4000) and not lifted
            self.world.cliff = edge
            if edge:
                self.world.edge_t = time.time()
            if edge and not self._edge and self.has_control and time.time() > self.edge_until:
                self.edge_events += 1
                asyncio.ensure_future(self._edge_reflex())
                self.bus.emit("raw.edge")
            self._edge = edge
        elif kind == "robot_observed_face":
            if msg.pose.origin_id:                 # Vector also knows where the face is in the room
                fp = msg.pose
                self.bus.emit("raw.face_pose", pos=(fp.x / 1000, fp.y / 1000, fp.z / 1000), origin=int(fp.origin_id))
            r = msg.img_rect
            self.bus.emit("raw.face", face=Face(
                id=int(msg.face_id),
                cx=(r.x_top_left + r.width / 2) / CAMERA_W,
                cy=(r.y_top_left + r.height / 2) / CAMERA_H,
                size=r.width / CAMERA_W,
                t=time.time(),
            ))
        elif kind == "wake_word" and msg.WhichOneof("wake_word_type") == "wake_word_begin":
            self.bus.emit("raw.wake_word")

    EDGE_BACK_MMPS = -70.0             # back-off speed (Body reads the sign/size of edge_v)

    async def _edge_reflex(self) -> None:
        """Get clear of an edge, worked out from what it was doing when it hit (Vector only says "edge",
        not which sensor): driving forward = edge in front, so reverse; reversing = edge behind, so roll
        forward; curving = edge on that side. Distance grows with speed. Keeps going until the sensor
        clears (3 tries), then says where the edge was so behaviors turn away and avoid it later.
        While it runs it owns the wheels (Body enforces it) so nothing can cancel it."""
        c, w = self.client, self.world
        if c is None:
            return
        l, r = w.left_mmps or 0.0, w.right_mmps or 0.0
        fwd, curve = (l + r) / 2, (r - l)                  # mm/s; curve > 0 = was turning left
        sign = 1.0 if fwd < -10 else -1.0                  # move away from where it was heading
        speed = 75.0
        self.edge_v = sign * speed
        x, y, yaw = w.pose or (0.0, 0.0, 0.0)
        ahead = 45.0 * (1 if sign < 0 else -1)             # the edge is ~4.5 cm past the sensor it hit
        w.edges.append((x + ahead * math.cos(yaw), y + ahead * math.sin(yaw), time.time()))
        del w.edges[:-200]
        cleared = False
        start = w.pose
        try:
            for attempt in range(3):
                dur = 0.7 + min(0.6, abs(fwd) / 150 * 0.6)   # faster in = further out
                self.edge_until = time.time() + dur
                end = self.edge_until
                while time.time() < end:                    # re-send: nothing gets to cancel it
                    await c.drive_wheels(self.edge_v, self.edge_v, 300)
                    await asyncio.sleep(0.15)
                if not w.cliff:
                    cleared = True
                    break
            await c.drive_wheels(0, 0, 300)
        except Exception as e:
            log.warning("edge reflex: %s", e)
        finally:
            self.edge_until = 0.0
        side = "left" if curve > 25 else "right" if curve < -25 else "ahead"
        self.last_edge = {"t": time.time(), "side": side, "behind": sign > 0, "cleared": cleared}
        self.bus.emit("edge.cleared", **self.last_edge)
        moved = math.dist(start[:2], w.pose[:2]) if start and w.pose else -1
        log.info("edge (%s%s): %s, moved %.0f mm, wheels read %.0f/%.0f mm/s", side, ", behind" if sign > 0 else "",
                 "backed off" if cleared else "couldn't get clear", moved, w.left_mmps or 0, w.right_mmps or 0)

    def _on_frame(self, img: Any, image_id: int) -> None:
        self.world.frame = img
        self.world.frame_id = image_id
