"""RobotLink: owns the robot connection. Discover via wire-pod, connect, watchdog, reconnect, clean shutdown.

Built on our own gRPC client (cloakos.vector); no Anki SDK. Streams (events, camera,
behavior control) run as tasks on the same asyncio loop as the brain.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from ..bus import Bus
from ..vector import VectorClient, discover
from ..world import CAMERA_H, CAMERA_W, Face, World

log = logging.getLogger(__name__)


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
        self.control_gen = 0                  # bumps on every control grant; Body resyncs its motor state
        self._want_control = False
        self._stop = asyncio.Event()

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
            try:
                await self._connect()
                backoff = 1.0
                await self._watch()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                msg = " ".join(str(e).split()) or type(e).__name__
                self.hint = hint_for(msg)
                log.warning("robot link: %s", msg)
            finally:
                await self._disconnect()
            if self._stop.is_set():
                break
            self._set_state("waiting", retry_in=backoff)
            try:
                await asyncio.wait_for(self._stop.wait(), backoff)
            except asyncio.TimeoutError:
                pass
            backoff = min(backoff * 2, float(self.cfg.get("max_backoff_s", 30)))

    async def stop(self) -> None:
        self._stop.set()

    async def _connect(self) -> None:
        self._set_state("connecting")
        bot = await discover(self.full_cfg, cert_dir=self.cfg.get("cert_dir", "data/certs"))
        client = VectorClient(bot)
        self.client = client
        await client.connect(float(self.cfg.get("connect_timeout_s", 10)))
        client.spawn(client.control_loop(), "control")
        client.spawn(client.events(self._on_event), "events")
        client.spawn(client.camera(self._on_frame), "camera")
        try:
            await client.face_detection(True)
        except Exception as e:
            log.warning("face detection: %s", e)
        try:
            self.anim_triggers = await client.anim_triggers()
        except Exception as e:
            log.warning("could not list animation triggers: %s", e)
        self.hint = None
        self._set_state("connected", name=bot.name, serial=bot.serial, host=bot.host)
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
                await self._request_control()
            try:
                await asyncio.wait_for(self._stop.wait(), interval)
            except asyncio.TimeoutError:
                pass

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
        ok = await self.client.request_control(self.cfg.get("control_priority", "default"),
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
        if kind == "robot_observed_face":
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

    def _on_frame(self, img: Any, image_id: int) -> None:
        self.world.frame = img
        self.world.frame_id = image_id
