"""RobotLink: owns the SDK connection. Connect, watchdog, reconnect with backoff, clean shutdown.

The SDK runs its own event loop on a background thread. AsyncRobot methods return
concurrent futures, which we await via asyncio.wrap_future; SDK callbacks hop back
onto our loop through bus.emit_threadsafe.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import time
from typing import Any

from ..bus import Bus
from ..world import CAMERA_H, CAMERA_W, Face, World

log = logging.getLogger(__name__)


class RobotLink:
    def __init__(self, cfg: dict, bus: Bus, world: World) -> None:
        self.cfg = cfg.get("robot", {}) or {}
        self.bus = bus
        self.world = world
        self.robot: Any = None
        self.state = "disconnected"
        self.anim_triggers: list[str] = []
        self._want_control = False
        self._stop = asyncio.Event()

    @property
    def connected(self) -> bool:
        return self.state == "connected" and self.robot is not None

    @property
    def has_control(self) -> bool:
        try:
            return self.connected and self.robot.conn.control_granted_event.is_set()
        except Exception:
            return False

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
                log.warning("robot link: %s", " ".join(str(e).split()) or type(e).__name__)
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
        import anki_vector  # lazy: the rest of the app (and tests) run without the SDK
        from anki_vector.events import Events

        self._set_state("connecting")
        robot = anki_vector.AsyncRobot(
            serial=self.cfg.get("serial") or None,
            ip=self.cfg.get("ip") or None,
            name=self.cfg.get("name") or None,
            default_logging=False,
            behavior_control_level=None,   # take control per mode, not at connect
            enable_face_detection=True,
            cache_animation_lists=False,
        )
        timeout = float(self.cfg.get("connect_timeout_s", 10))
        await asyncio.wait_for(asyncio.to_thread(robot.connect, timeout), timeout + 5)
        self.robot = robot
        loop = asyncio.get_running_loop()
        self.bus.bind_loop(loop)

        robot.camera.init_camera_feed()
        robot.events.subscribe(self._on_face, Events.robot_observed_face)
        robot.events.subscribe(self._on_wake_word, Events.wake_word)
        try:
            await self.call(robot.anim.load_animation_trigger_list(), timeout=15)
            self.anim_triggers = sorted(robot.anim.anim_trigger_list)
        except Exception as e:
            log.warning("could not load animation triggers: %s", e)

        self._set_state("connected")
        log.info("connected to Vector (%d anim triggers)", len(self.anim_triggers))
        if self._want_control:
            await self._request_control()

    async def _watch(self) -> None:
        """Heartbeat: battery read every few seconds; two misses in a row = dead link."""
        misses = 0
        interval = float(self.cfg.get("heartbeat_s", 3))
        while not self._stop.is_set():
            try:
                b = await self.call(self.robot.get_battery_state(), timeout=5)
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
        robot, self.robot = self.robot, None
        if robot is None:
            return
        self._set_state("disconnecting")
        try:
            try:
                await self.call(robot.motors.stop_all_motors(), timeout=2)
            except Exception:
                pass
            await asyncio.wait_for(asyncio.to_thread(robot.disconnect), 10)
        except Exception as e:
            log.debug("disconnect: %s", e)
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
        elif self.has_control:
            try:
                await asyncio.to_thread(self.robot.conn.release_control, 10.0)
            except Exception as e:
                log.warning("release control: %s", e)

    async def _request_control(self) -> None:
        try:
            await asyncio.to_thread(self.robot.conn.request_control, timeout=float(self.cfg.get("control_timeout_s", 10)))
        except Exception as e:
            log.warning("request control: %s", e)

    # ── calls + sensors ────────────────────────────────────────

    async def call(self, fut: Any, timeout: float | None = None) -> Any:
        """Await an SDK result (AsyncRobot returns concurrent futures)."""
        if isinstance(fut, concurrent.futures.Future):
            fut = asyncio.wrap_future(fut)
        if asyncio.isfuture(fut) or asyncio.iscoroutine(fut):
            return await asyncio.wait_for(fut, timeout)
        return fut

    def poll(self) -> None:
        """Copy the SDK's cached robot state into World. Cheap; called every tick."""
        r, w = self.robot, self.world
        w.has_control = self.has_control
        if r is None:
            return
        try:
            st = r.status
            w.picked_up = st.is_picked_up
            w.held = st.is_being_held
            w.cliff = st.is_cliff_detected
            w.button = st.is_button_pressed
            w.on_charger = st.is_on_charger
            w.charging = st.is_charging
            w.wheels_moving = st.are_wheels_moving
            w.animating = st.is_animating
            touch = r.touch.last_sensor_reading
            w.touching = bool(touch and touch.is_being_touched)
            prox = r.proximity.last_sensor_reading
            w.proximity_mm = None if prox is None or prox.unobstructed else round(prox.distance.distance_mm, 1)
            w.head_angle = round(r.head_angle_rad, 4)
            w.lift_mm = round(r.lift_height_mm, 1)
            w.left_mmps = round(r.left_wheel_speed_mmps, 1)
            w.right_mmps = round(r.right_wheel_speed_mmps, 1)
            a, g, p = r.accel, r.gyro, r.pose
            w.accel = (round(a.x, 1), round(a.y, 1), round(a.z, 1))
            w.gyro = (round(g.x, 3), round(g.y, 3), round(g.z, 3))
            w.pose = (round(p.position.x, 1), round(p.position.y, 1), round(p.rotation.angle_z.radians, 4))
            img = r.camera.latest_image
            if img is not None and r.camera.latest_image_id != w.frame_id:
                w.frame = img.raw_image
                w.frame_id = r.camera.latest_image_id
        except Exception as e:
            log.debug("poll: %s", e)

    # SDK-thread callbacks: never touch World here, hand off to our loop.

    def _on_face(self, _robot: Any, _event_type: Any, event: Any, *_: Any) -> None:
        rect = event.img_rect
        face = Face(
            id=int(event.face_id),
            cx=(rect.x_top_left + rect.width / 2) / CAMERA_W,
            cy=(rect.y_top_left + rect.height / 2) / CAMERA_H,
            size=rect.width / CAMERA_W,
            t=time.time(),
        )
        self.bus.emit_threadsafe("raw.face", face=face)

    def _on_wake_word(self, _robot: Any, _event_type: Any, event: Any, *_: Any) -> None:
        if event.WhichOneof("wake_word_type") == "wake_word_begin":
            self.bus.emit_threadsafe("raw.wake_word")
