"""SimClient: VectorClient's interface, backed by the MuJoCo sim instead of gRPC.

RobotLink can't tell the difference, so the whole brain (drives, behaviors, teleop,
recorder, web UI) runs against the sim unchanged. Sensor messages are the real protobufs.
"""
from __future__ import annotations

import asyncio
import io
import math
import time
from typing import Any, Callable

from ..vector.client import STATUS, VectorClient, m
from ..vector.discovery import BotInfo
from .engine import Sim
from .vector_model import LIFT_MM, lift_angle

ANIMS = ["GreetAfterLongTime", "Happy", "GoToSleepGetIn", "Surprise", "Sad", "Curious"]


class SimClient(VectorClient):
    def __init__(self, sim: Sim) -> None:
        super().__init__(BotInfo(serial="sim", host="sim", guid="", cert=b"", name="Vector (sim)"))
        self.sim = sim
        self.screen: bytes | None = None

    # ── lifecycle ──────────────────────────────────────────────

    async def connect(self, timeout: float = 10) -> None:
        await asyncio.to_thread(self.sim.start)

    async def close(self) -> None:
        self.sim.stop_motors()
        await super().close()

    # ── streams ────────────────────────────────────────────────

    async def control_loop(self, priority: str = "default") -> None:
        await asyncio.Event().wait()          # nothing to hold open; control is instant in the sim

    async def request_control(self, priority: str = "default", timeout: float = 10) -> bool:
        if not self.has_control:
            self.has_control = True
            self.control_changed.set()
            self.control_changed = asyncio.Event()
        return True

    async def release_control(self) -> None:
        self.has_control = False
        self.sim.stop_motors()                # "stock Vector" in the sim just sits still

    async def events(self, on_event: Callable[[str, Any], None]) -> None:
        last_face = 0.0
        while True:
            s = self.sim.snapshot
            if s:
                self.state = self._robot_state(s)
                on_event("robot_state", self.state)
                f = s.get("face")
                now = time.monotonic()
                if f and now - last_face > 0.2:
                    last_face = now
                    on_event("robot_observed_face", m.RobotObservedFace(
                        face_id=1, img_rect=m.CladRect(x_top_left=f["x"], y_top_left=f["y"], width=f["w"], height=f["h"])))
            await asyncio.sleep(1 / 30)

    @staticmethod
    def _robot_state(s: dict) -> Any:
        bits = 0
        for flag, key in (("moving", "moving"), ("picked_up", "picked_up"), ("button", "button"),
                          ("falling", "falling"), ("cliff", "cliff")):
            if s.get(key):
                bits |= STATUS[flag]
        if s.get("moving"):
            bits |= STATUS["wheels_moving"]
        x, y, z = s["pos"]
        q = s["quat"]
        prox = s.get("prox_m")
        return m.RobotState(
            pose=m.PoseStruct(x=x * 1000, y=y * 1000, z=z * 1000, q0=q[0], q1=q[1], q2=q[2], q3=q[3]),
            pose_angle_rad=s["yaw"], pose_pitch_rad=s["pitch"],
            left_wheel_speed_mmps=s["left_mmps"], right_wheel_speed_mmps=s["right_mmps"],
            head_angle_rad=s["head_angle"], lift_height_mm=s["lift_mm"],
            accel=m.AccelData(x=s["accel"][0], y=s["accel"][1], z=s["accel"][2]),
            gyro=m.GyroData(x=s["gyro"][0], y=s["gyro"][1], z=s["gyro"][2]),
            status=bits,
            prox_data=m.ProxData(distance_mm=int(prox * 1000) if prox else 0, signal_quality=1.0 if prox else 0.0,
                                 unobstructed=prox is None, found_object=prox is not None),
            touch_data=m.TouchData(raw_touch_value=0, is_being_touched=False),
        )

    async def camera(self, on_frame: Callable[[Any, int], None]) -> None:
        from PIL import Image

        last = -1
        while True:
            fid, data = self.sim.frame_id, self.sim.frame
            if data is not None and fid != last:
                last = fid
                img = Image.open(io.BytesIO(data))
                img.load()
                on_frame(img, fid)
            await asyncio.sleep(1 / 30)

    # ── motors + actions ───────────────────────────────────────

    async def drive_wheels(self, left: float, right: float, accel: float = 0.0) -> None:
        self.sim.set_wheels(left, right, accel)

    async def move_head(self, rad_s: float) -> None:
        self.sim.set_head(None, rad_s)

    async def move_lift(self, rad_s: float) -> None:
        self.sim.set_lift(None, rad_s)

    async def stop_all(self) -> None:
        self.sim.stop_motors()

    async def head_angle(self, deg: float, speed: float = 10.0, duration: float = 0.0) -> None:
        self.sim.set_head(math.radians(deg))
        await self._settle(lambda s: abs(s["head_angle"] - self.sim.head_target) < 0.03)

    async def lift_height(self, frac: float, speed: float = 10.0, duration: float = 0.0) -> None:
        mm = LIFT_MM[0] + max(0.0, min(1.0, frac)) * (LIFT_MM[1] - LIFT_MM[0])
        self.sim.set_lift(lift_angle(mm))
        await self._settle(lambda s: abs(s["lift_mm"] - mm) < 3)

    async def turn(self, deg: float, speed_dps: float = 90.0) -> None:
        start = self.sim.snapshot["yaw"]
        goal = math.radians(deg)
        v = math.radians(abs(speed_dps)) * 0.0235 * 1000 * (1 if deg > 0 else -1)

        def done(s):
            d = (s["yaw"] - start + math.pi) % (2 * math.pi) - math.pi
            return abs(d) >= abs(goal) - math.radians(2)
        self.sim.set_wheels(-v, v, 600)
        try:
            await self._settle(done, timeout=abs(deg) / max(10.0, abs(speed_dps)) * 3 + 1)
        finally:
            self.sim.set_wheels(0, 0)

    async def straight(self, mm: float, speed_mmps: float = 80.0) -> None:
        x0, y0, _ = self.sim.snapshot["pos"]
        v = abs(speed_mmps) * (1 if mm > 0 else -1)
        self.sim.set_wheels(v, v, 400)
        try:
            await self._settle(lambda s: math.hypot(s["pos"][0] - x0, s["pos"][1] - y0) * 1000 >= abs(mm) - 2,
                               timeout=abs(mm) / max(10.0, abs(speed_mmps)) * 3 + 1)
        finally:
            self.sim.set_wheels(0, 0)

    async def _settle(self, ok: Callable[[dict], bool], timeout: float = 3.0) -> None:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if self.sim.snapshot and ok(self.sim.snapshot):
                return
            await asyncio.sleep(0.02)

    # ── animations, screen, speaker, misc ──────────────────────

    async def drive_on_charger(self) -> bool:
        await asyncio.sleep(1.0)                  # the sim has no charger yet
        return False

    async def drive_off_charger(self) -> None:
        await self.straight(-60, 60)

    async def anim_triggers(self) -> list[str]:
        return list(ANIMS)

    async def play_trigger(self, name: str) -> None:
        await asyncio.sleep(1.2)              # no stock animations in the sim; just take the time

    async def show_face(self, rgb565: bytes, duration_ms: int, interrupt: bool = True) -> None:
        self.screen = rgb565

    async def eye_color(self, hue: float, saturation: float) -> None:
        return None

    async def master_volume(self, volume: int) -> None:
        return None

    async def face_detection(self, on: bool = True) -> None:
        return None

    async def battery(self) -> Any:
        return m.BatteryStateResponse(battery_level=2, battery_volts=3.9)

    async def play_pcm(self, pcm: bytes, rate: int = 16000, volume: int = 75) -> None:
        async with self._audio_lock:
            await asyncio.sleep(len(pcm) / 2 / rate)
