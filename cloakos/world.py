"""World: the latest snapshot of what the body senses. Filled by connection/, read by everything else."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

CAMERA_W, CAMERA_H = 640, 360


@dataclass
class Face:
    id: int
    cx: float      # center x, 0..1 across the camera image
    cy: float
    size: float    # bbox width, 0..1; bigger = closer
    t: float = field(default_factory=time.time)


@dataclass
class World:
    connected: bool = False
    has_control: bool = False
    # body state
    picked_up: bool = False
    held: bool = False
    cliff: bool = False
    edges: list = field(default_factory=list)   # remembered edges: (x mm, y mm, when), in odometry
    edge_t: float = 0.0              # when an edge was last felt (the reflex backs off, so cliff clears fast)
    touching: bool = False
    button: bool = False
    on_charger: bool = False
    charging: bool = False
    wheels_moving: bool = False
    animating: bool = False
    proximity_mm: float | None = None
    head_angle: float = 0.0          # rad
    lift_mm: float = 0.0
    left_mmps: float = 0.0
    right_mmps: float = 0.0
    accel: tuple[float, float, float] = (0.0, 0.0, 0.0)
    gyro: tuple[float, float, float] = (0.0, 0.0, 0.0)
    pose: tuple[float, float, float] = (0.0, 0.0, 0.0)   # x mm, y mm, heading rad
    battery_level: int | None = None  # 1 low, 2 nominal, 3 full
    battery_volts: float | None = None
    # perception
    face: Face | None = None
    last_stimulus_t: float = field(default_factory=time.time)
    # camera (not serialized)
    frame: Any = None                 # PIL.Image
    frame_id: int = -1

    @property
    def battery_low(self) -> bool:
        return self.battery_level == 1 and not self.charging

    def face_visible(self, within_s: float = 1.0) -> bool:
        return self.face is not None and time.time() - self.face.t < within_s

    def idle_s(self) -> float:
        return time.time() - self.last_stimulus_t

    def stimulate(self) -> None:
        self.last_stimulus_t = time.time()

    def sensors(self) -> dict[str, Any]:
        """JSON-safe sensor dict for the logger and the web UI."""
        d = asdict(self)
        for k in ("frame", "frame_id", "last_stimulus_t", "face"):
            d.pop(k, None)
        d["face"] = asdict(self.face) if self.face_visible(1.0) else None
        d["battery_low"] = self.battery_low
        return d
