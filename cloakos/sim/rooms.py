"""Procedural worlds for the sim: random rooms (domain randomisation), a tabletop, a person.

build(world, seed) -> Scene: the full MJCF plus what the tester needs to know about it
(where the objects, the person and the table edge are). Same seed, same room.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .vector_model import vector_actuators, vector_mjcf

WORLDS = {
    "room": "Random cluttered room",
    "empty": "Empty floor, walls only",
    "table": "Tabletop with edges (cliffs)",
    "person": "Room with a person sitting on the floor",
    "goal": "Room with a goal marker to reach",
    "balance": "Flat floor, Vector starts tipped up on its back treads",
}

# things Vector can be asked to find later: name -> (shape, size, rgba)
OBJECTS = {
    "red mug": ("cylinder", (0.032, 0.042), (0.78, 0.16, 0.12, 1)),
    "blue ball": ("sphere", (0.028,), (0.15, 0.35, 0.85, 1)),
    "green cube": ("box", (0.022, 0.022, 0.022), (0.2, 0.65, 0.3, 1)),
    "yellow duck": ("sphere", (0.035,), (0.95, 0.8, 0.15, 1)),
    "white box": ("box", (0.05, 0.035, 0.03), (0.92, 0.92, 0.9, 1)),
    "black shoe": ("box", (0.12, 0.045, 0.04), (0.08, 0.08, 0.08, 1)),
}


@dataclass
class Scene:
    world: str
    seed: int
    xml: str
    room: tuple[float, float]                 # half extents (x, y), metres
    spawn: tuple[float, float, float, float]  # x, y, z, yaw
    objects: dict[str, tuple[float, float]] = field(default_factory=dict)   # name -> xy
    person: tuple[float, float] | None = None
    goal: tuple[float, float] | None = None
    floor_z: float = 0.0                      # what Vector stands on (table top for "table")
    start_pitch: float = 0.0                  # radians; >0 tips the nose up (balance)


def _f(*v: float) -> str:
    return " ".join(f"{x:.4g}" for x in v)


def _free_spot(rng, room, taken, r, margin=0.08):
    for _ in range(200):
        x = rng.uniform(-room[0] + r + margin, room[0] - r - margin)
        y = rng.uniform(-room[1] + r + margin, room[1] - r - margin)
        if all(math.hypot(x - tx, y - ty) > r + tr + 0.05 for tx, ty, tr in taken):
            taken.append((x, y, r))
            return x, y
    return None


def build(world: str = "room", seed: int | None = None) -> Scene:
    world = world if world in WORLDS else "room"
    seed = int(seed if seed is not None else np.random.default_rng().integers(1, 1_000_000))
    rng = np.random.default_rng(seed)
    hx, hy = (rng.uniform(0.9, 1.6), rng.uniform(0.9, 1.6)) if world not in ("table", "balance") else (1.0, 1.0)
    floor = rng.uniform(0.25, 0.75, 3)
    wall = rng.uniform(0.45, 0.95, 3)
    taken: list[tuple[float, float, float]] = [(0.0, 0.0, 0.18)]    # keep the spawn clear
    parts: list[str] = []
    scene = Scene(world, seed, "", (hx, hy), (0.0, 0.0, 0.0, float(rng.uniform(-math.pi, math.pi))))

    # walls
    t, h = 0.02, 0.25
    for name, pos, size in (("wall_n", (0, hy + t, h), (hx + 2 * t, t, h)), ("wall_s", (0, -hy - t, h), (hx + 2 * t, t, h)),
                            ("wall_e", (hx + t, 0, h), (t, hy, h)), ("wall_w", (-hx - t, 0, h), (t, hy, h))):
        parts.append(f'<geom name="{name}" type="box" pos="{_f(*pos)}" size="{_f(*size)}" rgba="{_f(*wall, 1)}"/>')

    if world == "table":
        tz = 0.38
        tw, td = rng.uniform(0.35, 0.5), rng.uniform(0.25, 0.4)
        parts.append(f'<geom name="table" type="box" pos="{_f(0, 0, tz - 0.015)}" size="{_f(tw, td, 0.015)}" rgba="0.55 0.4 0.28 1"/>')
        for sx in (-1, 1):
            for sy in (-1, 1):
                parts.append(f'<geom name="leg_{sx}{sy}" type="box" pos="{_f(sx * (tw - 0.03), sy * (td - 0.03), (tz - 0.03) / 2)}" '
                             f'size="{_f(0.02, 0.02, (tz - 0.03) / 2)}" rgba="0.45 0.32 0.22 1"/>')
        scene.floor_z = tz
        scene.spawn = (rng.uniform(-0.1, 0.1), rng.uniform(-0.05, 0.05), tz, scene.spawn[3])
        scene.room = (tw, td)
    elif world == "balance":
        scene.start_pitch = math.radians(62)
    else:
        if world in ("room", "person", "goal"):
            for i in range(int(rng.integers(3, 7))):        # furniture-ish blocks
                sx, sy, sz = rng.uniform(0.06, 0.3), rng.uniform(0.06, 0.3), rng.uniform(0.04, 0.3)
                spot = _free_spot(rng, (hx, hy), taken, math.hypot(sx, sy))
                if spot:
                    c = rng.uniform(0.2, 0.9, 3)
                    parts.append(f'<geom name="block_{i}" type="box" pos="{_f(*spot, sz)}" size="{_f(sx, sy, sz)}" '
                                 f'euler="0 0 {rng.uniform(0, 180):.1f}" rgba="{_f(*c, 1)}"/>')
            for i in range(int(rng.integers(2, 6))):        # chair / table legs
                spot = _free_spot(rng, (hx, hy), taken, 0.03)
                if spot:
                    parts.append(f'<geom name="post_{i}" type="cylinder" pos="{_f(*spot, 0.2)}" size="0.018 0.2" rgba="0.3 0.25 0.2 1"/>')
            names = list(OBJECTS)
            rng.shuffle(names)
            for name in names[: int(rng.integers(2, 5))]:
                shape, size, rgba = OBJECTS[name]
                spot = _free_spot(rng, (hx, hy), taken, max(size) + 0.02)
                if not spot:
                    continue
                z = size[-1] if shape != "sphere" else size[0]
                if shape == "cylinder":
                    z = size[1]
                slug = name.replace(" ", "_")
                parts.append(f'<body name="obj_{slug}" pos="{_f(*spot, z + 0.001)}"><freejoint/>'
                             f'<geom name="obj_{slug}" type="{shape}" size="{_f(*size)}" rgba="{_f(*rgba)}" mass="0.05"/></body>')
                scene.objects[name] = spot
        if world == "person":
            spot = _free_spot(rng, (hx, hy), taken, 0.22, margin=0.15) or (hx * 0.6, 0.0)
            yaw = math.atan2(-spot[1], -spot[0])           # facing the middle of the room
            skin = rng.choice([(0.95, 0.8, 0.68), (0.78, 0.57, 0.42), (0.45, 0.3, 0.22)])
            shirt = rng.uniform(0.15, 0.85, 3)
            parts.append(f"""<body name="person" mocap="true" pos="{_f(*spot, 0)}" euler="0 0 {math.degrees(yaw):.1f}">
          <geom name="person_legs" type="box" pos="0.12 0 0.06" size="0.16 0.17 0.06" rgba="0.2 0.25 0.4 1"/>
          <geom name="person_body" type="box" pos="0 0 0.33" size="0.1 0.17 0.22" rgba="{_f(*shirt, 1)}"/>
          <geom name="person_head" type="sphere" pos="0.01 0 0.66" size="0.095" rgba="{_f(*skin, 1)}"/>
          <geom name="person_eye_l" type="sphere" pos="0.09 0.032 0.68" size="0.012" rgba="0.05 0.05 0.05 1" contype="0" conaffinity="0"/>
          <geom name="person_eye_r" type="sphere" pos="0.09 -0.032 0.68" size="0.012" rgba="0.05 0.05 0.05 1" contype="0" conaffinity="0"/>
          <site name="face" pos="0.095 0 0.66"/>
        </body>""")
            scene.person = spot
        if world == "goal":
            spot = _free_spot(rng, (hx, hy), taken, 0.08, margin=0.15) or (hx * 0.6, 0.0)
            parts.append(f'<geom name="goal" type="cylinder" pos="{_f(*spot, 0.0006)}" size="0.07 0.0005" '
                         f'rgba="0.83 0.3 0.16 0.9" contype="0" conaffinity="0"/>')
            scene.goal = spot

    sx, sy, sz, yaw = scene.spawn
    # balance: tip it back so it rests on its rear tread wheels (pitch about y)
    robot = vector_mjcf((sx, sy, sz + 0.0005), yaw)
    light = rng.uniform(0.55, 1.0)
    scene.xml = f"""<mujoco model="cloakos-sim">
  <compiler angle="degree" autolimits="true"/>
  <option timestep="0.002" integrator="implicitfast" cone="elliptic" impratio="5"/>
  <visual>
    <global offwidth="1280" offheight="720"/>
    <quality shadowsize="2048"/>
    <headlight ambient="{_f(0.25 * light, 0.25 * light, 0.25 * light)}" diffuse="{_f(0.5 * light, 0.5 * light, 0.5 * light)}"/>
  </visual>
  <default>
    <geom contype="1" conaffinity="3" friction="0.9 0.005 0.0001" solref="0.004 1"/>
  </default>
  <asset>
    <texture name="floor" type="2d" builtin="checker" width="256" height="256" rgb1="{_f(*floor)}" rgb2="{_f(*(floor * 0.85))}"/>
    <material name="floor" texture="floor" texrepeat="{_f(hx * 4, hy * 4)}"/>
  </asset>
  <worldbody>
    <light pos="{_f(rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5), 2.5)}" dir="0 0 -1" diffuse="{_f(light, light, light)}" castshadow="true"/>
    <geom name="floor" type="plane" size="{_f(hx + 0.05, hy + 0.05, 0.05)}" material="floor" rgba="{_f(*floor, 1)}"/>
    {chr(10).join('    ' + p for p in parts)}
{robot}
  </worldbody>
{vector_actuators()}
</mujoco>"""
    return scene
