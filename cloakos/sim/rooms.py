"""Procedural worlds for the sim: realistic random rooms (domain randomisation), a tabletop,
a person, a goal, a balance start.

build(world, seed, assets) -> Scene: the full MJCF plus what the tester / RL / map need to
know about it (where the objects, the person and the table edge are). Same seed, same room.

Realism comes from real data where we can get it: photo-scanned household objects (Google
Scanned Objects) and generated wood / carpet / tile / paint / fabric materials, with random
furniture and 1-4 lights of different colour temperatures. Without assets (offline, tests)
it falls back to plain coloured shapes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from .vector_model import vector_actuators, vector_assets, vector_mjcf

if TYPE_CHECKING:
    from .assets import Assets

WORLDS = {
    "room": "Random realistic room",
    "empty": "Empty floor, walls only",
    "table": "Tabletop with edges (cliffs)",
    "person": "Room with a person sitting on the floor",
    "goal": "Room with a goal marker to reach",
    "balance": "Flat floor, Vector starts tipped back (recovery practice)",
}

# plain stand-ins when scanned objects aren't available: name -> (shape, size, rgba)
OBJECTS = {
    "red mug": ("cylinder", (0.032, 0.042), (0.78, 0.16, 0.12, 1)),
    "blue ball": ("sphere", (0.028,), (0.15, 0.35, 0.85, 1)),
    "green cube": ("box", (0.022, 0.022, 0.022), (0.2, 0.65, 0.3, 1)),
    "yellow duck": ("sphere", (0.035,), (0.95, 0.8, 0.15, 1)),
    "white box": ("box", (0.05, 0.035, 0.03), (0.92, 0.92, 0.9, 1)),
    "black shoe": ("box", (0.12, 0.045, 0.04), (0.08, 0.08, 0.08, 1)),
}

KELVIN = {2700: (1.0, 0.76, 0.52), 3500: (1.0, 0.85, 0.68), 4500: (1.0, 0.94, 0.86), 5500: (1.0, 1.0, 0.98),
          6500: (0.92, 0.95, 1.0)}


@dataclass
class Scene:
    world: str
    seed: int
    xml: str
    room: tuple[float, float]                 # half extents (x, y), metres
    spawn: tuple[float, float, float, float]  # x, y, z, yaw
    objects: dict[str, tuple[float, float]] = field(default_factory=dict)   # body name -> xy
    labels: dict[str, str] = field(default_factory=dict)                    # body name -> "mug", "shoe", ...
    person: tuple[float, float] | None = None
    goal: tuple[float, float] | None = None
    floor_z: float = 0.0                      # what Vector stands on (table top for "table")
    start_pitch: float = 0.0                  # radians; >0 tips the nose up (balance)
    obstacles: list[tuple[float, float, float]] = field(default_factory=list)   # x, y, radius (for RL / planners)


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


class _Mats:
    """Collects <texture>/<material> assets for one world."""

    def __init__(self, assets: "Assets | None", rng) -> None:
        self.assets, self.rng = assets, rng
        self.xml: list[str] = []
        self.n = 0

    def make(self, kind: str, rgba=(1, 1, 1, 1), repeat=(1, 1), specular=0.2, shininess=0.2, reflectance=0.0,
             fallback=(0.6, 0.6, 0.6)) -> str:
        self.n += 1
        name = f"m{self.n}"
        if self.assets is not None:
            tex = self.assets.texture(kind, int(self.rng.integers(0, 8)))
            self.xml.append(f'<texture name="t{self.n}" type="2d" file="{tex}"/>')
            self.xml.append(f'<material name="{name}" texture="t{self.n}" texrepeat="{_f(*repeat)}" texuniform="true" '
                            f'rgba="{_f(*rgba)}" specular="{specular}" shininess="{shininess}" reflectance="{reflectance}"/>')
        else:
            c = np.clip(np.array(fallback) * self.rng.uniform(0.8, 1.2, 3), 0, 1)
            self.xml.append(f'<material name="{name}" rgba="{_f(*c, 1)}" specular="{specular}" shininess="{shininess}"/>')
        return name


def _furniture(rng, mats: _Mats, room, taken, parts: list[str], scene: Scene, i: int) -> None:
    """One piece of furniture: sofa, table, shelf, chair, cabinet, or a plain block."""
    kind = rng.choice(["sofa", "table", "shelf", "chair", "cabinet", "box"])
    yaw = float(rng.choice([0, 90, 180, 270]) + rng.uniform(-8, 8))
    if kind == "sofa":
        L, D = rng.uniform(0.6, 1.0), rng.uniform(0.28, 0.36)
        spot = _free_spot(rng, room, taken, math.hypot(L, D) / 1.6)
        if not spot:
            return
        m = mats.make("fabric", repeat=(3, 3), fallback=(0.35, 0.38, 0.45))
        parts.append(f'<body name="sofa_{i}" pos="{_f(*spot, 0)}" euler="0 0 {yaw:.1f}">'
                     f'<geom type="box" pos="0 0 0.09" size="{_f(L / 2, D / 2, 0.09)}" material="{m}"/>'
                     f'<geom type="box" pos="{_f(0, -D / 2 + 0.04, 0.22)}" size="{_f(L / 2, 0.04, 0.13)}" material="{m}"/>'
                     f'<geom type="box" pos="{_f(L / 2 - 0.03, 0, 0.15)}" size="{_f(0.03, D / 2, 0.06)}" material="{m}"/>'
                     f'<geom type="box" pos="{_f(-L / 2 + 0.03, 0, 0.15)}" size="{_f(0.03, D / 2, 0.06)}" material="{m}"/></body>')
    elif kind in ("table", "chair"):
        L, D, H = (rng.uniform(0.4, 0.7), rng.uniform(0.3, 0.45), rng.uniform(0.35, 0.5)) if kind == "table" \
            else (0.2, 0.2, rng.uniform(0.2, 0.25))
        spot = _free_spot(rng, room, taken, 0.06)                   # legs only: Vector can drive underneath
        if not spot:
            return
        m = mats.make("wood", repeat=(2, 2), specular=0.4, shininess=0.5, reflectance=0.05, fallback=(0.5, 0.35, 0.22))
        legs = "".join(f'<geom type="box" pos="{_f(sx * (L / 2 - 0.02), sy * (D / 2 - 0.02), H / 2)}" size="{_f(0.012, 0.012, H / 2)}" material="{m}"/>'
                       for sx in (-1, 1) for sy in (-1, 1))
        parts.append(f'<body name="{kind}_{i}" pos="{_f(*spot, 0)}" euler="0 0 {yaw:.1f}">{legs}'
                     f'<geom type="box" pos="{_f(0, 0, H + 0.012)}" size="{_f(L / 2, D / 2, 0.012)}" material="{m}"/></body>')
        for sx in (-1, 1):
            for sy in (-1, 1):
                c, s_ = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
                lx, ly = sx * (L / 2 - 0.02), sy * (D / 2 - 0.02)
                scene.obstacles.append((spot[0] + c * lx - s_ * ly, spot[1] + s_ * lx + c * ly, 0.02))
        return
    elif kind == "shelf":
        L, D, H = rng.uniform(0.4, 0.8), 0.15, rng.uniform(0.4, 0.8)
        spot = _free_spot(rng, room, taken, math.hypot(L, D) / 1.6)
        if not spot:
            return
        m = mats.make("wood", repeat=(1, 2), fallback=(0.45, 0.3, 0.2))
        books = []
        for k in range(int(rng.integers(6, 14))):
            w = rng.uniform(0.015, 0.03)
            x = -L / 2 + 0.03 + k * (L - 0.06) / 14
            h = rng.uniform(0.08, 0.13)
            c = rng.uniform(0.15, 0.85, 3)
            books.append(f'<geom type="box" pos="{_f(x, 0, 0.03 + h / 2)}" size="{_f(w / 2, D / 2 - 0.01, h / 2)}" rgba="{_f(*c, 1)}"/>')
        parts.append(f'<body name="shelf_{i}" pos="{_f(*spot, 0)}" euler="0 0 {yaw:.1f}">'
                     f'<geom type="box" pos="{_f(0, 0, 0.015)}" size="{_f(L / 2, D / 2, 0.015)}" material="{m}"/>'
                     f'<geom type="box" pos="{_f(0, -D / 2 + 0.006, H / 2)}" size="{_f(L / 2, 0.006, H / 2)}" material="{m}"/>'
                     f'<geom type="box" pos="{_f(L / 2 - 0.01, 0, H / 2)}" size="{_f(0.01, D / 2, H / 2)}" material="{m}"/>'
                     f'<geom type="box" pos="{_f(-L / 2 + 0.01, 0, H / 2)}" size="{_f(0.01, D / 2, H / 2)}" material="{m}"/>'
                     + "".join(books) + "</body>")
    else:
        sx, sy, sz = rng.uniform(0.08, 0.25), rng.uniform(0.08, 0.25), rng.uniform(0.05, 0.3)
        spot = _free_spot(rng, room, taken, math.hypot(sx, sy))
        if not spot:
            return
        m = mats.make(rng.choice(["wood", "paint", "fabric"]), repeat=(1, 1), fallback=tuple(rng.uniform(0.2, 0.9, 3)))
        parts.append(f'<geom name="{kind}_{i}" type="box" pos="{_f(*spot, sz)}" size="{_f(sx, sy, sz)}" '
                     f'euler="0 0 {yaw:.1f}" material="{m}"/>')
        scene.obstacles.append((spot[0], spot[1], math.hypot(sx, sy)))
        return
    scene.obstacles.append((spot[0], spot[1], taken[-1][2]))


def build(world: str = "room", seed: int | None = None, assets: "Assets | None" = None, n_objects: int | None = None) -> Scene:
    world = world if world in WORLDS else "room"
    seed = int(seed if seed is not None else np.random.default_rng().integers(1, 1_000_000))
    rng = np.random.default_rng(seed)
    hx, hy = (rng.uniform(0.9, 1.6), rng.uniform(0.9, 1.6)) if world not in ("table", "balance") else (1.0, 1.0)
    mats = _Mats(assets, rng)
    taken: list[tuple[float, float, float]] = [(0.0, 0.0, 0.18)]    # keep the spawn clear
    parts: list[str] = []
    mesh_assets: list[str] = []
    scene = Scene(world, seed, "", (hx, hy), (0.0, 0.0, 0.0, float(rng.uniform(-math.pi, math.pi))))

    floor_kind = str(rng.choice(["wood", "wood", "carpet", "tile", "concrete"]))
    floor_mat = mats.make(floor_kind, repeat={"wood": (2.5, 2.5), "carpet": (4, 4), "tile": (1.5, 1.5)}.get(floor_kind, (2, 2)),
                          specular=0.35 if floor_kind in ("wood", "tile") else 0.05,
                          shininess=0.5 if floor_kind in ("wood", "tile") else 0.05,
                          reflectance=0.08 if floor_kind in ("wood", "tile") else 0.0,
                          fallback=tuple(rng.uniform(0.25, 0.75, 3)))
    wall_mat = mats.make("paint", repeat=(2, 1), specular=0.05, shininess=0.05, fallback=tuple(rng.uniform(0.45, 0.95, 3)))

    # walls + baseboards
    t, h = 0.02, 0.35
    for name, pos, size in (("wall_n", (0, hy + t, h), (hx + 2 * t, t, h)), ("wall_s", (0, -hy - t, h), (hx + 2 * t, t, h)),
                            ("wall_e", (hx + t, 0, h), (t, hy, h)), ("wall_w", (-hx - t, 0, h), (t, hy, h))):
        parts.append(f'<geom name="{name}" type="box" pos="{_f(*pos)}" size="{_f(*size)}" material="{wall_mat}"/>')
    parts.append(f'<geom name="baseboard" type="box" pos="{_f(0, hy - 0.004, 0.03)}" size="{_f(hx, 0.004, 0.03)}" '
                 f'rgba="0.93 0.92 0.9 1" contype="0" conaffinity="0"/>')

    if world == "table":
        tz = 0.38
        tw, td = rng.uniform(0.35, 0.5), rng.uniform(0.25, 0.4)
        tm = mats.make("wood", repeat=(1, 1), specular=0.4, shininess=0.5, reflectance=0.06, fallback=(0.55, 0.4, 0.28))
        parts.append(f'<geom name="table" type="box" pos="{_f(0, 0, tz - 0.015)}" size="{_f(tw, td, 0.015)}" material="{tm}"/>')
        for sx in (-1, 1):
            for sy in (-1, 1):
                parts.append(f'<geom name="leg_{sx}{sy}" type="box" pos="{_f(sx * (tw - 0.03), sy * (td - 0.03), (tz - 0.03) / 2)}" '
                             f'size="{_f(0.02, 0.02, (tz - 0.03) / 2)}" material="{tm}"/>')
        scene.floor_z = tz
        scene.spawn = (rng.uniform(-0.1, 0.1), rng.uniform(-0.05, 0.05), tz, scene.spawn[3])
        scene.room = (tw, td)
    elif world == "balance":
        scene.start_pitch = math.radians(62)
    elif world != "empty":
        if rng.random() < 0.6:                                     # a rug
            rx, ry = rng.uniform(0.3, 0.6), rng.uniform(0.25, 0.5)
            rm = mats.make("carpet", repeat=(2, 2), fallback=tuple(rng.uniform(0.2, 0.7, 3)))
            parts.append(f'<geom name="rug" type="box" pos="{_f(rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3), 0.0015)}" '
                         f'size="{_f(rx, ry, 0.0015)}" material="{rm}" euler="0 0 {rng.uniform(0, 180):.0f}"/>')
        for i in range(int(rng.integers(2, 6))):
            _furniture(rng, mats, (hx, hy), taken, parts, scene, i)
        _objects(rng, assets, (hx, hy), taken, parts, mesh_assets, scene, n_objects)
        if world == "person":
            spot = _free_spot(rng, (hx, hy), taken, 0.22, margin=0.15) or (hx * 0.6, 0.0)
            yaw = math.atan2(-spot[1], -spot[0])           # facing the middle of the room
            skin = rng.choice([(0.95, 0.8, 0.68), (0.78, 0.57, 0.42), (0.45, 0.3, 0.22)])
            hair = rng.choice([(0.1, 0.07, 0.05), (0.35, 0.22, 0.1), (0.75, 0.6, 0.35), (0.05, 0.05, 0.05)])
            shirt = mats.make("fabric", repeat=(2, 2), fallback=tuple(rng.uniform(0.15, 0.85, 3)))
            pants = mats.make("fabric", repeat=(2, 2), fallback=(0.2, 0.25, 0.4))
            parts.append(f"""<body name="person" mocap="true" pos="{_f(*spot, 0)}" euler="0 0 {math.degrees(yaw):.1f}">
          <geom name="person_legs" type="capsule" fromto="0 0.08 0.06 0.3 0.09 0.05" size="0.055" material="{pants}"/>
          <geom name="person_legs2" type="capsule" fromto="0 -0.08 0.06 0.3 -0.09 0.05" size="0.055" material="{pants}"/>
          <geom name="person_body" type="capsule" fromto="0 0 0.14 0 0 0.46" size="0.13" material="{shirt}"/>
          <geom name="person_arm_l" type="capsule" fromto="0 0.17 0.48 0.16 0.16 0.26" size="0.04" material="{shirt}"/>
          <geom name="person_arm_r" type="capsule" fromto="0 -0.17 0.48 0.16 -0.16 0.26" size="0.04" material="{shirt}"/>
          <geom name="person_neck" type="capsule" fromto="0 0 0.56 0 0 0.6" size="0.04" rgba="{_f(*skin, 1)}"/>
          <geom name="person_head" type="ellipsoid" pos="0.01 0 0.68" size="0.085 0.075 0.1" rgba="{_f(*skin, 1)}"/>
          <geom name="person_hair" type="ellipsoid" pos="-0.012 0 0.715" size="0.085 0.08 0.075" rgba="{_f(*hair, 1)}" contype="0" conaffinity="0"/>
          <geom name="person_eye_l" type="sphere" pos="0.085 0.03 0.69" size="0.011" rgba="0.05 0.05 0.05 1" contype="0" conaffinity="0"/>
          <geom name="person_eye_r" type="sphere" pos="0.085 -0.03 0.69" size="0.011" rgba="0.05 0.05 0.05 1" contype="0" conaffinity="0"/>
          <site name="face" pos="0.09 0 0.68" group="5"/>
        </body>""")
            scene.person = spot
            scene.obstacles.append((spot[0], spot[1], 0.2))
        if world == "goal":
            spot = _free_spot(rng, (hx, hy), taken, 0.08, margin=0.15) or (hx * 0.6, 0.0)
            parts.append(f'<geom name="goal" type="cylinder" pos="{_f(*spot, 0.0016)}" size="0.07 0.0005" '
                         f'rgba="0.83 0.3 0.16 0.9" contype="0" conaffinity="0"/>')
            scene.goal = spot

    # lights: 1-4, mixed colour temperatures; headlight kept dim so shadows read
    lights = []
    for k in range(int(rng.integers(1, 5)) if world != "empty" else 1):
        kel = KELVIN[int(rng.choice(list(KELVIN)))]
        power = rng.uniform(0.35, 0.8)
        lx, ly = rng.uniform(-hx, hx), rng.uniform(-hy, hy)
        lights.append(f'<light name="light_{k}" pos="{_f(lx, ly, rng.uniform(1.6, 2.6))}" dir="{_f(-lx * 0.3, -ly * 0.3, -1)}" '
                      f'diffuse="{_f(*(np.array(kel) * power))}" specular="0.15 0.15 0.15" castshadow="{"true" if k == 0 else "false"}" '
                      f'attenuation="1 0 0.05"/>')
    ambient = rng.uniform(0.12, 0.3)

    sx, sy, sz, yaw = scene.spawn
    robot = vector_mjcf((sx, sy, sz + 0.0005), yaw)
    scene.xml = f"""<mujoco model="cloakos-sim">
  <compiler angle="degree" autolimits="true"/>
  <option timestep="0.002" integrator="implicitfast" cone="elliptic" impratio="5"/>
  <visual>
    <global offwidth="1280" offheight="720"/>
    <quality shadowsize="4096" offsamples="4"/>
    <headlight ambient="{_f(ambient, ambient, ambient)}" diffuse="0.15 0.15 0.15" specular="0 0 0"/>
    <map znear="0.002"/>
  </visual>
  <default>
    <geom contype="1" conaffinity="3" friction="0.9 0.005 0.0001" solref="0.004 1"/>
  </default>
  <asset>
    {vector_assets()}
    {chr(10).join('    ' + x for x in mats.xml)}
    {chr(10).join('    ' + x for x in mesh_assets)}
  </asset>
  <worldbody>
    {chr(10).join('    ' + x for x in lights)}
    <geom name="floor" type="plane" size="{_f(hx + 0.05, hy + 0.05, 0.05)}" material="{floor_mat}"/>
    {chr(10).join('    ' + p for p in parts)}
{robot}
  </worldbody>
{vector_actuators()}
</mujoco>"""
    return scene


def _objects(rng, assets, room, taken, parts, mesh_assets, scene: Scene, n: int | None) -> None:
    """Small things on the floor to find: photo-scanned objects when we have them."""
    from .assets import label_for

    want = int(n if n is not None else rng.integers(2, 6))
    names = []
    if assets is not None:
        cat = assets.catalog()
        order = rng.permutation(len(cat))
        for idx in order[: want * 3]:
            if len(names) >= want:
                break
            d = assets.object(cat[int(idx)])
            if d is not None:
                names.append((cat[int(idx)], d))
    for k, (name, d) in enumerate(names):
        spot = _free_spot(rng, room, taken, 0.09)
        if not spot:
            break
        body = f"obj_{k}"
        mesh_assets.append(f'<texture name="{body}_tex" type="2d" file="{d / "texture.png"}"/>')
        mesh_assets.append(f'<material name="{body}_mat" texture="{body}_tex" specular="0.3" shininess="0.3"/>')
        mesh_assets.append(f'<mesh name="{body}_mesh" file="{d / "model.obj"}"/>')
        parts.append(f'<body name="{body}" pos="{_f(*spot, 0.2)}" euler="0 0 {rng.uniform(0, 360):.0f}"><freejoint/>'
                     f'<geom name="{body}" type="mesh" mesh="{body}_mesh" material="{body}_mat" density="400"/></body>')
        scene.objects[body] = spot
        scene.labels[body] = label_for(name)
    # plain shapes to make up the numbers (offline, or no assets)
    plain = list(OBJECTS)
    rng.shuffle(plain)
    for name in plain[: max(0, want - len(names))]:
        shape, size, rgba = OBJECTS[name]
        spot = _free_spot(rng, room, taken, max(size) + 0.02)
        if not spot:
            continue
        z = size[1] if shape == "cylinder" else size[0] if shape == "sphere" else size[-1]
        slug = name.replace(" ", "_")
        parts.append(f'<body name="obj_{slug}" pos="{_f(*spot, z + 0.001)}"><freejoint/>'
                     f'<geom name="obj_{slug}" type="{shape}" size="{_f(*size)}" rgba="{_f(*rgba)}" mass="0.05"/></body>')
        scene.objects[f"obj_{slug}"] = spot
        scene.labels[f"obj_{slug}"] = name
