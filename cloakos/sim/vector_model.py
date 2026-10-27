"""Vector's body as MuJoCo MJCF, built from its real measurements.

Published specs: ~100 x 60 x 69 mm (lift down), ~159 g, head -22..45 deg, lift 32..92 mm.
Everything else (tread wheel size and spacing, part masses, centre of mass) is measured
off photos/teardowns and approximate; tune the constants below if you measure your own.

Frame: x forward, y left, z up. The robot's origin sits on the floor midway between the
tread axles, which is the point Vector's own pose reports.
"""
from __future__ import annotations

import math

# ── measurements (metres, kilograms) ──────────────────────────
TREAD_Y = 0.0235          # tread centreline from the middle (outer width ~60 mm)
TREAD_W = 0.011           # tread width
WHEEL_R = 0.0105          # tread wheel radius (front and rear); treads wrap these
AXLE_X = 0.0215           # front/rear axle offset from the origin (wheelbase ~43 mm)
END_LIFT = 0.0008         # end wheels ride this much higher than the middle ones
CHASSIS = (0.064, 0.036, 0.030)     # body block between the treads (l, w, h)
CHASSIS_Z = 0.008                   # its underside, above the floor
HEAD = (0.034, 0.046, 0.026)        # head block (l, w, h)
HEAD_PIVOT = (-0.004, 0.0, 0.043)   # head hinge, relative to the origin
LIFT_PIVOT = (-0.026, 0.0, 0.032)   # lift arm hinge (back of the body)
LIFT_ARM = 0.064                    # hinge to fork
MASS = {"chassis": 0.088, "wheel": 0.002, "head": 0.032, "lift": 0.013}   # total ~0.159 kg

HEAD_RANGE = (-22.0, 45.0)          # degrees, from the SDK
LIFT_MM = (32.0, 92.0)              # fork height range, from the SDK
MAX_WHEEL_MMPS = 220.0
CAMERA_FOVY = 50.0                  # vertical field of view (90 deg horizontal at 16:9)


def lift_angle(height_mm: float) -> float:
    """Fork height above the floor (mm) -> lift hinge angle (rad)."""
    h = max(LIFT_MM[0], min(LIFT_MM[1], height_mm)) / 1000.0
    return math.asin(max(-1.0, min(1.0, (h - LIFT_PIVOT[2]) / LIFT_ARM)))


def lift_height_mm(angle: float) -> float:
    return (LIFT_PIVOT[2] + LIFT_ARM * math.sin(angle)) * 1000.0


LIFT_RANGE = (lift_angle(LIFT_MM[0]), lift_angle(LIFT_MM[1]))


def _f(*v: float) -> str:
    return " ".join(f"{x:.5g}" for x in v)


def rounded_box(l: float, w: float, h: float, r: float, n: int = 6,
                front_drop: float = 0.0, back_drop: float = 0.0) -> list[tuple[float, float, float]]:
    """Points whose convex hull is a box with rounded edges (MuJoCo builds the hull).
    front_drop/back_drop lower the top edge at the front/back, for Vector's sloped body."""
    pts = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                cx, cy, cz = sx * (l / 2 - r), sy * (w / 2 - r), sz * (h / 2 - r)
                if sz > 0:
                    cz -= front_drop if sx > 0 else back_drop
                for i in range(n + 1):
                    th = math.pi / 2 * i / n
                    for j in range(n + 1):
                        ph = math.pi / 2 * j / n
                        pts.append((cx + sx * r * math.cos(th) * math.cos(ph),
                                    cy + sy * r * math.cos(th) * math.sin(ph),
                                    cz + sz * r * math.sin(th)))
    return pts


def _verts(pts) -> str:
    return " ".join(f"{x:.5f} {y:.5f} {z:.5f}" for x, y, z in pts)


def vector_assets() -> str:
    """Meshes + materials for Vector's look (goes inside <asset>)."""
    cl, cw, ch = CHASSIS
    hl, hw, hh = HEAD
    return f"""
    <mesh name="vec_body" vertex="{_verts(rounded_box(cl, cw + 0.004, ch, 0.006, front_drop=0.006, back_drop=0.002))}"/>
    <mesh name="vec_head" vertex="{_verts(rounded_box(hl, hw, hh, 0.006))}"/>
    <mesh name="vec_pack" vertex="{_verts(rounded_box(0.026, 0.020, 0.006, 0.003, n=4))}"/>
    <mesh name="vec_fork" vertex="{_verts(rounded_box(0.008, 0.042, 0.016, 0.002, n=3))}"/>
    <material name="vec_matte" rgba="0.075 0.075 0.08 1" specular="0.25" shininess="0.3"/>
    <material name="vec_gloss" rgba="0.045 0.045 0.05 1" specular="0.9" shininess="0.9" reflectance="0.08"/>
    <material name="vec_rubber" rgba="0.04 0.04 0.045 1" specular="0.05" shininess="0.1"/>
    <material name="vec_hub" rgba="0.2 0.2 0.21 1" specular="0.5" shininess="0.6"/>
    <material name="vec_metal" rgba="0.24 0.24 0.26 1" specular="0.7" shininess="0.7"/>
    <material name="vec_light" rgba="0.25 1 0.6 1" emission="0.9"/>
    <material name="vec_screen" rgba="0 0 0 1" specular="1" shininess="1" reflectance="0.15"/>"""


# visual-only geoms: no collision, no mass
VIS = 'contype="0" conaffinity="0" mass="0" group="3"'
# collision-only geoms: invisible primitives that keep the physics stable
COL = 'rgba="0 0 0 0" group="3" contype="2" conaffinity="1"'


def vector_mjcf(pos: tuple[float, float, float] = (0, 0, 0), yaw: float = 0.0) -> str:
    """The <body> for Vector. Robot geoms are in group 3 so sensor rays can skip them."""
    cl, cw, ch = CHASSIS
    hl, hw, hh = HEAD
    wheels = []
    for side, y in (("l", TREAD_Y), ("r", -TREAD_Y)):
        # treads carry the weight on the middle; the end wheels sit a hair higher and only
        # bite when it rocks or tips (that's what lets a tread robot spin in place)
        for end, x, lift in (("f", AXLE_X, END_LIFT), ("m", 0.0, 0.0), ("b", -AXLE_X, END_LIFT)):
            hub = "" if end == "m" else f"""
        <geom name="hub_{side}{end}" type="cylinder" size="{_f(WHEEL_R * 0.55, TREAD_W / 2 + 0.0004)}" euler="90 0 0"
              material="vec_hub" {VIS}/>"""
            wheels.append(f"""
      <body name="wheel_{side}{end}" pos="{_f(x, y, WHEEL_R + lift)}">
        <joint name="wheel_{side}{end}" type="hinge" axis="0 1 0" damping="0.00002"/>
        <geom name="wheel_{side}{end}" type="cylinder" size="{_f(WHEEL_R, 0.0008)}" euler="90 0 0"
              mass="{MASS['wheel']}" friction="1.1 0.005 0.0001" rgba="0 0 0 0" group="3" contype="2" conaffinity="1"/>{hub}
      </body>""")
        # the rubber tread band around the wheels
        wheels.append(f"""
      <geom name="tread_{side}" type="capsule" fromto="{_f(-AXLE_X, y, WHEEL_R, AXLE_X, y, WHEEL_R)}"
            size="{_f(WHEEL_R + 0.0006)}" material="vec_rubber" {VIS}/>""")
    hy = HEAD_RANGE
    ly = LIFT_RANGE
    return f"""
    <body name="vector" pos="{_f(*pos)}" euler="0 0 {math.degrees(yaw):.4g}">
      <freejoint name="vector"/>
      <site name="imu" pos="0 0 0.03" group="5"/>
      <geom name="chassis" type="box" pos="{_f(-0.003, 0, CHASSIS_Z + ch / 2)}" size="{_f(cl / 2, cw / 2, ch / 2)}"
            mass="{MASS['chassis']}" {COL}/>
      <geom name="body" type="mesh" mesh="vec_body" pos="{_f(-0.003, 0, CHASSIS_Z + ch / 2)}" material="vec_matte" {VIS}/>
      <geom name="backpack" type="mesh" mesh="vec_pack" pos="{_f(-0.020, 0, CHASSIS_Z + ch + 0.001)}" material="vec_matte" {VIS}/>
      <geom name="backpack_light" type="box" pos="{_f(-0.020, 0, CHASSIS_Z + ch + 0.0042)}" size="0.009 0.0025 0.0004"
            material="vec_light" {VIS}/>
      <site name="prox" pos="{_f(cl / 2 - 0.003, 0, CHASSIS_Z + 0.012)}" group="5"/>
      <site name="cliff_fl" pos="{_f(AXLE_X + 0.006, 0.010, CHASSIS_Z)}" group="5"/>
      <site name="cliff_fr" pos="{_f(AXLE_X + 0.006, -0.010, CHASSIS_Z)}" group="5"/>
      <site name="cliff_bl" pos="{_f(-AXLE_X - 0.006, 0.010, CHASSIS_Z)}" group="5"/>
      <site name="cliff_br" pos="{_f(-AXLE_X - 0.006, -0.010, CHASSIS_Z)}" group="5"/>
      {''.join(wheels)}
      <body name="head" pos="{_f(*HEAD_PIVOT)}">
        <joint name="head" type="hinge" axis="0 -1 0" range="{_f(*hy)}" damping="0.002" armature="0.00005"/>
        <geom name="head" type="box" pos="{_f(hl / 2 - 0.004, 0, hh / 2 - 0.004)}" size="{_f(hl / 2, hw / 2, hh / 2)}"
              mass="{MASS['head']}" {COL}/>
        <geom name="head_shell" type="mesh" mesh="vec_head" pos="{_f(hl / 2 - 0.004, 0, hh / 2 - 0.004)}" material="vec_gloss" {VIS}/>
        <geom name="screen" type="box" pos="{_f(hl - 0.0038, 0, hh / 2 - 0.003)}" size="{_f(0.0004, 0.0165, 0.0087)}"
              material="vec_screen" {VIS}/>
        <camera name="eye" pos="{_f(hl - 0.003, 0, 0.0)}" xyaxes="0 -1 0 0 0 1" fovy="{CAMERA_FOVY}"/>
      </body>
      <body name="lift" pos="{_f(*LIFT_PIVOT)}">
        <joint name="lift" type="hinge" axis="0 -1 0" range="{math.degrees(ly[0]):.4g} {math.degrees(ly[1]):.4g}" damping="0.002"/>
        <geom name="arm_l" type="capsule" fromto="{_f(0, 0.019, 0, LIFT_ARM, 0.019, 0)}" size="0.0022"
              mass="{MASS['lift'] * 0.3}" material="vec_metal" group="3" contype="2" conaffinity="1"/>
        <geom name="arm_r" type="capsule" fromto="{_f(0, -0.019, 0, LIFT_ARM, -0.019, 0)}" size="0.0022"
              mass="{MASS['lift'] * 0.3}" material="vec_metal" group="3" contype="2" conaffinity="1"/>
        <geom name="fork" type="box" pos="{_f(LIFT_ARM + 0.003, 0, -0.004)}" size="0.004 0.021 0.008"
              mass="{MASS['lift'] * 0.4}" {COL}/>
        <geom name="fork_shell" type="mesh" mesh="vec_fork" pos="{_f(LIFT_ARM + 0.003, 0, -0.004)}" material="vec_metal" {VIS}/>
      </body>
    </body>"""


def vector_actuators() -> str:
    wheels = "".join(
        f'\n    <velocity name="wheel_{s}{e}" joint="wheel_{s}{e}" kv="0.004" forcerange="-0.012 0.012"/>'
        for s in "lr" for e in "fmb")
    return f"""
  <actuator>{wheels}
    <position name="head" joint="head" kp="0.25" kv="0.01" forcerange="-0.08 0.08"/>
    <position name="lift" joint="lift" kp="0.35" kv="0.01" forcerange="-0.12 0.12"/>
  </actuator>
  <sensor>
    <accelerometer name="accel" site="imu"/>
    <gyro name="gyro" site="imu"/>
  </sensor>"""
