"""Mapper: builds and remembers a 3D map of the space from Vector's camera + pose.

Pose comes from Vector itself (wheel odometry fused with its gyro), the head angle tilts the
camera. Each map belongs to a pose *origin*: Vector starts a new one when it's picked up and
loses track; when it recognises the old one again it reuses it, and so do we. Maps and the
objects found in them are saved to <data>/maps/ and reloaded next time.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .vision import CameraPose, DepthModel, Detector, backproject, camera_pose, quat_R
from .voxels import VoxelMap

if TYPE_CHECKING:
    from ..bus import Bus, Event
    from ..connection import RobotLink
    from ..world import World

log = logging.getLogger(__name__)
MERGE_M = 0.3          # sightings of the same label closer than this are the same object


class Mapper:
    def __init__(self, cfg: dict, link: "RobotLink", world: "World", bus: "Bus") -> None:
        c = cfg.get("map", {}) or {}
        self.enabled = bool(c.get("enabled", True))
        self.hz = float(c.get("hz", 2.0))
        self.use_models = bool(c.get("models", True))          # depth + detector on the real robot
        self.detect_every = int(c.get("detect_every", 2))
        self.dir = Path((cfg.get("datalog", {}) or {}).get("dir", "data")) / "maps"
        self.link, self.world, self.bus = link, world, bus
        self.key: str | None = None
        self.map: VoxelMap | None = None
        self.objects: list[dict] = []
        self.trail: deque[tuple[float, float, float]] = deque(maxlen=2000)
        self.status = "waiting for the robot"
        self.depth_model = DepthModel()
        self.detector = Detector(float(c.get("detect_threshold", 0.55)))
        self._last_frame = -1
        self._last_sim_t = 0.0
        self._frames = 0
        self.vision_needed = lambda: True       # the brain says when the sim camera must keep running
        self._saved = time.time()
        self._dirty = False
        self.cam: CameraPose | None = None          # where the camera was for the latest frame
        self.dets: list[dict] = []                  # what it recognised in the latest frame (image coords)
        self.det_size = (640, 360)
        self.floor_z = 0.0
        bus.on("raw.face_pose", self._on_face)

    # ── map selection / persistence ───────────────────────────

    def _use(self, key: str, center: tuple[float, float]) -> None:
        if key == self.key:
            return
        self.save()
        self.key = key
        path = self.dir / f"{key}.npz"
        try:
            self.map = VoxelMap.load(path)
            meta = json.loads((self.dir / f"{key}.json").read_text())
            self.objects = meta.get("objects", [])
            self.trail = deque((tuple(p) for p in meta.get("trail", [])), maxlen=2000)
            log.info("map %s loaded (%d objects remembered)", key, len(self.objects))
        except (OSError, ValueError, KeyError):
            self.map = VoxelMap(center=center)
            self.objects, self.trail = [], deque(maxlen=2000)
            log.info("new map %s", key)
        self.bus.emit("map.changed", key=key)

    def save(self) -> None:
        if not (self.map and self.key and self._dirty):
            return
        try:
            self.map.save(self.dir / f"{self.key}.npz")
            (self.dir / f"{self.key}.json").write_text(json.dumps(
                {"objects": self.objects, "trail": list(self.trail)[-2000:], "saved": time.time()}))
            self._dirty = False
        except OSError as e:
            log.warning("couldn't save the map: %s", e)

    def clear(self) -> None:
        if self.map:
            self.map.clear()
        self.objects, self.trail = [], deque(maxlen=2000)
        self._dirty = True
        self.save()

    # ── loop ───────────────────────────────────────────────────

    async def run(self) -> None:
        period = 1.0 / self.hz
        while True:
            t0 = time.monotonic()
            try:
                if self.enabled and self.link.connected:
                    if self.link.target == "sim" and self.link.sim is not None:
                        if self.vision_needed():
                            self.link.sim.touch()
                        await self._from_sim()
                    else:
                        await self._from_robot()
                elif self.link.sim is not None:
                    self.link.sim.want_depth = False
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("mapping step failed")
            if time.time() - self._saved > 30:
                await asyncio.to_thread(self.save)
                self._saved = time.time()
            await asyncio.sleep(max(0.05, period - (time.monotonic() - t0)))

    async def _from_sim(self) -> None:
        sim = self.link.sim
        sim.want_depth = True
        sim.depth_hz = self.hz
        f = sim.depth_frame
        if f is None or f["t"] == self._last_sim_t:
            return
        self._last_sim_t = f["t"]
        self._use(f["key"], tuple(f["pos"][:2]))
        cam = CameraPose(f["pos"], f["R"])
        depth = f["depth"]
        self.cam, self.floor_z = cam, f.get("floor_z", 0.0)
        self.dets, self.det_size = f["detections"], (depth.shape[1], depth.shape[0])
        await asyncio.to_thread(self._integrate, depth, cam, f["rgb"], f["detections"], 2)
        snap = sim.snapshot
        self.trail.append((round(snap["pos"][0], 3), round(snap["pos"][1], 3), round(snap["pos"][2], 3)))
        self.status = "mapping (sim depth)"

    async def _from_robot(self) -> None:
        w = self.world
        if w.frame is None or w.frame_id == self._last_frame or self.link.pose is None:
            return
        self._last_frame = w.frame_id
        p = self.link.pose
        key = f"vector-{self.link.client.bot.serial}-{p['origin']}"
        self._use(key, (p["pos"][0], p["pos"][1]))
        self.trail.append(tuple(round(v, 3) for v in p["pos"]))
        cam = camera_pose(p["pos"], quat_R(p["quat"]), p["head"])
        # the distance sensor always counts: a solid hit straight ahead
        if w.proximity_mm:
            R = quat_R(p["quat"])
            hit = np.asarray(p["pos"]) + R @ np.array([0.03 + w.proximity_mm / 1000, 0, 0.02])
            self.map.integrate(np.asarray(p["pos"]) + R @ np.array([0.03, 0, 0.02]), hit[None, :], carve=True)
            self._dirty = True
        if not (self.use_models and self.depth_model.available()):
            self.status = "mapping (distance sensor only: install the ml extra for camera depth)"
            return
        if self.depth_model.error:
            self.status = f"mapping (distance sensor only: {self.depth_model.error})"
            return
        img = w.frame.convert("RGB")
        try:
            depth = await asyncio.to_thread(self.depth_model.metric, img, cam,
                                            (w.proximity_mm / 1000) if w.proximity_mm else None)
        except Exception as e:                      # model download failed, out of memory, ...
            self.depth_model.error = f"depth model unavailable ({type(e).__name__}: {e})"[:200]
            log.warning(self.depth_model.error)
            return
        if depth is None:
            self.status = "mapping (waiting for a frame with floor in view to set the scale)"
            return
        dets = []
        self._frames += 1
        if self._frames % self.detect_every == 0 and not getattr(self.detector, "error", None):
            try:
                dets = await asyncio.to_thread(self.detector.detect, img)
            except Exception as e:
                self.detector.error = f"{type(e).__name__}: {e}"[:200]
                log.warning("object detector unavailable: %s", self.detector.error)
        await asyncio.to_thread(self._integrate, depth, cam, np.asarray(img), dets, 4)
        self.cam, self.dets, self.det_size = cam, dets, img.size
        self.status = "mapping (camera depth)"

    def _integrate(self, depth: np.ndarray, cam: CameraPose, rgb: np.ndarray, dets: list[dict], stride: int) -> None:
        pts, px = backproject(depth, cam, stride)
        cols = rgb[px[:, 0], px[:, 1]] if rgb is not None and len(px) else None
        self.map.integrate(cam.pos, pts, cols)
        self._dirty = True
        for d in dets:
            x0, y0, x1, y1 = (int(v) for v in d["box"])
            h, w = depth.shape
            x0, x1 = max(0, x0), min(w - 1, x1)
            y0, y1 = max(0, y0), min(h - 1, y1)
            if x1 <= x0 or y1 <= y0:
                continue
            patch = depth[y0:y1, x0:x1]
            z = float(np.nanmedian(patch)) if np.isfinite(patch).any() else math.nan
            if not math.isfinite(z) or z <= 0 or z > 3:
                continue
            u, v = (x0 + x1) / 2, (y0 + y1) / 2
            from .vision import intrinsics

            f, cx, cy = intrinsics(w, h)
            p = cam.pos + cam.R @ np.array([z, -(u - cx) * z / f, -(v - cy) * z / f])
            self.remember(d["label"], p, d.get("score", 1.0), "camera")

    def _on_face(self, ev: "Event") -> None:
        p = ev.data.get("pos")
        if p is not None and self.map is not None:
            self.remember("person", np.asarray(p), 1.0, "face")

    def remember(self, label: str, pos: np.ndarray, score: float, source: str) -> dict:
        now = time.time()
        for o in self.objects:
            if o["label"] == label and math.dist(o["pos"][:2], pos[:2]) < MERGE_M:
                n = o["seen"]
                o["pos"] = [round((o["pos"][i] * n + float(pos[i])) / (n + 1), 3) for i in range(3)]
                o["seen"], o["last_seen"] = n + 1, now
                o["score"] = round(max(o["score"], score), 2)
                self._dirty = True
                return o
        o = {"id": f"{label}-{len(self.objects) + 1}", "label": label, "pos": [round(float(v), 3) for v in pos],
             "seen": 1, "first_seen": now, "last_seen": now, "score": round(score, 2), "source": source}
        self.objects.append(o)
        self._dirty = True
        self.bus.emit("map.object", label=label, pos=o["pos"], id=o["id"])
        return o

    def rays(self, pos, yaw: float, n: int, fov: float, max_range: float) -> "np.ndarray":
        """Distance to the nearest mapped obstacle in n directions around the robot (at its
        height). Space the map hasn't seen counts as clear, like the gaps it trained with."""
        m = self.map
        out = np.full(n, max_range, np.float32)
        if m is None:
            return out
        k0 = max(0, int((self.floor_z + 0.015 - m.z0) / m.voxel))
        k1 = min(m.n_z, int((self.floor_z + 0.10 - m.z0) / m.voxel) + 1)
        steps = np.arange(0.03, max_range, m.voxel * 0.8)
        for i, a in enumerate(np.linspace(-fov / 2, fov / 2, n)):
            d = np.array([math.cos(yaw + a), math.sin(yaw + a)])
            pts = np.asarray(pos[:2]) + steps[:, None] * d
            ij = np.floor((pts - m.origin[:2]) / m.voxel).astype(int)
            ok = (ij >= 0).all(1) & (ij < m.n_xy).all(1)
            hit = np.zeros(len(steps), bool)
            hit[ok] = (m.logodds[ij[ok, 0], ij[ok, 1], k0:k1] > 6).any(1)
            if hit.any():
                out[i] = float(steps[np.argmax(hit)])
        return out

    def find(self, query: str) -> dict | None:
        """The best remembered object matching a word ("mug", "shoe", "person")."""
        q = query.lower().strip()
        hits = [o for o in self.objects if q in o["label"].lower()]
        return max(hits, key=lambda o: (o["seen"], o["last_seen"])) if hits else None

    # ── for the UI ─────────────────────────────────────────────

    def summary(self) -> dict[str, Any]:
        return {"enabled": self.enabled, "key": self.key, "status": self.status,
                "stats": self.map.stats() if self.map else None,
                "objects": sorted(self.objects, key=lambda o: -o["last_seen"]),
                "trail": list(self.trail)[-600:],
                "models": {"depth": self.depth_model.available(), "error": self.depth_model.error}}

    def live(self) -> dict[str, Any]:
        """What the FSD-style view needs every update: camera cone, current detections, object boxes."""
        out: dict[str, Any] = {"key": self.key, "version": self.map.version if self.map else 0, "status": self.status}
        if self.cam is not None:
            R = self.cam.R
            out["cam"] = {"pos": [round(float(v), 3) for v in self.cam.pos], "fwd": R[:, 0].round(4).tolist(),
                          "left": R[:, 1].round(4).tolist(), "up": R[:, 2].round(4).tolist()}
        out["dets"] = [{"label": d["label"], "box": [round(float(v), 1) for v in d["box"]], "score": round(d.get("score", 1), 2)}
                       for d in self.dets]
        out["frame_size"] = list(self.det_size)
        out["floor_z"] = self.floor_z
        boxes = []
        if self.map is not None:
            for o in self.objects:
                ext = self.map.extent(o["pos"])
                boxes.append({"id": o["id"], "label": o["label"], "pos": o["pos"], "box": ext,
                              "fresh": time.time() - o["last_seen"] < 3})
        out["objects"] = boxes
        return out

    def voxels_bin(self) -> bytes:
        """u32 n, f32 origin xyz + voxel, then n x (u16 i,j,k), n x rgb, n x age (u8),
        then u32 m, m x (u16 i,j) drivable floor cells."""
        if self.map is None:
            return np.array([0], np.uint32).tobytes() + np.zeros(4, np.float32).tobytes() + np.array([0], np.uint32).tobytes()
        ijk, col = self.map.occupied()
        age = np.minimum(self.map.age(ijk), 255).astype(np.uint8)
        floor = self.map.drivable(self.floor_z)[:150_000]
        head = np.array([len(ijk)], np.uint32).tobytes() + np.array([*self.map.origin, self.map.voxel], np.float32).tobytes()
        return (head + ijk.astype(np.uint16).tobytes() + col.astype(np.uint8).tobytes() + age.tobytes()
                + np.array([len(floor)], np.uint32).tobytes() + floor.astype(np.uint16).tobytes())
