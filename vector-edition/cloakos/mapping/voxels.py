"""VoxelMap: a dense 3D occupancy grid with colour, built from depth + pose.

2.5 cm voxels over 10 x 10 x 1.2 m (int8 log-odds + RGB, ~31 MB), centred on where the robot
started. Every depth frame marks the surface it saw as occupied and carves the space between
the camera and that surface as free, so things that move away fade out of the map.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

HIT, MISS, LO, HI, OCCUPIED = 4, -1, -20, 40, 6


class VoxelMap:
    def __init__(self, voxel: float = 0.025, size_xy: float = 10.0, z_range: tuple[float, float] = (-0.1, 1.1),
                 center: tuple[float, float] = (0.0, 0.0)) -> None:
        self.voxel = voxel
        self.n_xy = int(round(size_xy / voxel))
        self.z0, self.n_z = z_range[0], int(round((z_range[1] - z_range[0]) / voxel))
        self.origin = np.array([center[0] - size_xy / 2, center[1] - size_xy / 2, self.z0], np.float32)
        self.logodds = np.zeros((self.n_xy, self.n_xy, self.n_z), np.int8)
        self.color = np.zeros((self.n_xy, self.n_xy, self.n_z, 3), np.uint8)
        self.stamp = np.zeros((self.n_xy, self.n_xy, self.n_z), np.uint16)   # frame it was last seen (hit)
        self.frame = 1
        self.version = 0

    # ── indexing ───────────────────────────────────────────────

    def index(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """World points (N,3) -> voxel indices (N,3) and an in-bounds mask."""
        ijk = np.floor((pts - self.origin) / self.voxel).astype(np.int32)
        ok = (ijk >= 0).all(1) & (ijk[:, 0] < self.n_xy) & (ijk[:, 1] < self.n_xy) & (ijk[:, 2] < self.n_z)
        return ijk, ok

    def center_of(self, ijk: np.ndarray) -> np.ndarray:
        return self.origin + (ijk + 0.5) * self.voxel

    # ── updates ────────────────────────────────────────────────

    def integrate(self, cam: np.ndarray, pts: np.ndarray, colors: np.ndarray | None = None,
                  carve: bool = True, max_range: float = 2.5) -> int:
        """Add one frame: `pts` (N,3) surface points seen from camera position `cam` (3,)."""
        if len(pts) == 0:
            return 0
        d = np.linalg.norm(pts - cam, axis=1)
        keep = (d > 0.03) & (d < max_range)
        pts, d = pts[keep], d[keep]
        if colors is not None:
            colors = colors[keep]
        if carve and len(pts):
            # free space: samples along each ray up to one voxel short of the hit
            steps = np.arange(0.5, 200) * self.voxel
            steps = steps[steps < max_range]
            frac = steps[None, :] / d[:, None]
            mask = frac < 1.0 - self.voxel / np.maximum(d[:, None], 1e-6)
            samples = cam[None, None, :] + (pts - cam)[:, None, :] * frac[..., None]
            free = samples[mask]
            ijk, ok = self.index(free)
            ijk = np.unique(ijk[ok], axis=0)
            if len(ijk):
                cur = self.logodds[ijk[:, 0], ijk[:, 1], ijk[:, 2]].astype(np.int16)
                self.logodds[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = np.clip(cur + MISS, LO, HI).astype(np.int8)
        ijk, ok = self.index(pts)
        ijk = ijk[ok]
        if colors is not None:
            self.color[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = colors[ok]
        u, counts = np.unique(ijk, axis=0, return_counts=True)
        if len(u):
            cur = self.logodds[u[:, 0], u[:, 1], u[:, 2]].astype(np.int16)
            self.logodds[u[:, 0], u[:, 1], u[:, 2]] = np.clip(cur + HIT * np.minimum(counts, 3), LO, HI).astype(np.int8)
            self.stamp[u[:, 0], u[:, 1], u[:, 2]] = self.frame
        self.frame = self.frame + 1 if self.frame < 65000 else 1
        self.version += 1
        return int(len(u))

    def clear(self) -> None:
        self.logodds[:] = 0
        self.color[:] = 0
        self.stamp[:] = 0
        self.version += 1

    # ── queries ────────────────────────────────────────────────

    def occupied(self, limit: int = 120_000) -> tuple[np.ndarray, np.ndarray]:
        """Occupied voxel indices (N,3) and colours (N,3), most certain first if over `limit`."""
        ijk = np.argwhere(self.logodds > OCCUPIED)
        if len(ijk) > limit:
            conf = self.logodds[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
            ijk = ijk[np.argsort(-conf)[:limit]]
        return ijk, self.color[ijk[:, 0], ijk[:, 1], ijk[:, 2]]

    def age(self, ijk: np.ndarray) -> np.ndarray:
        """Frames since each voxel was last seen (0 = this frame)."""
        return (self.frame - 1 - self.stamp[ijk[:, 0], ijk[:, 1], ijk[:, 2]].astype(np.int32)) % 65001

    def drivable(self, floor_z: float = 0.0) -> np.ndarray:
        """(i, j) cells of floor that have been seen free with nothing standing on them."""
        g = self.floor_grid(floor_z + 0.01, floor_z + 0.12)
        k = max(0, int((floor_z - self.z0) / self.voxel))
        floor_seen = (self.logodds[:, :, max(0, k - 1):k + 2] > OCCUPIED).any(2) | (g == -1)
        return np.argwhere(floor_seen & (g != 1))

    def extent(self, center, radius: float = 0.12) -> tuple[list[float], list[float]] | None:
        """Bounding box (min, max corners) of occupied voxels around a point, above the floor."""
        c = np.asarray(center, float)
        lo, _ = self.index((c - [radius, radius, 0.0])[None])
        hi, _ = self.index((c + [radius, radius, 0.22])[None])
        lo, hi = np.clip(lo[0], 0, None), np.minimum(hi[0], [self.n_xy - 1, self.n_xy - 1, self.n_z - 1])
        k0 = max(int(lo[2]), int((max(c[2] - 0.02, self.z0) - self.z0) / self.voxel) + 1)
        sub = self.logodds[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1, k0:hi[2] + 1] > OCCUPIED
        idx = np.argwhere(sub)
        if len(idx) < 3:
            return None
        idx += [lo[0], lo[1], k0]
        mn, mx = self.center_of(idx.min(0)) - self.voxel / 2, self.center_of(idx.max(0)) + self.voxel / 2
        return [round(float(v), 3) for v in mn], [round(float(v), 3) for v in mx]

    def floor_grid(self, z_lo: float = 0.01, z_hi: float = 0.12) -> np.ndarray:
        """2D obstacle grid at Vector's height: 1 occupied, -1 seen free, 0 unknown."""
        k0 = max(0, int((z_lo - self.z0) / self.voxel))
        k1 = min(self.n_z, int((z_hi - self.z0) / self.voxel) + 1)
        band = self.logodds[:, :, k0:k1]
        out = np.zeros(band.shape[:2], np.int8)
        out[(band < -2).any(2)] = -1
        out[(band > OCCUPIED).any(2)] = 1
        return out

    def stats(self) -> dict:
        occ = int((self.logodds > OCCUPIED).sum())
        free = int((self.logodds < -2).sum())
        return {"occupied": occ, "free": free, "voxel": self.voxel, "version": self.version}

    # ── persistence ────────────────────────────────────────────

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, logodds=self.logodds, color=self.color, origin=self.origin,
                            voxel=self.voxel, z0=self.z0)            # freshness isn't saved: old maps start "old"
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> "VoxelMap":
        z = np.load(path)
        m = cls.__new__(cls)
        m.logodds, m.color = z["logodds"], z["color"]
        m.origin, m.voxel, m.z0 = z["origin"], float(z["voxel"]), float(z["z0"])
        m.n_xy, m.n_z = m.logodds.shape[0], m.logodds.shape[2]
        m.stamp = np.zeros(m.logodds.shape, np.uint16)
        m.frame, m.version = 1000, 1
        return m
