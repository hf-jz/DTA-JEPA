"""Environments: observation rendering, shift knobs, and image corruptions.

Observation = 64x64 RGB uint8. All entities are colour-coded exactly like the
AdaJEPA setup so that the three colour shifts (redAgent / redBlock / redAnchor)
transfer directly.
"""

from __future__ import annotations

import numpy as np

from .sim import CELL, PUSH_R, SHAPES, WS, MazeSim, PushSim, maze_expert, push_expert, shape_corners, shape_pieces, _rot

RES = 64
SS = 2                                   # supersampling factor

DEFAULT_COLORS = {
    "block": (0.62, 0.62, 0.62),
    "agent": (0.16, 0.36, 0.92),
    "anchor": (0.20, 0.68, 0.34),
    "goal": (0.20, 0.68, 0.34),
    "wall": (0.30, 0.30, 0.34),
    "bg": 0.93,
}
RED = (0.88, 0.16, 0.14)


def _grid():
    n = RES * SS
    lin = np.linspace(-1.0, 1.0, n)
    X, Y = np.meshgrid(lin, lin[::-1])       # Y decreases with the row index
    return X, Y


_X, _Y = _grid()


def _fill_quad(canvas, corners, colour, X=None, Y=None):
    """Rasterise a convex quad (4x2 world corners) onto the supersampled canvas."""
    X = _X if X is None else X
    Y = _Y if Y is None else Y
    inside = np.ones_like(X, dtype=bool)
    for i in range(4):
        a, b = corners[i], corners[(i + 1) % 4]
        e = b - a
        # consistent orientation for a convex polygon ordered around the boundary
        inside &= (e[0] * (Y - a[1]) - e[1] * (X - a[0])) <= 0
    if inside.any():
        canvas[inside] = colour
    return canvas


def _fill_disc(canvas, centre, radius, colour, X=None, Y=None):
    X = _X if X is None else X
    Y = _Y if Y is None else Y
    m = (X - centre[0]) ** 2 + (Y - centre[1]) ** 2 <= radius ** 2
    canvas[m] = colour
    return canvas


def _down(canvas, bg):
    img = canvas.reshape(RES, SS, RES, SS, 3).mean(axis=(1, 3))
    return img


def render_push(state, shape, colours=None, ref_pose=(0.0, 0.58, 0.0)):
    """Render a PushBlock frame. `ref_pose` = static reference marker (visual only)."""
    c = dict(DEFAULT_COLORS)
    c.update(colours or {})
    canvas = np.full((RES * SS, RES * SS, 3), c["bg"], dtype=np.float32)
    # static reference marker (green T by default; the anchor colour shift recolours it)
    rc = shape_pieces("T")
    Rr = _rot(ref_pose[2])
    for pc in rc:
        corners = (np.array(ref_pose[:2]) + (shape_corners(np.array([pc])) @ Rr.T)).reshape(4, 2)
        _fill_quad(canvas, corners, c["anchor"])
    # movable block
    R = _rot(state[5])
    for pc in shape_pieces(shape):
        corners = (state[4:6] + (shape_corners(np.array([pc])) @ R.T)).reshape(4, 2)
        _fill_quad(canvas, corners, c["block"])
    _fill_disc(canvas, state[:2], PUSH_R, c["agent"])
    return (_down(canvas, c["bg"]) * 255).astype(np.uint8)


def render_maze(state, walls, goal, colours=None):
    c = dict(DEFAULT_COLORS)
    c.update(colours or {})
    canvas = np.full((RES * SS, RES * SS, 3), c["bg"], dtype=np.float32)
    for lo, hi in _wall_aabbs_all(walls):
        corners = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])
        _fill_quad(canvas, corners, c["wall"])
    _fill_disc(canvas, np.asarray(goal)[:2], 0.09, c["goal"])
    _fill_disc(canvas, state[:2], 0.055, c["agent"])
    return (_down(canvas, c["bg"]) * 255).astype(np.uint8)


def _wall_aabbs_all(walls):
    out = []
    n = walls.shape[0]
    for r in range(n):
        for cc in range(n):
            if walls[r, cc]:
                lo = np.array([cc / n * 2 - 1, r / n * 2 - 1])
                out.append((lo, lo + 2.0 / n))
    return out


# ----------------------------------------------------------------------------
# image-level corruptions (AdaJEPA's visual shifts)
# ----------------------------------------------------------------------------
def corrupt(obs: np.ndarray, kind: str, level: float = 0.9, rng=None) -> np.ndarray:
    """Apply a test-time visual corruption to a uint8 HxWx3 frame."""
    rng = rng or np.random.default_rng(0)
    x = obs.astype(np.float32) / 255.0
    if kind == "blur":
        r = max(1, int(round(level * 3)))
        k = np.exp(-0.5 * (np.arange(-r, r + 1) / max(level, 1e-3)) ** 2)
        k /= k.sum()
        for ax in (0, 1):
            x = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), ax, x)
    elif kind in ("snp1", "snp5", "snp"):
        p = {"snp1": 0.02, "snp5": 0.05, "snp": 0.03}[kind] * (level / 0.9)
        m = rng.random(x.shape[:2])
        x[m < p / 2] = 0.0
        x[(m >= p / 2) & (m < p)] = 1.0
    elif kind == "dark":
        x = x * (1.0 - 0.72 * level)
    else:
        raise ValueError(kind)
    return np.clip(x * 255, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------------
# environments
# ----------------------------------------------------------------------------
class PushEnv:
    """PushBlock env with dynamics and colour shift knobs (AdaJEPA-style)."""

    def __init__(self, shape="T", dynamics=None, colors=None, expert_noise=0.25, seed=0):
        d = dict(force=6.0, agent_damp=9.0, block_damp=1.0, ang_damp=3.0, friction=0.7)
        d.update(dynamics or {})
        self.sim = PushSim(shape=shape, force=d["force"], agent_damp=d["agent_damp"],
                           block_damp=d["block_damp"], ang_damp=d["ang_damp"],
                           friction=d["friction"], dt=0.1)
        self.shape = shape
        self.colors = colors
        self.rng = np.random.default_rng(seed)
        self.expert_noise = expert_noise

    def reset(self, state=None):
        return self.sim.reset(state, self.rng)

    def step(self, action):
        return self.sim.step(np.asarray(action, dtype=np.float64))

    def obs(self, state=None):
        return render_push(self.sim.s if state is None else state, self.shape, self.colors)

    def expert_action(self, goal, noise=None):
        return push_expert(self.sim, self.sim.s, goal,
                           noise=self.expert_noise if noise is None else noise, rng=self.rng)

    def success(self, goal):
        return self.sim.success(goal)


class MazeEnv:
    """PointMaze env with mass/damping dynamics shifts and layout shift."""

    def __init__(self, walls, mass_scale=1.0, damping_scale=1.0, colors=None, seed=0,
                 expert_noise=0.15):
        self.sim = MazeSim(walls, mass=1.0 * mass_scale, damping=2.0 * damping_scale)
        self.walls = np.asarray(walls, dtype=bool)
        self.colors = colors
        self.rng = np.random.default_rng(seed)
        self.expert_noise = expert_noise
        self.goal = None

    def reset(self, state=None):
        return self.sim.reset(self.rng, state)

    def step(self, action):
        return self.sim.step(np.asarray(action, dtype=np.float64))

    def obs(self, state=None, goal=None):
        g = goal if goal is not None else (self.goal if self.goal is not None else np.zeros(4))
        return render_maze(self.sim.s if state is None else state, self.walls, g, self.colors)

    def expert_action(self, goal, noise=None):
        return maze_expert(self.sim, self.sim.s, goal,
                           noise=self.expert_noise if noise is None else noise, rng=self.rng)

    def success(self, goal):
        return self.sim.success(goal)
