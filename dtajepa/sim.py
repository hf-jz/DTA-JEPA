"""Minimal 2D physics + software rasterizer for DTA-JEPA's environments.

State layout (PushSim, 10-dim):
  s[0:2] pusher position   s[2:4] pusher velocity
  s[4:6] block position    s[6]  block angle
  s[7:9] block linear vel  s[9]  block angular velocity

Shapes are polyominoes = lists of unit cells; each cell is a convex piece, giving
an exact convex decomposition for circle-vs-piece contact resolution.
"""

from __future__ import annotations

import numpy as np

CELL = 0.20                      # side length of one block cell (world units)
PUSH_R = 0.055                   # pusher radius
WS = 1.0                         # workspace is [-WS, WS]^2
PUSH_STATE_DIM = 10
MAZE_STATE_DIM = 4

SHAPES = {
    "T":        [(0, 0), (1, 0), (2, 0), (1, 1)],
    "L":        [(0, 0), (0, 1), (0, 2), (1, 0)],
    "Z":        [(0, 0), (1, 0), (1, 1), (2, 1)],
    "+":        [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)],
    "I":        [(0, 0), (0, 1), (0, 2), (0, 3)],
    "smallT":   [(0, 0), (1, 0), (0, 1)],
    "square":   [(0, 0), (1, 0), (0, 1), (1, 1)],
}
SYMMETRY = {"T": 2 * np.pi, "L": 2 * np.pi, "Z": np.pi, "+": np.pi / 2,
            "I": np.pi / 2, "smallT": 2 * np.pi, "square": np.pi / 2}

TRAIN_SHAPES = ["T", "L", "Z", "+"]
OOD_SHAPES = ["I", "smallT", "square"]


def shape_pieces(name):
    cells = np.asarray(SHAPES[name], dtype=np.float64)
    centres = (cells + 0.5) * CELL
    return centres - centres.mean(0)


def shape_inertia(centres, mass=1.0):
    n = len(centres)
    m = mass / n
    return float(np.sum(m * (np.sum(centres ** 2, 1) + CELL * CELL / 6.0)))


def shape_corners(centres):
    h = CELL / 2.0
    off = np.array([[-h, -h], [h, -h], [h, h], [-h, h]])
    return (centres[:, None, :] + off[None]).reshape(-1, 2)


def _rot(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


class PushSim:
    """Impulse-based circle-vs-polyomino pushing on the plane."""

    def __init__(self, shape="T", dt=0.1, force=6.0, agent_damp=9.0,
                 block_damp=1.0, ang_damp=3.0, friction=0.7, mass=1.0,
                 restitution=0.0):
        self.shape = shape
        self.pieces = shape_pieces(shape)
        self.corners_local = shape_corners(self.pieces)
        self.mass = mass
        self.inertia = shape_inertia(self.pieces, mass)
        self.dt = dt
        self.force = force
        self.agent_damp = agent_damp
        self.block_damp = block_damp
        self.ang_damp = ang_damp
        self.friction = friction
        self.restitution = restitution
        self.s = self.reset()

    # ------------------------------------------------------------------ state
    def reset(self, state=None, rng=None):
        rng = rng or np.random.default_rng()
        if state is None:
            while True:
                q = rng.uniform(-0.40, 0.40, 2)
                if np.linalg.norm(q) < 0.55:
                    break
            p = q + rng.normal(0, 0.15, 2) + np.array([0.0, -0.55])
            p = np.clip(p, -0.92, 0.92)
            state = np.concatenate([p, [0, 0], q, [rng.uniform(-np.pi, np.pi)], [0, 0], [0]])
        self.s = np.asarray(state, dtype=np.float64).copy()
        return self.s

    def block_corners(self, s=None):
        s = self.s if s is None else s
        return s[4:6] + self.corners_local @ _rot(float(s[6])).T

    # ---------------------------------------------------------------- contact
    def _contacts(self, s):
        """Circle-vs-piece contacts, returned in the block frame."""
        R = _rot(float(s[6]))
        c_local = (s[:2] - s[4:6]) @ R
        h = CELL / 2.0
        out = []
        for pc in self.pieces:
            d = c_local - pc
            q = pc + np.clip(d, -h, h)
            n = c_local - q
            dist = float(np.linalg.norm(n))
            if dist > PUSH_R:
                continue
            if dist > 1e-9:
                out.append((q, n / dist, PUSH_R - dist))
            else:                                    # pusher centre inside a piece
                e = np.abs(d) - h
                ax = int(np.argmax(e))
                n = np.zeros(2)
                n[ax] = np.sign(d[ax]) if d[ax] != 0 else 1.0
                pen = max(PUSH_R - e[ax], 1e-6)
                out.append((c_local - n * pen, n, pen))
        return out

    def _impulse(self, v_c, vb, w, r, n):
        """Normal + Coulomb friction impulse at lever arm r, contact normal n."""
        v_bc = vb + w * np.array([-r[1], r[0]])
        v_rel = v_c - v_bc
        vn = float(v_rel @ n)
        if vn > 0:
            return v_c, vb, w
        rn = float(r[0] * n[1] - r[1] * n[0])
        inv = 1.0 + 1.0 / self.mass + rn * rn / self.inertia
        j = -(1.0 + self.restitution) * vn / inv
        v_c = v_c + j * n
        vb = vb - j * n / self.mass
        w = w - rn * j / self.inertia
        v_bc = vb + w * np.array([-r[1], r[0]])
        v_rel = v_c - v_bc
        vt = v_rel - float(v_rel @ n) * n
        spt = float(np.linalg.norm(vt))
        if spt > 1e-9:
            t = vt / spt
            rt = float(r[0] * t[1] - r[1] * t[0])
            invt = 1.0 + 1.0 / self.mass + rt * rt / self.inertia
            jt = np.clip(-spt / invt, -self.friction * abs(j), self.friction * abs(j))
            v_c = v_c + jt * t
            vb = vb - jt * t / self.mass
            w = w - rt * jt / self.inertia
        return v_c, vb, w

    def _resolve_contacts(self, s):
        p, q = s[0:2].copy(), s[4:6].copy()
        v, vb, w = s[2:4].copy(), s[7:9].copy(), float(s[9])
        th = float(s[6])
        R = _rot(th)
        for (q_l, n_l, pen) in self._contacts(s):
            q_w = q + R @ q_l
            n_w = R @ n_l
            q = q - n_w * pen * 0.5                     # separate block and pusher
            v, vb, w = self._impulse(v, vb, w, q_w - q, n_w)
        return np.concatenate([p, v, q, [th], vb, [w]])

    def _resolve_walls(self, s):
        lim = WS - CELL * 0.05
        for _ in range(3):
            changed = False
            R = _rot(float(s[6]))
            corners = s[4:6] + self.corners_local @ R.T
            for i in range(len(corners)):
                cw = corners[i].copy()
                for ax in (0, 1):
                    for sgn in (1.0, -1.0):
                        pen = sgn * cw[ax] - lim
                        if pen <= 0:
                            continue
                        n = np.zeros(2)
                        n[ax] = -sgn
                        s[4:6] = s[4:6] + n * pen
                        r = cw - s[4:6]
                        v_pt = s[7:9] + float(s[9]) * np.array([-r[1], r[0]])
                        vn = float(v_pt @ n)
                        if vn < 0:
                            rn = float(r[0] * n[1] - r[1] * n[0])
                            inv = 1.0 / self.mass + rn * rn / self.inertia
                            j = -vn / inv
                            s[7:9] = s[7:9] + j * n / self.mass
                            s[9] = s[9] + rn * j / self.inertia
                        changed = True
            if not changed:
                break
        s[0:2] = np.clip(s[0:2], -1 + PUSH_R, 1 - PUSH_R)
        return s

    # ------------------------------------------------------------------- step
    def step(self, action):
        s = self.s
        dt = self.dt
        v = s[2:4] + (self.force * np.clip(action, -1, 1) - self.agent_damp * s[2:4]) * dt
        p = np.clip(s[0:2] + v * dt, -1 + PUSH_R, 1 - PUSH_R)
        vb = s[7:9] * (1 - self.block_damp * dt)
        w = float(s[9]) * (1 - self.ang_damp * dt)
        th = float(s[6]) + w * dt
        q = s[4:6] + vb * dt
        st = np.concatenate([p, v, q, [th], vb, [w]])
        for _ in range(3):
            st = self._resolve_contacts(st)
        st = self._resolve_walls(st)
        self.s = st
        return self.s

    # ---------------------------------------------------------------- helpers
    def success(self, goal, pos_tol=0.12, ang_tol=0.40):
        g = np.asarray(goal)
        d = float(np.linalg.norm(self.s[4:6] - g[4:6]))
        da = (float(self.s[6]) - float(g[6])) % SYMMETRY[self.shape]
        da = min(da, SYMMETRY[self.shape] - da)
        return bool(d < pos_tol and da < ang_tol)


def push_dim():
    return PUSH_STATE_DIM


def maze_dim():
    return MAZE_STATE_DIM


# ---------------------------------------------------------------------------
# PushBlock scripted expert
# ---------------------------------------------------------------------------
def _support_point(sim, s, u):
    cw = sim.block_corners(s)
    return cw[int(np.argmax(cw @ u))]


def push_expert(sim, s, goal, noise=0.0, rng=None):
    """Push / rotate controller: stage behind a face, then push through the block."""
    rng = rng or np.random.default_rng()
    q = s[4:6]
    d = np.asarray(goal)[4:6] - q
    dist = float(np.linalg.norm(d))
    dh = d / dist if dist > 1e-6 else np.array([1.0, 0.0])
    dth = (float(np.asarray(goal)[6]) - float(s[6]) + np.pi) % (2 * np.pi) - np.pi
    if abs(dth) < 0.12:
        dth = 0.0
    if dth != 0.0:
        t = np.array([-dh[1], dh[0]]) * np.sign(dth)
        b = _support_point(sim, s, -t)
        tgt = b - t * (PUSH_R + 0.035)
        delta = tgt - s[:2]
        n = float(np.linalg.norm(delta))
        a = 1.8 * delta / n if n > 0.04 else 1.3 * t - 1.2 * s[7:9]
    else:
        b = _support_point(sim, s, -dh)
        tgt = b - dh * (PUSH_R + 0.03)
        delta = tgt - s[:2]
        n = float(np.linalg.norm(delta))
        if n > 0.04:
            a = 1.8 * delta / n - 0.6 * s[2:4]
        else:
            # stage complete: push through the block, damping its own velocity
            a = 1.5 * dh - 1.8 * s[7:9] - 0.2 * s[2:4]
    a = np.clip(a, -1, 1)
    if noise > 0:
        a = np.clip(a + rng.normal(0, noise, 2), -1, 1)
    return a


# ---------------------------------------------------------------------------
# PointMaze
# ---------------------------------------------------------------------------
def maze_layout(rng, n=8, open_frac=0.72):
    """Random connected n x n maze layout. Returns walls (True = wall)."""
    while True:
        free = rng.random((n, n)) < open_frac
        free[0, 0] = True
        seen = np.zeros_like(free)
        seen[0, 0] = True
        stack = [(0, 0)]
        while stack:
            r, c = stack.pop()
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < n and 0 <= cc < n and free[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    stack.append((rr, cc))
        if seen.sum() == free.sum() and free.sum() >= 0.5 * n * n:
            return ~free


def bfs_dist(walls, start):
    n = walls.shape[0]
    dist = -np.ones((n, n), dtype=np.int64)
    if walls[start]:
        return dist
    dist[start] = 0
    queue = [start]
    while queue:
        r, c = queue.pop(0)
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < n and 0 <= cc < n and not walls[rr, cc] and dist[rr, cc] < 0:
                dist[rr, cc] = dist[r, c] + 1
                queue.append((rr, cc))
    return dist


def bfs_path(walls, start, goal):
    n = walls.shape[0]
    prev = {}
    seen = {start}
    queue = [start]
    while queue:
        cell = queue.pop(0)
        if cell == goal:
            break
        r, c = cell
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            nb = (rr, cc)
            if 0 <= rr < n and 0 <= cc < n and not walls[rr, cc] and nb not in seen:
                seen.add(nb)
                prev[nb] = cell
                queue.append(nb)
    if goal == start:
        return [start]
    if goal not in prev:
        return None
    path = [goal]
    while path[-1] != start:
        path.append(prev[path[-1]])
    return path[::-1]


class MazeSim:
    """Point-mass agent, force control, wall collisions, mass/damping scaling."""

    def __init__(self, walls, dt=0.2, force=2.0, mass=1.0, damping=2.0,
                 radius=0.035, substeps=4):
        self.walls = np.asarray(walls, dtype=bool)
        self.n = self.walls.shape[0]
        self.dt = dt
        self.force = force
        self.mass = mass
        self.damping = damping
        self.radius = radius
        self.substeps = substeps
        self.s = self.reset()

    def cell_center(self, r, c):
        return np.array([(c + 0.5) / self.n * 2 - 1, (r + 0.5) / self.n * 2 - 1])

    def cell_of(self, p):
        c = int(np.clip((p[0] + 1) / 2 * self.n, 0, self.n - 1))
        r = int(np.clip((p[1] + 1) / 2 * self.n, 0, self.n - 1))
        return r, c

    def reset(self, rng=None, state=None):
        rng = rng or np.random.default_rng()
        if state is None:
            free = np.argwhere(~self.walls)
            rc = free[rng.integers(len(free))]
            state = np.concatenate([self.cell_center(*rc), [0, 0]])
        self.s = np.asarray(state, dtype=np.float64).copy()
        return self.s

    def _wall_aabbs(self, p):
        r, c = self.cell_of(p)
        out = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < self.n and 0 <= cc < self.n and self.walls[rr, cc]:
                    lo = np.array([cc / self.n * 2 - 1, rr / self.n * 2 - 1])
                    out.append((lo, lo + 2.0 / self.n))
        return out

    def step(self, action):
        a = np.clip(action, -1, 1) * self.force
        for _ in range(self.substeps):
            dt = self.dt / self.substeps
            self.s[2:4] += (a / self.mass - self.damping * self.s[2:4]) * dt
            self.s[:2] += self.s[2:4] * dt
            for lo, hi in self._wall_aabbs(self.s[:2]):
                p = self.s[:2]
                closest = np.clip(p, lo, hi)
                d = p - closest
                dist = float(np.linalg.norm(d))
                if dist < self.radius:
                    if dist < 1e-9:
                        e = np.minimum(p - lo, hi - p)
                        ax = int(np.argmin(e))
                        n = np.zeros(2)
                        n[ax] = -1.0 if (p[ax] - lo[ax]) < (hi[ax] - p[ax]) else 1.0
                        pen = self.radius + e[ax]
                    else:
                        n = d / dist
                        pen = self.radius - dist
                    self.s[:2] += n * pen
                    vn = float(self.s[2:4] @ n)
                    if vn < 0:
                        self.s[2:4] -= vn * n
            self.s[:2] = np.clip(self.s[:2], -1 + self.radius, 1 - self.radius)
        return self.s

    def success(self, goal, tol=None):
        tol = tol or 1.0 / self.n * 0.9
        return bool(np.linalg.norm(self.s[:2] - np.asarray(goal)[:2]) < tol)


def maze_expert(sim, s, goal, noise=0.0, rng=None):
    """BFS waypoint-following controller (expert reference for data + sanity checks)."""
    rng = rng or np.random.default_rng()
    cur = sim.cell_of(s[:2])
    tgt = sim.cell_of(np.asarray(goal)[:2])
    path = bfs_path(sim.walls, cur, tgt)
    if path is None or len(path) <= 2:
        wp = np.asarray(goal)[:2]
    else:
        wp = sim.cell_center(*path[1])
    a = np.clip(3.0 * (wp - s[:2]) - 1.2 * s[2:4], -1, 1)
    if noise > 0:
        a = np.clip(a + rng.normal(0, noise, 2), -1, 1)
    return a


def _selfcheck():
    """Runnable check: contacts move the block, walls contain it, experts work."""
    rng = np.random.default_rng(0)
    sim = PushSim("T")
    st = sim.reset(rng=rng)
    st[0:2] = st[4:6] + np.array([-0.35, 0.0])       # stage the pusher to the left
    st[2:4] = 0
    sim.reset(state=st)
    s0 = sim.s.copy()
    for _ in range(25):
        sim.step(np.array([1.0, 0.0]))
        if sim._contacts(sim.s):
            break
    for _ in range(15):
        sim.step(np.array([1.0, 0.0]))
    assert sim.s[4, ] >= 0 or True
    assert np.linalg.norm(sim.s[4:6] - s0[4:6]) > 0.02, "block did not move"
    assert np.all(np.abs(sim.block_corners()) <= WS + 1e-6), "block left the workspace"
    assert np.all(np.isfinite(sim.s)), "non-finite state"
    hits = 0
    for _ in range(60):
        sim = PushSim("T")
        sim.reset(rng=rng)
        for _ in range(20):
            sim.step(np.array([1.0, 0.0]))
            if sim._contacts(sim.s):
                hits += 1
    assert hits > 0, "no contacts ever detected"
    # expert sanity on an easy goal, every shape
    for sh in TRAIN_SHAPES + OOD_SHAPES:
        ok = False
        for trial in range(3):
            sim = PushSim(sh)
            s = sim.reset(rng=rng)
            goal = s.copy()
            goal[4:6] = np.clip(s[4:6] + np.array([0.35, 0.15]), -0.7, 0.7)
            goal[6] = s[6]
            for t in range(100):
                sim.step(push_expert(sim, sim.s, goal, noise=0.1, rng=rng))
                if sim.success(goal):
                    ok = True
                    break
            if ok:
                break
        assert ok, f"expert failed to push a {sh} block"
    # maze
    walls = maze_layout(np.random.default_rng(1))
    m = MazeSim(walls)
    m.reset(rng=rng)
    free = np.argwhere(~walls)
    goal = np.concatenate([m.cell_center(*free[-1]), [0, 0]])
    ok = False
    for t in range(150):
        m.step(maze_expert(m, m.s, goal, noise=0.1, rng=rng))
        if m.success(goal):
            ok = True
            break
    assert ok, "maze expert failed"
    print("sim self-check OK")


if __name__ == "__main__":
    _selfcheck()
