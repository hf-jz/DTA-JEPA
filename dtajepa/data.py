"""Dataset generation for DTA-JEPA's benchmark suites.

Every suite mirrors one AdaJEPA shift family:

  push_four_shapes  : PushObj-TLZ+- analogue  (train 4 shapes, eval 7)
  push_T_visual     : PushT visual-shift analogue (train on T, eval corrupted T)
  maze_medium       : PointMaze-Medium analogue (one layout, eval default/low-mass/high-damping)
  maze_diverse      : Diverse-PointMaze analogue (25 train layouts, 5 held-out layouts)

Trajectories are stored as (obs uint8 [N,T,64,64,3], act float32 [N,T-1,2],
state float32 [N,T,D]) in compressed .npz. Eval goals are always a *future frame
of the same held-out trajectory* (goal_source='segments' in AdaJEPA), sampled 25
steps ahead with a contact/motion filter so the target is non-trivial.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from .envs import MazeEnv, PushEnv
from .sim import OOD_SHAPES, TRAIN_SHAPES, bfs_dist, maze_layout

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
HISTORY = 3
FUTURE = 4
GOAL_GAP = 25                      # eval goals are this many steps ahead
PUSH_STEPS = 44
MAZE_STEPS = 56


# ---------------------------------------------------------------------------
# PushBlock rollouts
# ---------------------------------------------------------------------------
def _push_rollout(shape, rng, steps=PUSH_STEPS, dyn=None, colors=None, expert_p=0.7):
    """One trajectory: mostly expert pushing, interleaved with exploratory pushes."""
    env = PushEnv(shape, dynamics=dyn, colors=colors, seed=int(rng.integers(1 << 30)))
    env.reset()
    goal = env.sim.s.copy()
    d = rng.uniform(0.25, 0.55)
    th = rng.uniform(-np.pi, np.pi)
    goal[4:6] = np.clip(goal[4:6] + d * np.array([np.cos(th), np.sin(th)]), -0.55, 0.55)
    obs, acts, states = [env.obs()], [], [env.sim.s.copy()]
    for t in range(steps):
        if rng.random() < expert_p:
            a = env.expert_action(goal)
        else:                                    # exploration: aim somewhere else
            g2 = env.sim.s.copy()
            g2[4:6] = np.clip(g2[4:6] + rng.uniform(-0.35, 0.35, 2), -0.6, 0.6)
            a = env.expert_action(g2)
        if t % 12 == 11:                         # resample a fresh target regularly
            goal[4:6] = np.clip(env.sim.s[4:6] + rng.uniform(-0.4, 0.4, 2), -0.55, 0.55)
        s = env.step(a)
        obs.append(env.obs())
        acts.append(a)
        states.append(s.copy())
    return np.array(obs), np.array(acts, dtype=np.float32), np.array(states, dtype=np.float32)


def gen_push(shape, n, seed, dyn=None, colors=None):
    rng = np.random.default_rng(seed)
    O, A, S = [], [], []
    for i in range(n):
        o, a, s = _push_rollout(shape, rng, dyn=dyn, colors=colors)
        O.append(o); A.append(a); S.append(s)
    return np.stack(O), np.stack(A), np.stack(S)


# ---------------------------------------------------------------------------
# PointMaze rollouts
# ---------------------------------------------------------------------------
def _maze_rollout(walls, rng, steps=MAZE_STEPS, mass_scale=1.0, damping_scale=1.0,
                  colors=None, min_dist=3, max_dist=5):
    env = MazeEnv(walls, mass_scale=mass_scale, damping_scale=damping_scale,
                  colors=colors, seed=int(rng.integers(1 << 30)))
    free = np.argwhere(~walls)
    start_cell = tuple(free[rng.integers(len(free))])
    dist = bfs_dist(walls, start_cell)
    cand = np.argwhere((dist >= min_dist) & (dist <= max_dist))
    if len(cand) == 0:
        cand = np.argwhere(dist >= 1)
    goal_cell = tuple(cand[rng.integers(len(cand))])
    start = np.concatenate([env.sim.cell_center(*start_cell), [0, 0]])
    goal = np.concatenate([env.sim.cell_center(*goal_cell), [0, 0]])
    env.goal = goal
    env.reset(state=start)
    obs, acts, states = [env.obs(goal=goal)], [], [np.concatenate([start, goal[:2]])]
    for t in range(steps):
        a = env.expert_action(goal, noise=0.18 if rng.random() < 0.4 else 0.0)
        s = env.step(a)
        obs.append(env.obs(goal=goal))
        acts.append(a)
        states.append(np.concatenate([s, goal[:2]]))
    return np.array(obs), np.array(acts, dtype=np.float32), np.array(states, dtype=np.float32)


def gen_maze(walls, n, seed, mass_scale=1.0, damping_scale=1.0, colors=None):
    rng = np.random.default_rng(seed)
    O, A, S = [], [], []
    for i in range(n):
        o, a, s = _maze_rollout(walls, rng, mass_scale=mass_scale,
                                damping_scale=damping_scale, colors=colors)
        O.append(o); A.append(a); S.append(s)
    return np.stack(O), np.stack(A), np.stack(S)


# ---------------------------------------------------------------------------
# dataset assembly
# ---------------------------------------------------------------------------
def save(name, obs, act=None, state=None, meta=None):
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, f"{name}.npz")
    payload = dict(obs=obs)
    if act is not None:
        payload["act"] = act
    if state is not None:
        payload["state"] = state
    if meta:
        payload["meta"] = np.array([meta], dtype=object)
    np.savez_compressed(path, **payload)
    print(f"  saved {path}  obs={np.shape(obs)} {np.asarray(obs).nbytes/1e6:.0f}MB raw", flush=True)
    return path


def load(name):
    z = np.load(os.path.join(DATA, f"{name}.npz"), allow_pickle=True)
    meta = z["meta"][0] if "meta" in z else {}
    return z["obs"], z["act"], z["state"], meta


def _maze_layouts(seed=0, n_train=25, n_test=5):
    rng = np.random.default_rng(seed)
    train, test = [], []
    for _ in range(n_train):
        train.append(maze_layout(rng))
    for _ in range(n_test):
        test.append(maze_layout(rng))
    return np.stack(train), np.stack(test)


def build_all(small=False, n_push=350, n_maze=16, n_eval=30, seed=0, only="all"):
    """Generate every suite used in the paper."""
    nt, nv = (60, 8) if small else (n_push, n_maze)
    ne = 8 if small else n_eval
    if only in ("all", "push"):
        print("[1/4] push_four_shapes (train T,L,Z,+ | eval all 7 shapes)")
        O, A, S, tags = [], [], [], []
        for k, sh in enumerate(TRAIN_SHAPES):
            o, a, s = gen_push(sh, nt, seed + 100 * (k + 1))
            O.append(o); A.append(a); S.append(s); tags += [sh] * len(o)
        save("push_four_shapes_train", np.concatenate(O), np.concatenate(A),
             np.concatenate(S), {"shapes": tags, "shapes_list": TRAIN_SHAPES})
        for k, sh in enumerate(TRAIN_SHAPES + OOD_SHAPES):
            o, a, s = gen_push(sh, ne, seed + 7000 + 100 * k)
            save(f"push_eval_{sh}", o, a, s, {"shape": sh})

        print("[2/4] push_T_visual (eval: blur/snp/dark/red agent/red block/red anchor)")
        o, a, s = gen_push("T", ne + 20, seed + 3333)
        save("push_eval_T_visual", o, a, s, {"shape": "T"})

    if only in ("all", "maze"):
        print("[3/4] maze_diverse (25 train layouts | 5 held-out layouts)")
        layouts, test_layouts = _maze_layouts(seed + 11)
        O, A, S, tags = [], [], [], []
        for k, w in enumerate(layouts):
            o, a, s = gen_maze(w, nv, seed + 4000 + k)
            O.append(o); A.append(a); S.append(s); tags += [k] * len(o)
        save("maze_diverse_train", np.concatenate(O), np.concatenate(A),
             np.concatenate(S), {"layouts": tags})
        np.savez_compressed(os.path.join(DATA, "maze_layouts.npz"),
                            train=layouts, test=test_layouts)
        for k, w in enumerate(test_layouts):
            o, a, s = gen_maze(w, max(12, ne // 2), seed + 9000 + k)
            save(f"maze_layout_test_{k}", o, a, s, {"layout": k})

        print("[4/4] maze_medium (dynamics shift eval: default | mass x0.2 | damping x20)")
        walls = np.load(os.path.join(DATA, "maze_layouts.npz"))["train"][0]
        save("maze_medium_layout", walls)
        for tag, ms, ds in (("default", 1.0, 1.0), ("lowmass", 0.2, 1.0), ("highdamp", 1.0, 20.0)):
            o, a, s = gen_maze(walls, ne, seed + 5555, mass_scale=ms, damping_scale=ds)
            save(f"maze_dyneval_{tag}", o, a, s, {"mass_scale": ms, "damping_scale": ds})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", action="store_true")
    ap.add_argument("--only", default="all", choices=["all", "push", "maze"])
    ap.add_argument("--n_push", type=int, default=350)
    ap.add_argument("--n_maze", type=int, default=16)
    ap.add_argument("--n_eval", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    build_all(a.small, a.n_push, a.n_maze, a.n_eval, a.seed, a.only)
