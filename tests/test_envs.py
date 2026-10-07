"""Smoke test: physics + expert + rendering. Saves a montage for visual inspection."""
import os
import sys
import time

import numpy as np

sys.path.insert(0, "/Users/sheng/LLM/DTA-JEPA")
from dtajepa.envs import MazeEnv, PushEnv, corrupt
from dtajepa.sim import OOD_SHAPES, TRAIN_SHAPES, maze_layout

OUT = "/Users/sheng/LLM/DTA-JEPA/scratch"
os.makedirs(OUT, exist_ok=True)


def rollout_push(shape, seed=0, steps=60, rng=None, dyn=None):
    rng = rng or np.random.default_rng(seed)
    env = PushEnv(shape, dynamics=dyn, seed=seed)
    env.reset()
    goal = env.sim.s.copy()
    goal[4:6] = np.clip(goal[4:6] + rng.uniform(-0.6, 0.6, 2), -0.6, 0.6)
    goal[6] = goal[6] + rng.uniform(-np.pi, np.pi)
    frames, states, acts, reached = [], [], [], 0
    for t in range(steps):
        a = env.expert_action(goal)
        s = env.step(a)
        frames.append(env.obs())
        states.append(s.copy())
        acts.append(a)
        if env.success(goal) and reached == 0:
            reached = t + 1
    return np.array(frames), np.array(states), np.array(acts), goal, reached


if __name__ == "__main__":
    t0 = time.time()
    tiles = []
    succ = []
    for sh in TRAIN_SHAPES + OOD_SHAPES:
        for seed in range(4):
            f, s, a, g, r = rollout_push(sh, seed=seed)
            succ.append(r > 0)
            if seed == 0:
                tiles += [f[0], f[-1]]
        print(f"push {sh:7s} expert success(4 trials)={sum(succ[-4:])}/4")
    print("overall expert success: %d/%d" % (sum(succ), len(succ)))

    base = rollout_push("T", seed=1)[0][0]
    for kind in ("blur", "snp", "dark"):
        tiles.append(corrupt(base, kind, 0.9, np.random.default_rng(0)))
    tiles.append(base)

    rng = np.random.default_rng(3)
    walls = maze_layout(rng)
    env = MazeEnv(walls, seed=0)
    env.reset()
    free = np.argwhere(~walls)
    goal = np.concatenate([env.sim.cell_center(*free[rng.integers(len(free))]), [0, 0]])
    frames, ok = [], False
    for t in range(120):
        s = env.step(env.expert_action(goal))
        frames.append(env.obs(goal=goal))
        if env.success(goal):
            ok = True
            break
    print("maze expert success:", ok, "steps:", len(frames))
    tiles += [frames[0], frames[-1]]

    n, cols = len(tiles), 6
    rows = (n + cols - 1) // cols
    mont = np.zeros((rows * 64, cols * 64, 3), dtype=np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        mont[r * 64:(r + 1) * 64, c * 64:(c + 1) * 64] = t
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.imsave(f"{OUT}/env_montage.png", mont)
    print("montage ->", f"{OUT}/env_montage.png", "elapsed %.1fs" % (time.time() - t0))
