"""Evaluation harness: AdaJEPA's four shift families + continual (sequential) suites.

Every condition runs closed-loop MPC on real environments (one episode at a time,
each with its own adapted model copy), compares Frozen / AdaJEPA / DTA-JEPA and the
DTA-JEPA ablations, and logs success after every MPC replanning step, final success,
adaptation statistics and wall-clock latency.

Usage
-----
  python -m dtajepa.evaluate --suite shape --n_ep 30 --methods frozen adajepa dtajepa
  python -m dtajepa.evaluate --suite all --methods frozen adajepa dtajepa --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import time

import numpy as np
import torch

from . import data as D
from .adapt import Adapter, Controller, EpisodicMemory, ParamMemory, SafetyLayer
from .envs import MazeEnv, PushEnv, corrupt, render_maze, render_push
from .models import fold_lora
from .plan import MPCConfig, cem_plan, gd_plan
from .sim import OOD_SHAPES, TRAIN_SHAPES
from .train import CKPT, load_model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RED = (0.88, 0.16, 0.14)

METHODS = {
    "frozen":        dict(mode="frozen"),
    "adajepa":       dict(mode="adajepa", lr=5e-4, enc_lr=1e-5, steps=1, buffer=5, fold=True),
    "dtajepa":       dict(mode="dtajepa", lr=5e-4, enc_lr=1e-5, steps=1, buffer=5,
                          memory=256, param_memory=True, safety=True, controller=True,
                          adaptive_compute=True, slow=True, w_unc=0.5, w_safe=0.5, fold=True),
    "dtajepa_lora":  dict(mode="dtajepa", lr=5e-4, enc_lr=1e-5, steps=1, buffer=5,
                          memory=256, param_memory=True, safety=True, controller=True,
                          adaptive_compute=True, slow=True, w_unc=0.5, w_safe=0.5, fold=False),
    "dtajepa_nounc": dict(mode="dtajepa", fold=True, lr=5e-4, enc_lr=1e-5, steps=1, buffer=5,
                          memory=256, param_memory=True, safety=True, controller=False,
                          adaptive_compute=False, slow=True, w_unc=0.0, w_safe=0.0),
    "dtajepa_nomem": dict(mode="dtajepa", fold=True, lr=5e-4, enc_lr=1e-5, steps=1, buffer=5,
                          memory=0, param_memory=False, safety=True, controller=True,
                          adaptive_compute=True, slow=False, w_unc=0.5, w_safe=0.5),
    "dtajepa_nosafe": dict(mode="dtajepa", fold=True, lr=5e-4, enc_lr=1e-5, steps=1, buffer=5,
                           memory=256, param_memory=True, safety=False, controller=True,
                           adaptive_compute=True, slow=True, w_unc=0.5, w_safe=0.0),
    "dtajepa_fixedk": dict(mode="dtajepa", fold=True, lr=5e-4, enc_lr=1e-5, steps=1, buffer=5,
                           memory=256, param_memory=True, safety=True, controller=True,
                           adaptive_compute=False, slow=True, w_unc=0.5, w_safe=0.5),
    "dtajepa_noslow": dict(mode="dtajepa", fold=True, lr=5e-4, enc_lr=1e-5, steps=1, buffer=5,
                           memory=256, param_memory=False, safety=True, controller=True,
                           adaptive_compute=True, slow=False, w_unc=0.5, w_safe=0.5),
    "no_adapt":      dict(mode="frozen"),        # alias used in the ablations table
}

_MODEL_CACHE = {}


def get_model(name, fold=True):
    key = (name, bool(fold))
    if key not in _MODEL_CACHE:
        m, ck = load_model(name)
        if fold:
            fold_lora(m)
        m.eval()
        _MODEL_CACHE[key] = (m, ck)
    return _MODEL_CACHE[key]


# ---------------------------------------------------------------------------
# episode containers
# ---------------------------------------------------------------------------
def make_episode(**kw):
    ep = type("Episode", (), {})()
    ep.__dict__.update(kw)
    ep.last_actions = None
    return ep


# ---------------------------------------------------------------------------
# episode builders
# ---------------------------------------------------------------------------
def _obs_push(shape, state, colors, corrupt_kind, corrupt_level, seed=0):
    frame = render_push(state, shape, colors)
    if corrupt_kind:
        frame = corrupt(frame, corrupt_kind, corrupt_level,
                        np.random.default_rng(abs(hash((seed, corrupt_kind))) % (1 << 31)))
    return frame


def _obs_maze(walls, state, goal, colors, corrupt_kind=None, corrupt_level=0.9, seed=0):
    frame = render_maze(state, walls, np.concatenate([np.asarray(goal)[:2], [0, 0]]), colors)
    if corrupt_kind:
        frame = corrupt(frame, corrupt_kind, corrupt_level,
                        np.random.default_rng(abs(hash((seed, corrupt_kind))) % (1 << 31)))
    return frame


def build_push_episodes(eval_file, n_ep, seed, shape, dynamics=None, colors=None,
                        corrupt_kind=None, corrupt_level=0.9, goal_gap=D.GOAL_GAP,
                        min_move=0.06, k=D.HISTORY):
    obs, act, state, meta = D.load(eval_file)
    N, T = obs.shape[:2]
    rng = np.random.default_rng(seed)
    eps, tries = [], 0
    while len(eps) < n_ep and tries < n_ep * 60:
        tries += 1
        j = int(rng.integers(N))
        t0 = int(rng.integers(k, max(k + 1, T - goal_gap)))
        t1 = t0 + goal_gap
        if t1 >= T:
            continue
        if np.linalg.norm(state[j, t1, 4:6] - state[j, t0, 4:6]) < min_move:
            continue
        env = PushEnv(shape, dynamics=dynamics, colors=colors,
                      seed=int(rng.integers(1 << 30)))
        env.reset(state[j, t0])
        goal_state = state[j, t1].copy()
        eps.append(make_episode(
            env=env, kind="push", shape=shape, start=state[j, t0].copy(),
            goal_state=goal_state,
            goal_obs=_obs_push(shape, goal_state, colors, corrupt_kind, corrupt_level, len(eps)),
            goal_prop=goal_state[:4].astype(np.float32),
            hist_obs=np.stack([_obs_push(shape, state[j, t0 - k + 1 + a], colors, corrupt_kind,
                                         corrupt_level, len(eps)) for a in range(k)]),
            hist_act=act[j, t0 - k + 1:t0 + 1].copy(),
            hist_prop=state[j, t0 - k + 1:t0 + 1, :4].copy(),
            done=False, first_success=None, steps=0))
    if tries >= n_ep * 60:
        print(f"    warning: only {len(eps)}/{n_ep} push episodes passed the motion filter",
              flush=True)
    return eps


def build_maze_episodes(eval_file, n_ep, seed, walls, mass_scale=1.0, damping_scale=1.0,
                        colors=None, corrupt_kind=None, corrupt_level=0.9, k=D.HISTORY):
    obs, act, state, meta = D.load(eval_file)
    N, T = obs.shape[:2]
    rng = np.random.default_rng(seed)
    eps = []
    for i in range(n_ep):
        j = int(rng.integers(N))
        t0 = int(rng.integers(k, max(k + 1, T // 2)))
        goal = state[j, t0, 4:6].copy()
        start = state[j, t0, :4].copy().astype(np.float32)
        env = MazeEnv(walls, mass_scale=mass_scale, damping_scale=damping_scale,
                      colors=colors, seed=int(rng.integers(1 << 30)))
        env.goal = np.concatenate([goal, [0, 0]])
        env.reset(state=start)
        goal_state = np.concatenate([goal, [0, 0]]).astype(np.float32)
        eps.append(make_episode(
            env=env, kind="maze", start=start, goal_state=goal_state,
            goal_obs=_obs_maze(walls, goal_state, goal, colors, corrupt_kind, corrupt_level, i),
            goal_prop=goal_state[:4],
            hist_obs=np.stack([_obs_maze(walls, state[j, t0 - k + 1 + a, :4], goal, colors,
                                         corrupt_kind, corrupt_level, i) for a in range(k)]),
            hist_act=act[j, t0 - k + 1:t0 + 1].copy(),
            hist_prop=state[j, t0 - k + 1:t0 + 1, :4].copy(),
            done=False, first_success=None, steps=0))
    return eps


# ---------------------------------------------------------------------------
# one (condition, method) run
# ---------------------------------------------------------------------------
class Runner:
    def __init__(self, method, model, cfg, pmean, pstd, zmean=None, zstd=None,
                 maha_thresh=None):
        self.m = METHODS[method]
        self.name = method
        self.base = model
        self.cfg = cfg
        self.pmean, self.pstd = np.asarray(pmean, dtype=np.float32), np.asarray(pstd, dtype=np.float32)
        self.zmean = None if zmean is None else torch.as_tensor(zmean, dtype=torch.float32)
        self.zstd = None if zstd is None else torch.as_tensor(zstd, dtype=torch.float32)
        self.maha_thresh = maha_thresh
        self.device = torch.device("cpu")
        self.memory = EpisodicMemory(capacity=self.m.get("memory", 0)) \
            if self.m.get("memory", 0) else None
        self.param_memory = ParamMemory(capacity=8) if self.m.get("param_memory") else None
        self.timing = {"plan": 0.0, "env": 0.0, "adapt": 0.0}

    def make_model(self):
        if self.m["mode"] == "frozen":
            return self.base
        return copy.deepcopy(self.base)

    def make_adapter(self, model, k):
        if self.m["mode"] == "frozen":
            return None
        safety = SafetyLayer(model, eps_anchor=self.cfg.anchor_eps,
                             val_tol=self.cfg.val_tol) if self.m.get("safety") else None
        ctrl = Controller() if self.m.get("controller") else None
        return Adapter(model, mode=self.m["mode"], lr=self.m["lr"], enc_lr=self.m["enc_lr"],
                       steps=self.m["steps"], buffer=self.m["buffer"], memory=self.memory,
                       fold=self.m.get("fold", True),
                       param_memory=self.param_memory, safety=safety, controller=ctrl,
                       n_refine=None, mc_samples=4 if self.m.get("adaptive_compute") else 0)

    # ---- helpers ----------------------------------------------------------
    def encode_batch(self, model, frames, props):
        with torch.no_grad():
            arr = np.asarray(frames)
            if arr.ndim == 4:                       # (N,H,W,C) -> (N,C,H,W)
                arr = np.ascontiguousarray(arr.transpose(0, 3, 1, 2))
            o = torch.as_tensor(arr, dtype=torch.float32, device=self.device) / 255.0
            p = torch.as_tensor((np.asarray(props) - self.pmean) / self.pstd,
                                dtype=torch.float32, device=self.device)
            return model.encode(o, p)

    def plan(self, model, z_hist, u_hist, z_goal, init=None, n_refine=None):
        t0 = time.time()
        kw = dict(w_unc=self.cfg.w_unc, w_safe=self.cfg.w_safe, zmean=self.zmean,
                  zstd=self.zstd, maha_thresh=self.maha_thresh, n_refine=n_refine)
        if self.cfg.planner == "gd":
            a = gd_plan(model, z_hist, u_hist, z_goal, horizon=self.cfg.horizon,
                        opt_steps=self.cfg.opt_steps, lr=self.cfg.gd_lr, init=init, **kw)
        else:
            a = cem_plan(model, z_hist, u_hist, z_goal, horizon=self.cfg.horizon,
                         n_samples=self.cfg.cem_samples, n_iter=self.cfg.cem_iter,
                         elite=self.cfg.cem_elite, init=init, **kw)
        self.timing["plan"] += time.time() - t0
        return a

    def surprise(self, model, z, u, z_n, k):
        with torch.no_grad():
            zh = z.unsqueeze(1).repeat(1, k, 1)
            uh = u.unsqueeze(1).repeat(1, k, 1)
            mean, logvar = model.predictor(zh, uh, u)[:2]
            return float(((mean - z_n) ** 2).sum(-1).mean())

    def step_env(self, ep, actions, corrupt_kind=None, corrupt_level=0.9, idx=0):
        """Execute an action chunk; returns transitions (obs, prop, a, obs_next, prop_next)."""
        trans = []
        prev_frame = ep.env.obs(goal=ep.goal_state) if ep.kind == "maze" else ep.env.obs()
        prev_prop = ep.env.sim.s[:4].copy()
        for a in actions:
            s = ep.env.step(a)
            ep.steps += 1
            frame = ep.env.obs(goal=ep.goal_state) if ep.kind == "maze" else ep.env.obs()
            if ep.kind == "push":
                frame = _corrupt_live(frame, corrupt_kind, corrupt_level, idx, ep.steps)
            prop = s[:4].copy()
            trans.append((prev_frame, prev_prop, np.asarray(a, dtype=np.float32), frame, prop))
            prev_frame, prev_prop = frame, prop
            if ep.env.success(ep.goal_state) and not ep.done:
                ep.done = True
                ep.first_success = ep.steps
        return trans


def _corrupt_live(frame, kind, level, ep_idx, step):
    if not kind:
        return frame
    return corrupt(frame, kind, level,
                   np.random.default_rng(abs(hash((ep_idx, step, kind))) % (1 << 31)))


def run_condition(cond, method, n_ep, seed, cfg, verbose=False, keep_models=False):
    t_start = time.time()
    model, ck = get_model(cond["ckpt"], fold=METHODS[method].get("fold", True))
    r = Runner(method, model, cfg, ck["pmean"], ck["pstd"],
               ck.get("zmean"), ck.get("zstd"), ck.get("maha_thresh"))
    k = model.k
    eps = cond["builder"](n_ep, seed)
    models = [r.make_model() for _ in eps]
    adapters = [r.make_adapter(m, k) for m in models]
    done_flags = [False] * len(eps)
    success_curve = np.zeros(cfg.max_replans)
    for rep in range(cfg.max_replans):
        for i, ep in enumerate(eps):
            m, ad = models[i], adapters[i]
            z_hist = r.encode_batch(m, ep.hist_obs, ep.hist_prop)[None]
            with torch.no_grad():     # inputs of the planner: no graph, reused every replan
                u_hist = m.action(torch.as_tensor(ep.hist_act, dtype=torch.float32)[None])
            z_g = r.encode_batch(m, ep.goal_obs[None], ep.goal_prop[None])
            fp = z_hist[0].mean(0).detach().numpy()
            if ad is not None:
                ad.start_episode(fingerprint=fp)
                n_ref = ad.n_refine
                init = ep.last_actions if (cfg.reuse_actions and ep.last_actions is not None) else None
            else:
                n_ref, init = None, (ep.last_actions if cfg.reuse_actions else None)
            a = r.plan(m, z_hist, u_hist, z_g, init=init, n_refine=n_ref)
            ep.last_actions = a[0, cfg.chunk:].detach()
            t0 = time.time()
            trans = r.step_env(ep, a[0, :cfg.chunk].cpu().numpy(), cond.get("corrupt"),
                               cond.get("corrupt_level", 0.9), idx=i)
            r.timing["env"] += time.time() - t0
            if ad is not None and trans:
                t0 = time.time()
                z = r.encode_batch(m, [t[0] for t in trans], [t[1] for t in trans])
                z_n = r.encode_batch(m, [t[3] for t in trans], [t[4] for t in trans]).detach()
                u = m.action(torch.as_tensor(np.stack([t[2] for t in trans]),
                                             dtype=torch.float32))
                err = r.surprise(m, z, u, z_n, k)
                use_mem = r.m.get("memory", 0) > 0
                for j, t in enumerate(trans):
                    ad.observe(t[0], t[1], t[2], t[3], t[4], err=err, unc=0.0,
                               z=z[j].detach().numpy() if use_mem else None,
                               u=u[j].detach().numpy() if use_mem else None,
                               zn=z_n[j].numpy() if use_mem else None)
                ad.update()
                r.timing["adapt"] += time.time() - t0
            if ep.done:
                done_flags[i] = True
        n_done = int(sum(done_flags))
        success_curve[rep] = n_done / len(eps)
        if verbose:
            print(f"    rep{rep}: {n_done}/{len(eps)} ({time.time()-t_start:.0f}s)", flush=True)
        if n_done == len(eps):
            success_curve[rep:] = 1.0
            break
    stats = {}
    if adapters[0] is not None:
        agg, keys = {}, set()
        for ad in adapters:
            keys |= set(ad.summary().keys())
        for kk in keys:
            vals = []
            for ad in adapters:
                v = ad.summary().get(kk, 0.0)
                if isinstance(v, (int, float)):
                    vals.append(float(v))
            agg[kk] = float(np.mean(vals)) if vals else 0.0
        stats = agg
        if r.param_memory is not None:
            stats["param_mem_size"] = len(r.param_memory)
        if len(adapters) and adapters[0].safety is not None:
            stats["anchor_proj_total"] = float(sum(ad.safety.events["anchor_proj"]
                                                   for ad in adapters if ad.safety))
    res = dict(suite=cond["suite"], label=cond["label"], method=method, ckpt=cond["ckpt"], n_ep=len(eps),
               n_requested=n_ep, seed=seed, success=float(success_curve[-1]),
               success_curve=[float(x) for x in success_curve],
               steps_mean=float(np.mean([e.steps for e in eps])),
               first_success_mean=float(np.mean([e.first_success or 0 for e in eps])),
               timing=r.timing, adapt_stats=stats, planner=cfg.planner, horizon=cfg.horizon,
               opt_steps=cfg.opt_steps, max_replans=cfg.max_replans, chunk=cfg.chunk,
               goal_gap=cond.get("goal_gap"), wall=time.time() - t_start)
    print(f"  [{cond['suite']}/{cond['label']}/{method}] success={res['success']*100:.1f}% "
          f"({res['wall']:.0f}s plan={r.timing['plan']:.0f}s env={r.timing['env']:.0f}s "
          f"adapt={r.timing['adapt']:.0f}s)", flush=True)
    return res


# ---------------------------------------------------------------------------
# suites
# ---------------------------------------------------------------------------
def suite_specs(suite, ckpt_push="push4", ckpt_maze="maze25", goal_gap=D.GOAL_GAP):
    specs = []
    if suite in ("shape", "all"):
        for sh in TRAIN_SHAPES + OOD_SHAPES:
            specs.append(dict(suite="shape", label=sh, ckpt=ckpt_push, kind="push", shape=sh,
                              eval_file=f"push_eval_{sh}", seen=sh in TRAIN_SHAPES,
                              goal_gap=goal_gap))
    if suite in ("visual", "all"):
        for label, ck, colors, lvl in (("default", None, None, 0.9), ("blur", "blur", None, 0.9),
                                       ("blurStrong", "blur", None, 1.8), ("snp", "snp", None, 0.9),
                                       ("dark", "dark", None, 0.9),
                                       ("redAgent", None, {"agent": RED}, 0.9),
                                       ("redBlock", None, {"block": RED}, 0.9),
                                       ("redAnchor", None, {"anchor": RED}, 0.9)):
            specs.append(dict(suite="visual", label=label, ckpt=ckpt_push, kind="push",
                              shape="T", eval_file="push_eval_T_visual", corrupt=ck,
                              corrupt_level=lvl, colors=colors, goal_gap=goal_gap))
    if suite in ("dyn", "all"):
        for label, ms, ds in (("default", 1.0, 1.0), ("lowMass", 0.2, 1.0),
                              ("highDamping", 1.0, 20.0)):
            fname = {"default": "default", "lowMass": "lowmass",
                     "highDamping": "highdamp"}[label]
            specs.append(dict(suite="dyn", label=label, ckpt=ckpt_maze, kind="maze",
                              eval_file=f"maze_dyneval_{fname}", mass_scale=ms,
                              damping_scale=ds, layout="train0"))
    if suite in ("layout", "all"):
        for kk in range(5):
            specs.append(dict(suite="layout", label=f"layout{kk}", ckpt=ckpt_maze, kind="maze",
                              eval_file=f"maze_layout_test_{kk}", layout=f"test{kk}"))
    return specs


def attach_builders(specs, min_move=0.06):
    layouts = np.load(os.path.join(D.DATA, "maze_layouts.npz"))
    for s in specs:
        if s["kind"] == "push":
            s["builder"] = (lambda n, sd, s=s: build_push_episodes(
                s["eval_file"], n, sd, s["shape"], dynamics=s.get("dynamics"),
                colors=s.get("colors"), corrupt_kind=s.get("corrupt"),
                corrupt_level=s.get("corrupt_level", 0.9),
                goal_gap=s.get("goal_gap", D.GOAL_GAP), min_move=min_move))
        else:
            walls = layouts["train"][0] if s.get("layout") == "train0" \
                else layouts["test"][int(s["layout"][4:])]
            s["builder"] = (lambda n, sd, s=s, walls=walls: build_maze_episodes(
                s["eval_file"], n, sd, walls, mass_scale=s.get("mass_scale", 1.0),
                damping_scale=s.get("damping_scale", 1.0), colors=s.get("colors")))
    return specs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="shape")
    ap.add_argument("--methods", nargs="+", default=["frozen", "adajepa", "dtajepa"])
    ap.add_argument("--n_ep", type=int, default=30)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--planner", default="gd")
    ap.add_argument("--horizon", type=int, default=15)
    ap.add_argument("--chunk", type=int, default=5)
    ap.add_argument("--max_replans", type=int, default=10)
    ap.add_argument("--opt_steps", type=int, default=40)
    ap.add_argument("--ckpt_push", default="push4")
    ap.add_argument("--ckpt_maze", default="maze25")
    ap.add_argument("--goal_gap", type=int, default=D.GOAL_GAP)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default="results/raw.jsonl")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.set_grad_enabled(True)

    specs = attach_builders(suite_specs(a.suite, a.ckpt_push, a.ckpt_maze, a.goal_gap))
    cfg = MPCConfig(planner=a.planner, horizon=a.horizon, chunk=a.chunk,
                    max_replans=a.max_replans, opt_steps=a.opt_steps, gd_lr=0.1)
    cfg.anchor_eps, cfg.val_tol = 0.8, 0.05
    out = os.path.join(ROOT, a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    for seed in a.seeds:
        for cond in specs:
            for meth in a.methods:
                res = run_condition(cond, meth, a.n_ep, seed, cfg, verbose=a.verbose)
                with open(out, "a") as f:
                    f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
