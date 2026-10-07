"""Latent planners (GD / CEM) for batched goal-conditioned MPC.

The cost is AdaJEPA's eq. (3) plus the two DTA-JEPA terms:

    C(a) = sum_k [ ||y_k - z_g||^2              (goal distance)
                 + beta * tr(Sigma_k)           (uncertainty penalty)
                 + gamma * relu(Maha(y_k) - tau) ]   (safe-region penalty)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


def cost_fn(wm, z_hist, u_hist, actions, z_goal, n_refine=None, w_unc=0.0, w_safe=0.0,
            zmean=None, zstd=None, maha_thresh=None, want_info=False):
    m, lv = wm.rollout(z_hist, u_hist, actions, n_refine=n_refine, return_unc=True)
    goal = ((m - z_goal.unsqueeze(1)) ** 2).sum(-1)                        # (B,H)
    total = goal.sum(1)
    info = {"goal": goal.mean(1).detach()}
    if w_unc > 0:
        var = torch.exp(lv)
        total = total + w_unc * var.sum(1)
        info["unc"] = var.mean(1).detach()
    if w_safe > 0 and zmean is not None:
        maha = (((m - zmean) / zstd) ** 2).sum(-1)
        thr = maha_thresh if maha_thresh is not None else 1e9
        total = total + w_safe * F.relu(maha - thr).sum(1)
        info["maha"] = maha.mean(1).detach()
    return (total, info) if want_info else total


def gd_plan(wm, z_hist, u_hist, z_goal, horizon=15, opt_steps=60, lr=0.1, init=None, **kw):
    """Gradient-descent action optimisation (AdaJEPA's GD planner), batched."""
    B = z_hist.shape[0]
    a = torch.zeros(B, horizon, 2, device=z_hist.device, dtype=z_hist.dtype)
    if init is not None:                       # init: (T_i, 2) warm start from the last replan
        n = min(init.shape[0], horizon)
        a[:, :n] = init[:n].to(a.dtype)
    a = a.clone().requires_grad_(True)
    opt = torch.optim.Adam([a], lr=lr)
    for _ in range(opt_steps):
        opt.zero_grad(set_to_none=True)
        cost = cost_fn(wm, z_hist, u_hist, a, z_goal, **kw)
        cost.sum().backward()
        opt.step()
    return a.detach()


def cem_plan(wm, z_hist, u_hist, z_goal, horizon=15, n_samples=200, n_iter=10, elite=20,
             std0=1.0, init=None, generator=None, **kw):
    """Cross-entropy-method planner, batched (per-episode candidate sets)."""
    B = z_hist.shape[0]
    dev, dt = z_hist.device, z_hist.dtype
    mean = torch.zeros(B, horizon, 2, device=dev, dtype=dt)
    if init is not None:
        n = min(init.shape[0], horizon)
        mean[:, :n] = init[:n].to(dt)
    std = torch.full_like(mean, std0)
    for _ in range(n_iter):
        zz = z_hist.repeat_interleave(n_samples, 0)
        uu = u_hist.repeat_interleave(n_samples, 0)
        eps = torch.randn(B, n_samples, horizon, 2, device=dev, dtype=dt, generator=generator)
        cand = mean.unsqueeze(1) + std.unsqueeze(1) * eps
        cost = cost_fn(wm, zz, uu, cand.reshape(B * n_samples, horizon, 2), z_goal=z_goal, **kw)
        cost = cost.reshape(B, n_samples)
        idx = torch.argsort(cost, 1)[:, :elite]
        best = torch.gather(cand.reshape(B, n_samples, horizon, 2), 1,
                            idx[..., None, None].expand(-1, -1, horizon, 2))
        mean = best.mean(1)
        std = best.std(1) + 1e-3
    return mean.detach()


@dataclass
class MPCConfig:
    planner: str = "gd"
    horizon: int = 15
    chunk: int = 5
    max_replans: int = 12
    opt_steps: int = 60
    gd_lr: float = 0.1
    cem_samples: int = 200
    cem_iter: int = 10
    cem_elite: int = 20
    reuse_actions: bool = True
    w_unc: float = 0.0
    w_safe: float = 0.0
    n_refine: int | None = None
