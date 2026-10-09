"""Training objectives for DTA-JEPA.

LeWM-style two-term recipe (few hyper-parameters, stable from scratch):
  L = L_pred (multi-step latent prediction, deep-supervised over refinement depth)
      + lam_unc * L_unc  (diagonal-Gaussian NLL, aleatoric)
      + lam_reg * L_reg  (Epps-Pulley / SIGReg isotropic-Gaussian latent regulariser)
      + lam_inv * L_inv  (inverse dynamics, action grounding)
Stop-gradient on all targets prevents collapse (as in AdaJEPA/DINO-WM).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def straightness(z, log=False):
    """Temporal straightening: penalise the second difference of latent trajectories.

    z: (B,T,d) latents of one trajectory. A straight latent trajectory has zero second
    difference, which is what makes a *linear* latent cost a faithful proxy for the true
    cost --- the property AdaJEPA's base model obtains with this criterion and which our
    planner-limited regime suggests we are missing.
    """
    if z.shape[1] < 3:
        return torch.zeros((), device=z.device, dtype=z.dtype)
    d2 = z[:, 2:] - 2 * z[:, 1:-1] + z[:, :-2]
    # per-element mean so the term is on the same scale as the prediction MSE.
    # (The STLS controller's curvature gate uses the sum over dimensions instead, and
    # compares against a median fitted on the same definition -- the two are internally
    # consistent, which is what matters.)
    return (d2 ** 2).mean()


def variance_floor(z, target=1.0, eps=1e-4):
    """VICReg-style hinge: relu(target - std) per dimension.

    SIGReg's restoring force is proportional to the latent scale, so once the latents
    collapse (std ~ 1e-2) it is too weak to escape on its own. This hinge supplies a
    scale-free gradient and is the practical stabiliser of the two-term recipe.
    """
    std = torch.sqrt(z.var(0) + eps)
    return torch.relu(target - std).mean()


def sigreg(z, n_proj=64, t_max=3.0, n_t=17, seed=0):
    """SIGReg (LeJEPA): Epps-Pulley normality test on random 1-D projections of z.

    Zero when (and only when) the projected distribution is N(0,1); this is the
    single anti-collapse hyper-parameter of the training recipe.
    """
    N, d = z.shape
    g = torch.Generator(device=z.device).manual_seed(seed)
    A = torch.randn(d, n_proj, generator=g, device=z.device, dtype=z.dtype)
    A = A / A.norm(dim=0, keepdim=True).clamp_min(1e-8)
    x = z @ A                                              # (N,P)
    t = torch.linspace(1e-3, t_max, n_t, device=z.device, dtype=z.dtype)
    w = torch.exp(-0.5 * t ** 2) * (t_max / n_t)
    c = torch.cos(t[:, None, None] * x[None]).mean(1)      # (n_t,P)
    s = torch.sin(t[:, None, None] * x[None]).mean(1)
    tgt = torch.exp(-0.5 * t ** 2)[:, None]
    ep = ((c - tgt) ** 2 + s ** 2) * w[:, None]
    return ep.sum(0).mean() * N


def gaussian_nll(pred, target, logvar):
    """Diagonal Gaussian NLL: 0.5 * e^T Sigma^-1 e + 0.5 * logdet Sigma."""
    e = pred - target
    iv = torch.exp(-logvar)
    return (0.5 * (e ** 2 * iv).sum(-1) + 0.5 * logvar.sum(-1)).mean()


def jepa_loss(model, obs, prop, act, future=4, lam_unc=0.1, lam_reg=0.1, lam_inv=0.1,
              lam_str=0.0,
              deep_supervision=True, n_refine=None, sigreg_seed=0):
    """obs (B,T,3,H,W) uint8/float, prop (B,T,p), act (B,T-1,a).

    History length = model.k; the first ``k`` frames are context and the next
    ``future`` frames are prediction targets.
    """
    B, T = obs.shape[:2]
    k = model.k
    if obs.dtype == torch.uint8:
        obs = obs.float() / 255.0
    z = model.encode(obs.reshape(B * T, *obs.shape[2:]), prop.reshape(B * T, -1))
    z = z.reshape(B, T, -1)
    u = model.action(act.reshape(B * (T - 1), -1)).reshape(B, T - 1, -1)

    z_hist = z[:, :k]
    u_hist = u[:, :k]
    tgts = z[:, k:k + future]
    u_fut = u[:, k - 1:k - 1 + future]

    # ---- 1-step, deep supervision over refinement depth --------------------
    mean1, logvar1, depth, lvs, trace = model.predictor(
        z_hist, u_hist, u_fut[:, 0], n_refine=n_refine, return_trace=True)
    tgt1 = tgts[:, 0].detach()
    if deep_supervision and trace is not None:
        w = torch.linspace(0.5, 1.0, trace.shape[1], device=z.device, dtype=z.dtype)
        l_pred = torch.stack([w[i] * F.mse_loss(trace[:, i], tgt1) for i in range(trace.shape[1])]).sum()
        l_pred = l_pred / w.sum()
    else:
        l_pred = F.mse_loss(mean1, tgt1)
    l_unc = gaussian_nll(mean1, tgt1, logvar1)

    # ---- multi-step rollout (detached latents fed back) --------------------
    ms_pred = 0.0
    zs = [z_hist[:, i] for i in range(k)]
    us = [u_hist[:, i] for i in range(k)]
    for j in range(future):
        m, lv, _, _, _ = model.predictor(torch.stack(zs[-k:], 1), torch.stack(us[-k:], 1), u_fut[:, j])
        ms_pred = ms_pred + F.mse_loss(m, tgts[:, j].detach())
        zs.append(m.detach())
        us.append(u_fut[:, j])
    l_pred_multi = ms_pred / future

    # ---- regularisers ------------------------------------------------------
    l_reg = sigreg(z.reshape(-1, z.shape[-1]), seed=sigreg_seed) \
        + sigreg(mean1, seed=sigreg_seed) \
        + variance_floor(z.reshape(-1, z.shape[-1])) + variance_floor(mean1)
    inv_in = torch.cat([z[:, :-1], z[:, 1:].detach()], -1)
    l_inv = F.mse_loss(model.inv(inv_in.reshape(-1, 2 * model.d)), act.reshape(-1, act.shape[-1]))

    # straightness of the encoded window (encoder side) and of the one-step prediction
    # continued from it (predictor side) -- the two quantities the planner actually rolls out
    l_str = straightness(z) + straightness(torch.cat([z[:, k - 2:], mean1.unsqueeze(1)], 1))
    total = (l_pred + l_pred_multi + lam_unc * l_unc + lam_reg * l_reg + lam_inv * l_inv
             + lam_str * l_str)

    return {"loss": total, "pred": l_pred.detach(), "pred_multi": l_pred_multi.detach(),
            "unc": l_unc.detach(), "reg": l_reg.detach(), "inv": l_inv.detach(),
            "depth": depth.float().mean().detach(),
            "latent_std": z.reshape(-1, z.shape[-1]).std(0).mean().detach()}