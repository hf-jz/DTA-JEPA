"""DTA-JEPA model: end-to-end JEPA encoder + dual-timescale recursive predictor.

Design (see docs/DTA-JEPA-v2.md):

  * Encoder E_s : from-scratch CNN (LeWM-style, no pretrained weights), split into
    a frozen-ish ``body`` and an adaptable ``head``; proprioception is fused with
    the visual features inside the head.
  * Predictor F  : two coupled timescales.
      - S-module (slow): K-history transformer producing the dynamics context c_t.
        Consolidation only (episode boundaries), never adapted online.
      - R-module (fast): recursively refines the next latent y^(k) from c_t,
        k = 1..K_max, with deep supervision and uncertainty-gated halting
        (TRM/HRM-style recursion, LeWM-style 2-term training).
    Uncertainty: diagonal Gaussian head (aleatoric) + MC-dropout spread (epistemic).
  * LoRA adapters live on the R-module: the fast parameter set of the dual-timescale
    parameterisation (theta_f).  theta_s = encoder body + S-module.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------


class LoRALinear(nn.Module):
    """Frozen base weight + low-rank update: W' = W + scale * B @ A."""

    def __init__(self, base: nn.Linear, rank: int = 4, scale: float = 1.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.rank = rank
        self.scale = scale
        if rank > 0:
            self.A = nn.Parameter(torch.zeros(rank, base.in_features).normal_(0, 1e-3))
            self.B = nn.Parameter(torch.zeros(base.out_features, rank))
        else:
            self.register_parameter("A", None)
            self.register_parameter("B", None)

    def forward(self, x):
        y = self.base(x)
        if self.rank > 0:
            y = y + self.scale * F.linear(F.linear(x, self.A), self.B)
        return y

    # nn.MultiheadAttention reads .weight/.bias off its out_proj, so expose the
    # composed weight (gradients still flow to A and B).
    @property
    def weight(self):
        if self.rank > 0 and self.A is not None:
            return self.base.weight + self.scale * (self.B @ self.A)
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def lora_parameters(self):
        return [p for p in (self.A, self.B) if p is not None]


def _replace_linear(module, rank, scale=1.0, target=lambda name: True, prefix=""):
    for name, child in list(module.named_children()):
        full = f"{prefix}{name}"
        if isinstance(child, nn.Linear) and target(full):
            setattr(module, name, LoRALinear(child, rank=rank, scale=scale))
        else:
            _replace_linear(child, rank, scale, target, full + ".")


def lora_parameters(module):
    return [p for m in module.modules() if isinstance(m, LoRALinear) for p in m.lora_parameters()]


def fold_lora(model):
    """Fold trained LoRA deltas into the base weights and switch to direct updates.

    The folded model is numerically identical to the LoRA model at the moment of
    folding, so a checkpoint trained with adapters can be evaluated with either fast
    parameterisation -- which is what makes the AdaJEPA/DTA-JEPA comparison isolate the
    adaptation *policy* instead of confounding it with the parameterisation.
    """
    for m in model.modules():
        if isinstance(m, LoRALinear) and m.rank > 0:
            with torch.no_grad():
                m.base.weight.add_(m.scale * (m.B @ m.A))
            m.scale = 0.0
            for p_ in m.lora_parameters():
                p_.requires_grad_(False)
    return model


def set_lora_scale(module, scale):
    for m in module.modules():
        if isinstance(m, LoRALinear):
            m.scale = scale


# ---------------------------------------------------------------------------


class Encoder(nn.Module):
    """64x64 RGB (+proprio) -> d-dim latent.  body = conv trunk, head = projection."""

    def __init__(self, d=128, proprio=4, ch=(32, 64, 96, 128), norm="group"):
        super().__init__()
        layers = []
        cin = 3
        for c in ch:
            layers += [nn.Conv2d(cin, c, 3, stride=2, padding=1),
                       nn.GroupNorm(8, c) if norm == "group" else nn.BatchNorm2d(c),
                       nn.GELU()]
            cin = c
        self.body = nn.Sequential(*layers)
        self.feat_dim = ch[-1] * 4 * 4
        self.proprio = nn.Sequential(nn.Linear(proprio, 64), nn.GELU(), nn.Linear(64, 64))
        self.head = nn.Sequential(nn.Linear(self.feat_dim + 64, 256), nn.GELU(), nn.Linear(256, d))
        # Output normalisation: without it the head's cheapest way to reduce the
        # prediction loss is to shrink every latent toward zero (collapse). LayerNorm
        # bounds the per-sample scale so the encoder must encode *direction*.
        self.out_norm = nn.LayerNorm(d)
        self.d = d
        self.proprio_dim = proprio

    def forward_body(self, x):
        return self.body(x).flatten(1)

    def forward(self, x, prop=None):
        f = self.forward_body(x)
        if prop is None:
            prop = torch.zeros(x.shape[0], self.proprio_dim, device=x.device, dtype=x.dtype)
        return self.out_norm(self.head(torch.cat([f, self.proprio(prop)], -1)))

    def forward_head(self, f, prop=None):
        if prop is None:
            prop = torch.zeros(f.shape[0], self.proprio_dim, device=f.device, dtype=f.dtype)
        return self.out_norm(self.head(torch.cat([f, self.proprio(prop)], -1)))


class ActionEncoder(nn.Module):
    def __init__(self, a_dim=2, d=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(a_dim, 64), nn.GELU(), nn.Linear(64, d))

    def forward(self, a):
        return self.net(a)


class Block(nn.Module):
    """Pre-norm transformer block over a token sequence."""

    def __init__(self, d=128, heads=4, mlp=2, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True, dropout=dropout)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, mlp * d), nn.GELU(), nn.Linear(mlp * d, d),
                                 nn.Dropout(dropout))

    def forward(self, x):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        return x + self.mlp(self.ln2(x))


class DualTimescalePredictor(nn.Module):
    """S-module (slow context) + R-module (fast recursive refinement)."""

    def __init__(self, d=128, heads=4, k_history=3, s_blocks=2, max_refine=4,
                 mc_dropout=0.1, rank=4, lora=True):
        super().__init__()
        self.d = d
        self.k = k_history
        self.max_refine = max_refine
        self.s_in = nn.Linear(2 * d, d)                       # (z_i, u_i) -> token
        self.s_module = nn.ModuleList([Block(d, heads) for _ in range(s_blocks)])
        self.c_proj = nn.Linear(d + d, d)                      # context + u_t
        # R-module: 2 tokens (y, c) + u injected -> refined latent
        self.r_block = Block(d, heads, dropout=mc_dropout)     # MC-dropout = epistemic probe
        self.r_norm = nn.LayerNorm(d)
        self.mean_head = nn.Linear(d, d)
        self.logvar_head = nn.Linear(d, 1)
        self.register_buffer("_lora_flag", torch.tensor(1 if lora else 0))
        if lora:
            _replace_linear(self.r_block, rank)
        self.mc_dropout = mc_dropout

    # ---- slow module -------------------------------------------------------
    def context(self, z_hist, u_hist):
        """z_hist,u_hist: (B,K,d) -> c_t (B,d) from the last token."""
        tok = self.s_in(torch.cat([z_hist, u_hist], -1))
        for blk in self.s_module:
            tok = blk(tok)
        return tok[:, -1]

    # ---- fast module -------------------------------------------------------
    def refine(self, c, y0, u_next, n_refine=None, return_trace=False):
        """Recursively refine the next latent.

        Returns (mean, logvar, depth, lvs, trace) where ``lvs`` is the (B,K) trace of
        predicted log-variance across refinement depths and ``trace`` the (B,K,d)
        latent trace (None unless requested).
        """
        B = c.shape[0]
        n_refine = self.max_refine if n_refine is None else n_refine
        y = y0
        trace, logvars = [], []
        depth = torch.zeros(B, dtype=torch.long, device=c.device)
        prev_lv = None
        for _ in range(n_refine):
            tok = torch.stack([y, c], 1)
            h = self.r_block(tok + u_next.unsqueeze(1))[:, 0]
            mean = self.mean_head(self.r_norm(h)) + y            # residual refinement
            logvar = self.logvar_head(mean)
            step_norm = (mean - y).norm(dim=-1) / (y.norm(dim=-1) + 1e-6)
            stalled = (step_norm < 1e-3) if prev_lv is not None else \
                torch.zeros_like(depth, dtype=torch.bool)
            depth = depth + (~stalled).long()
            prev_lv = step_norm.detach()
            y = mean
            logvars.append(logvar)
            trace.append(mean)
        lvs = torch.stack([l.squeeze(-1) for l in logvars], 1)   # (B,K)
        return mean, logvar, depth, lvs, (torch.stack(trace, 1) if return_trace else None)

    # ---- one-step / multi-step prediction ---------------------------------
    def forward(self, z_hist, u_hist, u_next, n_refine=None, return_trace=False):
        c = self.context(z_hist, u_hist)
        y0 = c + u_next
        mean, logvar, depth, lvs, trace = self.refine(c, y0, u_next, n_refine, return_trace)
        return mean, logvar, depth, lvs, trace

    def mc_epistemic(self, z_hist, u_hist, u_next, samples=4, n_refine=None):
        """Epistemic variance from MC-dropout in the R-module (train mode = dropout on)."""
        was_training = self.r_block.training
        self.r_block.train()
        c = self.context(z_hist, u_hist)
        outs = []
        for _ in range(samples):
            m, _, _, _, _ = self.refine(c, c + u_next, u_next, n_refine)
            outs.append(m)
        self.r_block.train(was_training)
        o = torch.stack(outs, 0)
        return o.mean(0), o.var(0, unbiased=False).mean(-1)


# ---------------------------------------------------------------------------


class WorldModel(nn.Module):
    """Encoder + action encoder + dual-timescale predictor (+ inverse dynamics)."""

    def __init__(self, d=128, proprio=4, k_history=3, a_dim=2, s_blocks=2,
                 max_refine=4, rank=4, lora=True, mc_dropout=0.1, enc_ch=(32, 64, 96, 128)):
        super().__init__()
        self.encoder = Encoder(d=d, proprio=proprio, ch=enc_ch)
        self.action = ActionEncoder(a_dim, d)
        self.predictor = DualTimescalePredictor(d=d, heads=4, k_history=k_history,
                                                s_blocks=s_blocks, max_refine=max_refine,
                                                mc_dropout=mc_dropout, rank=rank, lora=lora)
        self.inv = nn.Sequential(nn.Linear(2 * d, 128), nn.GELU(), nn.Linear(128, a_dim))
        self.d = d
        self.k = k_history
        self.lora = lora

    # ---- encoding ---------------------------------------------------------
    def encode(self, obs, prop=None):
        return self.encoder(obs, prop)

    def feature(self, obs):
        return self.encoder.forward_body(obs)

    def encode_from_feature(self, f, prop=None):
        return self.encoder.forward_head(f, prop)

    # ---- rollout ----------------------------------------------------------
    def rollout(self, z_hist, u_hist, actions, n_refine=None, return_unc=False):
        """actions: (B,H,a_dim) -> latent means (B,H,d) (+ logvars)."""
        B, H, _ = actions.shape
        zs = [z_hist[:, i] for i in range(z_hist.shape[1])]
        us = [u_hist[:, i] for i in range(u_hist.shape[1])]
        means, logvars = [], []
        for hh in range(H):
            u_next = self.action(actions[:, hh])
            z_hist_t = torch.stack(zs[-self.k:], 1)
            u_hist_t = torch.stack(us[-self.k:], 1)
            mean, logvar, depth, _, _ = self.predictor(z_hist_t, u_hist_t, u_next, n_refine)
            means.append(mean)
            logvars.append(logvar)
            zs.append(mean)
            us.append(u_next)
        m = torch.stack(means, 1)
        lv = torch.stack(logvars, 1).squeeze(-1)
        return (m, lv) if return_unc else m

    # ---- parameter groups -------------------------------------------------
    def slow_parameters(self):
        return list(self.encoder.body.parameters()) + list(self.predictor.s_module.parameters()) \
            + list(self.predictor.s_in.parameters()) + list(self.predictor.c_proj.parameters())

    def fast_parameters(self):
        """theta_f: the online-adaptable set (R-module + encoder head)."""
        extra = list(self.predictor.mean_head.parameters()) \
            + list(self.predictor.r_norm.parameters()) \
            + list(self.encoder.head.parameters())
        if self.lora:
            return lora_parameters(self.predictor.r_block) + extra
        return list(self.predictor.r_block.parameters()) + extra

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def n_fast(self):
        return sum(p.numel() for p in self.fast_parameters())
