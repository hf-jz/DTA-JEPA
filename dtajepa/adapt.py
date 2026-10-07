"""Online adaptation machinery: dual memory, safety layer, adaptive controller.

  * EpisodicMemory : priority memory of high-surprise latent transitions
                     (priority = alpha*error + beta*uncertainty + gamma*novelty).
  * ParamMemory    : snapshots of the fast parameters theta_f keyed by an
                     environment fingerprint, retrieved at episode start.
  * SafetyLayer    : anchor ball on theta_f, validation-gated rollback, event log.
  * Adapter        : the online update itself, in either
                       mode='adajepa'  -> direct update of the predictor's last block
                                          + encoder head, fixed 1 step, fixed lr
                       mode='dtajepa'  -> uncertainty-driven steps/lr, memory retrieval,
                                          anchor + rollback, refinement-depth allocation,
                                          optional EWC slow consolidation at episode end.
"""

from __future__ import annotations

import copy
import math

import numpy as np
import torch
import torch.nn.functional as F

from .models import LoRALinear


# ---------------------------------------------------------------------------
class EpisodicMemory:
    """Priority memory over latent transitions (AdaJEPA-style buffer, but persistent)."""

    def __init__(self, capacity=256, tau=1.0, w_err=1.0, w_unc=1.0, w_nov=1.0):
        self.capacity = capacity
        self.tau = tau
        self.w = (w_err, w_unc, w_nov)
        self.z, self.u, self.zn, self.prio = [], [], [], []
        self.age = 0

    def __len__(self):
        return len(self.z)

    def novelty(self, z):
        if not len(self.z):
            return 1.0
        Z = np.stack(self.z)
        d = np.linalg.norm(Z - z[None], axis=1)
        return float(np.exp(-(d.min() ** 2) / self.tau))

    def add(self, z, u, z_next, err, unc):
        p = (self.w[0] * float(err) + self.w[1] * float(unc)
             + self.w[2] * self.novelty(np.asarray(z)))
        if len(self.z) >= self.capacity:
            j = int(np.argmin(self.prio))
            if self.prio[j] >= p:
                return
            self.z[j], self.u[j], self.zn[j], self.prio[j] = z, u, z_next, p
        else:
            self.z.append(np.asarray(z)); self.u.append(np.asarray(u))
            self.zn.append(np.asarray(z_next)); self.prio.append(p)
        self.age += 1

    def retrieve(self, z_query, k=4, device="cpu"):
        if not len(self.z):
            return None
        Z = np.stack(self.z)
        sim = np.exp(-((Z - np.asarray(z_query)[None]) ** 2).sum(1) / self.tau)
        idx = np.argsort(-sim)[:k]
        t = lambda arr: torch.as_tensor(np.stack([arr[i] for i in idx]),
                                        dtype=torch.float32, device=device)
        return t(self.z), t(self.u), t(self.zn)

    def stats(self):
        return {"size": len(self.z), "mean_prio": float(np.mean(self.prio)) if self.prio else 0.0}


# ---------------------------------------------------------------------------
class ParamMemory:
    """theta_f snapshots keyed by an environment fingerprint (mean episode latent)."""

    def __init__(self, capacity=8):
        self.capacity = capacity
        self.keys, self.params, self.hits, self.misses = [], [], 0, 0

    def __len__(self):
        return len(self.keys)

    def add(self, key, params):
        key = np.asarray(key, dtype=np.float32)
        if len(self.keys) >= self.capacity:
            self.keys.pop(0); self.params.pop(0)
        self.keys.append(key)
        self.params.append({k: v.detach().clone() for k, v in params.items()})

    def retrieve(self, key, blend=0.0):
        if not self.keys:
            self.misses += 1
            return None
        K = np.stack(self.keys)
        d = np.linalg.norm(K - np.asarray(key, dtype=np.float32)[None], axis=1)
        i = int(np.argmin(d))
        self.hits += 1
        return self.params[i]


# ---------------------------------------------------------------------------
class SafetyLayer:
    """Anchor ball + validation gate + rollback bookkeeping."""

    def __init__(self, model, eps_anchor=1.0, val_tol=0.02):
        self.eps = eps_anchor
        self.val_tol = val_tol
        self.theta0 = {n: p.detach().clone() for n, p in model.named_parameters()
                       if p.requires_grad}
        self.events = {"anchor_proj": 0, "rollback": 0, "updates": 0}
        self.max_dev = 0.0

    def deviation(self, model):
        tot = 0.0
        for n, p in model.named_parameters():
            if n in self.theta0:
                tot += float(((p.detach() - self.theta0[n]) ** 2).sum())
        return math.sqrt(tot)

    def project(self, model):
        """Project theta_f back into the anchor ball ||theta_f - theta_f^0|| <= eps."""
        dev = self.deviation(model)
        self.max_dev = max(self.max_dev, dev)
        if dev > self.eps > 0:
            with torch.no_grad():
                for n, p in model.named_parameters():
                    if n in self.theta0:
                        p.copy_(self.theta0[n] + (p - self.theta0[n]) * (self.eps / dev))
            self.events["anchor_proj"] += 1
            return True
        return False

    def snapshot(self, model):
        return {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}

    def restore(self, model, snap):
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in snap:
                    p.copy_(snap[n])


# ---------------------------------------------------------------------------
class Controller:
    """Surprise/OOD -> (gradient steps, lr scale, refinement depth)."""

    def __init__(self, u_min=1, u_max=3, lr_min=0.2, lr_max=5.0, refine_min=1, refine_max=4,
                 ema=0.7, gain=2.0):
        self.u_min, self.u_max = u_min, u_max
        self.lr_min, self.lr_max = lr_min, lr_max
        self.refine_min, self.refine_max = refine_min, refine_max
        self.ema, self.gain = ema, gain
        self.mean_err = None
        self.last = {}

    def intensity(self, err, unc=0.0):
        err = float(err)
        if self.mean_err is None:
            self.mean_err = err
        ratio = err / (self.mean_err + 1e-6)
        self.mean_err = self.ema * self.mean_err + (1 - self.ema) * err
        s = 1.0 / (1.0 + math.exp(-self.gain * (ratio - 1.0)))
        steps = int(round(self.u_min + s * (self.u_max - self.u_min)))
        lr_scale = self.lr_min + s * (self.lr_max - self.lr_min)
        n_refine = int(round(self.refine_min + s * (self.refine_max - self.refine_min)))
        self.last = {"surprise": err, "ratio": ratio, "s": s, "steps": steps,
                     "lr_scale": lr_scale, "n_refine": n_refine, "unc": float(unc)}
        return self.last


# ---------------------------------------------------------------------------
class Adapter:
    """Per-episode online adaptation (one copy of the model per episode)."""

    def __init__(self, model, mode="dtajepa", lr=5e-4, enc_lr=1e-5, buffer=5, steps=1,
                 opt="adam", buffer_mode="recent", memory=None, param_memory=None,
                 safety=None, controller=None, ret_weight=0.5, n_refine=None,
                 mc_samples=0, grad_clip=5.0, fold=True):
        self.model = model
        self.mode = mode
        self.base_lr, self.enc_lr = lr, enc_lr
        self.steps = steps
        self.buffer_cap = buffer
        self.buffer_mode = buffer_mode
        self.buf = []
        self.memory = memory
        self.param_memory = param_memory
        self.safety = safety
        self.controller = controller if controller is not None else Controller()
        self.ret_weight = ret_weight
        self.n_refine = n_refine
        self.mc_samples = mc_samples
        self.grad_clip = grad_clip
        self.params = self._param_set(model, mode, fold)
        self.opt = torch.optim.Adam(self.params, lr=lr) if opt == "adam" else \
            torch.optim.SGD(self.params, lr=lr)
        self.info = {"updates": 0, "steps": 0, "rollback": 0, "surprise": [], "depth": [],
                     "retrieved": 0, "lr": []}

    # ---- episode lifecycle -------------------------------------------------
    @staticmethod
    def _param_set(model, mode, fold=True):
        """theta_f for each adaptation mode.

        'adajepa'  (baseline) : direct updates of the predictor's last transformer
                    block (AdaJEPA's predlast) + the encoder head (enclast).
        'dtajepa'             : LoRA adapters on the same predictor block + the
                    encoder head -> a deliberately decoupled fast parameterisation.
        """
        heads = list(model.predictor.mean_head.parameters()) \
            + list(model.predictor.r_norm.parameters()) \
            + list(model.encoder.head.parameters())
        if mode == "adajepa" or fold:
            # direct updates of the predictor's last block (AdaJEPA's predlast):
            # the LoRA path is off, so its parameters are excluded
            for m in model.predictor.r_block.modules():
                if isinstance(m, LoRALinear):
                    for p in m.lora_parameters():
                        p.requires_grad_(False)
            for p in model.predictor.r_block.parameters():
                p.requires_grad_(True)
            return [p for p in model.predictor.r_block.parameters() if p.requires_grad] + heads
        return [p for p in model.fast_parameters() if p.requires_grad]

    def start_episode(self, fingerprint=None):
        self.buf = []
        self.fingerprint = fingerprint
        for g in self.opt.param_groups:
            g["lr"] = self.base_lr
        if self.mode == "dtajepa" and self.param_memory is not None and fingerprint is not None:
            snap = self.param_memory.retrieve(fingerprint)
            if snap is not None:
                with torch.no_grad():
                    for n, p in self.model.named_parameters():
                        if n in snap:
                            p.copy_(snap[n])

    def end_episode(self, consolidate=True, lr_slow=1e-6, ewc=None, batches=4):
        """DTA-JEPA: store theta_f in parametric memory; optionally consolidate theta_s."""
        if self.mode != "dtajepa":
            return {}
        out = {}
        if self.param_memory is not None and getattr(self, "fingerprint", None) is not None:
            self.param_memory.add(self.fingerprint,
                                  {n: p for n, p in self.model.named_parameters()
                                   if p.requires_grad})
        if consolidate and self.memory is not None and len(self.memory):
            params = [p for p in self.model.slow_parameters() if p.requires_grad]
            names = {n for n, p in self.model.named_parameters()
                     if any(p is q for q in params)}
            anchor = {n: p.detach().clone() for n, p in self.model.named_parameters()
                      if n in names}
            opt = torch.optim.Adam(params, lr=lr_slow)
            loss = None
            for _ in range(batches):
                for z, u, zn in self._memory_batches():
                    opt.zero_grad(set_to_none=True)
                    loss = self._pred_loss(z, u, zn)
                    if ewc is not None:
                        loss = loss + ewc.penalty(self.model, anchor)
                    loss.backward()
                    opt.step()
            out["slow_update"] = float(loss) if loss is not None else 0.0
        return out

    def _memory_batches(self, n=2, bs=16):
        mem = self.memory
        if mem is None or not len(mem):
            return
        idx = np.random.permutation(len(mem))[:bs]
        t = lambda arr: torch.as_tensor(np.stack([arr[i] for i in idx]), dtype=torch.float32,
                                        device=self._device)
        yield t(mem.z), t(mem.u), t(mem.zn)
        for _ in range(n - 1):
            idx = np.random.permutation(len(mem))[:bs]
            yield t(mem.z), t(mem.u), t(mem.zn)

    @property
    def _device(self):
        return next(self.model.parameters()).device

    # ---- observation -------------------------------------------------------
    def observe(self, obs, prop, act, obs_next, prop_next, err=0.0, unc=0.0, z=None, u=None,
                zn=None):
        """Append a transition; keeps recent-N or hard-N."""
        rec = (obs, prop, act, obs_next, prop_next, float(err))
        if self.buffer_mode == "hard":
            self.buf.append(rec)
            self.buf = sorted(self.buf, key=lambda r: -r[-1])[:self.buffer_cap]
        else:
            self.buf.append(rec)
            self.buf = self.buf[-self.buffer_cap:]
        if self.mode == "dtajepa" and self.memory is not None and z is not None:
            self.memory.add(z, u, zn, err, unc)

    def _pred_loss(self, z, u, zn):
        k = self.model.k
        zh = z.unsqueeze(1).repeat(1, k, 1)
        uh = u.unsqueeze(1).repeat(1, k, 1)
        out = self.model.predictor(zh, uh, u, n_refine=self.n_refine)
        mean, logvar = out[0], out[1]
        mse = ((mean - zn) ** 2).sum(-1).mean()
        nll = (0.5 * ((mean - zn) ** 2 * torch.exp(-logvar)).sum(-1)
               + 0.5 * logvar.sum(-1)).mean()
        return mse + 0.1 * nll

    def update(self):
        """One adaptation event; returns the info dict of this event."""
        if not self.buf:
            return None
        dev = self._device
        nhwc = lambda arrs: (lambda a: np.ascontiguousarray(a.transpose(0, 3, 1, 2))
                             if a.ndim == 4 else a)(np.asarray(arrs))
        obs = torch.as_tensor(nhwc([b[0] for b in self.buf]), dtype=torch.float32,
                              device=dev) / 255.0
        prop = torch.as_tensor(np.stack([b[1] for b in self.buf]), dtype=torch.float32, device=dev)
        act = torch.as_tensor(np.stack([b[2] for b in self.buf]), dtype=torch.float32, device=dev)
        obs_n = torch.as_tensor(nhwc([b[3] for b in self.buf]), dtype=torch.float32,
                                device=dev) / 255.0
        prop_n = torch.as_tensor(np.stack([b[4] for b in self.buf]), dtype=torch.float32,
                                 device=dev)
        mode = self.mode
        # --- surprise-driven or fixed step/lr budget ------------------------
        err = float(np.mean([b[5] for b in self.buf]))
        unc = 0.0
        if mode == "dtajepa":
            if self.mc_samples > 0:
                z_all = self.model.encode(obs, prop)
                z_n = self.model.encode(obs_n, prop_n).detach()
                u_all = self.model.action(act)
                zh = z_all.unsqueeze(1).repeat(1, self.model.k, 1)
                uh = u_all.unsqueeze(1).repeat(1, self.model.k, 1)
                with torch.no_grad():
                    _, epi = self.model.predictor.mc_epistemic(zh, uh, u_all,
                                                               samples=self.mc_samples,
                                                               n_refine=self.n_refine)
                unc = float(epi.mean())
            ctrl = self.controller.intensity(err, unc)
            steps, lr_scale = ctrl["steps"], ctrl["lr_scale"]
            self.n_refine = ctrl["n_refine"]
        else:
            ctrl = {"surprise": err, "s": 0.0, "steps": self.steps, "lr_scale": 1.0}
            steps, lr_scale = self.steps, 1.0
        for g in self.opt.param_groups:
            g["lr"] = self.base_lr * lr_scale
        # --- forward / backward ---------------------------------------------
        snap = self.safety.snapshot(self.model) if self.safety is not None else None
        self.model.train(False)
        with torch.no_grad():            # the action encoder is never adapted online
            u = self.model.action(act)
        val_old = None
        if mode == "dtajepa" and self.safety is not None and len(self.buf) > 2:
            with torch.no_grad():
                z0 = self.model.encode(obs, prop)
                z0_n = self.model.encode(obs_n, prop_n)
                val_old = float(self._pred_loss(z0[:-1], u[:-1], z0_n[:-1]))
        losses = []
        for s in range(steps):
            self.opt.zero_grad(set_to_none=True)
            # re-encode each step: the encoder head is part of theta_f, so the graph
            # must be rebuilt after every in-place parameter update
            z = self.model.encode(obs, prop)
            z_n = self.model.encode(obs_n, prop_n).detach()
            loss = self._pred_loss(z, u, z_n)
            if mode == "dtajepa" and self.memory is not None and len(self.memory):
                ret = self.memory.retrieve(z.mean(0).detach().cpu().numpy(), k=4, device=dev)
                if ret is not None:
                    zz, uu, zz_n = ret
                    loss = loss + self.ret_weight * self._pred_loss(zz, uu, zz_n)
                    self.info["retrieved"] += 1
            loss.backward()
            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.params, self.grad_clip)
            self.opt.step()
            losses.append(float(loss.detach()))
            if mode == "dtajepa" and self.safety is not None:
                self.safety.project(self.model)
        # --- validation gate + rollback -------------------------------------
        rollback = False
        if mode == "dtajepa" and self.safety is not None and val_old is not None:
            with torch.no_grad():
                z2 = self.model.encode(obs, prop)
                z2_n = self.model.encode(obs_n, prop_n).detach()
                u2 = self.model.action(act)
                val_new = float(self._pred_loss(z2[:-1], u2[:-1], z2_n[:-1]))
            if val_new > val_old + self.safety.val_tol:
                self.safety.restore(self.model, snap)
                self.safety.events["rollback"] += 1
                self.info["rollback"] += 1
                rollback = True
        self.info["updates"] += 1
        self.info["steps"] += steps
        self.info["surprise"].append(err)
        self.info["lr"].append(self.base_lr * lr_scale)
        if self.safety is not None and self.safety.events["anchor_proj"]:
            self.info["anchor_proj"] = self.safety.events["anchor_proj"]
        return {"err": err, "unc": unc, "steps": steps, "lr": self.base_lr * lr_scale,
                "loss": float(np.mean(losses)), "rollback": rollback,
                "n_refine": self.n_refine, **ctrl}

    def summary(self):
        s = dict(self.info)
        s["surprise"] = float(np.mean(s["surprise"])) if s["surprise"] else 0.0
        s["lr"] = float(np.mean(s["lr"])) if s["lr"] else 0.0
        return s


# ---------------------------------------------------------------------------
class EWC:
    """Diagonal Fisher penalty used by the slow (episode-boundary) update."""

    def __init__(self, model, lam=1e3, n_batches=4, bs=16, device="cpu"):
        self.lam = lam
        self.F = {n: torch.zeros_like(p) for n, p in model.named_parameters()
                  if p.requires_grad}
        self.star = {n: p.detach().clone() for n, p in model.named_parameters()
                     if p.requires_grad}
        self._estimate(model, n_batches, bs, device)

    def _estimate(self, model, n_batches, bs, device):
        # LAZY: a one-shot random-input Fisher estimate (fine for a bounded slow LR).
        params = [p for p in model.parameters() if p.requires_grad]
        for _ in range(n_batches):
            z = torch.randn(bs, model.d, device=device)
            u = torch.randn(bs, model.d, device=device)
            zn = torch.randn(bs, model.d, device=device)
            loss = ((model.predictor(z.unsqueeze(1).repeat(1, model.k, 1),
                                     u.unsqueeze(1).repeat(1, model.k, 1), u)[0] - zn) ** 2).mean()
            model.zero_grad(set_to_none=True)
            loss.backward()
            for n, p in model.named_parameters():
                if n in self.F and p.grad is not None:
                    self.F[n] += p.grad.detach() ** 2 / n_batches
        model.zero_grad(set_to_none=True)

    def penalty(self, model, anchor=None):
        pen = 0.0
        for n, p in model.named_parameters():
            if n in self.F:
                star = self.star[n] if anchor is None or n not in anchor else anchor[n]
                pen = pen + (self.F[n] * (p - star) ** 2).sum()
        return self.lam * pen
