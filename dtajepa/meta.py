"""Stage-2 meta-training: FOMAML over the fast-parameter initialisation theta_f^0.

Only the fast parameters (LoRA on the R-module + encoder head + heads) are meta-learned;
the slow parameters (encoder body, S-module) are frozen, exactly as in deployment.

    python -m dtajepa.meta --data push_four_shapes_train --base push4 --name push4_meta
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from . import data as D
from .losses import jepa_loss
from .train import CKPT, batch_tensors, load_model, window_index


def task_groups(meta, n_traj):
    """Split training trajectories into meta-tasks (leave-one-shape/layout-out)."""
    if not meta:
        k = max(2, n_traj // 4)
        return [(np.arange(i * k, min((i + 1) * k, n_traj))) for i in range(4)]
    key = "shapes" if "shapes" in meta else "layouts"
    tags = dict(zip(range(len(meta[key])), meta[key]))
    groups = {}
    for i, t in enumerate(meta[key]):
        groups.setdefault(t, []).append(i)
    return [np.array(v) for v in groups.values()]


def meta_train(args):
    device = torch.device("mps" if torch.cuda.is_available() or torch.backends.mps.is_available()
                          else "cpu")
    device = torch.device("cpu") if args.cpu else device
    model, ck = load_model(args.base, device=str(device))
    pmean, pstd = ck["pmean"], ck["pstd"]
    obs, act, state, meta = D.load(args.data)
    N, T = obs.shape[:2]
    groups = task_groups(meta, N)
    rng = np.random.default_rng(args.seed)

    params = [p for p in model.fast_parameters() if p.requires_grad]
    # the inner step must mirror deployment: the encoder head uses a learning rate
    # two orders of magnitude smaller than the predictor's fast parameters
    enc_ids = {id(p) for p in model.encoder.head.parameters()}
    inner_lr = [args.enc_inner_lr if id(p) in enc_ids else args.inner_lr for p in params]
    meta_opt = torch.optim.Adam(params, lr=args.meta_lr)
    model.train(False)
    print(f"meta-training on {len(groups)} tasks, {len(params)} fast tensors", flush=True)

    for it in range(args.iters):
        gi = int(rng.integers(len(groups)))
        idx_tr = groups[gi]
        idx_other = np.concatenate([g for j, g in enumerate(groups) if j != gi]) \
            if len(groups) > 1 else idx_tr
        sup = rng.choice(idx_tr, args.bs, replace=len(idx_tr) < args.bs)
        qry = rng.choice(idx_tr, args.bs, replace=len(idx_tr) < args.bs)
        # clip window starts so every sample has a valid (history, future) span
        starts = np.arange(0, T - 6 - 4 + 1)
        sup = np.stack([sup, rng.choice(starts, len(sup))], 1)
        qry = np.stack([qry, rng.choice(starts, len(qry))], 1)

        p0 = [p.detach().clone() for p in params]
        o, pr, a = batch_tensors(obs, act, state, sup, device, pmean, pstd)
        loss_s = jepa_loss(model, o, pr, a, lam_reg=1.0)["loss"]
        grads = torch.autograd.grad(loss_s, params, allow_unused=True)
        with torch.no_grad():
            for p, g, lr in zip(params, grads, inner_lr):
                if g is not None:
                    p.add_(g, alpha=-lr)
        o, pr, a = batch_tensors(obs, act, state, qry, device, pmean, pstd)
        loss_q = jepa_loss(model, o, pr, a, lam_reg=1.0)["loss"]
        meta_opt.zero_grad(set_to_none=True)
        loss_q.backward()
        with torch.no_grad():                      # restore theta_f^0 before the meta step
            for p, q in zip(params, p0):
                p.copy_(q)
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        meta_opt.step()
        if it % max(1, args.log_every) == 0:
            print(f"it{it} support={float(loss_s):.4f} query={float(loss_q):.4f} "
                  f"[{(time.time()-args._t0)/60:.1f}min]", flush=True)
    ck["state"] = model.state_dict()
    ck["meta_trained"] = True
    ck["meta_args"] = {k: v for k, v in vars(args).items() if k != "_t0"}
    torch.save(ck, os.path.join(CKPT, f"pretrain_{args.name}.pt"))
    print("saved", os.path.join(CKPT, f"pretrain_{args.name}.pt"), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="push_four_shapes_train")
    ap.add_argument("--base", default="push4")
    ap.add_argument("--name", default="push4_meta")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--inner_lr", type=float, default=5e-4)
    ap.add_argument("--enc_inner_lr", type=float, default=1e-5)
    ap.add_argument("--meta_lr", type=float, default=1e-4)
    ap.add_argument("--log_every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    a._t0 = time.time()
    meta_train(a)
