"""Phase-1 pretraining of the DTA-JEPA world model.

    python -m dtajepa.train --data push_four_shapes_train --name push4 --epochs 6
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
from .models import WorldModel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(ROOT, "checkpoints")
HISTORY, FUTURE = D.HISTORY, D.FUTURE


def pick_device(name="auto"):
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def window_index(n_traj, T, k=HISTORY, future=FUTURE):
    starts = np.arange(0, T - k - future + 1)
    idx = np.stack(np.meshgrid(np.arange(n_traj), starts, indexing="ij"), -1).reshape(-1, 2)
    return idx


def batch_tensors(obs, act, state, idx, device, pmean, pstd):
    frames = np.arange(idx.shape[0])
    o = obs[idx[:, 0]]
    a = act[idx[:, 0]]
    s = state[idx[:, 0]]
    t0 = idx[:, 1]
    L = HISTORY + FUTURE
    win_o = np.stack([o[i, t0[i]:t0[i] + L] for i in range(len(idx))])
    win_a = np.stack([a[i, t0[i]:t0[i] + L - 1] for i in range(len(idx))])
    win_s = np.stack([s[i, t0[i]:t0[i] + L] for i in range(len(idx))])
    prop = (win_s[..., :4] - pmean) / pstd
    return (torch.as_tensor(np.ascontiguousarray(win_o.transpose(0, 1, 4, 2, 3)), device=device),
            torch.as_tensor(prop, dtype=torch.float32, device=device),
            torch.as_tensor(win_a, dtype=torch.float32, device=device))


def train(args):
    device = pick_device(args.device)
    obs, act, state, meta = D.load(args.data)
    N, T = obs.shape[:2]
    print(f"data {args.data}: obs={obs.shape} act={act.shape}", flush=True)
    pmean = state[..., :4].reshape(-1, 4).mean(0)
    pstd = state[..., :4].reshape(-1, 4).std(0) + 1e-6

    idx = window_index(N, T)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(idx)
    n_val = max(1, int(0.05 * len(idx)))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    model = WorldModel(d=args.d, proprio=4, k_history=HISTORY, a_dim=act.shape[-1],
                       s_blocks=args.s_blocks, max_refine=args.max_refine,
                       rank=args.rank, lora=args.lora, mc_dropout=0.1,
                       enc_ch=tuple(args.ch)).to(device)
    print(f"params total={model.n_params()/1e6:.2f}M  fast={model.n_fast()/1e3:.1f}k", flush=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, args.epochs * (len(tr_idx) // args.bs + 1)), eta_min=args.lr * 0.1)

    os.makedirs(CKPT, exist_ok=True)
    hist = []
    step = 0
    t_start = time.time()
    for ep in range(args.epochs):
        model.train()
        perm = rng.permutation(len(tr_idx))
        run = {}
        for i in range(0, len(perm) - args.bs + 1, args.bs):
            sel = tr_idx[perm[i:i + args.bs]]
            o, p, a = batch_tensors(obs, act, state, sel, device, pmean, pstd)
            out = jepa_loss(model, o, p, a, future=FUTURE,
                            lam_unc=args.lam_unc, lam_reg=args.lam_reg, lam_inv=args.lam_inv,
                            sigreg_seed=step % 7)
            opt.zero_grad(set_to_none=True)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            step += 1
            for k, v in out.items():
                run[k] = run.get(k, 0.0) + float(v.detach() if torch.is_tensor(v) else v)
            if step % args.log_every == 0:
                msg = " ".join(f"{k}={run[k]/args.log_every:.4f}"
                               for k in ("loss", "pred", "pred_multi", "unc", "reg", "inv"))
                print(f"ep{ep} step{step} {msg} lat_std={run['latent_std']/args.log_every:.3f} "
                      f"depth={run['depth']/args.log_every:.2f} "
                      f"[{(time.time()-t_start)/60:.1f}min]", flush=True)
                run = {}
            if args.max_iters and step >= args.max_iters:
                break
        # validation
        model.eval()
        vloss = 0.0
        with torch.no_grad():
            for i in range(0, len(val_idx) - args.bs + 1, args.bs):
                sel = val_idx[i:i + args.bs]
                o, p, a = batch_tensors(obs, act, state, sel, device, pmean, pstd)
                out = jepa_loss(model, o, p, a, future=FUTURE, lam_unc=args.lam_unc,
                                lam_reg=args.lam_reg, lam_inv=args.lam_inv)
                vloss += float(out["loss"]) * len(sel)
        vloss /= max(1, len(val_idx))
        hist.append({"epoch": ep, "val": vloss})
        print(f"== ep{ep} val={vloss:.4f} ({(time.time()-t_start)/60:.1f}min)", flush=True)
        zmean, zstd, maha = latent_stats(model, obs, act, state, val_idx, device, pmean, pstd)
        torch.save({"state": model.state_dict(),
                    "cfg": vars(args), "pmean": pmean, "pstd": pstd,
                    "zmean": zmean, "zstd": zstd, "maha_thresh": maha,
                    "meta": meta, "hist": hist,
                    "history": HISTORY, "future": FUTURE},
                   os.path.join(CKPT, f"pretrain_{args.name}.pt"))
        if args.max_iters and step >= args.max_iters:
            break
    print("done", json.dumps(hist), flush=True)


def latent_stats(model, obs, act, state, idx, device, pmean, pstd, bs=64, max_batches=12):
    """Latent mean/std and the 95th-percentile Mahalanobis radius used by the safety term."""
    model.eval()
    zs = []
    with torch.no_grad():
        for i in range(0, min(len(idx), bs * max_batches), bs):
            sel = idx[i:i + bs]
            o, p, a = batch_tensors(obs, act, state, sel, device, pmean, pstd)
            o = o.float() / 255.0
            zs.append(model.encode(o.reshape(-1, *o.shape[2:]), p.reshape(-1, p.shape[-1])))
    z = torch.cat(zs).cpu()
    zmean, zstd = z.mean(0).numpy(), z.std(0).clamp_min(1e-4).numpy()
    maha = (((z - torch.as_tensor(zmean)) / torch.as_tensor(zstd)) ** 2).sum(-1)
    return zmean, zstd, float(torch.quantile(maha, 0.95))


def load_model(name, device="cpu", lora=None):
    ck = torch.load(os.path.join(CKPT, f"pretrain_{name}.pt"), map_location="cpu",
                    weights_only=False)
    cfg = ck["cfg"]
    model = WorldModel(d=cfg["d"], proprio=4, k_history=ck["history"], a_dim=2,
                       s_blocks=cfg["s_blocks"], max_refine=cfg["max_refine"],
                       rank=cfg["rank"], lora=cfg["lora"] if lora is None else lora,
                       enc_ch=tuple(cfg["ch"]))
    model.load_state_dict(ck["state"])
    model.to(device).eval()
    return model, ck


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="push_four_shapes_train")
    ap.add_argument("--name", default="push4")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--ch", type=int, nargs="+", default=[32, 64, 96, 128])
    ap.add_argument("--s_blocks", type=int, default=2)
    ap.add_argument("--max_refine", type=int, default=4)
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--lora", type=int, default=1)
    ap.add_argument("--lam_unc", type=float, default=0.1)
    ap.add_argument("--lam_reg", type=float, default=0.1)
    ap.add_argument("--lam_inv", type=float, default=0.1)
    ap.add_argument("--log_every", type=int, default=25)
    ap.add_argument("--max_iters", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    train(ap.parse_args())
