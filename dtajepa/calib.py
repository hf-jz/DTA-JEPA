"""Uncertainty diagnostics: calibration, surprise under shift, and the depth-error curve.

These are the three measurements the method's claims rest on, and none of them needs the
planner:

  1. calibration  : does predicted variance track realised squared error? (reliability curve,
                    Pearson/Spearman correlation, ECE)
  2. surprise     : is in-distribution surprise lower than shifted surprise? (the controller's
                    input signal must actually be informative)
  3. depth curve  : does recursive refinement reduce error monotonically in depth, and how much
                    of the gain is already obtained at depth 1? (Proposition 2)

    python -m dtajepa.calib --ckpt push4 --id push_eval_T --ood push_eval_I
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from . import data as D
from .envs import corrupt
from .train import CKPT, batch_tensors, load_model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@torch.no_grad()
def collect(model, ck, name, n=384, corrupt_kind=None, level=0.9, seed=0):
    """Per-transition predicted variance, squared error, and the full depth trace."""
    obs, act, state, meta = D.load(name)
    N, T = obs.shape[:2]
    rng = np.random.default_rng(seed)
    idx = np.stack([rng.integers(N, size=n), rng.integers(0, T - 7, size=n)], 1)
    o, p, a = batch_tensors(obs, act, state, idx, device="cpu", pmean=ck["pmean"], pstd=ck["pstd"])
    o = o.float() / 255.0
    if corrupt_kind:
        o2 = torch.empty_like(o)
        for i in range(o.shape[0]):
            for t in range(o.shape[1]):
                frame = (o[i, t].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                frame = corrupt(frame, corrupt_kind, level, np.random.default_rng(seed + i * 31 + t))
                o2[i, t] = torch.as_tensor(frame.transpose(2, 0, 1)).float() / 255.0
        o = o2
    B, L = o.shape[:2]
    k = model.k
    z = model.encode(o.reshape(B * L, *o.shape[2:]), p.reshape(B * L, -1)).reshape(B, L, -1)
    u = model.action(a.reshape(B * (L - 1), -1)).reshape(B, L - 1, -1)
    mean, logvar, depth, lvs, trace = model.predictor(z[:, :k], u[:, :k], u[:, k - 1],
                                                     return_trace=True)
    tgt = z[:, k].detach()
    err = ((mean - tgt) ** 2).sum(-1)
    var = torch.exp(logvar.squeeze(-1))          # per-sample predicted variance
    # error at every refinement depth
    depth_err = ((trace - tgt[:, None]) ** 2).sum(-1)          # (B, K_max)
    return dict(err=err.numpy(), var=var.numpy(), depth_err=depth_err.numpy(),
                _o=o, _p=p, _a=a,
                z=z.detach().numpy(), lat_std=float(z.reshape(-1, z.shape[-1]).std(0).mean()))





def geometry(ckpt_name, name, n=192, out="results/geometry.json", append=True):
    """Planning-free latent geometry panel for one checkpoint.

    Separates the two axes that a straightening weight moves in opposite directions:
      pred_err   -- next-step prediction error (the "accuracy" axis)
      curv       -- mean squared second difference per dim (what straightening minimises)
      sep        -- mean squared distance between distinct timesteps in a window (state
                    separation: what the goal cost needs to tell states apart)
      sep_over_curv -- the ratio the planner actually reads; a straight but uninformative
                    latent has a low ratio
      eff_dim    -- participation ratio of the latent covariance (how many dimensions carry
                    variance)
      ju_norm    -- mean ||dz_{t+1}/du_t|| (how much one action moves the latent)
    """
    model, ck = load_model(ckpt_name)
    model.eval()
    x = collect(model, ck, name, n=n)
    z = x["z"]
    B, L, d = z.shape
    err = float(np.mean(x["err"]))
    var = float(np.mean(x["var"]))
    d2 = np.diff(z, n=2, axis=1)
    curv = float((d2 ** 2).mean())
    if L > 1:
        iu = np.triu_indices(L, 1)
        sep = float(np.mean([((z[b, i] - z[b, j]) ** 2).mean() for b, i, j in
                             zip(range(B), iu[0], iu[1])]))
    else:
        sep = float("nan")
    zf = z.reshape(-1, d)
    cov = np.cov(zf.T)
    ev = np.linalg.eigvalsh(cov)
    ev = np.clip(ev, 1e-12, None)
    eff_dim = float(ev.sum() ** 2 / (ev ** 2).sum())
    ju = float("nan")
    try:
        # ||d z_hat_{t+1} / d u_t||: how much one action moves the predicted latent
        with torch.no_grad():
            za = model.encode(x["_o"].reshape(B * L, *x["_o"].shape[2:]),
                              x["_p"].reshape(B * L, -1)).reshape(B, L, -1)
            u = model.action(x["_a"].reshape(B * (L - 1), -1)).reshape(B, L - 1, -1)
        u_next = u[:, model.k - 1].detach().clone().requires_grad_(True)
        pred = model.predictor(za[:, :model.k].detach(), u[:, :model.k].detach(), u_next)[0]
        g = torch.autograd.grad(pred.mean(), u_next)[0]
        ju = float(g.norm(dim=-1).mean())
    except Exception as e:                        # geometry must never break the panel
        print("  ju_norm unavailable:", e)
    res = dict(ckpt=ckpt_name, dset=name, n=n, pred_err=err, pred_var=var, curv=curv,
               sep=sep, sep_over_curv=sep / max(curv, 1e-12), eff_dim=eff_dim, ju_norm=ju,
               lat_std=x["lat_std"])
    rows = []
    fp = os.path.join(ROOT, out)
    if append and os.path.exists(fp):
        rows = [json.loads(l) for l in open(fp) if l.strip()]
    rows = [r for r in rows if (r["ckpt"], r["dset"]) != (ckpt_name, name)] + [res]
    with open(fp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print("geometry", json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                                  for k, v in res.items()}))
    return res

def fit_from_records(records, out):
    """Fit the STLS constants on the *closed-loop* surprise trace of the frozen baseline.

    The open-loop (teacher-forced) stream is not the stream the controller sees: closed-loop
    next-step surprise runs several times higher (measured 169 vs 25 on the strengthened model),
    so thresholds fitted open-loop are miscalibrated for the loop they govern. Reusing the frozen
    baseline of the same sweep costs nothing extra.

    trace rows are (err, predicted_variance, n_refine, curvature).
    """
    tr = [t for r in records
          for t in ((r.get("adapt_stats") or {}).get("surprise_trace") or [])]
    if len(tr) < 8:
        raise SystemExit("not enough trace rows to fit (%d)" % len(tr))
    err = np.array([t[0] for t in tr], dtype=float)
    unc = np.maximum(np.array([t[1] for t in tr], dtype=float), 1e-6)
    curv = np.array([t[3] if t[3] is not None else np.nan for t in tr], dtype=float)
    ratio = err / unc
    tau = float(np.mean(ratio))
    s_cal = ratio / tau
    res = dict(source="closed_loop_frozen", n_updates=int(len(tr)), tau=tau,
               q_mid=float(np.quantile(s_cal, 0.75)), q_high=float(np.quantile(s_cal, 0.95)),
               curv_mid=float(np.nanmedian(curv)) if np.isfinite(curv).any() else None,
               mean_surprise_raw=float(np.mean(err)), mean_ratio=float(np.mean(ratio)))
    with open(os.path.join(ROOT, out), "w") as f:
        json.dump(res, f, indent=1)
    print("STLS calibration (closed loop):", json.dumps(
        {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()}))
    return res

def fit_stls(ckpt_name, id_name, n=384, out="results/calib_stls.json"):
    """Fit the STLS allocation constants on in-distribution held-out transitions.

    tau     : temperature for the predicted variance, tau = mean(e^2 / sigma^2), so that
              calibrated surprise has mean one on data the model was trained for;
    q_mid/q_high : quantiles of the calibrated surprise (the allocation thresholds);
    curv_mid     : median trajectory curvature (second difference), the depth gate.
    """
    model, ck = load_model(ckpt_name)
    model.eval()
    x = collect(model, ck, id_name, n=n)
    ratio = x["err"] / np.maximum(x["var"], 1e-9)
    tau = float(np.mean(ratio))
    s_cal = ratio / tau
    d2 = np.diff(x["z"], n=2, axis=1) if x["z"].shape[1] >= 3 else np.zeros_like(x["z"][:, :1])
    curv = (d2 ** 2).sum(-1).mean(1)
    res = dict(ckpt=ckpt_name, id=id_name, tau=tau,
               q_mid=float(np.quantile(s_cal, 0.75)), q_high=float(np.quantile(s_cal, 0.95)),
               curv_mid=float(np.median(curv)), n=n,
               mean_surprise_raw=float(np.mean(ratio)),
               mean_surprise_cal=float(np.mean(s_cal)))
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, out), "w") as f:
        json.dump(res, f, indent=1)
    print("STLS calibration:", json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                                           for k, v in res.items()}))
    return res

def calibrate(x):
    """Reliability + correlation between predicted variance and realised squared error."""
    err, var = x["err"], var_to_std(x["var"])
    order = np.argsort(var)
    bins = np.array_split(order, 10)
    pred = np.array([var[b].mean() for b in bins])
    real = np.array([err[b].mean() for b in bins])
    ece = float(np.mean(np.abs(pred - real)) / (real.mean() + 1e-12))
    pear = float(np.corrcoef(var, err)[0, 1])
    spear = float(np.corrcoef(np.argsort(np.argsort(var)), np.argsort(np.argsort(err)))[0, 1])
    return dict(ece_rel=ece, pearson=pear, spearman=spear,
                bins_pred=pred.tolist(), bins_real=real.tolist())


def var_to_std(v):
    return np.sqrt(np.maximum(v, 0.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="push4")
    ap.add_argument("--id", default="push_eval_T")
    ap.add_argument("--ood", default="push_eval_I")
    ap.add_argument("--corrupt", default="blur")
    ap.add_argument("--n", type=int, default=384)
    ap.add_argument("--out", default="results/calib.json")
    ap.add_argument("--fit_stls", action="store_true")
    ap.add_argument("--fit_records", action="store_true")
    ap.add_argument("--geometry", action="store_true")
    ap.add_argument("--in", dest="inp", default=None)
    a = ap.parse_args()
    if a.geometry:
        geometry(a.ckpt, a.id, n=a.n, out=a.out)
        raise SystemExit(0)
    if a.fit_records:
        recs = [json.loads(l) for l in open(a.inp) if l.strip()]
        fit_from_records(recs, a.out)
        raise SystemExit(0)
    if a.fit_stls:
        fit_stls(a.ckpt, a.id, n=a.n, out=a.out)
        raise SystemExit(0)
    torch.set_num_threads(4)
    model, ck = load_model(a.ckpt)
    model.eval()
    res = {}
    for tag, name, ck_kind in (("in_distribution", a.id, None),
                               ("unseen_shape", a.ood, None),
                               ("visual_blur", a.id, a.corrupt)):
        x = collect(model, ck, name, n=a.n, corrupt_kind=ck_kind)
        res[tag] = dict(calibrate(x), mean_surprise=float(np.mean(x["err"] / (x["var"] + 1e-6))),
                        mean_err=float(np.mean(x["err"])), mean_var=float(np.mean(x["var"])),
                        depth_curve=[float(v) for v in x["depth_err"].mean(0)],
                        lat_std=x["lat_std"])
        print(f"{tag:16s} err={res[tag]['mean_err']:.4f} var={res[tag]['mean_var']:.4f} "
              f"pearson={res[tag]['pearson']:.3f} spearman={res[tag]['spearman']:.3f} "
              f"ece_rel={res[tag]['ece_rel']:.3f}")
        print("   depth curve:", " ".join(f"{v:.4f}" for v in res[tag]["depth_curve"]))
    ratio = res["unseen_shape"]["mean_err"] / max(res["in_distribution"]["mean_err"], 1e-9)
    res["surprise_ratio_unseen_over_id"] = float(ratio)
    print(f"unseen/in-distribution error ratio: {ratio:.2f}x")
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, a.out), "w") as f:
        json.dump(res, f, indent=1)
    print("wrote", os.path.join(ROOT, a.out))


if __name__ == "__main__":
    main()
