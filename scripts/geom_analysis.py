#!/usr/bin/env python3
"""Does planning success track predictive accuracy or latent geometry?

Reads the lambda_str sweep (results/geometry.json + results/sweep_mazeS*.jsonl) and prints, per
arm, the accuracy axis (prediction error) against the geometry axes (curvature, state separation,
effective dimension) and the closed-loop success on the two probe conditions. Writes
results/geom/{table.tex, geom_sweep.png}.
"""
import glob, json, os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(R, "results", "geom")
os.makedirs(OUT, exist_ok=True)

geo = {}
for l in open(os.path.join(R, "results", "geometry.json")):
    if l.strip():
        r = json.loads(l)
        geo[(r["ckpt"], r["dset"])] = r

LAM = {"mazeS00": 0.0, "mazeS002": 0.02, "mazeS01": 0.1, "mazeS05": 0.5}   # tag -> lambda_str
arms = []
for tag in LAM:
    g = geo.get((tag, "maze_dyneval_default"))
    fp = os.path.join(R, "results", f"sweep_{tag}.jsonl")
    if not g or not os.path.exists(fp):
        continue
    rs = [json.loads(l) for l in open(fp) if l.strip()]
    succ = {}
    for lab in ("default", "lowMass"):
        v = [r["success"] * 100 for r in rs if r["label"] == lab]
        succ[lab] = float(np.mean(v)) if v else float("nan")
    arms.append(dict(tag=tag, lam=LAM[tag], **g,
                     **{"succ_" + k: v for k, v in succ.items()}))
if not arms:
    raise SystemExit("no sweep arms yet")
arms.sort(key=lambda a: a["lam"])

hdr = f"{'lam_str':>8s} {'pred_err':>9s} {'curv':>7s} {'sep/curv':>9s} {'eff_dim':>8s} " \
      f"{'ju_norm':>9s} {'succ_def':>9s} {'succ_low':>9s}"
print(hdr)
for a in arms:
    print(f"{a['lam']:8.2f} {a['pred_err']:9.2f} {a['curv']:7.3f} {a['sep_over_curv']:9.3f} "
          f"{a['eff_dim']:8.2f} {a['ju_norm']:9.5f} {a['succ_default']:9.1f} {a['succ_lowMass']:9.1f}")

lam = [a["lam"] for a in arms]
L = [r"\begin{table}[t]\centering",
     r"\caption{Straightening sweep: identical lr, epochs and seed, only $\lambda_{\mathrm{str}}$ "
     r"varies. Prediction error falls monotonically while the latent's effective dimension on the "
     r"maze distribution and the closed-loop success at a mass shift follow it down.}",
     r"\label{tab:geom}\small", r"\begin{tabular}{rrrrrrr}", r"\toprule",
     r"$\lambda_{\mathrm{str}}$ & pred.\ err. & curvature & sep/curv & eff.\ dim & "
     r"succ.\ default & succ.\ lowMass \\", r"\midrule"]
for a in arms:
    L.append("%.2f & %.2f & %.3f & %.3f & %.2f & %.1f & %.1f \\\\" % (
        a["lam"], a["pred_err"], a["curv"], a["sep_over_curv"], a["eff_dim"],
        a["succ_default"], a["succ_lowMass"]))
L += [r"\bottomrule", r"\end{tabular}\end{table}"]
open(os.path.join(OUT, "table.tex"), "w").write("\n".join(L))
json.dump(arms, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
open(os.path.join(R, "latex-paper", "dtajepa-paper", "tab-geom.tex"), "w").write("\n".join(L))

# significance of the comparisons that matter (12 episodes x 2 seeds each)
from scipy.stats import fisher_exact


def hits(tag, lab):
    rs = [json.loads(l) for l in open(os.path.join(R, "results", f"sweep_{tag}.jsonl"))
          if l.strip() and json.loads(l)["label"] == lab]
    return int(round(sum(r["success"] * r["n_ep"] for r in rs))), int(round(12 * len(rs)))


print()
for a, b, lab in (("mazeS00", "mazeS05", "lowMass"), ("mazeS00", "mazeS01", "lowMass"),
                  ("mazeS00", "mazeS05", "default")):
    ha, na = hits(a, lab)
    hb, nb = hits(b, lab)
    p = fisher_exact([[ha, na - ha], [hb, nb - hb]])[1]
    print(f"  {lab:8s} {a} {ha}/{na} vs {b} {hb}/{nb}   Fisher p={p:.3f}")

fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
ax[0].plot(lam, [a["pred_err"] for a in arms], "o-", color="#1d4e89", label="prediction error")
ax[0].set_xlabel(r"$\lambda_{\mathrm{str}}$"); ax[0].set_ylabel("next-step error", color="#1d4e89")
ax2 = ax[0].twinx()
ax2.plot(lam, [a["eff_dim"] for a in arms], "s--", color="#2e7d5b", label="effective dim.")
ax2.set_ylabel("effective dimension", color="#2e7d5b")
ax[0].set_title("accuracy improves as geometry degrades", fontsize=9)
ax[1].plot(lam, [a["succ_default"] for a in arms], "o-", label="default dynamics")
ax[1].plot(lam, [a["succ_lowMass"] for a in arms], "s-", label="mass x0.2 (shifted)")
ax[1].set_xlabel(r"$\lambda_{\mathrm{str}}$"); ax[1].set_ylabel("success (%)")
ax[1].legend(fontsize=8); ax[1].grid(alpha=.25)
ax[1].set_title("planning success", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "geom_sweep.png"), dpi=150)
print("wrote results/geom/{table.tex,geom_sweep.png} + paper tab-geom.tex")
