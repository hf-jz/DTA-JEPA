#!/usr/bin/env python3
"""STLS-JEPA vs DTA-JEPA/AdaJEPA/frozen on the same (straightened) models.

Writes results/stls/summary.json + tables.tex + fig_stls.png, and prints the four
pre-registered observables from docs/STLS-JEPA.md.

Adapter summary keys (see dtajepa/adapt.py): updates (count), steps (cumulative GD steps),
lr (mean over updates), surprise (mean raw MSE), anchor_proj (count), depth (list).
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(R, "results", "stls")
os.makedirs(OUT, exist_ok=True)
N_EP = 12
NAME = {"frozen": "Frozen", "adajepa": "AdaJEPA", "dtajepa": "DTA-JEPA", "stls": "STLS-JEPA"}
METHODS = ["frozen", "adajepa", "dtajepa", "stls"]
COLOR = {"frozen": "#8a8f98", "adajepa": "#c98a2e", "dtajepa": "#1d4e89", "stls": "#2e7d5b"}


def mean_of(rs, method, fn):
    v = [fn(r) for r in rs if r["method"] == method and fn(r) is not None]
    v = [x for x in v if isinstance(x, (int, float)) and not np.isnan(x)]
    return float(np.mean(v)) if v else float("nan")


def _st(r):
    return r.get("adapt_stats") or {}


def steps_per_update(r):
    u = _st(r).get("updates") or 0
    return (_st(r).get("steps", 0) / u) if u else None


def anchor_per_ep(r):
    a = _st(r).get("anchor_proj_total")
    return (a / N_EP) if a is not None else None


rows = {}
for suite, fn in (("shape", "stls_shape.jsonl"), ("dyn", "stls_dyn.jsonl")):
    p = os.path.join(R, "results", fn)
    if not os.path.exists(p):
        print("missing", fn); continue
    rs = [json.loads(l) for l in open(p) if l.strip()]
    print(f"{fn}: {len(rs)} rows, suites={sorted({r['suite'] for r in rs})}, "
          f"labels={sorted({r.get('label') for r in rs})}, methods={sorted({r['method'] for r in rs})}")
    rows[suite] = rs
if not rows:
    raise SystemExit("no STLS results yet")

METRICS = {"success": ("success (%)", lambda r: r["success"] * 100),
           "steps": ("GD steps / update", steps_per_update),
           "lr": ("mean lr", lambda r: _st(r).get("lr")),
           "depth": ("refine depth", lambda r: _st(r).get("depth")),
           "anchor": ("anchor proj. / episode", anchor_per_ep),
           "rollback": ("rollbacks / episode", lambda r: _st(r).get("rollback"))}

summary = {}
for suite, rs in rows.items():
    entry = {"n_conditions": len({(r["label"]) for r in rs})}
    for key, (_, fn) in METRICS.items():
        entry[key] = {m: round(mean_of(rs, m, fn), 4) for m in METHODS}
    summary[suite] = entry
json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1)

for suite, s in summary.items():
    print(f"\n== {suite} ({s['n_conditions']} conditions x {N_EP} ep x 2 seeds) ==")
    for key, (label, _) in METRICS.items():
        print(f"  {label:24s}", {k: s[key][k] for k in METHODS})

L = [r"\begin{table}[t]\centering",
     r"\caption{STLS-JEPA against the three baselines on identical models and episodes. "
     r"Diagnostics are the pre-registered observables of Table~\ref{tab:stls}.}",
     r"\label{tab:stlsres}\small", r"\begin{tabular}{llrrrrr}", r"\toprule",
     r"Family & Method & Success (\%) & Steps/upd & Depth & Anchor/ep & Rollback \\", r"\midrule"]
def fmt(x, nd=2):
    return "--" if (x is None or (isinstance(x, float) and np.isnan(x)) or x < 0) else f"{x:.{nd}f}"


for suite, s in summary.items():
    for m in METHODS:
        v = s["success"][m]
        if np.isnan(v):
            continue
        L.append("%s & %s & %.1f & %s & %s & %s & %s \\\\" % (
            suite, NAME[m], v, fmt(s["steps"][m]), fmt(s["depth"][m]),
            fmt(s["anchor"][m], 1), fmt(s["rollback"][m])))
    L.append(r"\midrule")
L[-1] = r"\bottomrule"
L += [r"\end{tabular}\end{table}"]
open(os.path.join(OUT, "tables.tex"), "w").write("\n".join(L))
open(os.path.join(R, "latex-paper", "dtajepa-paper",
                  "tab-stlsres.tex"), "w").write("\n".join(L))

fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
suites = list(summary)
xs = np.arange(len(suites)); w = 0.2
ax = axes[0]
for i, m in enumerate(METHODS):
    vals = [summary[s]["success"].get(m, np.nan) for s in suites]
    b = ax.bar(xs + (i - 1.5) * w, vals, w, label=NAME[m], color=COLOR[m])
    ax.bar_label(b, fmt="%.0f", fontsize=7)
ax.set_xticks(xs); ax.set_xticklabels(suites); ax.set_ylabel("success (%)"); ax.set_ylim(0, 105)
ax.legend(fontsize=7, ncol=2); ax.grid(axis="y", alpha=.25)
ax.set_title("goal reached on identical models/episodes", fontsize=9)
ax = axes[1]
for i, m in enumerate(METHODS):
    vals = [summary[s]["steps"].get(m, np.nan) for s in suites]
    b = ax.bar(xs + (i - 1.5) * w, vals, w, color=COLOR[m])
    ax.bar_label(b, fmt="%.1f", fontsize=7)
ax.axhline(2.2, ls="--", lw=1, color="#555")
ax.text(0.02, 2.25, "DTA baseline 2.2", fontsize=7, color="#555")
ax.set_xticks(xs); ax.set_xticklabels(suites)
ax.set_ylabel("GD steps per adaptation update"); ax.grid(axis="y", alpha=.25)
ax.set_title("allocation: ratio (saturating) vs thresholded", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_stls.png"), dpi=150)   # site.py stages this into docs/figs
print("\nwrote results/stls/{summary.json,tables.tex,fig_stls.png} + docs/figs/fig_stls.png")
print(open(os.path.join(OUT, "tables.tex")).read())
