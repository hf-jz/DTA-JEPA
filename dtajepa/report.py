"""Aggregate evaluation results into tables (markdown + LaTeX) and figures.

    python -m dtajepa.report --in results/raw.jsonl --out results
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAMILIES = [("shape", "Shape"), ("visual", "Visual"), ("dyn", "Dynamics"), ("layout", "Layout")]
METHOD_ORDER = ["frozen", "adajepa", "dtajepa", "dtajepa_nounc", "dtajepa_nomem",
                "dtajepa_nosafe", "dtajepa_fixedk", "dtajepa_noslow"]
PRETTY = {"frozen": "Frozen", "adajepa": "AdaJEPA", "dtajepa": "DTA-JEPA",
          "dtajepa_nounc": "DTA-JEPA w/o unc.", "dtajepa_nomem": "DTA-JEPA w/o memory",
          "dtajepa_nosafe": "DTA-JEPA w/o safety", "dtajepa_fixedk": "DTA-JEPA fixed K",
          "dtajepa_noslow": "DTA-JEPA w/o slow"}


def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    if "success" in df:
        df["success"] = df["success"] * 100.0
        df["curve"] = df["success_curve"].apply(lambda c: [100.0 * x for x in c])
    return df


def agg_table(df, group_cols):
    g = df.groupby(group_cols)
    out = g.agg(success=("success", "mean"), std=("success", "std"), n=("success", "size"),
                wall=("wall", "mean")).reset_index()
    out["std"] = out["std"].fillna(0.0)
    return out


def fmt(v, s=None):
    return f"{v:.1f}" + (f" $\\pm$ {s:.1f}" if s is not None else "")


def main_table(df):
    """Per-family, per-method success (main results table)."""
    rows = []
    for fam, label in FAMILIES:
        sub = df[df.suite == fam]
        if not len(sub):
            continue
        for meth in METHOD_ORDER:
            m = sub[sub.method == meth]
            if not len(m):
                continue
            rows.append({"family": label, "method": PRETTY.get(meth, meth),
                         "success": m.success.mean(), "std": m.success.std(ddof=1) if len(m) > 1 else 0.0,
                         "conds": len(m)})
    t = pd.DataFrame(rows)
    return t.pivot(index="method", columns="family", values="success")


def seen_unseen_table(df):
    sub = df[df.suite == "shape"]
    if not len(sub):
        return None
    sub = sub.copy()
    sub["split"] = sub.label.map(lambda s: "seen" if s in ("T", "L", "Z", "+") else "unseen")
    out = sub.pivot_table(index="method", columns="split", values="success", aggfunc="mean")
    return out


def curves(df, suite, labels=None, methods=("frozen", "adajepa", "dtajepa")):
    sub = df[(df.suite == suite)]
    if labels:
        sub = sub[sub.label.isin(labels)]
    out = {}
    for meth in methods:
        m = sub[sub.method == meth]
        if not len(m):
            continue
        L = min(len(c) for c in m["curve"])
        out[meth] = np.mean([c[:L] for c in m["curve"]], axis=0)
    return out


def figures(df, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)
    colors = {"frozen": "#666666", "adajepa": "#1f77b4", "dtajepa": "#d62728"}
    names = {"frozen": "Frozen", "adajepa": "AdaJEPA", "dtajepa": "DTA-JEPA"}

    # ---- Fig: success-vs-replan curves, 4 shift families -------------------
    panels = [("shape", ["T", "L", "Z", "+"], "Shape shift (seen)"),
              ("shape", ["I", "smallT", "square"], "Shape shift (unseen)"),
              ("visual", ["default", "blur", "snp", "dark"], "Visual shift (corruptions)"),
              ("visual", ["redAgent", "redBlock", "redAnchor"], "Visual shift (colour)"),
              ("dyn", None, "Dynamics shift (PointMaze)"),
              ("layout", None, "Layout shift (unseen mazes)")]
    have = [(a, b, c) for a, b, c in panels if len(df[df.suite == a])]
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5))
    for ax, (suite, labels, title) in zip(axes.ravel(), have):
        for meth, c in curves(df, suite, labels).items():
            x = np.arange(1, len(c) + 1)
            ax.plot(x, c, marker="o", ms=3, color=colors.get(meth, "k"),
                    label=names.get(meth, meth))
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("MPC replanning step")
        ax.set_ylabel("Planning success (%)")
        ax.set_ylim(0, 105)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    for ax in axes.ravel()[len(have):]:
        ax.axis("off")
    fig.tight_layout()
    p = os.path.join(outdir, "fig_success_curves.png")
    fig.savefig(p, dpi=170)
    plt.close(fig)
    print("figure ->", p)

    # ---- Fig: per-condition bars ------------------------------------------
    conds, groups, data = [], [], []
    for suite, _ in FAMILIES:
        cols = []
        for lab in df[df.suite == suite].label.unique():
            conds.append(f"{suite}:{lab}")
            cols.append(lab)
        groups.append((suite, cols))
    methods = [m for m in ("frozen", "adajepa", "dtajepa") if m in set(df.method)]
    fig, axes = plt.subplots(len(groups), 1, figsize=(11, 2.4 * len(groups)),
                             squeeze=False)
    for ax, (suite, labs) in zip(axes.ravel(), groups):
        sub = df[df.suite == suite]
        order = [l for l in labs]
        w = 0.8 / max(1, len(methods))
        for i, meth in enumerate(methods):
            vals, errs = [], []
            for lab in order:
                m = sub[(sub.method == meth) & (sub.label == lab)]
                vals.append(m.success.mean() if len(m) else np.nan)
                errs.append(m.success.std(ddof=1) if len(m) > 1 else 0.0)
            ax.bar(np.arange(len(order)) + i * w, vals, w, yerr=errs, capsize=2,
                   color=colors[meth], label=names[meth])
        ax.set_xticks(np.arange(len(order)) + 0.4 - w / 2)
        ax.set_xticklabels(order, fontsize=8)
        ax.set_ylabel("Success (%)")
        ax.set_ylim(0, 105)
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=8, ncol=3)
        ax.set_title(f"{suite} conditions", fontsize=10)
    fig.tight_layout()
    p = os.path.join(outdir, "fig_bars.png")
    fig.savefig(p, dpi=170)
    plt.close(fig)
    print("figure ->", p)

    # ---- Fig: adaptation statistics ---------------------------------------
    if "adapt_stats" in df and df.adapt_stats.notna().any():
        sub = df[(df.method == "dtajepa") & df.adapt_stats.apply(lambda s: bool(s))]
        if len(sub):
            stats = ["surprise", "steps", "lr", "updates", "retrieved"]
            fig, axes = plt.subplots(1, len(stats), figsize=(3.0 * len(stats), 2.6))
            for ax, s in zip(np.atleast_1d(axes), stats):
                vals = [float(d.get(s, np.nan)) for d in sub.adapt_stats]
                ax.hist(vals, bins=min(12, max(3, len(vals))), color="#d62728", alpha=0.8)
                ax.set_title(s, fontsize=9)
                ax.grid(alpha=0.3)
            fig.tight_layout()
            p = os.path.join(outdir, "fig_adapt_stats.png")
            fig.savefig(p, dpi=170)
            plt.close(fig)
            print("figure ->", p)


def latex_table(t, caption, label, float_fmt="%.1f"):
    cols = list(t.columns)
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{%s}" % caption, r"\label{%s}" % label,
             r"\begin{tabular}{l" + "c" * len(cols) + "}", r"\toprule",
             " & ".join([""] + [str(c) for c in cols]) + r" \\", r"\midrule"]
    for idx, row in t.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            vals.append("--" if pd.isna(v) else (float_fmt % v))
        lines.append(" & ".join([str(idx)] + vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="results/raw.jsonl")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    df = load(os.path.join(ROOT, a.inp))
    print(f"{len(df)} result rows, suites={sorted(df.suite.unique())}, "
          f"methods={sorted(df.method.unique())}")
    os.makedirs(os.path.join(ROOT, a.out), exist_ok=True)

    fam = main_table(df)
    su = seen_unseen_table(df)
    latin = []
    if len(fam):
        desc = fam.round(1).to_string()
        print("\n=== success by family (mean over conditions/seeds) ===\n" + desc)
        latin.append(latex_table(fam.round(1), "Planning success (\\%) by shift family, averaged over all conditions and seeds of each family.", "tab:family"))
    if su is not None and len(su):
        print("\n=== shape shift: seen vs unseen ===\n" + su.round(1).to_string())
        latin.append(latex_table(su.round(1), "Shape-shift success (\\%): seen (train) vs unseen (held-out) geometries.", "tab:seenunseen"))
    per_cond = agg_table(df, ["suite", "label", "method"]).round(1)
    per_cond.to_csv(os.path.join(ROOT, a.out, "per_condition.csv"), index=False)
    fam.to_csv(os.path.join(ROOT, a.out, "by_family.csv"))
    with open(os.path.join(ROOT, a.out, "tables.tex"), "w") as f:
        f.write("\n\n".join(latin) + "\n")
    df.to_csv(os.path.join(ROOT, a.out, "all_runs.csv"), index=False)
    figures(df, os.path.join(ROOT, a.out, "figs"))
    print("\nwrote", os.path.join(ROOT, a.out, "tables.tex"), "and figures")


if __name__ == "__main__":
    main()
