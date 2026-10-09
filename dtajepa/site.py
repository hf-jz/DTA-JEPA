"""Generate the GitHub Pages project site (docs/index.html) from the evaluation results.

    python -m dtajepa.site            # writes docs/index.html + docs/figs/*.png
"""
from __future__ import annotations

import json
import os
import shutil

import numpy as np

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
DOCS = os.path.join(ROOT, "docs")

PRETTY = {"frozen": "Frozen", "adajepa": "AdaJEPA", "dtajepa": "DTA-JEPA",
          "dtajepa_meta": "DTA-JEPA + meta init", "dtajepa_nometa": "DTA-JEPA w/o meta init",
          "dtajepa_lora": "LoRA fast set (r=4)",
          "dtajepa_nounc": "w/o uncertainty allocation", "dtajepa_nomem": "w/o dual memory",
          "dtajepa_nosafe": "w/o safety layer", "dtajepa_fixedk": "fixed refinement depth",
          "dtajepa_noslow": "w/o cross-episode consolidation"}
FAM = ["Shape", "Visual", "Dynamics", "Layout"]


def table_html(df, index_name="", highlight="DTA-JEPA"):
    head = "".join(f"<th>{c}</th>" for c in [index_name] + list(df.columns))
    rows = []
    for idx, r in df.iterrows():
        tds = []
        for c in df.columns:
            v = r[c]
            txt = "—" if pd.isna(v) else f"{v:.1f}"
            cls = ' class="hl"' if str(idx) == highlight else ""
            tds.append(f"<td{cls}>{txt}</td>")
        rows.append(f"<tr><th scope=\"row\">{idx}</th>{''.join(tds)}</tr>")
    return (f'<table><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table>')



def parity_html(parity, traj_figs, figs_dst):
    """Verification inventory + the parity runs, rendered from raw JSONL."""
    P = {"frozen": "Frozen", "adajepa": "AdaJEPA", "dtajepa": "DTA-JEPA"}
    out = []
    inv = [
        ("Fig. 2 top --- 7 shapes, curves to 30 MPC steps", "success vs step",
         "reproduced (GD, 30 steps, 6 episodes/condition); CEM spot-check below"),
        ("Fig. 2 bottom --- visual shifts (blur/snp/dark/red&nbsp;&times;3)", "success vs step",
         "reproduced, but degenerate: all eight conditions coincide, see the no-op finding"),
        ("Table 1 --- PointMaze dynamics + layout shifts", "success",
         "reproduced for GD (main sweep); CEM not run on the maze"),
        ("Fig. 4/5 --- planning trajectories", "qualitative",
         "reproduced below (agent and object paths per replan, goal marked)"),
        ("Fig. 8 --- which parameters to adapt", "env-average success",
         "partly: we ablate the fast set (direct vs LoRA), memory, safety, refinement depth, "
         "slow consolidation and meta-init; not their predfirst / encfrozen taps"),
        ("Fig. 9 --- test-time lr, steps, replay buffer", "success vs knob",
         "reproduced below on the same axes"),
        ("Fig. 6 --- data scale (shapes K &times; trajectories N)", "success vs data",
         "not run: needs 4--8 additional pretrainings (~30 min each on this machine)"),
        ("Fig. 7 --- decoded imagination", "qualitative",
         "not run: needs a decoder trained on the latent space"),
        ("Table 2 --- across JEPA implementations", "success",
         "not run: requires DINOv2 weights and their released checkpoints"),
        ("their released checkpoints / eval pickles", "---",
         "unobtainable here: <code>drive.google.com</code> unreachable, "
         "<code>dl.fbaipublicfiles.com</code> 403, and <code>torchvision / einops / hydra / "
         "mujoco</code> are not installed"),
    ]
    rows = "".join(f"<tr><th scope=\"row\">{a}</th><td>{b}</td><td>{c}</td></tr>"
                   for a, b, c in inv)
    out.append("<h2>Verification against the paper's examples</h2>"
               "<p class=\"sub\">Every figure and table of AdaJEPA, with what we were able to "
               "put on the same yardstick. Running their code on their examples was not possible "
               "in this environment: the probe below is the evidence, not an excuse.</p>"
               "<table><thead><tr><th>AdaJEPA artifact</th><th>Yardstick</th>"
               f"<th>Status here</th></tr></thead><tbody>{rows}</tbody></table>")

    if "curves" in parity:
        rows_ = parity["curves"]
        conds = sorted({r["label"] for r in rows_})
        meths = ["frozen", "adajepa", "dtajepa"]
        hdr = "".join(f"<th>{P[m]}</th>" for m in meths)
        body = ""
        for c in conds:
            tds = ""
            for m in meths:
                r = [x for x in rows_ if x["label"] == c and x["method"] == m]
                tds += f"<td>{r[0]['success']:.0f}</td>" if r else "<td>--</td>"
            body += f"<tr><th scope=\"row\">{c}</th>{tds}</tr>"
        out.append("<h3>Success at 30 MPC steps &mdash; their Fig. 2 x-axis</h3>"
                   "<p class=\"sub\">Final success after 30 replanning steps (6 episodes per "
                   "condition, GD planner, our affordable planner budget).</p>"
                   f"<table><thead><tr><th>Shape</th>{hdr}</tr></thead><tbody>{body}</tbody></table>")
        if os.path.exists(os.path.join(figs_dst, "fig_parity_curves.png")):
            out.append("<figure><img src=\"figs/fig_parity_curves.png\" alt=\"30-step curves\">"
                       "<figcaption>Success versus MPC replanning step over 30 steps, averaged "
                       "over the 7 shapes (their Fig. 2 top x-axis).</figcaption></figure>")
        if traj_figs:
            out.append("<h3>Planning trajectories &mdash; their Fig. 4/5</h3>"
                       "<p class=\"sub\">Agent path per replanning step under each method "
                       "(dotted = episode failed, solid = reached the goal); gold star = goal, "
                       "thin lines = manipulated object. Shift family in each panel title.</p>")
            for f in traj_figs:
                nm = os.path.basename(f)
                out.append(f'<figure><img src="figs/{nm}" alt="{nm}"></figure>')

    if "sweep" in parity:
        rows_ = parity["sweep"]
        keys = sorted({(r.get("adapt", {}).get("lr"), r.get("adapt", {}).get("steps"),
                        r.get("adapt", {}).get("buffer")) for r in rows_},
                      key=lambda t: (t[0] or 0, t[1] or 0, t[2] or 0))
        meths = ["adajepa", "dtajepa"]
        body = ""
        for lr, st, bf in keys:
            if None in (lr, st, bf):
                continue
            tds = ""
            for m in meths:
                sel = [r for r in rows_ if r.get("adapt", {}).get("lr") == lr
                       and r.get("adapt", {}).get("steps") == st
                       and r.get("adapt", {}).get("buffer") == bf and r["method"] == m]
                if sel:
                    m_ = sum(x["success"] for x in sel) / len(sel)
                    tds += f"<td>{m_:.0f}</td>"
                else:
                    tds += "<td>--</td>"
            body += (f"<tr><th scope=\"row\">lr {lr/5e-4:.1f}&times;, {st} step"
                     f"{'s' if st != 1 else ''}, buffer {bf}</th>{tds}</tr>")
        out.append("<h3>Test-time adaptation hyper-parameters &mdash; their Fig. 9 axes</h3>"
                   "<p class=\"sub\">Success (%) on the shape suite, averaged over the "
                   "conditions of that configuration (6 episodes each).</p>"
                   "<table><thead><tr><th>Configuration</th><th>AdaJEPA</th><th>DTA-JEPA</th>"
                   f"</tr></thead><tbody>{body}</tbody></table>")

    if "cem" in parity:
        rows_ = parity["cem"]
        conds = sorted({r["label"] for r in rows_})
        meths = ["frozen", "adajepa", "dtajepa"]
        hdr = "".join(f"<th>{P[m]}</th>" for m in meths)
        body = ""
        for c in conds:
            tds = "".join(
                (lambda r: f"<td>{r[0]['success']:.0f}</td>" if r else "<td>--</td>")(
                    [x for x in rows_ if x["label"] == c and x["method"] == m]) for m in meths)
            body += f"<tr><th scope=\"row\">{c}</th>{tds}</tr>"
        out.append("<h3>CEM planner &mdash; they report GD and CEM for every family</h3>"
                   "<p class=\"sub\">Cross-entropy-method spot check (64 samples, 5 iterations, "
                   "4 episodes, 4 replans): the ordering between methods should not depend on the "
                   "planner.</p>"
                   f"<table><thead><tr><th>Shape</th>{hdr}</tr></thead><tbody>{body}</tbody></table>")
    return "".join(out)



def parity_curves_figure(outdir):
    """One panel per shape, success vs replanning step, 3 methods (AdaJEPA Fig. 2 layout)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fp = os.path.join(RES, "parity_curves.jsonl")
    if not os.path.exists(fp):
        return None
    rows = [json.loads(l) for l in open(fp) if l.strip()]
    shts = sorted({r["label"] for r in rows})
    n = max(1, len(shts))
    cols = 4
    nrow = (n + cols - 1) // cols
    fig, axes = plt.subplots(nrow, cols, figsize=(3.1 * cols, 2.5 * nrow), squeeze=False,
                             sharex=True)
    colr = {"frozen": "#9aa1ab", "adajepa": "#4a86c8", "dtajepa": "#c0504d"}
    nam = {"frozen": "Frozen", "adajepa": "AdaJEPA", "dtajepa": "DTA-JEPA"}
    for ax, sh in zip(axes.ravel(), shts):
        for m in ("frozen", "adajepa", "dtajepa"):
            sel = [r for r in rows if r["label"] == sh and r["method"] == m]
            if not sel:
                continue
            cur = 100.0 * np.mean([r["success_curve"] for r in sel], axis=0)
            ax.plot(np.arange(1, len(cur) + 1), cur, lw=1.7, color=colr[m], label=nam[m])
        ax.set_title(sh, fontsize=10)
        ax.set_ylim(0, 105)
        ax.grid(alpha=.25)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    for ax in axes.ravel()[max(0, n - cols):n]:
        ax.set_xlabel("MPC replanning step")
    for r_ in range(nrow):
        axes[r_][0].set_ylabel("Success (%)")
    axes.ravel()[0].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    out = os.path.join(outdir, "fig_parity_curves.png")
    fig.savefig(out, dpi=165)
    plt.close(fig)
    print("figure ->", out)
    return out

def traj_figures(outdir):
    """Agent/object trajectories per MPC replan, in the style of AdaJEPA's Fig. 4/5."""
    import glob
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)
    made = []
    for kind, tag in (("shape", "push"), ("dyn", "maze"), ("layout", "maze-layouts")):
        files = sorted(glob.glob(os.path.join(RES, "traj", f"{kind}_*_*.npz")))
        if not files:
            continue
        labels = [os.path.splitext(os.path.basename(f))[0].replace(f"{kind}_", "").rsplit("_", 1)
                  for f in files]
        conds = sorted({a for a, _ in labels})
        methods = [m for m in ("frozen", "adajepa", "dtajepa") if any(b == m for _, b in labels)]
        fig, axes = plt.subplots(1, min(4, len(conds)), figsize=(3.3 * min(4, len(conds)), 3.4),
                                 squeeze=False)
        for ax, cond in zip(axes.ravel(), conds[:4]):
            for meth, col in zip(methods, ("#9aa1ab", "#8fb4e3", "#d64541")):
                f = os.path.join(RES, "traj", f"{kind}_{cond}_{meth}.npz")
                if not os.path.exists(f):
                    continue
                d = np.load(f)
                keys = [k for k in d.files if k.startswith("ep")]
                ok = d["success"]
                for i, k in enumerate(keys[:4]):
                    tr = d[k]
                    sty = dict(color=col, lw=1.6, alpha=.9 if ok[i] else .45,
                               ls="-" if ok[i] else ":")
                    ax.plot(tr[:, 0], tr[:, 1], **sty)
                    if kind == "shape":                      # block path
                        ax.plot(tr[:, 2], tr[:, 3], color=col, lw=1.0, alpha=.35)
                    ax.plot(tr[0, 0], tr[0, 1], marker="o", ms=4, color=col)
                    g = d["goals"][i]
                    ax.plot(g[0] if kind == "shape" else g[0], g[1] if kind == "shape" else g[1],
                            marker="*", ms=11, color="#d8b24a", markeredgecolor="#0f1114")
            ax.set_title(f"{tag}: {cond}", fontsize=10)
            ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05)
            ax.set_aspect("equal"); ax.grid(alpha=.18)
            handles = [plt.Line2D([], [], color=c, lw=2, label={"frozen": "Frozen",
                                                                "adajepa": "AdaJEPA",
                                                                "dtajepa": "DTA-JEPA"}[m])
                       for m, c in zip(methods, ("#9aa1ab", "#8fb4e3", "#d64541"))]
            handles.append(plt.Line2D([], [], color="#d8b24a", marker="*", ls="", ms=10,
                                      label="goal"))
            ax.legend(handles=handles, fontsize=7)
        fig.tight_layout()
        out = os.path.join(outdir, f"fig_traj_{tag.replace('-', '_')}.png")
        fig.savefig(out, dpi=170); plt.close(fig)
        made.append(out)
        print("figure ->", out)
    return made


def stls_html():
    """STLS-JEPA section: the four repairs, the pre-registered observables and the outcome.

    Reads results/stls/summary.json (written by scripts/stls_compare.py); returns "" when the
    comparison has not been run so the site still builds.
    """
    fp = os.path.join(ROOT, "results", "stls", "summary.json")
    if not os.path.exists(fp):
        return ""
    # site.py rebuilds docs/figs from results/*/figs, so a figure written straight into
    # docs/figs by another script gets wiped: stage it here instead (same as layout_figure)
    fig_src = os.path.join(ROOT, "results", "stls", "fig_stls.png")
    if os.path.exists(fig_src):
        shutil.copy(fig_src, os.path.join(ROOT, "docs", "figs", "fig_stls.png"))
    fig_g = os.path.join(ROOT, "results", "geom", "geom_sweep.png")
    if os.path.exists(fig_g):
        shutil.copy(fig_g, os.path.join(ROOT, "docs", "figs", "fig_geom_sweep.png"))
    S = json.load(open(fp))
    NAME = {"frozen": "Frozen", "adajepa": "AdaJEPA", "dtajepa": "DTA-JEPA", "stls": "STLS-JEPA"}
    METHODS = [m for m in ("frozen", "adajepa", "dtajepa", "stls")]
    rows = ""
    for suite, s in S.items():
        for m in METHODS:
            v = s["success"].get(m)
            if v is None or v != v:
                continue
            cls = ' class="hl"' if m == "stls" else ""
            rows += (f"<tr{cls}><td>{suite}</td><td>{NAME[m]}</td><td>{v:.1f}</td>"
                     f"<td>{s['steps'][m]:.2f}</td><td>{s['depth'][m]:.2f}</td>"
                     f"<td>{s['anchor'][m]:.1f}</td><td>{s['rollback'][m]:.1f}</td></tr>")
    gfp = os.path.join(ROOT, "results", "geom", "summary.json")
    geom_tbl = ""
    if os.path.exists(gfp):
        A = json.load(open(gfp))
        tr = "".join(
            f"<tr><td>{a['lam']:.2f}</td><td>{a['pred_err']:.2f}</td><td>{a['curv']:.3f}</td>"
            f"<td>{a['sep_over_curv']:.3f}</td><td>{a['eff_dim']:.2f}</td>"
            f"<td>{a['succ_default']:.1f}</td><td>{a['succ_lowMass']:.1f}</td></tr>" for a in A)
        geom_tbl = ("<table><thead><tr><th>&lambda;str</th><th>pred. err</th><th>curv</th>"
                    "<th>sep/curv</th><th>eff. dim</th><th>succ default</th>"
                    "<th>succ lowMass</th></tr></thead><tbody>" + tr + "</tbody></table>")
    return f"""
<h2>STLS-JEPA: four repairs, each from a measured failure</h2>
<p class="sub">DTA-JEPA's diagnostics said <em>where</em> it fails. STLS-JEPA
(<code>Surprise-Thresholded Latent Straightening</code>) acts on exactly those four points and
changes nothing else — same encoder, same dual-timescale predictor, same planner, same episodes.
The observables below were pre-registered in <code>docs/STLS-JEPA.md</code> before the run.</p>
<table>
<thead><tr><th>Diagnosis (measured)</th><th>STLS change</th></tr></thead>
<tbody>
<tr><td>Planner-limited regime: families saturate, methods do not separate</td>
<td>Latent <b>straightening</b> in pretraining (second difference through encoder and predictor)</td></tr>
<tr><td>Variance head conditionally compressed; controller normalises surprise by its own lagging
EMA and pins 4&times; lr at 2.2 steps</td>
<td><b>Temperature</b> &tau; plus <b>quantile thresholds</b> on the calibrated surprise, so the
reference is external</td></tr>
<tr><td>Anchor radius 0.8 equals one Adam step (&eta;&radic;n<sub>f</sub>=0.77) &rarr; 159
projections/episode, the trust region sets the step size</td>
<td>Anchor radius <b>calibrated to an episode budget</b>, c&middot;&eta;&radic;n<sub>f</sub>&middot;U&middot;R</td></tr>
<tr><td>Recursion saturates at depth 3-4 and is harmful under a dynamics shift</td>
<td><b>Curvature-gated depth</b>, capped at 3</td></tr>
</tbody></table>
<figure><img src="figs/fig_stls.png" alt="STLS comparison">
<figcaption>Left: goal reached on identical models and episodes. Right: gradient steps per
adaptation update — the thresholded controller should sit below the saturating baseline.</figcaption></figure>
<table>
<thead><tr><th>Family</th><th>Method</th><th>Success (%)</th><th>Steps/update</th><th>Depth</th>
<th>Anchor proj./ep</th><th>Rollback/ep</th></tr></thead>
<tbody>{rows}</tbody></table>
<h3>Does planning success track predictive accuracy? A controlled sweep says no</h3>
<p class="sub">Four maze world models trained with identical lr, epochs and seed, differing only in
the straightening weight &lambda;<sub>str</sub>. Next-step error falls monotonically (37.96 &rarr;
10.67, &minus;72%%) while success under the <em>unshifted</em> dynamics is byte-for-byte identical
(15/24 at every &lambda;, Fisher p=1.000), and success under the mass shift is flat to
&lambda;=0.1 then drops non-significantly at &lambda;=0.5 (9/24 &rarr; 4/24, p=0.193). None of the
four geometry metrics tracks that movement: effective dimension steps down at the first non-zero
&lambda; and then stays put, curvature is non-monotone, and action sensitivity <em>rises</em> with
&lambda;. Accuracy and planning are decoupled; the geometry mechanism is not identified by these
metrics.</p>
{geom_tbl}
<figure><img src="figs/fig_geom_sweep.png" alt="straightening sweep">
<figcaption>Left: prediction error falls as the latent's effective dimension collapses. Right:
planning success is flat on the unshifted condition and only moves (non-significantly) at the
largest weight, so the accuracy axis and the planning axis are not the same axis.</figcaption>
</figure>
"""


def main():
    os.makedirs(DOCS, exist_ok=True)
    figs_src = os.path.join(RES, "figs")
    figs_dst = os.path.join(DOCS, "figs")
    if os.path.isdir(figs_src):
        shutil.rmtree(figs_dst, ignore_errors=True)
        shutil.copytree(figs_src, figs_dst)

    fam_tbl, seen_tbl, abl_tbl, calib = "", "", "", {}
    fp = os.path.join(RES, "by_family.csv")
    if os.path.exists(fp):
        df = pd.read_csv(fp)
        if "method" in df:
            df["method"] = df["method"].map(lambda m: PRETTY.get(m, m))
            fam_tbl = table_html(df.set_index("method"))
    ap = os.path.join(RES, "all_runs.csv")
    if os.path.exists(ap):
        df = pd.read_csv(ap)
        sub = df[df.suite == "shape"].copy()
        if len(sub):
            sub["split"] = sub.label.map(lambda s: "seen" if s in ("T", "L", "Z", "+") else "unseen")
            p = sub.pivot_table(index="method", columns="split", values="success", aggfunc="mean")
            p.index = [PRETTY.get(i, i) for i in p.index]
            seen_tbl = table_html(p)
        parts = []
        ab = df[df.method.str.startswith("dtajepa_")].copy()
        if len(ab):
            parts.append(ab)
        for extra, relabel in ((os.path.join(RES, "ablation", "all_runs.csv"), None),
                               (os.path.join(RES, "meta", "all_runs.csv"), "dtajepa_meta")):
            if os.path.exists(extra):
                e = pd.read_csv(extra)
                if relabel:
                    e["method"] = relabel
                parts.append(e)
        if parts:
            ab = pd.concat(parts)
            ab["method"] = ab["method"].map(lambda m: PRETTY.get(m, m))
            q = ab.pivot_table(index="method", columns="suite", values="success", aggfunc="mean")
            abl_tbl = table_html(q)
    parity = {}
    for tag, fn in (("curves", "parity_curves.jsonl"), ("sweep", "parity_sweep.jsonl"),
                    ("cem", "parity_cem.jsonl")):
        fp2 = os.path.join(RES, fn)
        if os.path.exists(fp2):
            rows = [json.loads(l) for l in open(fp2) if l.strip()]
            for r in rows:
                r["success"] *= 100
            parity[tag] = rows
    cp = os.path.join(RES, "calib_push.json")
    if os.path.exists(cp):
        calib = json.load(open(cp))
    parity_curves_figure(figs_dst)
    traj_figs = traj_figures(figs_dst)
    extra = parity_html(parity, traj_figs, figs_dst)
    stls_sec = stls_html()

    def fmt_cal(tag, label):
        d = calib.get(tag)
        if not d:
            return ""
        return (f"<tr><th scope=\"row\">{label}</th><td>{d['mean_err']:.3f}</td>"
                f"<td>{d['mean_var']:.3f}</td><td>{d['spearman']:+.2f}</td>"
                f"<td>{d['ece_rel']*100:.0f}%</td></tr>")

    depth_row = ""
    if calib.get("in_distribution"):
        dc = calib["in_distribution"]["depth_curve"]
        depth_row = "".join(f"<td>{v:.3f}</td>" for v in dc)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DTA-JEPA · Dual-Timescale Recursive Adaptive Latent World Models</title>
<meta name="description" content="Uncertainty-allocated test-time adaptation for latent world models, with persistent memory and a safety layer.">
<style>
:root{{--bg:#f4f4f2;--panel:#fdfdfc;--panel2:#fdfdfc;--ink:#15171b;--muted:#575c65;
--line:#dcdcd6;--accent:#1d4e89;--accent-ink:#f4f4f2;--soft:#e9eef4;
--shadow:0 18px 40px rgba(21,23,27,.08);
--sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;
--serif:"Iowan Old Style",Palatino,"Palatino Linotype",Georgia,serif;
--mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
/* aliases used by the rest of this sheet */
--fg:var(--ink);--dim:var(--muted);--acc:var(--accent);--acc2:var(--accent);--ok:var(--accent);}}
@media (prefers-color-scheme: dark){{:root{{--bg:#0f1114;--panel:#16191d;--panel2:#16191d;
--ink:#eceef1;--muted:#9aa1ab;--line:#282d34;--accent:#8fb4e3;--accent-ink:#0f1114;
--soft:#17212c;--shadow:0 18px 40px rgba(0,0,0,.5);}}}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);
font-size:16.5px;line-height:1.62;-webkit-font-smoothing:antialiased}}
a{{color:var(--accent);text-decoration:none;border-bottom:1px solid rgba(143,180,227,.35)}}
a:hover{{color:var(--ink);border-bottom-color:var(--accent)}}
header{{padding:56px 24px 32px;border-bottom:1px solid var(--line);
background:radial-gradient(900px 320px at 18% -12%,var(--soft) 0%,var(--bg) 72%)}}
.wrap{{max-width:1120px;margin:0 auto}}
h1{{font-family:var(--serif);font-size:33px;line-height:1.22;margin:0 0 8px;letter-spacing:-.01em}}
h2{{font-family:var(--serif);font-size:22px;font-weight:600;margin:44px 0 12px;
padding-bottom:8px;border-bottom:1px solid var(--line)}}
h3{{font-size:16.5px;margin:26px 0 8px;color:var(--accent)}}
.sub{{color:var(--dim);font-size:15px;margin:0 0 18px}}
.links a{{display:inline-block;margin:0 10px 10px 0;padding:8px 14px;border:1px solid var(--line);
border-radius:8px;background:var(--panel);font-size:13.5px;box-shadow:var(--shadow)}}
.tldr{{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:14px;margin:18px 0}}
.card{{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:16px}}
.card h4{{margin:0 0 6px;font-size:14.5px;color:var(--accent);letter-spacing:.01em}}
.card p{{margin:0;font-size:14px;color:var(--dim)}}
table{{width:100%;border-collapse:collapse;margin:14px 0;font-size:14px;background:var(--panel);
border:1px solid var(--line);border-radius:8px;overflow:hidden}}
th,td{{padding:9px 12px;text-align:center;border-bottom:1px solid var(--line)}}
thead th{{background:var(--soft);color:var(--ink);font-weight:600;font-size:13px;
letter-spacing:.02em;text-transform:uppercase}}
tbody th{{text-align:left;color:var(--ink);font-weight:500}}
td.hl,tbody th.hl{{color:var(--accent);background:var(--soft);font-weight:700}}
figure{{margin:22px 0}} img{{width:100%;border-radius:10px;border:1px solid var(--line);
background:#fff;box-shadow:var(--shadow)}}
figcaption{{color:var(--muted);font-size:13px;margin-top:8px}}
pre{{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px;
overflow:auto;font-size:12.5px;font-family:var(--mono);color:var(--muted)}}
code{{font-family:var(--mono);font-size:.92em;color:var(--accent)}}
.eq{{background:var(--panel);border-left:3px solid var(--acc);border-radius:0 8px 8px 0;
padding:12px 16px;margin:14px 0;font-family:ui-monospace,Menlo,monospace;font-size:14px;overflow:auto}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}
@media(max-width:820px){{.grid2{{grid-template-columns:1fr}}}}
footer{{margin-top:48px;padding:28px 24px;border-top:1px solid var(--line);
background:var(--soft);color:var(--muted);font-size:13.5px}}
</style>
</head>
<body>
<header><div class="wrap">
<h1>DTA-JEPA: Dual-Timescale Recursive Adaptive Latent World Models</h1>
<p class="sub">Uncertainty-allocated test-time adaptation with persistent memory and a safety
layer &nbsp;·&nbsp; Hefei Jiuzhai Big Data Technology Co., Ltd.</p>
<div class="links">
<a href="paper.pdf">📄 Paper (PDF)</a>
<a href="https://github.com/hf-jz/DTA-JEPA">💻 Code</a>
<a href="https://github.com/hf-jz/DTA-JEPA/blob/main/docs/DTA-JEPA-v2.md">📐 Scheme document</a>
<a href="figs/fig_success_curves.png">📊 Success curves</a>
</div>
</div></header>

<div class="wrap">

<h2>The idea in one paragraph</h2>
<p>A latent world model plans by rolling out a learned predictor, and that predictor is almost
always frozen after training — so under test-time shift, MPC optimises the wrong objective.
AdaJEPA fixed the symptom with <em>one</em> gradient step per replan. DTA-JEPA fixes the unit of
adaptation instead: a <strong>slow S-module</strong> produces a dynamics context and is never
touched online, a <strong>fast R-module</strong> recursively refines the next latent with deep
supervision at every refinement depth, and a single <strong>surprise</strong> signal decides how
many gradient steps, how much learning rate, how deep to refine, how conservative to plan, what
to remember and when to roll back.</p>

<div class="tldr">
<div class="card"><h4>Computational dual timescale</h4><p>Slow context + fast recursion.
Adaptation lives in a rank-4 LoRA set (0.6M of 1.29M parameters) on the fast module and the
encoder head; the pretrained trunk is an immutable anchor.</p></div>
<div class="card"><h4>Uncertainty as a resource</h4><p>Mahalanobis surprise under the model's
own Gaussian allocates steps <code>U∈[1,3]</code>, learning rate <code>[0.2,5]×</code> and
refinement depth <code>K∈[1,4]</code>. In distribution it is as cheap as AdaJEPA; out of
distribution it spends where the model is measurably wrong.</p></div>
<div class="card"><h4>Dual memory</h4><p>A priority queue of high-surprise latent transitions
that persists across episodes, plus a skill bank of fast-parameter snapshots indexed by
environment fingerprint, consolidated at episode boundaries under EWC.</p></div>
<div class="card"><h4>Safety layer</h4><p>Anchor ball projection, validation-gated rollback,
immutable slow anchor, and a Mahalanobis support penalty inside the planning cost.</p></div>
</div>

<h3>The online loop</h3>
<div class="eq">
plan with c<sub>t</sub>=S(history) → ŷ<sub>1:H</sub>=Rollout<sub>K_t</sub>(c<sub>t</sub>, a) →
execute chunk → s<sub>t</sub> = Mahalanobis(ŷ<sub>t+1</sub>, z<sub>t+1</sub>) →
(U<sub>t</sub>, η<sub>t</sub>, K<sub>t</sub>) = Controller(s<sub>t</sub>) →
LoRA step on B ∪ M<sub>ret</sub> → project + validate → replan
</div>

<h2>Results</h2>
<p class="sub">Frozen vs a faithful AdaJEPA re-implementation vs DTA-JEPA, identical planner,
horizon, replay buffer, checkpoint and episode set. Success in percent, averaged over the
conditions of each shift family.</p>
{fam_tbl if fam_tbl else '<p><em>Results are being generated — see the repository.</em></p>'}

<figure><img src="figs/fig_success_curves.png" alt="success versus replan curves">
<figcaption>Planning success versus MPC replanning step for all four shift families
(seen/unseen shapes, corruptions, colour shifts, dynamics shifts, unseen mazes).</figcaption></figure>

<figure><img src="figs/fig_bars.png" alt="per-condition bars">
<figcaption>Final success per condition, mean ± s.d. over seeds. Empty categories are genuine zeros: no method reached the goal in the <code>highDamping</code> (damping &times;20) maze within the replanning budget.</figcaption></figure>

<h3>Seen versus unseen geometries</h3>
{seen_tbl if seen_tbl else ''}

<h3>Ablations</h3>
{abl_tbl if abl_tbl else ''}

<h3>Is the surprise signal actually informative?</h3>
<table><thead><tr><th>Stream</th><th>mean squared error</th><th>mean predicted variance</th>
<th>Spearman ρ(var, err)</th><th>relative ECE</th></tr></thead><tbody>
{fmt_cal("in_distribution", "in distribution")}
{fmt_cal("unseen_shape", "unseen shape")}
{fmt_cal("visual_blur", "blurred observations")}
</tbody></table>
{f'<p class="sub">Mean squared latent error per refinement depth (depth 1 → 4): {depth_row}</p>' if depth_row else ''}

{extra}
{stls_sec}

<h2>Protocol replication</h2>
<p>Shift families, segment-based goal sampling (goals are future frames of the same held-out
trajectory, 25 steps ahead), MPC budget (horizon 15, chunk 5, ≤10 replans), planner (GD with
Adam, lr 0.1), adaptation rates (5e-4 / 1e-5) and success criteria follow AdaJEPA. The
environments are a self-contained numpy re-implementation: impulse-based 2D contact physics and
a software rasteriser, with no simulator, dataset or checkpoint download.</p>

<pre><code>python -m dtajepa.data
python -m dtajepa.train --data push_four_shapes_train --name push4 --epochs 6
python -m dtajepa.meta  --data push_four_shapes_train --base push4 --name push4_meta --iters 300
python -m dtajepa.evaluate --suite all --methods frozen adajepa dtajepa \
       --n_ep 20 --max_replans 8 --opt_steps 25 --horizon 12 --threads 8
python -m dtajepa.report</code></pre>

<h2>What we do not claim</h2>
<p>The benchmark is a physics-based 2D re-implementation with 64×64 observations and ~1.3M
parameter world models trained on a few hundred trajectories per family. Absolute success rates
are <em>not</em> comparable to PushT/PointMaze numbers in the literature; the method comparison
is, because every method sees the identical planner, budget, checkpoint and episode set.</p>

</div>
<footer><div class="wrap">
DTA-JEPA · code and results by Hefei Jiuzhai Big Data Technology Co., Ltd. ·
<a href="https://github.com/hf-jz/DTA-JEPA">github.com/hf-jz/DTA-JEPA</a>
</div></footer>
</body></html>
"""
    with open(os.path.join(DOCS, "index.html"), "w") as f:
        f.write(html)
    print("wrote", os.path.join(DOCS, "index.html"))


if __name__ == "__main__":
    main()
