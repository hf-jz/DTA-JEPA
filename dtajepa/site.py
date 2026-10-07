"""Generate the GitHub Pages project site (docs/index.html) from the evaluation results.

    python -m dtajepa.site            # writes docs/index.html + docs/figs/*.png
"""
from __future__ import annotations

import json
import os
import shutil

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
    cp = os.path.join(RES, "calib_push.json")
    if os.path.exists(cp):
        calib = json.load(open(cp))

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
:root{{--bg:#0d1117;--panel:#151b23;--panel2:#1b222c;--fg:#e6edf3;--dim:#9aa7b4;
--acc:#d64541;--acc2:#4c8fd6;--ok:#2e9e5b;--line:#252d38;}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--fg);
font:16px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,"PingFang SC",sans-serif}}
a{{color:var(--acc2);text-decoration:none}} a:hover{{text-decoration:underline}}
header{{padding:56px 24px 32px;border-bottom:1px solid var(--line);
background:radial-gradient(1200px 400px at 20% -10%,#1d2735 0%,var(--bg) 70%)}}
.wrap{{max-width:1120px;margin:0 auto}}
h1{{font-size:34px;line-height:1.25;margin:0 0 8px}}
h2{{font-size:22px;margin:44px 0 12px;padding-bottom:8px;border-bottom:1px solid var(--line)}}
h3{{font-size:17px;margin:24px 0 8px;color:#c9d5e1}}
.sub{{color:var(--dim);font-size:15px;margin:0 0 18px}}
.links a{{display:inline-block;margin:0 10px 10px 0;padding:8px 14px;border:1px solid var(--line);
border-radius:8px;background:var(--panel);font-size:14px}}
.tldr{{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:14px;margin:18px 0}}
.card{{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:16px}}
.card h4{{margin:0 0 6px;font-size:15px;color:#fff}}
.card p{{margin:0;font-size:14px;color:var(--dim)}}
table{{width:100%;border-collapse:collapse;margin:14px 0;font-size:14px;background:var(--panel);
border:1px solid var(--line);border-radius:8px;overflow:hidden}}
th,td{{padding:9px 12px;text-align:center;border-bottom:1px solid var(--line)}}
thead th{{background:var(--panel2);color:#c9d5e1;font-weight:600}}
tbody th{{text-align:left;color:#c9d5e1;font-weight:500}}
td.hl,tbody th.hl{{color:#fff;background:#2a1c1c}}
figure{{margin:22px 0}} img{{width:100%;border-radius:10px;border:1px solid var(--line);background:#fff}}
figcaption{{color:var(--dim);font-size:13px;margin-top:8px}}
pre{{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px;
overflow:auto;font-size:13px}}
code{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}}
.eq{{background:var(--panel);border-left:3px solid var(--acc);border-radius:0 8px 8px 0;
padding:12px 16px;margin:14px 0;font-family:ui-monospace,Menlo,monospace;font-size:14px;overflow:auto}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}
@media(max-width:820px){{.grid2{{grid-template-columns:1fr}}}}
footer{{margin-top:48px;padding:28px 24px;border-top:1px solid var(--line);color:var(--dim);font-size:14px}}
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
<figcaption>Final success per condition, mean ± s.d. over seeds.</figcaption></figure>

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
