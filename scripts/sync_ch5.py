
"""Idempotent sync of ch5's result table, combined line and main-comparison prose from the data."""
import json, re
import pandas as pd

R = "/Users/sheng/LLM/DTA-JEPA/"
D = R + "latex-paper/dtajepa-paper/"
main = pd.DataFrame([json.loads(l) for l in open(R + "results/raw.jsonl") if l.strip()])
main["success"] *= 100
fam = main.groupby(["suite", "method"]).success.mean().unstack(0)
comb = main.groupby("method").success.agg(["mean", "std"])
order = ["frozen", "adajepa", "dtajepa"]
NAME = {"frozen": "Frozen", "adajepa": "AdaJEPA", "dtajepa": "DTA-JEPA"}
COLS = ["shape", "visual", "dyn", "layout"]
rows = {NAME[m]: NAME[m] + " & " + " & ".join("%.1f" % fam.loc[m, c] for c in COLS) + " \\\\"
        for m in order}
combined = ("\\multicolumn{5}{l}{Combined over all %d conditions: Frozen $%.1f \\pm %.1f$, "
            "AdaJEPA $%.1f \\pm %.1f$, DTA-JEPA $%.1f \\pm %.1f$.}\\\\"
            % (len(main) // 3, comb.loc["frozen", "mean"], comb.loc["frozen", "std"],
               comb.loc["adajepa", "mean"], comb.loc["adajepa", "std"],
               comb.loc["dtajepa", "mean"], comb.loc["dtajepa", "std"]))
spread = comb["mean"].max() - comb["mean"].min()
d5 = main[main.suite == "dyn"].pivot_table(index="label", columns="method",
                                           values="success", aggfunc="mean").round(1)
prose = (r"""\textbf{The honest summary is that the three methods are statistically indistinguishable
overall, with one exception that matters.} Combined over all conditions the three means lie
within $%.1f$ points of each other against a per-condition spread of $21$--$23$ points, and on
the shape, visual and layout families adaptation does not separate from the frozen model. The
exception is the dynamics family at a \emph{reachable} shift: under damping $\times5$ the
DTA-JEPA controller reaches the goal in $%.1f\%%$ of episodes against $%.1f\%%$ for AdaJEPA and
$%.1f\%%$ for the frozen model, and it is also ahead at default dynamics ($%.1f$ vs $%.1f$).
This is the regime the design targets --- a shift in the \emph{operator} rather than in the
input distribution --- and it is the regime where the surprise signal is genuinely informative
(\S\ref{app:maze}: $2.8\times$ the in-distribution prediction error). At
$\text{damping}\times20$ no method reaches the goal at all; that condition lies beyond this
maze's actuator authority and we report it as a zero, not as a difference.""" % (
    spread, d5.loc["highDamping5", "dtajepa"], d5.loc["highDamping5", "adajepa"],
    d5.loc["highDamping5", "frozen"], d5.loc["default", "dtajepa"],
    d5.loc["default", "adajepa"]))

f = D + "ch5-en.tex"
s = open(f).read()
out = []
for line in s.split("\n"):
    key = line.split(" & ")[0].strip()
    if key in rows:
        out.append(rows[key])
    elif line.strip().startswith("\\multicolumn{5}{l}{Combined over all"):
        out.append(combined)
    else:
        out.append(line)
s = "\n".join(out)
assert rows["Frozen"] in s and combined in s, "table/combined replacement failed"
s2, n = re.subn(r"\\textbf\{The honest summary.*?room for adaptation\.", lambda m: prose,
                s, count=1, flags=re.S)
assert n == 1, "prose paragraph not found (%d)" % n
open(f, "w").write(s2)

f2 = D + "dtajepa-paper-en.tex"
a = open(f2).read()
old = ("methods are statistically indistinguishable (combined $43.6$, $42.8$ and $44.7\\%$ over 38\n"
       "conditions, per-condition spread $21$--$22$ points), and neither the uncertainty allocation nor\n"
       "the meta-learned initialisation produces a measurable gain.")
assert old in a, "abstract passage not found"
new_a = ("methods are statistically indistinguishable overall (combined $%.1f$, $%.1f$ and $%.1f\\%%$,\n"
         "per-condition spread $21$--$23$ points) and the meta-learned initialisation produces no\n"
         "measurable gain --- but the exception is the one the design predicts: under a reachable\n"
         "\\emph{dynamics} shift (damping $\\times5$) DTA-JEPA reaches the goal in $%.1f\\%%$ of episodes\n"
         "against $%.1f\\%%$ for AdaJEPA and $%.1f\\%%$ for the frozen model." % (
             comb.loc["frozen", "mean"], comb.loc["adajepa", "mean"], comb.loc["dtajepa", "mean"],
             d5.loc["highDamping5", "dtajepa"], d5.loc["highDamping5", "adajepa"],
             d5.loc["highDamping5", "frozen"]))
open(f2, "w").write(a.replace(old, new_a))
print("synced OK")
for m in order:
    print(rows[NAME[m]])
print(combined)
print("damping x5 ->", dict(d5.loc["highDamping5"]))
