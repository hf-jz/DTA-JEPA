
import json, numpy as np, pandas as pd
R = "/Users/sheng/LLM/DTA-JEPA/"
load = lambda p: pd.DataFrame([json.loads(l) for l in open(R+p) if l.strip()])
main, ab, me = load("results/raw.jsonl"), load("results/ablation.jsonl"), load("results/meta.jsonl")
for d in (main, ab, me): d["success"] *= 100
P = {"frozen":"Frozen","adajepa":"AdaJEPA","dtajepa":"DTA-JEPA","dtajepa_lora":"LoRA r=4",
     "dtajepa_nounc":"w/o alloc","dtajepa_nomem":"w/o mem","dtajepa_nosafe":"w/o safety",
     "dtajepa_fixedk":"fixedK","dtajepa_noslow":"w/o slow"}
print("rows main=%d meta=%d abl=%d seeds=%s n_ep=%s" % (len(main),len(me),len(ab),sorted(main.seed.unique()),sorted(main.n_ep.unique())))
print("\n=== per-condition (mean over seeds) ===")
print(main.pivot_table(index=["suite","label"],columns="method",values="success",aggfunc="mean").round(1).to_string())
print("\n=== family x method ===")
print(main.groupby(["suite","method"]).success.agg(["mean","std"]).round(1).unstack(0).to_string())
print("\n=== combined ===")
print(main.groupby("method").success.agg(["mean","std","size"]).round(1).to_string())
s = main[main.suite=="shape"].copy(); s["split"]=np.where(s.label.isin(["T","L","Z","+"]),"seen","unseen")
print("\n=== shape seen/unseen ==="); print(s.pivot_table(index="method",columns="split",values="success",aggfunc="mean").round(1).to_string())
v = main[main.suite=="visual"].copy(); v["kind"]=np.where(v.label.fillna("").str.startswith("red"),"colour","photometric")
print("\n=== visual by kind ==="); print(v.pivot_table(index="method",columns="kind",values="success",aggfunc="mean").round(1).to_string())
print("\n=== visual per condition ==="); print(v.pivot_table(index="label",columns="method",values="success",aggfunc="mean").round(1).to_string())
print("\n=== ablations (shape+dyn) ===")
ab["m"]=ab.method.map(lambda x:P.get(x,x))
print(ab.pivot_table(index="m",columns="suite",values="success",aggfunc="mean").round(1).to_string())
print("\n=== ablation vs main baseline on same conditions (shape seeds0 only) ===")
sh = main[(main.suite=="shape")&(main.seed==0)]
print("main: " + "  ".join(f"{P[m]}={sh[sh.method==m].success.mean():.1f}" for m in ("frozen","adajepa","dtajepa")))
print("\n=== meta init (shape, 2 seeds) ===")
me["m"]="DTA-JEPA+meta"; mm=main[(main.suite=="shape")&(main.method=="dtajepa")].copy(); mm["m"]="DTA-JEPA plain-init"
allm = pd.concat([me,mm]); print(allm.pivot_table(index="m",columns="suite",values="success",aggfunc="mean").round(1).to_string())
print("\n=== adapt stats (dtajepa, main) ===")
st = main[main.method=="dtajepa"].adapt_stats.apply(pd.Series)
print(st.mean(numeric_only=True).round(3).to_string())
print("\n=== timing (s, mean per condition) ===")
tm = main.copy(); tm["plan"]=tm.timing.apply(lambda d:d["plan"]); tm["adapt"]=tm.timing.apply(lambda d:d["adapt"]); tm["wall"]=tm.wall
print(tm.groupby("method")[["plan","adapt","wall"]].mean().round(1).to_string())
