
import json, numpy as np, pandas as pd
R = "/Users/sheng/LLM/DTA-JEPA/"
load = lambda p: pd.DataFrame([json.loads(l) for l in open(R+p) if l.strip()])
cur = load("results/parity_curves.jsonl"); sw = load("results/parity_sweep.jsonl"); cem = load("results/parity_cem.jsonl")
for d in (cur, sw, cem): d["success"] *= 100
P = {"frozen":"Frozen","adajepa":"AdaJEPA","dtajepa":"DTA-JEPA"}
print("=== A. curves to 30 steps (7 shapes x 3 methods x 6 ep) ===")
t = cur.pivot_table(index="label", columns="method", values="success", aggfunc="mean").round(0)
print(t.to_string()); print("mean:", t.mean().round(1).to_dict())
# does success keep rising, or saturate? compare step 6 vs step 30 on the per-replan curve
for m in ("frozen","adajepa","dtajepa"):
    c = np.mean([r for r in cur[cur.method==m].success_curve], 0)
    print(f"  {P[m]:9s} curve@1,6,12,20,30 = " + " ".join("%.0f" % c[i-1] for i in (1,6,12,20,30)))
print("\n=== B. Fig.9-style sweep (shape suite, 7 conditions x 6 ep) ===")
sw["cfg"] = sw.apply(lambda r: (r["adapt"]["lr"]/5e-4, r["adapt"]["steps"], r["adapt"]["buffer"]), axis=1)
q = sw.pivot_table(index="cfg", columns="method", values="success", aggfunc="mean").round(1)
print(q.to_string())
print("\n=== C. CEM spot check (4 ep, 4 replans, 64 samples/5 iters) ===")
print(cem.pivot_table(index="label", columns="method", values="success", aggfunc="mean").round(0).to_string())
print("wall(s) per condition:", cem.groupby("method").wall.mean().round(1).to_dict())
