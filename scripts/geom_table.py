#!/usr/bin/env python3
"""Print the latent-geometry panel as a table (results/geometry.json)."""
import json, sys, os
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
fp = os.path.join(R, "results", "geometry.json")
rows = [json.loads(l) for l in open(fp) if l.strip()]
if len(sys.argv) > 1:
    rows = [r for r in rows if r["ckpt"] in sys.argv[1:]]
w = max(len(r["ckpt"]) for r in rows)
print(f"{'ckpt':{w}s} {'dset':26s} {'pred_err':>8s} {'curv':>7s} {'sep':>7s} "
      f"{'sep/curv':>8s} {'eff_dim':>7s} {'ju_norm':>9s}")
for r in rows:
    print(f"{r['ckpt']:{w}s} {r['dset']:26s} {r['pred_err']:8.2f} {r['curv']:7.3f} "
          f"{r['sep']:7.3f} {r['sep_over_curv']:8.3f} {r['eff_dim']:7.2f} {r['ju_norm']:9.5f}")
