"""Merge the re-measured dynamics family (4 conditions at R=15) into results/raw.jsonl.

- dyn rows are replaced wholesale by results/maze_hd5.jsonl (default, lowMass, highDamping5,
  highDamping -- one consistent run);
- layout rows stay as merged earlier (also R=15);
- push families keep R=6.
"""
import json, os, shutil

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
raw = os.path.join(R, "results", "raw.jsonl")
src = os.path.join(R, "results", "maze_hd5.jsonl")

rows = [json.loads(l) for l in open(raw) if l.strip()]
new = [json.loads(l) for l in open(src) if l.strip()]
assert new, "maze_hd5.jsonl is empty -- did the rerun finish?"
conds = sorted({r["label"] for r in new})
assert len(conds) == 4, "expected 4 dynamics conditions, got %s" % conds
shutil.copy(raw, raw + ".3replan-dyn.bak")
kept = [r for r in rows if r["suite"] != "dyn"]
with open(raw, "w") as f:
    for r in kept + new:
        f.write(json.dumps(r) + "\n")
print("kept %d rows (push R=6 + layout R=15), dyn replaced with %d rows over %s"
      % (len(kept), len(new), conds))
