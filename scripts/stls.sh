#!/bin/bash
# STLS-JEPA comparison: probe (reference stream) -> fit thresholds -> four methods, per family.
# A stage is skipped only when its output already holds the expected number of result rows, so a
# crashed stage (which leaves a partial file) is always redone.
set -u
cd "$(dirname "$0")/.." || exit 1
PY=/usr/bin/python3
LOG=$PWD/scratch/stls.log
: > "$LOG"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
run() { local out=$1 n=$2; shift 2
        local have=$(wc -l < "$out" 2>/dev/null || echo 0)
        if [ "$have" -ge "$n" ]; then say "skip $out ($have/$n rows)"; return; fi
        say "$*"; rm -f "$out"
        "$@" >>"$LOG" 2>&1 || { say "STAGE FAILED: $*"; exit 1; }; }
COMMON="--n_ep 12 --seeds 0 1 --ckpt_push push4s --ckpt_maze maze25s --opt_steps 10 --horizon 8 --chunk 5 --threads 8 --goal_gap 15"

run results/stls_shape_probe.jsonl 14 $PY -m dtajepa.evaluate --suite shape --methods probe \
    $COMMON --max_replans 6 --out results/stls_shape_probe.jsonl
run results/calib_stls.json 1 $PY -m dtajepa.calib --fit_records \
    --in results/stls_shape_probe.jsonl --out results/calib_stls.json
run results/stls_shape.jsonl 56 $PY -m dtajepa.evaluate --suite shape \
    --methods frozen adajepa dtajepa stls $COMMON --max_replans 6 --stls_calib results/calib_stls.json \
    --out results/stls_shape.jsonl

run results/stls_dyn_probe.jsonl 8 $PY -m dtajepa.evaluate --suite dyn --methods probe \
    $COMMON --max_replans 15 --out results/stls_dyn_probe.jsonl
run results/calib_stls_maze.json 1 $PY -m dtajepa.calib --fit_records \
    --in results/stls_dyn_probe.jsonl --out results/calib_stls_maze.json
run results/stls_dyn.jsonl 32 $PY -m dtajepa.evaluate --suite dyn \
    --methods frozen adajepa dtajepa stls $COMMON --max_replans 15 --stls_calib results/calib_stls_maze.json \
    --out results/stls_dyn.jsonl
say "STLS DONE"
