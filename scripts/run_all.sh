#!/bin/bash
# One-shot experiment driver: wait for pretraining + metas to finish, then run the
# whole evaluation programme (main sweep, meta ablation, ablations, calibration, report).
set -u
cd "$(dirname "$0")/.." || exit 1
ROOT=$(pwd)
PY=/usr/bin/python3
LOG=$ROOT/scratch/run_all.log
: > "$LOG"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

# ---------------------------------------------------------------- wait for training
say "waiting for pretrain_maze25.pt (pretraining chain)"
for i in $(seq 1 240); do
  [ -f checkpoints/pretrain_maze25.pt ] && break
  sleep 30
done
# the meta jobs come after; wait for them too (they write *_meta.pt)
for i in $(seq 1 240); do
  [ -f checkpoints/pretrain_maze25_meta.pt ] && [ -f checkpoints/pretrain_push4_meta.pt ] && break
  sleep 30
done
say "checkpoints ready: $(ls checkpoints/)"

E="--n_ep 12 --max_replans 6 --opt_steps 10 --horizon 8 --chunk 5 --threads 8 --goal_gap 15"

# ---------------------------------------------------------------- 1. main sweep
say "main sweep (frozen / adajepa / dtajepa)"
$PY -m dtajepa.evaluate --suite all --methods frozen adajepa dtajepa $E \
    --out results/raw.jsonl >>"$LOG" 2>&1
say "main sweep done"

# ---------------------------------------------------------------- 2. meta ablation
say "meta-initialisation ablation (shape suite)"
$PY -m dtajepa.evaluate --suite shape --methods dtajepa $E --ckpt_push push4_meta \
    --out results/meta.jsonl >>"$LOG" 2>&1
say "meta ablation done"

# ---------------------------------------------------------------- 3. ablations
say "ablations (shape + dyn)"
$PY -m dtajepa.evaluate --suite shape --methods dtajepa_nounc dtajepa_nomem dtajepa_nosafe dtajepa_fixedk dtajepa_noslow $E \
    --out results/ablation.jsonl >>"$LOG" 2>&1
$PY -m dtajepa.evaluate --suite dyn --methods dtajepa_nounc dtajepa_nomem dtajepa_nosafe dtajepa_fixedk dtajepa_noslow $E \
    --out results/ablation.jsonl >>"$LOG" 2>&1
say "ablations done"

# ---------------------------------------------------------------- 4. uncertainty diagnostics
say "uncertainty diagnostics"
$PY -m dtajepa.calib --ckpt push4 --id push_eval_T --ood push_eval_I --n 192 \
    --out results/calib_push.json >>"$LOG" 2>&1
$PY -m dtajepa.calib --ckpt maze25 --id maze_dyneval_default --ood maze_dyneval_lowmass --n 256 \
    --corrupt "" --out results/calib_maze.json >>"$LOG" 2>&1
say "calibration done"

# ---------------------------------------------------------------- 5. aggregate
say "report"
$PY -m dtajepa.report --in results/raw.jsonl --out results >>"$LOG" 2>&1
cat results/per_condition.csv >>"$LOG" 2>&1
say "ALL DONE"
