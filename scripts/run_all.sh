#!/bin/bash
# One-shot experiment driver: waits for checkpoints, then runs the whole evaluation programme.
set -u
cd "$(dirname "$0")/.." || exit 1
PY=/usr/bin/python3
LOG=$PWD/scratch/run_all.log
: > "$LOG"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

for i in $(seq 1 240); do
  [ -f checkpoints/pretrain_maze25.pt ] && [ -f checkpoints/pretrain_push4_meta.pt ] \
    && [ -f checkpoints/pretrain_maze25_meta.pt ] && break
  sleep 20
done
say "checkpoints ready"

E="--max_replans 6 --opt_steps 10 --horizon 8 --chunk 5 --threads 8 --goal_gap 15"
M="--methods frozen adajepa dtajepa"

say "shape (2 seeds)"
$PY -m dtajepa.evaluate --suite shape   $M --n_ep 12 --seeds 0 1 $E --out results/raw.jsonl >>"$LOG" 2>&1 || { say "STAGE FAILED -- see log"; exit 1; }
say "visual"
$PY -m dtajepa.evaluate --suite visual  $M --n_ep 12 --seeds 0   $E --out results/raw.jsonl >>"$LOG" 2>&1 || { say "STAGE FAILED -- see log"; exit 1; }
say "dynamics + layout (2 seeds)"
$PY -m dtajepa.evaluate --suite dyn     $M --n_ep 12 --seeds 0 1 $E --out results/raw.jsonl >>"$LOG" 2>&1 || { say "STAGE FAILED -- see log"; exit 1; }
$PY -m dtajepa.evaluate --suite layout  $M --n_ep 12 --seeds 0 1 $E --out results/raw.jsonl >>"$LOG" 2>&1 || { say "STAGE FAILED -- see log"; exit 1; }
say "meta-initialisation ablation (shape, 2 seeds)"
$PY -m dtajepa.evaluate --suite shape --methods dtajepa --ckpt_push push4_meta \
    --n_ep 12 --seeds 0 1 $E --out results/meta.jsonl >>"$LOG" 2>&1 || { say "STAGE FAILED -- see log"; exit 1; }
say "ablations (shape seeds 0, dyn seeds 0)"
A="--methods dtajepa_lora dtajepa_nounc dtajepa_nomem dtajepa_nosafe dtajepa_fixedk dtajepa_noslow"
$PY -m dtajepa.evaluate --suite shape  $A --n_ep 12 --seeds 0 $E --out results/ablation.jsonl >>"$LOG" 2>&1 || { say "STAGE FAILED -- see log"; exit 1; }
$PY -m dtajepa.evaluate --suite dyn    $A --n_ep 12 --seeds 0 $E --out results/ablation.jsonl >>"$LOG" 2>&1 || { say "STAGE FAILED -- see log"; exit 1; }
say "uncertainty diagnostics"
$PY -m dtajepa.calib --ckpt push4 --id push_eval_T --ood push_eval_I --n 192 --out results/calib_push.json >>"$LOG" 2>&1 || { say "STAGE FAILED -- see log"; exit 1; }
$PY -m dtajepa.calib --ckpt maze25 --id maze_dyneval_default --ood maze_dyneval_lowmass --n 192 --out results/calib_maze.json >>"$LOG" 2>&1 || { say "STAGE FAILED -- see log"; exit 1; }
say "report"
$PY -m dtajepa.report --in results/raw.jsonl --out results >>"$LOG" 2>&1 || { say "STAGE FAILED -- see log"; exit 1; }
for f in results/meta.jsonl results/ablation.jsonl; do
  [ -f $f ] && $PY -m dtajepa.report --in $f --out results/$(basename $f .jsonl) >>"$LOG" 2>&1 || { say "STAGE FAILED -- see log"; exit 1; }
done
say "ALL DONE"
