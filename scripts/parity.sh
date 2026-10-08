#!/bin/bash
# AdaJEPA figure-parity runs: the examples the paper reports, on our implementation.
# Every stage appends raw JSONL; nothing here re-tunes toward a better number.
set -u
cd "$(dirname "$0")/.." || exit 1
PY=/usr/bin/python3
LOG=$PWD/scratch/parity.log
: > "$LOG"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
run() { say "$1"; shift; "$@" >>"$LOG" 2>&1 || { say "STAGE FAILED: $*"; exit 1; }; }

E="--max_replans 6 --opt_steps 10 --horizon 8 --chunk 5 --threads 8 --goal_gap 15"

# A. AdaJEPA Fig. 2 x-axis: success curves to 30 MPC steps, all 7 shapes (GD)
run "A: shape curves to 30 replans (7 shapes x 3 methods x 6 episodes)" \
  $PY -m dtajepa.evaluate --suite shape --methods frozen adajepa dtajepa \
      --n_ep 6 --seeds 0 --max_replans 30 --opt_steps 10 --horizon 8 --chunk 5 \
      --threads 8 --goal_gap 15 --out results/parity_curves.jsonl

# B. AdaJEPA Fig. 9 axes: TTA learning-rate scale, adaptation steps, replay buffer
for scale in 0.2 0.5 1.0 2.0 5.0; do
  run "B: lr scale $scale" \
    $PY -m dtajepa.evaluate --suite shape --methods adajepa dtajepa --n_ep 6 --seeds 0 \
        --ckpt_push push4 $E --adapt_lr_scale $scale --out results/parity_sweep.jsonl
done
for st in 1 2 5; do
  run "B: adapt steps $st" \
    $PY -m dtajepa.evaluate --suite shape --methods adajepa dtajepa --n_ep 6 --seeds 0 \
        --ckpt_push push4 $E --adapt_steps $st --out results/parity_sweep.jsonl
done
for buf in 1 5 20; do
  run "B: replay buffer $buf" \
    $PY -m dtajepa.evaluate --suite shape --methods adajepa dtajepa --n_ep 6 --seeds 0 \
        --ckpt_push push4 $E --adapt_buffer $buf --out results/parity_sweep.jsonl
done

# C. planner independence (AdaJEPA report GD and CEM for every family)
run "C: CEM planner spot check (2 shapes x 3 methods)" \
  $PY -m dtajepa.evaluate --suite shape --methods frozen adajepa dtajepa --n_ep 4 --seeds 0 \
      --planner cem --cem_samples 64 --cem_iter 5 --cem_elite 8 \
      --max_replans 4 --horizon 8 --chunk 5 --threads 8 --goal_gap 15 \
      --out results/parity_cem.jsonl

# D. trajectory dumps for AdaJEPA Fig. 4/5-style qualitative figures
run "D: trajectory dumps (shape -I + square)" \
  $PY -m dtajepa.evaluate --suite shape --methods frozen adajepa dtajepa --n_ep 4 --seeds 0 \
      --max_replans 8 --opt_steps 10 --horizon 8 --chunk 5 --threads 8 --goal_gap 15 \
      --record_traj --out results/parity_traj_shape.jsonl
run "D: trajectory dumps (maze dynamics + layouts)" \
  $PY -m dtajepa.evaluate --suite dyn --methods frozen adajepa dtajepa --n_ep 4 --seeds 0 \
      --max_replans 8 --opt_steps 10 --horizon 8 --chunk 5 --threads 8 --goal_gap 15 \
      --record_traj --out results/parity_traj_maze.jsonl
run "D: trajectory dumps (held-out layouts)" \
  $PY -m dtajepa.evaluate --suite layout --methods frozen adajepa dtajepa --n_ep 3 --seeds 0 \
      --max_replans 8 --opt_steps 10 --horizon 8 --chunk 5 --threads 8 --goal_gap 15 \
      --record_traj --out results/parity_traj_layout.jsonl

say "PARITY DONE"
