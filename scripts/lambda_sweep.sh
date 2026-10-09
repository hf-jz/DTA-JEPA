#!/bin/bash
# Decides whether planning success tracks predictive accuracy or latent geometry: four maze
# world models trained with identical lr/epochs/seed, differing only in the straightening weight.
# For each arm: the planning-free geometry panel, then a frozen closed-loop probe on the two
# conditions that carry the signal (default: easy for both models; lowMass: where straightening
# cost 50 points).
set -u
cd "$(dirname "$0")/.." || exit 1
PY=/usr/bin/python3
LOG=$PWD/scratch/lambda_sweep.log
: > "$LOG"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
run() { local skip=$1; shift; if [ -e "$skip" ]; then say "skip ($skip)"; return; fi
        say "$*"; "$@" >>"$LOG" 2>&1 || { say "STAGE FAILED: $*"; exit 1; }; }

for L in 0.0 0.02 0.1 0.5; do
  TAG="mazeS$(echo $L | tr -d '.')"
  run "checkpoints/pretrain_$TAG.pt" $PY -m dtajepa.train --data maze_diverse_train \
      --name $TAG --epochs 6 --bs 48 --lr 5e-4 --d 128 --ch 32 64 96 128 --s_blocks 2 \
      --lam_reg 1.0 --lam_str $L --log_every 200
  say "geometry $TAG (lam_str=$L)"
  $PY -m dtajepa.calib --geometry --ckpt $TAG --id maze_dyneval_default --n 192 \
      --out results/geometry.json >>"$LOG" 2>&1 || say "STAGE FAILED: geometry $TAG"
  run "results/sweep_$TAG.jsonl" $PY -m dtajepa.evaluate --suite dyn --methods frozen \
      --labels default lowMass --n_ep 12 --seeds 0 1 --ckpt_push push4s --ckpt_maze $TAG \
      --max_replans 15 --opt_steps 10 --horizon 8 --chunk 5 --threads 8 --goal_gap 15 \
      --out results/sweep_$TAG.jsonl
done
say "LAMBDA SWEEP DONE"
