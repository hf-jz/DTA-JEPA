# DTA-JEPA: Dual-Timescale Recursive Adaptive Latent World Models

**Uncertainty-allocated test-time adaptation for latent world models, with persistent memory and a safety layer.**

AdaJEPA showed that one self-supervised gradient step per MPC replan recovers much of the
planning performance lost under test-time distribution shift. DTA-JEPA argues that a *fixed*
one-step update is the wrong unit of adaptation, and replaces it with three separations:

| separation | mechanism |
|---|---|
| **slow vs fast dynamics** | a slow **S-module** (history → dynamics context, never modified online) and a fast **R-module** that recursively refines the next latent, `y^(k) = y^(k-1) + g([y^(k-1), c_t], u_t)` for `k = 1..K_t`, with deep supervision at every depth |
| **right vs wrong** | one **surprise** signal (Mahalanobis residual under the model's own predicted Gaussian) allocates gradient steps, learning rate, refinement depth, planning conservatism, memory priority and rollback |
| **this episode vs so far** | persistent **episodic memory** (high-surprise latent transitions) + a **skill bank** of fast-parameter snapshots indexed by environment fingerprint, and EWC slow consolidation at episode boundaries — all inside an **anchor ball + validation-gated rollback** safety layer |

Online plasticity is restricted to a rank-4 LoRA fast parameter set on the R-module plus the
encoder head (~0.6M of 1.29M parameters); the pretrained encoder trunk and S-module are an
immutable anchor.

## Quickstart

```bash
python -m dtajepa.data                       # generate the five suites (~8 min, one core)
python -m dtajepa.train --data push_four_shapes_train --name push4 --epochs 6
python -m dtajepa.train --data maze_diverse_train     --name maze25 --epochs 8
python -m dtajepa.meta  --data push_four_shapes_train --base push4  --name push4_meta --iters 300
python -m dtajepa.evaluate --suite all --methods frozen adajepa dtajepa \
       --n_ep 20 --max_replans 8 --opt_steps 25 --horizon 12 --threads 8
python -m dtajepa.report                     # tables (markdown + LaTeX) and figures
```

`scripts/run_all.sh` runs the whole programme end to end. Everything runs on a laptop CPU or
Apple MPS; there is no simulator, dataset or checkpoint download.

## Repository layout

```
dtajepa/
  sim.py        2D impulse-based contact physics + PointMaze (numpy only)
  envs.py       64x64 software rasteriser, shift knobs, image corruptions
  data.py       data generation for the five suites (shape/visual/dyn/layout/continual)
  models.py     encoder, dual-timescale recursive predictor (S/R modules, LoRA, uncertainty)
  losses.py     SIGReg + variance floor, Gaussian NLL, deep-supervised JEPA loss
  train.py      stage-1 pretraining
  meta.py       stage-2 FOMAML fast-parameter initialisation
  adapt.py      episodic/parametric memory, safety layer, adaptive controller, adapter, EWC
  plan.py       GD and CEM planners, uncertainty + support penalised cost
  evaluate.py   shift suites, closed-loop MPC, Frozen / AdaJEPA / DTA-JEPA / ablations
  calib.py      uncertainty calibration, surprise under shift, depth-error curve
  report.py     aggregation into LaTeX tables and figures
docs/           the scheme document (DTA-JEPA-v2.md) and this project page's source
latex-paper/    the paper (LaTeX, template-compatible)
```

## Evaluation protocol

We replicate AdaJEPA's protocol rather than its pixels: goals are future frames of the same
held-out trajectory (25 steps ahead), MPC uses horizon 15 and executes chunks of 5 actions for
at most 10 replans, GD planning uses Adam at lr 0.1, adaptation is one update per replan on a
recent-5 buffer with learning rates 5e-4 / 1e-5, and success is reported per replan and at the
end of the episode. Shift families: **shape** (4 training shapes → 3 unseen), **visual**
(blur / salt-and-pepper / dark / red agent / red block / red anchor), **dynamics**
(mass ×0.2, damping ×20) and **layout** (25 training mazes → 5 held-out, BFS goals at
distance 3–5).

## Results

See `results/tables.tex`, `results/figs/` and the project page. Headline numbers and the full
ablation table are regenerated from `results/raw.jsonl` by `python -m dtajepa.report`.

## What we do not claim

The benchmark is a physics-based 2D re-implementation with $64\times64$ observations and ~1.3M
parameter world models, trained on a few hundred trajectories per family. Absolute success
rates are not comparable to PushT/PointMaze results in the literature; the method comparison
is, because every method sees the identical planner, budget, checkpoint and episode set.

## License

MIT (see LICENSE).
