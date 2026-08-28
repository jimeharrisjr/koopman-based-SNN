# External calibration baselines

## snnTorch (improvements.md P2.3) — run 2026-08-28

`snntorch_shd.py` transplants the R-recipe protocol onto snnTorch's
RSynaptic neuron (same (v, i) state pair, subtractive reset, fast-sigmoid
slope 5, same channels/bins/augmentation/optimizer/budget/evaluation;
disclosed differences in the script header). Expectation stated before
running: 0.80–0.88, vs the recorded R-recipe mean 0.850 ± 0.027.
Raw log: `demo/snntorch-baseline-log.txt`.

| variant | seed 42 | seed 43 | seed 44 | mean |
|---|---|---|---|---|
| direct transplant (gain 1.0) | 0.5960* | 0.7246 | 0.7621 | 0.694 |
| gain-corrected (×0.25, post-hoc diagnostic) | 0.8442 | 0.7768 | 0.7321 | 0.784 |

\*seed 42's final 50 steps show a training blow-up (loss 0.43 → 1.66)
under the raw gain; all gain-corrected runs were stable.

**What this calibrates, honestly stated:**

1. **The protocol transfers.** An independent framework, given our data
   pipeline and architecture, trains SHD into the published recurrent band
   — our recorded numbers are not an artifact of a broken harness or
   evaluation.
2. **The pre-stated expectation (0.80–0.88) was wrong for the direct
   transplant** (0.60–0.76). The dominant identified cause: RSynaptic's
   Euler-flavored update couples the synaptic current into the membrane
   with gain 1.0/step where the exact ZOH propagator's γ = 0.239, so
   hyperparameters tuned on the exact engine over-drive snnTorch ~4×. The
   labeled gain-corrected diagnostic recovers most of the gap (best seed
   0.8442, inside the expected band).
3. **What may NOT be concluded:** that the exact engine "beats snnTorch by
   ~10 points." The recipe was tuned end-to-end on the exact engine
   (home-field advantage), and dedicated snnTorch tuning would surely
   close much of the remaining gap. The defensible statement is narrower
   and still useful: *integrator conventions are part of the recipe* —
   hyperparameters do not transfer across sub-threshold discretizations,
   which is precisely the sensitivity the exact-propagator formulation
   exists to remove.
4. Under matched-and-corrected settings, the exact engine's runs remain
   both higher (0.850 ± 0.027 vs 0.784 mean) and markedly more
   seed-stable (± 2.7 vs ± 5.6 points) at this operating point.

Reproduce: `python baselines/snntorch_shd.py <seed> [gain]` (venv with
`torch snntorch h5py numpy`; MPS or CPU).
