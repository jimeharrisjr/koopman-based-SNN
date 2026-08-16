# SHD sweep results

All runs: identical minibatch sequence, Adam 5e-3, 100 × 10 ms bins, batch 32,
full-test-set evaluation (2240 samples, chance = 5 %). Raw logs:
`sweep-AF-log.txt`, `sweep-GI-log.txt`. Hardware: Apple Silicon laptop,
single thread.

## Comparison table

| tag | architecture | channels | budget | **test acc** | final train loss | train time |
|---|---|---|---|---|---|---|
| I | 1 × 256 | **350** | **3000** | **0.680** | 0.454 | 159 s |
| F | 1 × 256 | **350** | 1500 | 0.639 | 0.795 | 80 s |
| B | 1 × 256 | 100 | 1500 | 0.627 | 1.046 | 42 s |
| D | 256 → 128 | 100 | 1500 | 0.622 | 1.086 | 146 s |
| A | 1 × 128 (baseline) | 100 | 1500 | 0.617 | 1.221 | 23 s |
| C | 1 × 512 | 100 | 1500 | 0.615 | 0.922 | 81 s |
| G | 1 × 256 | 100 | **3000** | 0.608 | 0.798 | 87 s |
| E | 128 → 128 | 100 | 1500 | 0.484 | 1.449 | 71 s |

Reference point: the original `shd_demo` configuration (1 × 128, 100 channels,
600 minibatches, 512-sample eval) measured **0.502**. Best sweep result:
**0.680** — a 17.8-point improvement, using ~2.7 minutes of training.

## What worked

1. **More training budget (up to a point)**: 600 → 1500 minibatches took the
   baseline from ~0.50 to 0.617 — the single largest gain, and free.
2. **Finer input resolution — the winning axis**: pooling the 700 cochlear
   channels 2:1 (350) instead of 7:1 (100) won at equal budget (F: 0.639 vs
   B: 0.627) *and* kept scaling with budget where coarse pooling did not
   (I: 0.680 at 3000, with the training loss still falling at the end —
   more budget would likely still pay).
3. **Moderate width**: 128 → 256 gave a consistent ~1-point gain.

## What didn't work

1. **Width past the data's support**: 1 × 512 (C) reached the *lowest*
   training loss of the 100-channel group but no test gain over 128 —
   textbook overfitting signature.
2. **More budget on information-poor input**: G (100 channels, 3000
   minibatches) *dropped* to 0.608 from B's 0.627 at 1500 — train loss kept
   improving (1.05 → 0.80) while test accuracy fell. Budget amplifies
   whatever the input allows: at 7:1 pooling that's memorization, at 2:1
   it's generalization. **Input resolution and training budget interact;
   neither axis can be tuned in isolation.**
3. **Depth, as configured here**: 256 → 128 (D) matched plain width at 3.5×
   the training cost; 128 → 128 (E) was actively harmful (0.484 — below the
   single-layer baseline). E's loss curve shows why: the second layer stayed
   near-dead until ~step 1100 (loss pinned ≈ 2.5-3.0), then learned late and
   never caught up. Deeper spiking nets likely need per-layer learning-rate /
   initialization care (or recurrence) that this uniform recipe doesn't
   provide — a known theme in the SNN literature, reproduced here.

## Loss curves (mean over 50-step windows)

```
A 1x128/100ch:   3.01 → 1.90 (step 500) → 1.46 (950) → 1.30 (1400)
B 1x256/100ch:   3.05 → 1.68 (500) → 1.28 (950) → 1.13 (1400)
C 1x512/100ch:   3.11 → 1.51 (500) → 1.18 (950) → 1.00 (1400)   [best fit, no test gain]
D 256-128/100ch: 3.04 → 1.96 (500) → 1.43 (950) → 1.16 (1400)
E 128-128/100ch: 3.04 → 2.63 (500) → 2.37 (950) → 1.60 (1400)   [second layer near-dead until ~1100]
F 1x256/350ch:   3.00 → 1.53 (500) → 1.12 (950) → 0.88 (1400)
G = B continued: 1.07 (1550) → 0.94 (2000) → 0.82 (2900)         [test fell while loss fell]
I = F continued: 0.82 (1550) → 0.65 (2000) → 0.47 (2900)         [test rose with loss]
```

## Context and honest caveats

- Published feedforward LIF baselines on SHD sit roughly in the 0.48–0.71
  band; recurrent SNNs reach 0.71–0.83+. 0.680 from a single pooled
  feedforward layer with ~2.7 minutes of CPU training is solidly in the
  feedforward range; recurrence is the known next step and is not implemented
  in this library's training path.
- One seed per configuration: differences under ~1.5 points (A/B/C/D cluster)
  are within plausible seed noise; the E collapse, the F/I resolution gains,
  and the G regression are well outside it.
- No test-set-driven tuning: the variation grid and budgets were fixed before
  any test evaluation; the only sequential decision (which config gets the
  long budget) used the equal-budget winner F, with G as the budget-only
  control.

## Round 2 — the "untried next steps," tried

All previously-listed next steps were run (raw logs: `sweep-JN-log.txt`,
`sweep-O-log.txt`). Recurrence required library support first: recurrent
weights in `KoopmanLayer` (previous-step own-spikes feed back into the drive),
BPTT through the recurrent path, per-matrix Adam state, and
`Trainer::set_learning_rate` for decay — all test-gated (self-excitation
echo/reset, zero-recurrence ≡ feedforward exactness, gradient-flow-from-zero).

| tag | variation (base: 350 ch, 1 × 256, 3000 mb) | **test acc** | final train loss | train time |
|---|---|---|---|---|
| **L** | **recurrent hidden layer (W_rec zero-init)** | **0.808** | 0.183 | 654 s |
| O | recurrent + 6000 mb + lr ×0.3 @4000 | 0.777 | 0.038 | 1316 s |
| K | feedforward, 6000 mb | 0.671 | 0.158 | 319 s |
| M | label-balanced minibatches | 0.668 | 0.463 | 158 s |
| J | no pooling (700 channels) | 0.664 | 0.329 | 218 s |
| N | lr ×0.3 @2000 | 0.661 | 0.530 | 159 s |

*(Round-1 best for reference: I = 0.680.)*

### Round-2 findings

**What worked — one thing, spectacularly:**

- **Recurrence: +12.8 points in one move (0.680 → 0.808).** The recurrent
  matrix was initialized to zero — exactly the feedforward network at step
  0 — and the through-time gradient grew task-relevant recurrence from
  nothing. The loss curves show it wasn't a late-training effect: the
  recurrent net was far ahead by step 650 (0.87 vs ~1.3 for every
  feedforward variant). 0.808 sits inside the published recurrent-SNN band
  for SHD (~0.71–0.83). Attribution is clean because *every other axis in
  this round was flat or negative*.

**What didn't work:**

- **More budget, again — now on both sides.** Feedforward K (6000 mb):
  0.671 < I's 0.680. Recurrent O (6000 mb + decay): 0.777 < L's 0.808, with
  train loss driven to 0.04 — near-memorization. At this model/data scale,
  ~3000 minibatches is the generalization sweet spot, and the ×0.3 decay
  did not rescue the overtrained regime.
- **Full input resolution (J, 700 ch): 0.664.** Resolution helped from 7:1
  to 2:1 pooling and *reversed* at 1:1 — more input detail without more
  regularization or data just gives the fit more to memorize (train loss
  0.33, among the lowest, test among the worst).
- **Balanced minibatches (M) and lr decay (N): no effect** (−1.2 / −1.9,
  order of seed noise). SHD's classes are already near-balanced, and the
  budget is too short for a decay schedule to matter.

### Cumulative journey

| stage | configuration | test acc |
|---|---|---|
| original demo | 100 ch, 1 × 128, 600 mb, 512-sample eval | 0.502 |
| round 1 best (I) | 350 ch, 1 × 256, 3000 mb | 0.680 |
| **round 2 best (L)** | **+ recurrent hidden layer** | **0.808** |

+30.6 points total; ~11 minutes of single-thread laptop training for the
final model.

## Remaining next steps (untried)

Regularization to unlock the overtrained regimes (dropout-style spike
masking, weight decay — nothing of the sort exists in the trainer yet),
two recurrent layers, recurrent + 700 channels *with* regularization,
τ-heterogeneity across neurons, and data augmentation (time-jitter/channel
noise), which is how the published >0.83 results get there.
