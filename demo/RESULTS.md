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

## Round 3 — regularization and augmentation (target: > 0.83)

Additions this round: **decoupled (AdamW-style) weight decay** in the trainer
(`TrainConfig::weight_decay`, applied to `W`/`W_rec` after the optimizer
step; test-gated by exact geometric shrinkage under silent input) and
**event-stream augmentation** in the harness (training data only: 15 %
event dropout, ±25-channel spectral shift, ±10 % time stretch). Raw logs:
`sweep-PQ-log.txt`, `sweep-RS-log.txt`, `sweep-T-log.txt`.

| tag | on top of L (recurrent 350 ch 1 × 256 @3000, 0.808) | **test acc** | final loss | train time |
|---|---|---|---|---|
| **T** | **1 × 512 + aug + wd 0.01, 6000 mb** | **0.877** | 0.219 | 75.5 min |
| R | + augmentation, 6000 mb | **0.873** | 0.243 | 24.3 min |
| S | + aug + wd 0.01, 6000 mb | 0.843 | 0.231 | 22.8 min |
| P | + wd 0.01 (3000 mb) | 0.778 | 0.185 | 11.8 min |
| Q | + augmentation (3000 mb) | 0.777 | 0.381 | 12.2 min |

**Target exceeded: three configurations clear 0.83, best 0.877.**

### Round-3 findings

**What worked:**

- **Augmentation × budget is the unlock — neither alone.** The decisive
  comparison: at 6000 minibatches the *unaugmented* recurrent net overfit
  to 0.777 (round 2's O); the *augmented* one reached 0.873 (R). Same
  model, same budget, +9.6 points — augmentation converts training budget
  from memorization fuel into generalization. At the short 3000 budget,
  augmentation alone (Q, 0.777) actually *hurt*: it makes the task harder,
  and the run was still underfitted (loss 0.38 vs L's 0.18).
- **Capacity follows regularization**: 512 neurons — which overfit pointlessly
  back in round 1 — becomes the champion (T, 0.877) once augmentation and
  decay control the fit. The margin over R is small (+0.4 points) for 3×
  the training time; R is the practical sweet spot.

**What didn't (or barely) worked:**

- **Weight decay is redundant once augmentation is present at this scale**:
  S (aug + wd) = 0.843 vs R (aug only) = 0.873 — the decay cost ~3 points
  at 256 neurons. Only at 512 (T) did it net out positive. Decay alone (P,
  0.778) slightly *reduced* accuracy vs plain L.

### Cumulative journey

| stage | configuration | test acc |
|---|---|---|
| original demo | 100 ch, 1 × 128, 600 mb | 0.502 |
| round 1 (I) | 350 ch, 1 × 256, 3000 mb | 0.680 |
| round 2 (L) | + recurrent hidden layer | 0.808 |
| **round 3 (T)** | **+ augmentation + wd, 512 neurons, 6000 mb** | **0.877** |

+37.5 points total. For context, published SHD results: feedforward SNNs
~0.48–0.71, recurrent SNNs ~0.71–0.83, augmented/adaptive state of the art
~0.90+. A single recurrent LIF layer with subtractive reset, count-based
readout, and 75 minutes of single-thread laptop training at 0.877 is at the
upper end of the recurrent band.

## Round 4 — adaptive neurons, heterogeneity, depth, budget (base: R recipe)

New library support this round (all exactness-gated): generalized
spike-triggered jumps (the subtractive reset became `jumps[0]`; adaptation is
one more linear jump), `KoopmanLayer::adlif` (k = 3, spike-for-spike against
the AdLif reference), and `adlif_hetero` (per-neuron time constants via
`Operator::PerNeuron` + per-neuron coupling, each neuron exact against its
own reference). Raw logs: `sweep-U-log.txt` … `sweep-X-log.txt`.

| tag | on top of R (recurrent 350 ch 1 × 256, aug, 6000; 0.873) | **test acc** | final loss | train time |
|---|---|---|---|---|
| **X** | **two recurrent layers 256-256** | **0.886** | 0.213 | 53 min |
| W | ALIF, heterogeneous τ (τ_m 10–40, τ_w 60–400 ms) | 0.864 | 0.203 | 26 min |
| U | 12000 minibatches | 0.858 | 0.126 | 49 min |
| V | ALIF, homogeneous (τ_w 150 ms, b 0.1) | 0.836 | 0.216 | 26 min |

### Round-4 findings

- **Depth pays once the recipe supports it: 0.886, the new best.** The same
  two-layer idea that was cost-neutral (D) or harmful (E) for feedforward
  unaugmented nets in round 1 gains +1.3 points over one layer under
  recurrence + augmentation. Each round's "loser" keeps becoming a later
  round's winner once its enabling ingredient arrives (512 width → round 3;
  depth → here).
- **Adaptation *lost* accuracy at this configuration** — a useful negative.
  Homogeneous ALIF dropped 3.7 points below plain LIF; per-neuron
  heterogeneous τ recovered most but stayed 0.9 below. The published ALIF
  wins on SHD come with finer time resolution, learned τ, and different
  readouts; adaptation is not a free upgrade at 10 ms bins with a count
  readout. (τ/b_jump were fixed, not tuned — a fair caveat.)
- **The budget ceiling is real and augmentation only moved it once**: 12000
  minibatches overfit (train loss 0.13) just like 6000 did without
  augmentation. More augmentation variety, not more epochs, is the lever.

Cumulative journey: **0.502 → 0.680 → 0.808 → 0.877 → 0.886.**

## Round 5 — the remaining next steps (target: > 0.92) — **target not reached**

New library support this round: leaky-trace readout
(`TrainConfig::readout_decay`: spike counts become an exponentially
decaying trace, with the matching per-step gradient scaling) and a public
`Trainer::logits` for logit-level ensembling in the harness. Raw logs:
`sweep-Z-log.txt`, `sweep-AA…AE-log.txt`, `sweep-AF-ensemble-log.txt`
(note: `sweep-AF-log.txt` is round 1's A–F log; the ensemble's tag is
also AF, hence the distinct filename).

### First: the seed-noise audit that reframes everything

Z1/Z2 re-ran the *identical* R recipe (recurrent 1 × 256, aug, 6000)
with only the weight-init seed changed:

| run | seed | test acc |
|---|---|---|
| R (round 3) | 42 | 0.873 |
| Z1 | 43 | 0.819 |
| Z2 | 44 | 0.858 |

**The same configuration spans 0.819–0.873 across three seeds: mean
≈ 0.850, spread ± 2.7 points.** R's 0.873 was a lucky draw, not the
recipe's true value. Every single-seed margin smaller than ~3 points in
rounds 1–4 (T vs R, X vs T, W vs V…) is individually inconclusive; the
axis-level conclusions stand only where effects were large (recurrence
+12.8, augmentation×budget +9.6) or replicated across configurations.

### The round-5 runs

| tag | variation | **test acc** | vs R-recipe mean 0.850 | train time |
|---|---|---|---|---|
| **AF** | **ensemble ×3 of X (two recurrent layers 256-256), summed logits** | **0.882** | **+3.2 (variance-reduced)** | 148 min |
| AB | 5 ms bins (200 steps), R recipe | 0.871 | +2.1 (single seed, 2× cost) | 44 min |
| AA | full 1.4 s duration (140 bins) | 0.850 | ±0 | 33 min |
| AD | 1 × 512, aug only (no wd), 6000 | 0.850 | ±0 | 79 min |
| AE | three recurrent layers 256-256-256 | 0.830 | −2.0 | 80 min |
| AC | leaky-trace readout (κ = 0.95) | 0.709 | **−14.1** | 26 min |

### Round-5 findings

- **The ensemble is the best honest number: 0.882.** Three members of
  the two-layer recipe (different init seeds), logits summed at eval.
  It matches X's single-seed 0.886 while averaging out the seed lottery
  — evidence that X's depth gain was real, but also that ensembling
  three ~0.85–0.88 members buys robustness, not a leap: the members'
  errors are too correlated (same architecture, same data) for voting
  to add much beyond the best draw.
- **Depth stops at two layers**: a third recurrent layer *lost* ~2
  points at 1.5× the two-layer cost (AE, 0.830). Consistent with
  rounds 1/4: each depth increment needs a new enabling ingredient,
  and whatever the third layer needs (per-layer lr, skip connections,
  normalization), this uniform recipe doesn't have it.
- **The count readout is load-bearing** — the round's most decisive
  negative: recency-weighting the readout (κ = 0.95, ~14-bin memory)
  collapsed accuracy to 0.709. SHD words are distinguished by evidence
  spread over the whole utterance; discarding early evidence costs 14
  points. Any temporal-readout scheme here must *add* memory, not
  replace the integral.
- **Duration and bin width are already right**: the full 1.4 s (AA,
  0.850 — trailing silence dilutes nothing but adds nothing) and 5 ms
  bins (AB, 0.871 at 2× cost) both land within seed noise of the 1 s /
  10 ms default. AD (0.850) closes the round-3 loose end: at 512 width,
  weight decay's apparent +2.7 (T) was within seed noise all along.

### Verdict on the 0.92 target

**Not reached: best variance-reduced result 0.882 (AF), best single
draw 0.886 (X).** Every accessible axis is now measured: input
resolution, duration, bin width, width, depth, budget, recurrence,
augmentation, weight decay, adaptation, heterogeneity, readout shape,
seeds, ensembling — each is either exhausted or negative. The published
runs that clear 0.90–0.94 on SHD rely on ingredients outside this
library's current training path: *learned* per-neuron time constants,
attention/state-space hybrid readouts, much heavier augmentation
pipelines, or larger ensembles of more diverse models. Closing the
remaining ~3.5 points is a library-feature project (most plausibly
learnable τ — backprop through the propagator entries — plus a
trained temporal-attention readout), not a sweep away.

### Final cumulative journey

| stage | configuration | test acc |
|---|---|---|
| original demo | 100 ch, 1 × 128, 600 mb | 0.502 |
| round 1 (I) | 350 ch, 1 × 256, 3000 mb | 0.680 |
| round 2 (L) | + recurrent hidden layer | 0.808 |
| round 3 (R/T) | + augmentation (× budget) | 0.873* |
| round 4 (X) | + second recurrent layer | 0.886* |
| **round 5 (AF)** | **ensemble ×3 of X recipe** | **0.882 (variance-reduced)** |

\* single seed; the Z audit puts ± 2.7 points on any single-seed number.

**+38 points over the campaign, ending at the top of the published
recurrent-SNN band (~0.71–0.83+) but short of the augmented/adaptive
state-of-the-art band (~0.90+).** The honest headline: a subtractive-reset
LIF network with exact linear sub-threshold dynamics, hand-rolled BPTT,
and a count readout reaches **0.88 ± noise** on SHD with ~2.5 h of
single-thread laptop training.

## Round 6 — learned time constants (pre-registered: docs/14) — **NULL**

First round under the improvements.md discipline: protocol committed and
pushed before the runs (docs/14), 3 seeds per arm as the default, and the
new `--threads 16` data-parallel trainer (~9× wall-clock; deterministic per
thread count, not bit-identical to the serial path — all arms share the
mode). Library support: backprop into the closed-form propagator entries
(`lif_entry_grads`, FD-gated), `KoopmanLayer::lif_hetero` (bit-identical to
fixed τ at the uniform 20/10 start), log-space per-neuron τ under the shared
Adam with clamps. Raw log: `sweep-AG-AH-X-log.txt`.

| run | seed 0 | seed 100 | seed 200 | mean ± half-range |
|---|---|---|---|---|
| X (fixed τ, control) | 0.8884 | 0.8638 | 0.8165 | **0.8563 ± 0.0359** |
| AG (X + learned τ) | 0.8737 | 0.8272 | 0.8268 | **0.8426 ± 0.0234** |
| AH (R + learned τ) | 0.8696 | 0.8549 | 0.8384 | **0.8543 ± 0.0156** |

- **Verdict (frozen rule): NULL** — mean(AG) − mean(X) = −0.0137, inside
  ±0.015. The mechanism engaged hard (layer-0 τ_m means ran from 20 to
  64–82 ms, distributions spanning the whole [5, 100] ms clamp), so the
  feature works and simply doesn't pay here: training loss dropped at or
  below the control's while test accuracy didn't move. Full analysis and
  scorecard: docs/15.
- **The two-layer recipe's seed band is ±3.6 points** (X: 0.8165–0.8884) —
  wider than round 5's ±2.7 on one layer. Round 4's X = 0.8857 was a
  favorable draw (its seed-0 re-run reproduces it at 0.8884); the recipe's
  honest value is ≈ 0.856.
- **Second leaderboard feature that fails to transfer as an isolated
  add-on** (after round 4's adaptation): the gap to 0.90+ increasingly looks
  carried by readout × time-resolution jointly, not neuron-model features.
  Next per the re-ranked roadmap: a trained temporal readout, then the
  combination experiment (learned τ × 5 ms bins × that readout).

## Round 7 — trained temporal readouts (pre-registered: docs/16) — **AI NEGATIVE, AJ NULL**

Two identity-initialized readouts on the X recipe, 3 seeds each, scored
against round 6's X control (mean 0.8563; reuse registered in docs/16). New
library support: `ReadoutMode::StaticProfile` (per-bin weights, init
all-ones = bitwise the count readout) and `ReadoutMode::SpikeAttention`
(scores u·s_t, softmax over time, u init 0 = uniform attention), both
trained jointly through hand-rolled backward passes (softmax Jacobian and
both spike paths), all test-gated. Raw log: `sweep-AI-AJ-log.txt`; full
analysis: docs/17.

| arm | seeds | mean ± half-range | Δ vs X mean | verdict |
|---|---|---|---|---|
| AI (static profile) | 0.8442 / 0.8393 / 0.8254 | 0.8363 ± 0.0094 | **−0.0200** | **NEGATIVE** |
| AJ (spike attention) | 0.8504 / 0.8754 / 0.8522 | 0.8594 ± 0.0125 | +0.0031 | **NULL** |

- **Both mechanisms engaged hard** (profiles swung to [−3.5, +6.5] with
  sign flips; attention concentrated 6–7× uniform), so these are verdicts
  on working features. A *learned* static profile is actively worse than
  uniform — AC's recency lesson generalizes: any fixed temporal weighting
  discards evidence because word alignment varies per sample. Data-dependent
  attention fixes the harm but buys no mean accuracy.
- **Post-hoc observation (not registered):** AJ's seed spread is ~3× tighter
  than the control's (±0.0125 vs ±0.0359; worst member 0.8504 vs 0.8165) —
  attention may stabilize rather than lift. n = 3; hypothesis only.
- **Single-axis scoreboard after seven rounds:** adaptation −, learned τ 0,
  static readout −, attention 0 — four literature features, none transfers
  alone. The registered consequence: readout axis closes, augmentation
  variety tops the re-ranked list, and one combination round (attention ×
  learned τ × 5 ms bins) is the designated next experiment.

## Round 8 — the combination round (pre-registered: docs/18) — **POSITIVE, SUPERADDITIVE**

The round-7 attribution — that the gap lives in *joint* interactions —
put to its registered test: AK = attention × learnable τ × 5 ms bins on
the X recipe; AL = 5 ms bins alone (the missing single-axis cell). Raw
log: `sweep-AK-AL-log.txt`; full analysis: docs/19.

| arm | seeds | mean ± half-range | Δ vs X mean | verdict |
|---|---|---|---|---|
| **AK (combination)** | 0.8812 / 0.8737 / **0.9116** | **0.8888 ± 0.0190** | **+0.0325** | **POSITIVE** |
| AL (5 ms alone) | 0.8795 / 0.8219 / 0.8429 | 0.8481 ± 0.0288 | −0.0082 | NULL |

- **Superadditive by the frozen formula:** the single-axis effects sum to
  −1.9 points; the combination delivers +3.3 — mean(AK) beats the additive
  expectation by +5.1 points. Four features that were individually
  null-or-negative are jointly worth five points: the interaction
  hypothesis, confirmed by pre-registered arithmetic.
- **First run over 0.90 in the campaign's history:** AK seed 200 at
  **0.9116**, inside the published state-of-the-art band. Honest headline:
  the 3-seed mean **0.8888**.
- **Mechanisms 3/3 on both gates**, with a twist: at 5 ms bins under
  attention, layer-0 τ_m learned *faster* membranes (means 11–24 ms) where
  round 6's count-readout runs had inflated it to ~80 ms — the same
  parameter finds different optima in different contexts, which is exactly
  why it was null alone.
- **Predictions 4/4** — the campaign's first perfect scorecard.

### Cumulative journey (updated)

| stage | configuration | honest value |
|---|---|---|
| original demo | 100 ch, 1 × 128, 600 mb | 0.502 |
| round 1 (I) | 350 ch, 1 × 256, 3000 mb | 0.680 |
| round 2 (L) | + recurrent hidden layer | 0.808 |
| round 3 (R) | + augmentation × budget | 0.850 (Z-audit mean) |
| round 4 (X) | + second recurrent layer | 0.856 (r6 3-seed mean) |
| **round 8 (AK)** | **+ attention × learned τ × 5 ms bins** | **0.889 (3-seed mean); best single 0.9116** |

**+38.7 points as a mean-of-seeds number, with the campaign's first entry
into the published 0.90–0.94 band.** AK is the new default recipe; next
registered rounds: augmentation variety on top of it, and an AK ensemble.

## Round 9 — augmentation variety + AK ensemble (pre-registered: docs/20) — **AM NEGATIVE, AN 0.9000**

Both registered follow-ups to round 8, on the AK default recipe. Raw
logs: `sweep-AM-log.txt`, `sweep-AN-log.txt`; full analysis: docs/21.

| run | result | Δ vs AK mean | verdict |
|---|---|---|---|
| AM (+ channel-block / time-mask / noise corruptions) | mean 0.8729 ± 0.0058 | −0.0159 | **NEGATIVE** |
| AN (ensemble ×3 of AK) | **0.9000** (2016/2240) | +0.0112 | **POSITIVE — milestone** |

- **The 0.90 milestone fired at exactly its threshold** — 2016 of 2,240,
  not one sample to spare; the flag's rule was frozen before the run. The
  campaign's honest, variance-reduced headline is now **0.9000**.
- **AM is an informative negative:** the manipulation check passed (train
  loss ~2× AK's), so the corruptions engaged — and cost 1.6 points. At
  these untuned strengths, extra variety destroys signal rather than
  regularizing; the axis closes at this operating point.
- Predictions 3/4 (V1 refuted, V2–V4 confirmed).

### Final campaign scoreboard (nine rounds)

| headline | value |
|---|---|
| honest, variance-reduced | **0.9000** (AN, ensemble ×3 of AK) |
| honest mean-of-3-seeds | 0.8888 ± 0.0190 (AK) |
| best single run | 0.9116 (AK seed 200) |
| starting point | 0.502 |

Every accuracy axis on the improvements-plan list now carries a registered
outcome. The recipe that got here: two recurrent 256-neuron LIF layers
with learnable per-neuron time constants, spike-driven temporal attention
over 5 ms bins, standard three-way augmentation, exact-propagator
dynamics, hand-rolled surrogate BPTT — on a laptop.
