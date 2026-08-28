# Learned Time Constants — Results

**Protocol:** `docs/14-learned-tau-preregistration.md` (frozen, committed and
pushed before these runs) · **Date run:** 2026-08-27 ·
**Raw log:** `demo/sweep-AG-AH-X-log.txt` · **Command:** exactly as
registered (`AG X AH --seeds 3 --threads 16`). **Deviations: none.**

## VERDICT: NULL

Per the pre-registered decision rule: mean(AG) − mean(X) = 0.8426 − 0.8563 =
**−0.0137**, inside the ±0.015 band. Learnable per-neuron time constants
produced **no detectable accuracy effect** on this recipe at this power —
with the mechanism gate clearly **passed**, so this is a genuine null on an
engaged feature, not a dead implementation.

## The numbers

| run | seed 0 | seed 100 | seed 200 | mean ± half-range |
|---|---|---|---|---|
| X (fixed τ, control) | 0.8884 | 0.8638 | 0.8165 | **0.8563 ± 0.0359** |
| AG (X + learned τ) | 0.8737 | 0.8272 | 0.8268 | **0.8426 ± 0.0234** |
| AH (R + learned τ) | 0.8696 | 0.8549 | 0.8384 | **0.8543 ± 0.0156** |

Paired per-seed differences AG − X: −0.0147, −0.0366, **+0.0103** — the sign
itself is seed-dependent. Secondary comparison: mean(AH) = 0.8543 vs the
recorded Z-audit R mean 0.850 → +0.004, also well inside the null band.

## Mechanism gate: PASSED (τ moved, a lot)

Every AG/AH member moved its time constants far beyond the 0.5 ms gate.
The consistent, striking pattern (final per-layer means; init was uniform
τ_m = 20, τ_s = 10 ms):

- **Layer 0 τ_m inflated hugely in AG** (means 64–82 ms; many neurons pinned
  at the 100 ms clamp) — the input layer, given the choice, wants very slow
  membranes. τ_s roughly doubled (means ≈ 19 ms).
- **Layer 1 τ_m stayed near its init** (means ≈ 22–23 ms), while its τ_s
  rose to ≈ 15 ms.
- AH's single layer landed in between (τ_m means ≈ 24–25 ms).
- Final distributions span the entire clamp region (ranges ≈ [5, 100] ms) —
  prediction P2's heterogeneity, delivered.

Interpretation (post-hoc, non-registered): under a spike-count readout,
slower membranes mean longer integration and more spikes — a path to lower
training loss (AG's final losses were at or below X's) that evidently buys
no generalization. The published learned-τ wins pair the feature with finer
time resolution and richer readouts; τ alone, on 10 ms bins with a count
readout, rearranges the solution without improving it.

## Predictions scorecard

| # | Prediction | Outcome |
|---|---|---|
| P1 | mean(AG) > mean(X) by +1 to +3 points | **REFUTED** — point estimate −1.4, verdict NULL |
| P2 | learned τ_m spreads > 5 ms within a layer | **CONFIRMED** — spans the full clamp region |
| P3 | no member below 0.80 | **CONFIRMED** — minimum 0.8165 (an X control member) |

## Two findings beyond the headline

1. **The two-layer recipe's seed noise is even larger than round 5
   measured on one layer:** X spans 0.8165–0.8884 across three seeds
   (± 3.6 points vs the Z audit's ± 2.7). The recorded single-seed X = 0.8857
   was, like R's 0.873 before it, a favorable draw — its seed-0 re-run under
   threads-16 (0.8884) replicates it within execution-mode noise, but the
   recipe's honest value is ≈ 0.856 ± 0.036. Every future comparison on this
   recipe inherits that band.
2. **This is now the second published-leaderboard feature (after round 4's
   adaptation) that does not transfer as an isolated add-on to this
   pipeline.** The pattern strengthens the roadmap's standing hypothesis:
   the remaining gap is carried by the readout and time resolution jointly,
   not by neuron-model features alone.

## Consequence (per docs/14 §6)

- Learned τ does **not** join the default recipe; the implementation stays
  in the library (gated, inert when off) for future combination experiments
  — the obvious registered follow-up is learned τ × 5 ms bins × a trained
  temporal readout, where the published wins live.
- `improvements.md` P1 re-ranks: the **trained temporal readout** is now the
  top accuracy item; learned τ drops to "retry only in combination."
