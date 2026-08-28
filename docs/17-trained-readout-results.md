# Trained Temporal Readout — Results

**Protocol:** `docs/16-trained-readout-preregistration.md` (frozen, committed
and pushed before these runs) · **Date run:** 2026-08-27 ·
**Raw log:** `demo/sweep-AI-AJ-log.txt` · **Command:** exactly as registered
(`AI AJ --seeds 3 --threads 16`). **Deviations: none.**

## VERDICTS: AI NEGATIVE · AJ NULL

Against the registered X control mean 0.8563 (round 6, seeds 0/100/200,
threads 16):

| arm | seed 0 | seed 100 | seed 200 | mean ± half-range | Δ vs control | verdict |
|---|---|---|---|---|---|---|
| AI (static profile) | 0.8442 | 0.8393 | 0.8254 | **0.8363 ± 0.0094** | **−0.0200** | **NEGATIVE** |
| AJ (spike attention) | 0.8504 | 0.8754 | 0.8522 | **0.8594 ± 0.0125** | **+0.0031** | **NULL** |
| X (control, round 6) | 0.8884 | 0.8638 | 0.8165 | 0.8563 ± 0.0359 | — | — |

Both mechanisms engaged decisively, so these are verdicts on working
features, not dead code:

- **AI's profiles left the identity violently** (3/3 members): final ranges
  [−1.0, 6.5], [−3.5, 4.7], [−1.1, 6.0] against the all-ones start — some
  bins' evidence was even sign-flipped. The freedom was used, and it cost
  two points: a learned static temporal profile is *worse* than uniform,
  the same direction as round 5's fixed recency weighting (AC, −14) at
  smaller magnitude. Static temporal structure on SHD is not merely
  useless — reweighting by it actively discards evidence, because word
  alignment varies sample to sample.
- **AJ's attention concentrated 6–7× above uniform** (mean max_t a_t =
  0.061–0.070 vs 0.010, 3/3 members ≥ the 0.020 gate) — and bought
  nothing on the mean.

## Predictions scorecard

| # | Prediction | Outcome |
|---|---|---|
| Q1 | AJ POSITIVE, +1 to +3 | **REFUTED** — NULL (+0.3) |
| Q2 | AI NULL (static timing carries little) | **REFUTED** — NEGATIVE (−2.0): static timing carries *harm*, not nothing |
| Q3 | mean(AJ) ≥ mean(AI) | **CONFIRMED** — data-dependence is the better half of the axis |
| Q4 | no member below 0.80 | **CONFIRMED** — minimum 0.8254; identity-grown supersets don't collapse |

## A post-hoc observation (non-registered, flagged as such)

AJ's seed spread (±0.0125, worst member 0.8504) is markedly tighter than the
control's (±0.0359, worst 0.8165), and the paired per-seed differences flip
sign with the control's luck (−3.8 / +1.2 / +3.6 points): attention appears
to *stabilize* rather than lift — pulling up bad draws while capping good
ones. With n = 3 this is a hypothesis, not a finding; if a future round
cares about worst-case robustness rather than mean accuracy, this is the
observation to pre-register against.

## Consequence (per docs/16 §6)

Both arms failed to reach POSITIVE, so the readout axis **closes at this
granularity**. The single-axis scoreboard now reads: adaptation (round 4)
negative, learned τ (round 6) null-with-engagement, static readout
(round 7) negative-with-engagement, attention readout (round 7)
null-with-engagement. Four literature-billed features, none of which
transfers alone into this recipe. Per the registered consequence,
`improvements.md` P1 re-ranks with **augmentation variety** at the top, and
the standing attribution is that the remaining ~4-point gap to 0.90+ lives
in *joint* feature interactions — motivating one final registered
**combination round** (attention × learned τ × 5 ms bins, all engaged
mechanisms even where individually null) before the campaign re-plans.
