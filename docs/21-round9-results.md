# Round 9: Augmentation Variety and the AK Ensemble — Results

**Protocol:** `docs/20-round9-preregistration.md` (frozen, committed and
pushed before these runs) · **Date run:** 2026-08-28 ·
**Raw logs:** `demo/sweep-AM-log.txt`, `demo/sweep-AN-log.txt` ·
**Commands:** exactly as registered. **Deviations: none.**

## VERDICTS: AM NEGATIVE · AN POSITIVE, MILESTONE REACHED

| run | result | Δ vs AK mean (0.8888) | verdict |
|---|---|---|---|
| AM (AK + extra corruptions) | 0.8772 / 0.8759 / 0.8656 → mean **0.8729 ± 0.0058** | **−0.0159** | **NEGATIVE** |
| AN (ensemble ×3 of AK) | **0.9000** (2016/2240) | +0.0112 | **POSITIVE** |

**The milestone flag fired at exactly its threshold: 0.9000.** The
campaign's first honest, variance-reduced number inside the published
0.90–0.94 band — by a margin of zero samples. (2016 correct of 2,240;
one fewer and the flag does not fire. Recorded with a smile and full
transparency: the flag's rule was frozen before the run.)

## AM: an informative negative at this operating point

The manipulation check **passed**: final train losses 0.2777 / 0.2755 /
0.2150 against AK's 0.12–0.14 — the corruptions engaged hard, roughly
doubling training difficulty — and test accuracy went *down* 1.6 points.
At these a priori strengths (channel blocks to 70/700, time masks to
100 ms, 2% noise events, each at p = 0.5), the extra variety crosses from
regularization into destruction of usable signal. Per the registered
consequence branch, the axis **closes at this operating point**: the
NEGATIVE-with-check-passed outcome is not the "too weak, retune" branch,
and any softer-strength retry requires its own registration. Notably, AM's
seed spread (±0.006) is the tightest of any arm in the campaign — heavy
augmentation homogenizes what the network can learn, in both senses.

## Predictions scorecard — 3 of 4

| # | Prediction | Outcome |
|---|---|---|
| V1 | AM POSITIVE, +1 to +2.5 | **REFUTED** — NEGATIVE, −1.6 |
| V2 | AN ≥ 0.90 | **CONFIRMED** — exactly 0.9000 |
| V3 | AM train loss rises above AK's | **CONFIRMED** — ~2× |
| V4 | no AM member below 0.84 | **CONFIRMED** — minimum 0.8656 |

## Campaign state after nine rounds

| headline | value | provenance |
|---|---|---|
| Honest, variance-reduced | **0.9000** | AN ensemble ×3 (this round) |
| Honest mean-of-3-seeds | 0.8888 ± 0.0190 | AK (round 8) |
| Best single run | 0.9116 | AK seed 200 (round 8) |

From 0.502 to an honest 0.90 in nine rounds, the last four pre-registered
with frozen rules, protocol-before-results commit ordering, and every
negative kept. The published band is entered; the remaining published
headroom (to ~0.94) sits beyond this recipe's measured axes — every one of
which is now either exhausted, negative, or absorbed into the AK default.

## Consequence (per docs/20 §6)

- `augment_extra` does **not** join the default recipe; the variety axis
  closes at this operating point (softer strengths would need a new
  registration, and the prior on them is now weak).
- The campaign's headline updates to **0.90 honest / 0.9116 best**; the
  draft paper's accuracy-ladder figure and abstract have a final rung to
  gain when next revised.
- With both round-9 axes resolved, the improvements-plan accuracy program
  (P1) is **complete as scoped**: items 1–4 all carry registered outcomes,
  and the remaining items (depth's enabler, diverse ensembles) are the only
  unexplored accuracy levers left on the list.
