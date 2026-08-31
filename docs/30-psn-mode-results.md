# Round 12: The PSN-Mode Study — Results

**Protocol:** `docs/29-psn-mode-preregistration.md` (frozen, committed and
pushed before these runs) · **Date run:** 2026-08-31 ·
**Raw log:** `demo/sweep-AT-AU-log.txt` · **Command:** exactly as
registered. **Deviations: none.**

## VERDICTS: AT NEGATIVE (−2.5) · AU NEGATIVE (−2.3) — and the tax is far smaller than predicted

| arm | seeds | mean ± half-range | Δ vs AK (0.8888) | verdict |
|---|---|---|---|---|
| AT (no reset, no recurrence) | 0.8589 / 0.8371 / 0.8942 | **0.8634 ± 0.0286** | **−0.0254** | NEGATIVE |
| AU (no reset, recurrent) | 0.8737 / 0.8304 / 0.8938 | **0.8659 ± 0.0317** | −0.0229 | NEGATIVE |

**The registered decomposition:**

```
reset effect                = mean(AU) − mean(AK) = −0.0229
recurrence (given no reset) = mean(AT) − mean(AU) = −0.0025  ≈ 0
```

Two findings, one expected and one not:

1. **The subtractive reset is worth ~2.3 points** — the first direct price
   tag on the library's central design decision, and almost exactly the
   registered point estimate (−2). Mechanistically consistent with the
   reset acting as per-neuron output normalization; without it, the free
   trajectory's threshold readout is noisier but evidently still largely
   usable under the attention readout.
2. **Recurrence contributes nothing in the modern recipe** (−0.25 points,
   deep inside noise; V3's "≥ 2 points" refuted decisively). Round 2's
   +12.8 for recurrence — the campaign's largest single effect — was
   measured on the count-readout/10 ms/fixed-τ recipe. Under attention,
   5 ms bins, and learned τ, that entire contribution has been absorbed:
   the temporal integration recurrence once provided is now supplied by
   the readout and the finer, adaptable dynamics. This closes a loop with
   study S-B (docs/26), which found recurrence dominating the *gradient*
   horizon: recurrence still shapes credit flow, but no longer adds
   task-relevant computation here.

**The decision the study existed to make (frozen rule):** AT is NEGATIVE,
so the docs/24 deferral of time-parallel engineering is **upheld** — but
now with a price tag of **2.5 points**, five-fold smaller than the
registered expectation, and with the incidental observation that the
parallelizable class already trains 2.4–2.8× faster *sequentially*
(feedforward steps: ~230 s vs ~560 s per 6,000 minibatches). A future
registered proposal may reasonably weigh 2.5 points against
order-of-magnitude training-cost reductions; the door docs/24 closed is
now measurably ajar.

## Predictions scorecard — 2 confirmed, 1 half, 1 refuted

| # | Prediction | Outcome |
|---|---|---|
| V1 | AT NEGATIVE by ≥ 3 points (est. −5 to −10) | **HALF** — verdict NEGATIVE confirmed; magnitude refuted (−2.5) |
| V2 | AU ≤ AK, est. −2 | **CONFIRMED** — −2.3 |
| V3 | AT < AU by ≥ 2 (recurrence matters without reset) | **REFUTED** — −0.25, ≈ zero |
| V4 | no member below 0.60 | **CONFIRMED** — minimum 0.8304 |

## Round-13 qualification (frozen bar: bump-0 ≥ 0.8700, docs/31)

- AT bump-0 = 0.8589 → **does not qualify** (as predicted, U3).
- AU bump-0 = 0.8737 → **QUALIFIES**: the no-reset recurrent network joins
  the ensemble pool as a distinct dynamics class.

## Consequence (per docs/29 §6)

The deferral stands per the letter of the frozen rule; the spirit is
revised in the record: the accuracy cost of the time-parallelizable class
is 2.5 points (nearly all of it the reset), recurrence is droppable in
modern recipes, and PSN-mode restructuring is a live candidate for the
next registered engineering proposal rather than a closed door.
