# The Combination Round — Results

**Protocol:** `docs/18-combination-preregistration.md` (frozen, committed and
pushed before these runs) · **Date run:** 2026-08-27 ·
**Raw log:** `demo/sweep-AK-AL-log.txt` · **Command:** exactly as registered
(`AK AL --seeds 3 --threads 16`). **Deviations: none.**

## VERDICT: POSITIVE and SUPERADDITIVE

Every frozen rule fired in the hypothesis's favor, and every mechanism gate
passed — the first fully positive, fully predicted round of the campaign.

| run | seed 0 | seed 100 | seed 200 | mean ± half-range | Δ vs X (0.8563) | verdict |
|---|---|---|---|---|---|---|
| AK (attention × τ × 5 ms) | 0.8812 | 0.8737 | **0.9116** | **0.8888 ± 0.0190** | **+0.0325** | **POSITIVE** |
| AL (5 ms bins alone) | 0.8795 | 0.8219 | 0.8429 | 0.8481 ± 0.0288 | −0.0082 | NULL |

**The attribution arithmetic (frozen in docs/18):**

```
E_add   = mean(AL) − 0.0106 = 0.8481 − 0.0106 = 0.8375
mean(AK) − E_add = 0.8888 − 0.8375 = +0.0513  ≥ +0.015  →  SUPERADDITIVE
```

The three single-axis effects *sum to −1.9 points below the control*; the
combination lands **+3.3 above it** — a five-point interaction. The features
the published 0.90+ systems ship together are, on this pipeline too, worth
something only together. This is the strongest possible vindication of the
round-7 attribution: the gap lived in joint interactions all along.

**Milestone:** AK seed 200 reached **0.9116** — the campaign's first run
above 0.90, inside the published state-of-the-art band (0.90–0.94). The
honest number remains the 3-seed mean **0.8888**, itself above every
previous *single-seed best* in the campaign's history.

## Mechanism gates: PASSED 3/3 on both

- **Attention:** concentration 0.0489 / 0.0434 / 0.0470 against the 0.010
  gate (uniform = 0.005) — ~9× uniform in every member.
- **τ:** distributions span the clamp region in every member (layer-0 ranges
  [5.0, 35.3], [5.0, 74.0], [5.0, 98.3]); the ≥ 10%-of-neurons movement
  criterion is met unambiguously.

A mechanistic note worth recording: at 5 ms bins under attention, layer-0
τ_m moved **down** (means 11.3–23.6 ms) — toward *faster* membranes — where
round 6's 10 ms-bin, count-readout runs had inflated it toward the 100 ms
clamp. The same learnable parameter finds qualitatively different optima
depending on its context, which is precisely why it was null in isolation.

## Predictions scorecard — 4 for 4

| # | Prediction | Outcome |
|---|---|---|
| S1 | AK POSITIVE, +1.5 to +4 | **CONFIRMED** — +3.3, inside the registered range |
| S2 | AL NULL | **CONFIRMED** — −0.8 |
| S3 | any AK win is superadditive | **CONFIRMED** — +5.1 over the additive expectation |
| S4 | no member < 0.80; both mechanisms engage | **CONFIRMED** — minimum 0.8219 (an AL member); gates 3/3 |

## Consequence (per docs/18 §6, branch 1)

- **The combination becomes the default recipe** (AK: two recurrent 256–256,
  augmentation, attention readout, learnable τ, 5 ms × 200 bins).
- The campaign's honest headline updates: **0.889 ± 0.019 (3-seed mean)**,
  best single run **0.9116**, at the bottom edge of the published
  state-of-the-art band — reached by a subtractive-reset LIF network with
  exact linear sub-threshold dynamics and hand-rolled BPTT, on a laptop.
- The next registered round, per the roadmap: **augmentation variety on top
  of the AK recipe** — the one axis with known headroom, now applied to a
  recipe whose training losses (0.12–0.14, the campaign's lowest) suggest
  capacity to absorb it. An AK-recipe ensemble is the cheap variance play
  alongside it.
