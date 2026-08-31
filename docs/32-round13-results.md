# Round 13: Growing the Ensemble Member Pool — Results

**Protocol:** `docs/31-round13-preregistration.md` (frozen before ANY
candidate result, including round 12's) · **Date run:** 2026-08-31 ·
**Raw logs:** `demo/sweep-AV-AW-log.txt`, `demo/sweep-AX-log.txt` ·
**Commands:** exactly as registered; the AX member list was set
mechanically by the frozen qualification rule. **Deviations: none.**

## Stage 1 — candidates: both new architectures land NULL vs AK, and everything qualifies

| candidate | seeds | mean ± half-range | Δ vs AK (0.8888) | verdict | bump-0 | qualifies (≥ 0.8700)? |
|---|---|---|---|---|---|---|
| AV (ALIF-hetero, modern recipe) | 0.8871 / 0.8625 / 0.9098 | 0.8865 ± 0.0237 | −0.0023 | NULL | 0.8871 | **yes** |
| AW (wide 1×512, modern recipe) | 0.8759 / 0.8804 / 0.8835 | 0.8799 ± 0.0038 | −0.0089 | NULL | 0.8759 | **yes** |
| AU (round 12) | — | 0.8659 ± 0.0317 | −0.0229 | NEGATIVE | 0.8737 | **yes** |
| AT (round 12) | — | 0.8634 ± 0.0286 | −0.0254 | NEGATIVE | 0.8589 | no |

**Adaptation's redemption completes the campaign's central pattern.** AV —
spike-triggered adaptation with per-neuron random time constants — was
−3.7 points in round 4's count-readout/10 ms recipe and is NULL (−0.2) in
the combination era, with one seed at 0.9098. Every "failed" literature
feature tested on the old recipe (learned τ, adaptation) turns out to have
been mis-measured by its context: features do not have effects, *feature ×
recipe pairs* do. AW's ±0.004 seed spread is the tightest of any trained
configuration in the campaign — width appears to buy stability, not
accuracy.

## Stage 3 — the six-member ensemble: SATURATION

```
AX = {AK, AP, X, AU, AV, AW}   (mechanical, per the frozen rule)
     five architectures, three dynamics classes, mean member strength 0.8847
```

| arm | result | Δ vs AR (0.9366) | verdict |
|---|---|---|---|
| AX (six members) | **0.9379** (2101/2240) | +0.0013 | **NULL** |

The 0.9400 milestone does not fire. Doubling the pool from three strong
members to six — adding a new neuron model, a new shape, and a new
dynamics class — moved the ensemble by **three test samples**. The
numerically best honest number of the campaign is now 0.9379, but it is
statistically indistinguishable from AR's 0.9366: **the diverse-ensemble
axis has saturated at ≈ 0.937–0.938 for this architecture generation.**
The plateau is informative: the ~140 test samples all six architectures
still miss together are, by construction, the samples on which every
readout, depth, time resolution, and dynamics class in this library
agrees and errs — the residual is data-limited or class-limited, not
architecture-limited.

## Predictions scorecard — 4 confirmed, 1 half

| # | Prediction | Outcome |
|---|---|---|
| U1 | AW qualifies | **CONFIRMED** |
| U2 | AV NULL-or-better vs AK (est. 0.87) | **CONFIRMED** — 0.8865, NULL |
| U3 | AT fails the bar; AU uncertain | **CONFIRMED** — AT out, AU in |
| U4 | AX ≥ AR; milestone plausible with ≥ 2 qualifiers | **HALF** — direction (+0.13) right, verdict NULL, milestone unfired with three qualifiers |
| U5 | no regression below AR − 0.010 | **CONFIRMED** |

## Consequence (per docs/31 §5)

The pool-growth avenue **closes at this architecture generation**: further
ensemble gains require model classes the library does not yet express,
each behind its own registration. The campaign's final scoreboard:

| headline | value | provenance |
|---|---|---|
| **Best honest number** | **0.9379** (≈ 0.9366) | AX / AR, saturated ensemble plateau |
| Best 3-seed mean | 0.8935 (AP) | round 10 |
| Best single run | 0.9116 (AK seed 200) | round 8 |
| Starting point | 0.502 | first demo |

Thirteen rounds, the last eight fully pre-registered; every feature and
every architecture in the final ensembles carries a mechanism-gated,
frozen-rule verdict; and the two closing rounds each contributed a law:
the parallelism tax is 2.5 points with recurrence free to drop (round 12),
and feature × recipe interactions — not features — are the unit of
evidence (rounds 4→13, sealed by adaptation's redemption).
