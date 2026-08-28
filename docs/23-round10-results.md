# Round 10: Depth's Enabler and the Diverse Ensemble — Results

**Protocol:** `docs/22-round10-preregistration.md` (frozen, committed and
pushed before these runs) · **Date run:** 2026-08-28 ·
**Raw logs:** `demo/sweep-AO-AP-log.txt`, `demo/sweep-AQ-log.txt` ·
**Commands:** exactly as registered. **Deviations: none.**

## VERDICTS: AO NEGATIVE · AP NULL vs AK, ENABLER CONFIRMED · AQ 0.9179 — DIVERSITY WINS

| run | result | comparison | verdict |
|---|---|---|---|
| AO (3rd layer, plain) | 0.8826 / 0.8210 / 0.8875 → mean **0.8637 ± 0.0333** | −0.0251 vs AK | **NEGATIVE** |
| AP (3rd layer + skips) | 0.9018 / 0.8741 / 0.9045 → mean **0.8935 ± 0.0152** | +0.0047 vs AK | **NULL** |
| AP − AO (enabler) | — | **+0.0298** | **ENABLER CONFIRMED** |
| AQ (diverse ensemble {AK, AJ, X}) | **0.9179** (2056/2240) | +0.0179 vs AN | **DIVERSITY WINS** |

## The depth story, closed with a mechanism in hand

The two-arm design did its job. A plain third layer loses 2.5 points on
the AK recipe (AO) — the round-5 AE result replicates on a much stronger
base, so depth-without-enabler is now a twice-confirmed negative. Adding
zero-init skip connections recovers 3.0 of those points (AP), with the
mechanism gate passed emphatically: `max |W_skip|` grew from 0 to 3.6–4.4
in every member — the bypass is not decoration, it is load-bearing. But
enabled depth only *ties* two layers (+0.5, well inside noise): per the
registered consequence, **the depth axis closes at this scale** — with the
enabler validated for any future attempt at larger data or width. A
noteworthy descriptive point (not a registered claim): AP's mean 0.8935 is
the highest 3-seed mean of the campaign, and two of its three members
crossed 0.90.

## The diversity story: the surprise of the round

AQ combined one member each of AK (0.8812), AJ (0.8504), and X (0.8884) —
member draws known in advance by determinism and disclosed in the
protocol, mean member strength 0.873, *weaker* than AN's three AKs. The
summed-logit ensemble scored **0.9179**: three points above its own best
member and 1.8 above the homogeneous AN. Prediction W4 (diversity ≈
member-weakness, NULL) is **refuted in the favorable direction** —
decorrelation across bin widths, readouts, and architectures is worth far
more here than member strength. This is the cleanest evidence yet for the
round-7 post-hoc observation that different readouts fail on different
samples.

**0.9179 is the campaign's new headline number** — solidly inside the
published 0.90–0.94 band.

## Predictions scorecard — 3 of 5, with both misses instructive

| # | Prediction | Outcome |
|---|---|---|
| W1 | AO null-to-negative | **CONFIRMED** — −2.5 |
| W2 | mean(AP) ≥ mean(AO) + 0.010 | **CONFIRMED** — +3.0 |
| W3 | AP NULL vs AK | **CONFIRMED** — +0.5 |
| W4 | AQ NULL vs AN | **REFUTED** — diversity won by +1.8 |
| W5 | no run below 0.84; skip gate passes | **HALF-REFUTED** — AO seed 100 hit 0.8210; the gate passed 3/3 |

## Campaign scoreboard — the P1 accuracy program concludes

| headline | value | provenance |
|---|---|---|
| **Best honest number** | **0.9179** | AQ, diverse ensemble {AK, AJ, X} |
| Homogeneous ensemble | 0.9000 | AN (round 9) |
| Best 3-seed mean | 0.8935 (AP) / 0.8888 (AK) | rounds 10 / 8 |
| Best single training run | 0.9116 | AK seed 200 (round 8) |
| Starting point | 0.502 | first demo |

Every item on the improvements-plan accuracy list now carries a registered
outcome across rounds 6–10: learned τ (null alone), readouts (static
negative, attention null alone), the combination (positive, superadditive),
augmentation variety (negative at tested strengths), depth (negative
plain, enabler confirmed, null enabled), ensembles (homogeneous 0.9000,
diverse 0.9179). Five of ten rounds produced negatives or nulls that are
in the record beside the wins that they made interpretable.

## Consequence (per docs/22 §6)

- The **diverse ensemble is the campaign headline**. The registered
  follow-up it unlocks — diversity WITH member strength (e.g., {AK, AP,
  AK-seed-variant} or attention/count mixtures at full strength) — is
  eligible but not scheduled; the accuracy program stands concluded.
- The depth axis closes; `with_skip` stays in the library, validated.
- Remaining open work is P2/P3 (parallel-in-time training, snnTorch
  baseline, spiking-regime ROM, V4 spectral regularization) and the **paper
  revision**: the draft's abstract, accuracy ladder, and campaign narrative
  still end at round 5's 0.88 and now trail the record by five rounds and
  three points.
