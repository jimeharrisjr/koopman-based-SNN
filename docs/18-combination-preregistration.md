# The Combination Round — Pre-registration

**Document:** 18-combination-preregistration.md · **Status: FROZEN before any
gated run** · **Date frozen:** 2026-08-27
**Origin:** improvements.md P1 item 2 — the registered consequence of rounds
6 and 7 (docs/15, docs/17). Protocol committed and pushed before results,
per the P0.1 discipline.

## 1 Hypothesis and rationale

Four literature-billed features have now been tested in isolation on this
pipeline, each with its mechanism demonstrably engaged, and none transferred:
adaptation (round 4, −3.7), learned τ (round 6, −1.4 NULL), learned static
readout (round 7, −2.0 NEGATIVE), attention readout (round 7, +0.3 NULL).
The published 0.90+ systems ship these ingredients **together** — learned
time constants at fine time resolution under data-dependent readouts. The
standing attribution, registered after round 7, is that the remaining
~4-point gap lives in *joint interactions* that single-axis rounds cannot
reach. This round tests that attribution directly, and is designed so that a
win can be attributed and a loss is terminal for the hypothesis.

**H1 (combination):** attention readout × learnable τ × 5 ms bins together
improve SHD accuracy over the fixed recipe.
**H2 (attribution):** any such improvement exceeds the *additive* sum of the
single-axis effects — i.e., it is an interaction, not accumulation.

## 2 Implementation under test

All three ingredients are merged and individually gated (docs/14, docs/16).
New for this round: the composition gate
`learned_tau_and_attention_compose` — learnable τ and the attention readout
had never run *simultaneously*; the gate proves joint training stays finite,
reduces loss, and engages both parameter groups under the threaded path.
At 5 ms bins the τ machinery uses dt = 5 in the same closed-form entry
gradients (FD-gated across operating points including dt ∈ [0.5, 10]).

## 3 Protocol (frozen)

Harness at the commit carrying this document; full-test-set evaluation
(2,240 samples); `--seeds 3` (seed_bump 0/100/200); `--threads 16`.

| run | config | role |
|---|---|---|
| `AK --seeds 3` | X recipe (two recurrent 256–256, aug, 6000 mb) + attention + learn_tau + 5 ms × 200 bins | the combination |
| `AL --seeds 3` | X recipe + 5 ms × 200 bins only | the missing single-axis cell |
| X | **reused** (round 6): mean **0.8563** | control |
| AG | **reused** (round 6, τ only): mean 0.8426 | single-axis term |
| AJ | **reused** (round 7, attention only): mean 0.8594 | single-axis term |

Control/single-axis reuse is disclosed and legitimate (identical recipe
lineage, seeds, data path, and execution mode; statistically independent of
the new arms). Round 5's AB (5 ms, +2.1) is **not** usable as the fine-bin
term: single seed, one-layer recipe — hence arm AL.

Command:

```
cargo run --release --features datasets --example shd_sweep -- AK AL --seeds 3 --threads 16
```

**Primary rule (H1), frozen:** mean(AK) − 0.8563:
- ≥ +0.015 → **POSITIVE** · ≤ −0.015 → **NEGATIVE** · else → **NULL**

**Attribution rule (H2), frozen:** the additive expectation is

```
E_add = 0.8563 + [mean(AG) − 0.8563] + [mean(AJ) − 0.8563] + [mean(AL) − 0.8563]
      = mean(AL) − 0.0106        (using the recorded AG/AJ means)
```

The interaction claim **SUPERADDITIVE** requires mean(AK) − E_add ≥ +0.015.
(If AK is positive but ≈ E_add, the driver is whichever single axis moved —
in practice AL.)

**Secondary rule:** mean(AL) − 0.8563, same ±0.015 bands — this scores the
fine-bin axis on the X recipe for the first time.

**Mechanism gates (required for any POSITIVE claim on AK), per the
registered per-run prints, in ≥ 2 of 3 members each:**
- τ engaged: in at least one layer, ≥ 10% of neurons end with
  |τ_m − 20| > 0.5 ms (as in docs/14).
- attention engaged: concentration (mean max_t a_t, first test batch)
  ≥ 0.010 = 2× the 200-bin uniform 0.005.

**Power caveat** (as docs/14/16): 3-seed mean differences carry SE ≈ 2
points; the ±1.5 band throttles claims, it does not confer significance.

## 4 Predictions (falsifiable, before running)

- **S1:** AK lands POSITIVE (+1.5 to +4). This is the hypothesis the round
  exists to test; stated with explicitly modest confidence (the single-axis
  record is 0-for-4), but it is the registered expectation — the published
  systems' recipes are evidence that the *joint* configuration works.
- **S2:** AL lands NULL — AB's +2.1 was a single seed on a different
  (one-layer) recipe; fine bins alone should not clear the band.
- **S3:** if AK is positive, it is SUPERADDITIVE (the singles sum to
  ≈ −1 point; any win must be interaction).
- **S4:** no member below 0.80; both mechanism gates pass in AK (they
  engaged in every prior single-axis run).

## 5 Disclosures

- AK and AL have never been run: no SHD accuracy has been computed for the
  attention × τ composition, nor for 5 ms bins on the X recipe, before this
  freeze. The composition ran only the synthetic-task gate.
- Expected wall-clock: 200-bin runs cost ≈ 2× the 100-bin runs; ~70–80 min
  for all six at 16 threads.
- Amendment policy: tighten-only; deviations recorded in the results
  document.

## 6 Consequence

- **AK POSITIVE + SUPERADDITIVE** → the interaction attribution is
  supported; the combination becomes the default recipe; the next
  registered round is augmentation variety on top of it.
- **AK POSITIVE but ≈ additive** (driven by AL) → fine time resolution is
  the active ingredient; the recipe adopts 5 ms bins and the roadmap
  re-ranks around it.
- **AK NULL/NEGATIVE** → the joint-interaction attribution is **falsified at
  this scale**: five literature mechanisms, alone and now in combination,
  fail to move this pipeline's mean. The accuracy program re-plans around
  the two axes with unexhausted headroom (augmentation variety, diverse
  ensembles), and the campaign records the honest conclusion that 0.90+ on
  SHD plausibly requires ingredients outside the current architecture class
  (e.g., trained input front-ends or sequence-model readouts), each of which
  would need its own registered proposal.
