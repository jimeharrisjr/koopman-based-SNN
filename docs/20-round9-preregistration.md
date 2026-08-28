# Round 9: Augmentation Variety and the AK Ensemble — Pre-registration

**Document:** 20-round9-preregistration.md · **Status: FROZEN before any
gated run** · **Date frozen:** 2026-08-28
**Origin:** the registered consequence of round 8 (docs/19 §consequence):
build on the new AK default recipe along the two axes with known headroom.
Protocol committed and pushed before results, per the P0.1 discipline.

## 1 Hypotheses and rationale

Round 8 established the combination recipe AK (two recurrent 256–256,
augmentation, attention readout, learnable τ, 5 ms × 200 bins) at mean
0.8888 ± 0.0190. Its final training losses (0.12–0.14) are the campaign's
lowest — the round-3/4 pattern says capacity that fits the training set this
well can absorb harder augmentation. And the AF precedent (round 5) says a
same-recipe ensemble buys variance reduction at ~zero design risk.

**U1 (variety):** adding SpecAugment-style corruptions — contiguous
channel-block dropout, contiguous time masking, additive noise events — to
the standard three improves the AK recipe.
**U2 (ensemble):** summing logits over three independently seeded AK members
yields an honest, variance-reduced number at or above the AK mean, with a
realistic shot at ≥ 0.90.

## 2 Implementation under test

Harness-only changes (no library code): `augment_extra` adds, per training
presentation and each with independent probability 0.5 —
- a contiguous **channel block mask**, width uniform in [1, 70] of 700;
- a contiguous **time mask**, width uniform in [0, 100] ms of the 1 s
  horizon (applied post-stretch);
- **additive noise events** at 2% of the kept event count, uniform in
  channel and time.

Strengths are a priori and untuned (disclosed limitation: a NULL under
these strengths does not close the axis — it bounds only this operating
point). Test data is never augmented. The ensemble arm reuses the existing
`ensemble` machinery (members at seed_bump +0/+1000/+2000, logits summed at
evaluation — the AF convention).

## 3 Protocol (frozen)

Harness at the commit carrying this document; full-test-set evaluation
(2,240 samples); `--threads 16`.

| run | config | seeds | role |
|---|---|---|---|
| `AM --seeds 3` | AK recipe + `augment_extra` | 0/100/200 | variety arm |
| `AN` | AK recipe, `ensemble: 3` (member bumps 0/1000/2000) | one evaluation | ensemble arm |
| AK | **reused** (round 8): mean **0.8888**, members 0.8812/0.8737/0.9116 | — | control |

Commands (two invocations, so the ensemble is a single evaluation rather
than being multiplied by `--seeds`):

```
cargo run --release --features datasets --example shd_sweep -- AM --seeds 3 --threads 16
cargo run --release --features datasets --example shd_sweep -- AN --threads 16
```

**Rules (frozen):**
- **AM:** mean over 3 seeds − 0.8888: ≥ +0.015 → **POSITIVE**;
  ≤ −0.015 → **NEGATIVE**; else **NULL**.
- **AN (single ensemble evaluation):** value − 0.8888: ≥ +0.010 →
  **POSITIVE**; ≤ −0.010 → **NEGATIVE**; else **NULL**. (Tighter band than
  a single training run deserves, because an ensemble evaluation is itself
  variance-reduced.)
- **Milestone flag:** AN ≥ 0.9000 is recorded as the campaign's first
  honest (variance-reduced) entry into the published band.

**Mechanism note:** AM needs no new mechanism gate (augmentation has no
learned parameter); its manipulation check is the training loss — harder
augmentation must *raise* final train loss vs AK's 0.12–0.14. If AM's train
loss does not rise, the corruptions were too weak to matter and a NULL is
uninformative about the axis.

**Power caveat** (as before): ±1.5 points on a 3-seed mean is a claim
throttle, not significance.

## 4 Predictions (falsifiable, before running)

- **V1:** AM POSITIVE (+1 to +2.5) — augmentation × capacity is the
  campaign's most reliable interaction (round 3: +9.6), and AK has fresh
  capacity headroom.
- **V2:** AN headline ≥ 0.90 (members drawn from a ≈ 0.889 ± 0.019
  distribution; AF matched its arm's best draw, and AK's best draw was
  0.9116).
- **V3:** AM's final train loss rises above AK's (manipulation check).
- **V4:** no AM member below 0.84.

## 5 Disclosures

- The extra corruptions have never been run on SHD; strengths were set a
  priori from SpecAugment-style conventions, untuned.
- AN retrains its members (the library has no model persistence); its
  members (seed bumps 0/1000/2000) are NOT round 8's AK runs (bumps
  0/100/200) — one member shares bump 0 with the control's first seed, a
  disclosed partial overlap inherent to the AF ensemble convention.
- Expected wall-clock: six ~11-minute trainings (AM 3 + AN members 3);
  ~70–80 min at 16 threads.
- Amendment policy: tighten-only; deviations recorded in the results
  document.

## 6 Consequence

- AM POSITIVE → `augment_extra` joins the default recipe; the final
  registered act of the campaign is an ensemble of the enriched recipe
  targeting round 5's unreached 0.92.
- AM NULL with the manipulation check failed → strengths get one registered
  retune; NULL with the check passed → the axis closes at this granularity.
- AN ≥ 0.90 → the campaign's headline becomes "0.90+ honest"; the paper's
  accuracy-ladder figure gains its final rung.
