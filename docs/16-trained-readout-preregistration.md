# Trained Temporal Readout — Pre-registration

**Document:** 16-trained-readout-preregistration.md · **Status: FROZEN before
any gated run** · **Date frozen:** 2026-08-27
**Origin:** improvements.md P1 item 1 (promoted to the top after the
learned-τ NULL, docs/15). Protocol committed and pushed before results, per
the P0.1 discipline.

## 1 Hypothesis and rationale

The count readout is load-bearing: round 5's AC (fixed exponential recency
weighting) collapsed accuracy by 14 points, proving evidence is spread across
the utterance — but *uniform* weighting is an assumption, not a result. Two
consecutive neuron-model features (adaptation, learned τ) failed to transfer
as isolated add-ons, and the published 0.90+ systems pair their neuron models
with *data-dependent* readouts. This round tests the readout axis directly,
with the round-2/6 discipline: every variant reduces **exactly** to the count
readout at initialization, so training can only *grow* temporal structure
that pays.

**H1 (attention):** letting the readout weight time bins by learned,
spike-driven attention improves SHD accuracy over the uniform count.
**H2 (static, the control-for-the-mechanism):** a learned *static* per-bin
profile — temporal reweighting without data dependence — captures little or
none of that gain.

## 2 Implementation under test (merged and gated)

`TrainConfig::readout_mode` (`ReadoutMode`):

- **StaticProfile**: `c = Σ_t w_t·s_t / T`, `w ≡ 1` at init — bitwise the
  count readout (gate: `trained_readouts_init_identical_to_count`, exact
  logits equality).
- **SpikeAttention**: scores `z_t = u·s_t`, per-sample softmax over time,
  `c = Σ_t a_t·s_t`; `u ≡ 0` at init — uniform attention, equal to the count
  readout to FP roundoff (same gate, ≤ 1e-12). Backward passes through the
  softmax Jacobian and both spike paths (weighted-sum and score).
- Both train jointly under the shared Adam with the standard clip; both learn
  and engage on the synthetic task
  (`trained_readouts_learn_and_engage`) and survive data-parallel chunking
  (`attention_readout_threaded_matches_serial`).

## 3 Protocol (frozen)

Harness at the commit carrying this document; full-test-set evaluation
(2,240 samples); `--seeds 3` (seed_bump 0/100/200); `--threads 16`.

| run | config | role |
|---|---|---|
| `AI --seeds 3` | X recipe (two recurrent 256–256, aug, 6000 mb) + learned static profile | static-reweighting arm |
| `AJ --seeds 3` | X recipe + spike-driven temporal attention | attention arm |
| X control | **reused from round 6** (docs/15): 0.8884 / 0.8638 / 0.8165, mean **0.8563** | control |

Control reuse is disclosed and legitimate: the round-6 X runs used the
identical recipe, seeds, data path, and execution mode (threads 16), and are
statistically independent of the new arms. No re-run.

Command:

```
cargo run --release --features datasets --example shd_sweep -- AI AJ --seeds 3 --threads 16
```

**Primary metric:** mean test accuracy over 3 seeds, per arm.

**Decision rule (frozen), per arm vs the X control mean 0.8563:**
- ≥ +0.015 → **POSITIVE**
- ≤ −0.015 → **NEGATIVE**
- otherwise → **NULL**

**Mechanism gates (required for any POSITIVE claim on that arm):**
- AI: the learned profile leaves the identity — printed range satisfies
  max ≥ 1.05 or min ≤ 0.95 in at least 2 of 3 members.
- AJ: attention concentrates — the printed probe (mean over the first full
  test batch of `max_t a_t`, uniform = 1/100 = 0.010) reaches ≥ 0.020 in at
  least 2 of 3 members.

**Power caveat** (unchanged from docs/14): single-seed sd ≈ 2.7–3.6 points on
this recipe; a 3-seed mean difference has SE ≈ 2 points. The ±1.5-point band
is a claim-throttle, not a significance test; NULL is a likely outcome under
small true effects and will be reported as such.

## 4 Predictions (falsifiable, before running)

- **Q1:** AJ lands POSITIVE, +1 to +3 points (the data-dependent mechanism
  the published readouts use). Confidence tempered by rounds 4 and 6:
  isolated features keep underperforming their literature billing here.
- **Q2:** AI lands NULL — AA (duration) and AC (recency) both said static
  timing structure carries little on SHD; a learned static profile should
  rediscover ≈ uniform.
- **Q3:** mean(AJ) ≥ mean(AI) (ordering: data-dependence is the active
  ingredient).
- **Q4:** no member below 0.80 — both modes are supersets of the count
  readout grown from identity, so catastrophic collapse à la AC should be
  impossible.

## 5 Disclosures

- Neither AI nor AJ has ever been run on SHD before this freeze (both modes
  ran only the synthetic-task test-suite gates and no accuracy was computed
  on SHD data).
- The X control is reused from round 6 rather than re-run (see §3).
- Amendment policy: tighten-only; deviations recorded in the results
  document.

## 6 Consequence

- AJ POSITIVE → attention joins the default recipe; the next registered
  round is the combination the literature actually ships: attention ×
  learned τ × 5 ms bins.
- Both arms NULL/NEGATIVE → the readout axis closes at this granularity;
  improvements.md P1 re-ranks with augmentation variety at the top, and the
  gap to 0.90+ is provisionally attributed to joint feature interactions
  none of our single-axis rounds can reach — motivating one final registered
  combination round before the campaign re-plans.
