# Round 12: The PSN-Mode Study — Pre-registration

**Document:** 29-psn-mode-preregistration.md · **Status: FROZEN before any
gated run** · **Date frozen:** 2026-08-31
**Origin:** docs/24 §4 deferred the reset-free "PSN-mode" — the one model
class in this library that permits time-parallel training — behind a
registration requirement. This is that registration. Protocol committed and
pushed before the runs, per the P0.1 discipline.

## 1 Hypothesis and rationale

docs/24 established that exact spiking dynamics with resets (and
recurrence, and per-step spike feedback generally) are inherently
sequential in time, and that the spiking-SSM literature obtains parallel
scans by *removing the reset*. The open question is the price: **what does
the time-parallelizable model class cost in accuracy on this pipeline?**
This study prices the trade; it does not implement the time-parallel
restructuring (that engineering follows only if the price is acceptable).

Two removals separate the AK recipe from the parallelizable class: the
subtractive reset (spike feedback into the own state) and recurrence
(spike feedback into the drive). A two-arm design decomposes their costs.

## 2 Implementation under test (merged and gated)

`KoopmanLayer::without_reset()` zeroes all spike-triggered jumps: the state
follows the free linear trajectory and spikes become a pure threshold
readout of it — Fang et al. (2023)'s Parallel Spiking Neuron in this
library's terms. Gates: `no_reset_state_follows_the_free_trajectory`
(state bitwise equal to a never-spiking layer's, while spikes still fire)
and `no_reset_mode_trains`. The reset's backward path vanishes
automatically with zero jumps (the jump-gradient loop skips zero entries),
so training needs no changes.

## 3 Protocol (frozen)

Harness at the commit carrying this document; full-test-set evaluation
(2,240); `--seeds 3` (bumps 0/100/200); `--threads 16`.

| run | config | role |
|---|---|---|
| `AT --seeds 3` | no reset, **no recurrence**, otherwise the modern recipe (two 256 layers, aug, 6000 mb, attention, learned τ, 5 ms × 200) | the time-parallelizable class |
| `AU --seeds 3` | no reset, **with recurrence** (= AK minus the reset only) | ablation isolating the reset |
| AK | **reused** (round 8): mean **0.8888 ± 0.019** | control |

Command:

```
cargo run --release --features datasets --example shd_sweep -- AT AU --seeds 3 --threads 16
```

**Rules (frozen), 3-seed means:**
- **AT vs AK** and **AU vs AK**, each: ≥ +0.015 → POSITIVE; ≤ −0.015 →
  NEGATIVE; else NULL.
- **Decomposition (reported):** reset effect = mean(AU) − mean(AK);
  recurrence-under-no-reset effect = mean(AT) − mean(AU).
- **The decision this study exists to make:** if AT lands NULL-or-better
  vs AK, the parallelism tax is negligible and a registered engineering
  project to restructure PSN-mode training time-parallel is justified;
  if AT is NEGATIVE, docs/24's deferral stands, now with a price tag.

## 4 Predictions (falsifiable, before running)

- **V1:** AT NEGATIVE by ≥ 3 points (point estimate −5 to −10): the
  parallelizable class pays heavily, chiefly for losing recurrence
  (round 2's +12.8 on the old recipe).
- **V2:** AU ≤ mean(AK) (the reset contributes positively — subtractive
  reset acts as per-neuron normalization; without it counts saturate),
  magnitude honestly uncertain, point estimate −2.
- **V3:** AT < AU by ≥ 2 points: recurrence matters even without a reset.
- **V4:** no member below 0.60 — even the fully stripped class, with
  modern ingredients, should beat the campaign's feedforward era.

## 5 Disclosures

- Neither arm has ever been run on SHD; the no-reset mode ran only its
  synthetic gates. AT's members will also be candidates for the round-13
  ensemble pool under that round's qualification bar (registered
  separately in docs/31).
- Expected wall-clock: 6 trainings ≈ 70 min at 16 threads.
- Amendment policy: tighten-only; deviations recorded in docs/30.

## 6 Consequence

- AT NULL-or-better → register the time-parallel engineering project.
- AT NEGATIVE (expected) → the docs/24 deferral is upheld with a measured
  price; the reset's share (via AU) is recorded as the exact-formulation's
  measured accuracy contribution — the first direct price tag on the
  library's central design decision.
