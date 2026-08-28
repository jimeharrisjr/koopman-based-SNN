# Learned Time Constants — Pre-registration

**Document:** 14-learned-tau-preregistration.md · **Status: FROZEN before any
gated run** · **Date frozen:** 2026-08-27
**Origin:** improvements.md P1.1 (the roadmap's highest-expected-value
accuracy feature), executed under the P0.1 discipline: this protocol commit
precedes the results commit, making the ordering externally verifiable —
closing the gap flagged in the README and the draft paper's limitations.

## 1 Hypothesis and rationale

Published SHD results above 0.90 learn per-neuron time constants; this
library's campaign fixed τ_m = 20 ms, τ_s = 10 ms throughout. Round 4's ALIF
negative result (V: −3.7 points) was attributed in `demo/RESULTS.md` to
exactly this gap. The exact-propagator formulation makes learned τ clean:
α, β, γ, δ, 1−β are closed-form functions of (τ_m, τ_s), so BPTT extends into
them analytically with no new approximation (the surrogate remains the only
one).

**H1:** letting each neuron learn its own (τ_m, τ_s) improves SHD test
accuracy over the identical fixed-τ recipe.

## 2 Implementation under test (already merged and gated)

- `KoopmanLayer::lif_hetero` — per-neuron LIF propagator; with uniform τ it
  is bit-identical to the fixed-τ layer (gate:
  `uniform_lif_hetero_matches_homogeneous_lif_exactly`).
- `lif_entry_grads` — analytic ∂{α,β,γ,δ,1−β}/∂τ, gated against central
  finite differences at five operating points
  (`entry_grads_match_central_finite_differences`).
- `TrainConfig::learn_tau` — log-space τ parameters, shared Adam, clamps
  τ_m ∈ [5, 100] ms, τ_s ∈ [2, 50] ms, τ_m ≥ 1.2 τ_s. Off = provably inert
  (`learn_tau_off_leaves_hetero_training_identical_to_fixed`, exact
  equality). End-to-end: `learn_tau_trains_and_moves_the_time_constants`.

## 3 Protocol (frozen)

All runs use the shd_sweep harness at the commit carrying this document, on
the full-test-set protocol (2,240 samples), with `--seeds 3` (seed_bump
offsets 0/100/200) and `--threads 16`. Threaded execution is deterministic
for fixed thread count but not bit-identical to the serial path (documented
on `TrainConfig::threads`); all compared runs share the same mode, so
comparisons are internally consistent.

| run | config | role |
|---|---|---|
| `AG --seeds 3` | X recipe (two recurrent 256–256, aug, 6000 mb) + learn_tau, τ init uniform 20/10 | treatment |
| `X --seeds 3` | identical, fixed τ | primary control |
| `AH --seeds 3` | R recipe (one recurrent 256, aug, 6000 mb) + learn_tau | secondary |

Commands:

```
cargo run --release --features datasets --example shd_sweep -- AG X AH --seeds 3 --threads 16
```

**Primary metric:** mean test accuracy over the 3 seeds.

**Decision rule (frozen):**
- mean(AG) − mean(X) ≥ +0.015 → **POSITIVE** (H1 supported)
- mean(AG) − mean(X) ≤ −0.015 → **NEGATIVE** (learned τ hurts here)
- otherwise → **NULL** (no detectable effect at this power)

**Mechanism gate (required for any POSITIVE claim):** the learned τ must
actually have moved — in at least one layer of each AG member, at least 10%
of neurons end with |τ_m − 20| > 0.5 ms. If accuracy rises but τ barely
moved, the difference is noise, not the feature.

**Secondary comparison (reported, not decision-driving):** mean(AH) vs the
recorded Z-audit mean of the R recipe (0.850, seeds 42/43/44, serial
execution — execution-mode difference disclosed), same ±0.015 bands.

**Power caveat, stated up front:** the round-5 audit measured a single-seed
sd of ≈ 2.7 points; the SE of a difference of two 3-seed means is ≈ 2.2
points, so this experiment can only detect large effects. The ±1.5-point
band is a claim-throttle (it prevents declaring noise a win), not a
significance test. A NULL outcome is expected to be the most likely result
under small true effects and will be reported as such.

## 4 Predictions (falsifiable, before running)

- **P1:** mean(AG) > mean(X), by +1 to +3 points (the feature the published
  0.90+ systems credit).
- **P2:** the learned τ_m distribution spreads substantially from its
  uniform start (final range > 5 ms within a layer) — heterogeneous
  timescales are the point of the feature.
- **P3:** no AG/AH member collapses (< 0.80): learning τ under clamps is a
  refinement, not a destabilizer.

## 5 Disclosures

- Before freezing, machinery smoke runs of AH/R executed a few hundred
  minibatches each for wall-clock measurement (loss prints only; **no test
  accuracy was computed on any learnable-τ configuration** before this
  freeze).
- The X seed-0 result under threads-16 will differ slightly from the
  recorded serial 0.8857; the control is re-run rather than reused precisely
  so both arms share the execution mode.
- Amendment policy: tighten-only; deviations recorded in the results
  document.

## 6 Consequence

POSITIVE → learned τ becomes part of the default recipe; follow-ups
(ALIF + learned τ, learned θ) get their own protocols. NULL/NEGATIVE → the
result is recorded in `demo/RESULTS.md` and improvements.md P1 is re-ranked
(the trained temporal readout moves up).
