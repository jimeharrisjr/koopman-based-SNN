# P3 Science Studies — Pre-registration

**Document:** 25-p3-science-preregistration.md · **Status: FROZEN before any
study runs** · **Date frozen:** 2026-08-28
**Origin:** improvements.md P3 items 1–3. These are *measurement studies*
(maps and laws, not accept/reject hypotheses), so the registration freezes
setups and falsifiable predictions rather than decision rules. Committed
before results, per the P0.1 discipline. Item 4 (nonlinear surrogates)
remains closed by the docs/08 standing decision; item 5 (operator staleness)
remains moot while training uses closed-form operators only.

---

## S-A: Probe richness for identification (P3.3)

**Question.** The paper's §3.2 showed excitation dominates identifiability
(constant drive → cond(X) ~ 1e19, unrecoverable). How much probe richness is
*enough* — specifically, how does known-B recovery of a dense 2N×2N `A`
degrade as the number of independently driven input channels `m` falls below
the layer width `N`?

**Setup (frozen).** numpy replication of the paper's identification
experiment (`analysis/probe_richness.py`): LIF layer, exact propagator
(τ = 20/10 ms, h = 1 ms), N ∈ {8, 32}, W ~ N(0, 0.35) with exactly `m`
active input channels, independent Poisson probes at rate 0.15/channel,
T = 4000 snapshots, state noise σ = 1e-4, known-B least squares. Sweep
m ∈ {1, 2, 4, …, N}. Metrics: relative Frobenius error of Â, cond(X), and
the error restricted to the excited subspace (top-2m left singular vectors
of X).

**Predictions (frozen).**
- **PA1:** full-A error > 0.1 whenever m < N/2 (the synaptic-current block
  is only excited on col-span(W), rank m).
- **PA2:** at m = N the error returns to ~1e-3–1e-4 (the paper's
  well-probed regime).
- **PA3:** the excited-subspace error stays ≤ 1e-2 even at small m —
  identification fails *globally*, not locally: what the probe reaches is
  learned well.
- **PA4:** reset feedback partially rescues the *voltage* block: v-row
  errors < i-row errors at small m.

---

## S-B: The V4 gradient-horizon measurement (P3.2)

**Question.** The paper's §4.5 corrected the premise: the spectrum
*diagnoses* gradient decay (‖∂L/∂x_{t+Δ}/∂x_t‖ ~ ρ(A)^Δ), it does not
prevent it, and surrogate factors add further attenuation. Measure this law
directly in the real trainer.

**Setup (frozen).** New instrumentation: `TrainConfig::record_grad_norms`
makes the backward pass record, per time step, the batch-mean L2 norm of
λ = ∂L/∂x_t for each layer (exposed via `Trainer::grad_norms`). Harness:
`examples/grad_horizon.rs` — one recurrent LIF layer, N = 64, synthetic
Poisson-pattern task, T = 60 steps, dt = 1 ms; for each
τ_m ∈ {10, 20, 40, 80} ms (τ_s = τ_m/2): run 30 train steps, then measure
the decay rate of ‖λ(t)‖ vs (T − t) by least-squares slope of log‖λ‖,
reported alongside the spectral prediction α = exp(−dt/τ_m).

**Predictions (frozen).**
- **PB1:** measured per-step decay factor ≤ α (the linear part sets the
  ceiling; surrogate factors σ′ ≤ 1 and spike-path fan-in only attenuate
  further or add noise, they cannot systematically amplify a leaky net
  trained to moderate activity).
- **PB2:** measured decay factor correlates monotonically with τ_m: the
  τ = 80 ms horizon (steps to decay by ×100) is ≥ 3× the τ = 10 ms one.
- **PB3:** the measured factor is within [0.5·α, α] for all τ — i.e., the
  spectral prediction is the right order, surrogate attenuation less than
  another factor-of-two per step at this activity level.

PB1–PB3 jointly convert §4.5 into a measured statement; PB3 failing low
(decay ≪ α) would mean surrogate attenuation dominates even at 60 steps —
also informative.

---

## S-C: Spiking-regime reduced-order models — a feasibility map (P3.1)

**Question.** V1 validated rank-r reduction sub-threshold only. Does a
POD-projected reduced model reproduce *spiking* trajectories, and where
does it break? V2's lesson (per-cycle error compounds into phase drift) is
the registered pessimist.

**Setup (frozen).** `analysis/spiking_rom.py`: recurrent LIF layer,
N = 64, exact propagator (τ = 20/10 ms, h = 1 ms), random W_rec
(N(0, 0.4/√N)), Poisson input at three drive levels giving low/medium/high
firing (targets ≈ 2%, 8%, 20% spikes/neuron/step, reported as measured).
Ground truth: full 2N spiking simulation, T = 1000. ROM: POD basis P_r
from ground-truth snapshots (rank r ∈ {8, 16, 32, 64, 128 = full}); reduced
step z ← (PᵀAP) z + Pᵀ B u with spikes computed from the lifted voltages
v = (P z)_v and fed back (reset + recurrence) exactly as in the full model.
Closed-loop rollout from the same initial state and input realization.
Metric: spike-train coincidence (±2 bins) against ground truth, plus
per-neuron rate error.

**Predictions (frozen).**
- **PC1:** r = 2N reproduces ground truth to machine precision (sanity
  gate — if this fails the harness is wrong, not the science).
- **PC2:** at fixed r < 2N, coincidence degrades monotonically with firing
  rate (docs/01 §5-Q7's prediction, never yet measured).
- **PC3 (the V2-informed pessimist):** at r ≤ N, coincidence < 0.8 at the
  medium drive — truncation error at spike times compounds through the
  reset feedback the way V2's per-step surrogate error did. A refutation
  (usable coincidence at r ≤ N) would be a genuinely publishable positive.
- **PC4:** rate error is far more forgiving than coincidence: ≤ 10% mean
  absolute rate error at r = N even where coincidence fails — reduced
  models lose *timing* before they lose *statistics*.

---

## Disclosures & execution

- None of the three studies has been run in any form before this freeze;
  S-A reuses the paper's fig-4 machinery (Poisson-probed regime) with new
  sweeps.
- All numeric setups above are frozen; deviations get recorded in the
  results document (docs/26). Tighten-only amendments.
- Compute: S-A and S-C are numpy, minutes; S-B is a new Rust example on
  the synthetic task, minutes. Seeds fixed in-script.
