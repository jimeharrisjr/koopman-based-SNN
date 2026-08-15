# V2 Experiment Results — Phase 4 Go/No-Go

**Date run:** 2026-08-14 · **Protocol:** `docs/04-v2-preregistration.md` (frozen) ·
**Harness:** `crates/kdmd-snn/examples/v2_rollout.rs` · **Raw output:** `docs/05-v2-results-raw.txt`

## VERDICT: FAIL

Per the pre-registered decision rule (§5, row 3): at least one RS interpolation gate was
missed at **every** (degree ∈ {2, 3}, Δt* ∈ {0.5, 1, 2} ms) configuration. Per the plan's
Phase 4 exit criterion, **work is paused for an owner decision informed by a skeptic
re-review** (`docs/06-skeptic-v2-review.md`). Under the registered consequence, V2 becomes
a documented negative result and the library's headline value cases reduce to V1
(reduced-order recurrent layers), V3 (spectral diagnostics), and V4 (spectral
transparency) — plus the exact-linear LIF/adLIF engine, whose oracle-precision
identification already works (Phases 2–3).

## Protocol deviations (recorded per the amendment policy)

1. **Ground-truth h.** The registered h = 0.025 ms failed its own self-convergence check
   (max spike shift at I = 13 under halving: 1.48 ms for RS). Per the §1 contingency, h
   was halved per type until convergence: **RS 0.0015625 ms, FS 0.000195 ms, CH
   0.00039 ms**. Thresholds were not adjusted. (Fast-firing FS accumulates O(h) Euler
   phase drift over ~194 spikes, hence the extreme requirement.)
2. **C3 wall-clock** was measured as single-neuron rollout micro-timing, not the
   registered N = 1024 zero-alloc criterion bench (which belongs to Phase 5
   infrastructure). Reported for context only; no gate outcome depended on C3.
3. The registered 200 ms error-growth curves were not produced (the 1000 ms tables
   already localize the failure); available on request from the archived traces.

## Headline numbers

**Cost side — both registered cost predictions held:**
- **P2 CONFIRMED:** no Euler h on the registered grid {0.5, 0.25, 0.1, 0.05 ms} passes
  the accuracy gates for RS or FS — coarse Euler drifts out of the ±2 ms window over
  1000 ms exactly as predicted. The C2 frontier was therefore wide open (limit = ∞), and
  degree-2 surrogates passed C1 easily (45–180 flops/ms vs the 300 cap).
  **The surrogate did not fail on cost. It failed on its own accuracy.**

**Accuracy side (RS, degree 2 — the registered best hope):**

| Δt* | I | ref/sur spikes | count err | coinc ±2ms (gate ≥ 0.80) | first-spike err | v-RMSE (gate ≤ 1 mV) |
|---|---|---|---|---|---|---|
| 0.5 | 10 | 23 / 22 | 4.3 % ✓ | **0.087 ✗** | 0.55 ms ✓ | 12.3 ✗ |
| 1.0 | 10 | 23 / 21 | 8.7 % ✓ | **0.130 ✗** | 0.20 ms ✓ | 11.5 ✗ |
| 1.0 | 6 | 14 / 12 | ✗ (|ΔN| = 2 > 1) | **0.071 ✗** | 0.02 ms ✓ | — |
| 2.0 | 6 | 14 / 141 | 907 % ✗ | 0.43 ✗ | 4.6 ms ✗ | 80 ✗ |

The signature is unmistakable: **counts and first-spike latency are nearly right, but
coincidence collapses** — a per-ISI period error of ~4–9 % that accumulates as phase
drift and blows through the ±2 ms window after the first couple of spikes. This is
precisely the cumulative-drift failure mode the pre-registration's 1000 ms horizon was
chosen to expose (§2), and it did.

- **Sub-rheobase I = 2 PASSED for nearly every degree-2 configuration** (no spurious
  spikes; v-RMSE 0.28–0.44 mV ≤ 0.5). The dictionary is fine on the slow manifold; the
  failure localizes to the upstroke/reset re-entry region — registered threat 1.
- **Degree 3 diverged at rollout** (NaN within tens of steps in most configurations;
  fitted ρ(K) up to 350) despite *better* one-step holdout residuals than degree 2 — the
  classic EDMD overfit-to-instability trade. P1's failure clause applies: dictionary
  work precedes any lifted path.
- **Fit preconditions failed universally**: held-out one-step residual 4.5e-3 – 4.3e-2
  (gate ≤ 1e-3) and ρ(K) 5.2 – 350 (gate < 1 + 1e-6). Note for any follow-up: the ρ
  precondition is arguably mis-calibrated for a flow that is genuinely locally expanding
  on the upstroke — but the tighten-only amendment policy means it stands as registered.
- **CH (reported, ungated):** degree 3 at Δt = 0.5, I = 6 reached coincidence 0.755 and
  v-RMSE 1.66 mV — the closest any configuration came to a gate; bursting CH was
  *predicted* (P3) to be the worst case, and was not. This inversion is worth
  understanding before any rescue attempt.

## Predictions scorecard

| # | Prediction | Outcome |
|---|---|---|
| P1 | degree-2 passes RS interpolation at Δt = 1 | **REFUTED** — coincidence 0.07–0.13 vs gate 0.80 |
| P2 | coarse Euler fails ±2 ms over 1000 ms | **CONFIRMED** — frontier wide open |
| P3 | accuracy degrades with firing rate; CH worst | **PARTIALLY REFUTED** — FS (194 Hz) did degrade, but CH outperformed RS at matched configs |
| P4 | no per-substep flop victory possible | CONFIRMED by construction (arithmetic) |

## What survives, unconditionally

Everything Phases 0–3 delivered is untouched by this result: the exact-linear LIF/adLIF
engine with closed-form propagators, oracle-precision DMD/DMDc identification (≤ 1e-8),
the subtractive-reset-as-control formulation validated on full spiking trajectories, the
koopman-dmd 0.2 `dmdc` branch, and the V1/V3/V4 value cases the reviews rated
independently defensible from the start.

## Options for the owner decision

1. **Pivot per the registered consequence** (the pre-registration's own recommendation):
   proceed to Phase 5 with `PerVariable`/`PerNeuron` exact-linear operators (LIF/adLIF),
   keep V1 (reduced-order recurrent) as the fitted-operator value case, document V2 as a
   negative result. Lowest risk; the plan's remaining phases work without V2.
2. **One bounded rescue attempt** before pivoting: the failure analysis points at
   specific, pre-registered follow-ups (foundations §5-Q2) — a dictionary containing
   upstroke-adapted observables (exponential/softplus of v, delay embedding), and/or
   fitting the ISI return map directly instead of the per-step flow. Any such attempt is
   a NEW experiment requiring a new pre-registration; the current thresholds stand.
3. **Drop the nonlinear-surrogate track entirely** and re-scope Phase 4's remaining time
   into Phase 5/6 depth.

The skeptic re-review (docs/06) audits whether any part of this failure is an artifact
of the harness or fitting pipeline rather than the method — read it before deciding.
