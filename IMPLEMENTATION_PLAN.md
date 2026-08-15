# kdmd-SNN — Phased Implementation Plan

**Status:** COMPLETE 2026-08-15 — Phases 0–6 delivered. V2 (lifted EDMD surrogates) and V2b (return-map rescue) **FAILED** their pre-registered gates and stand as documented negative results (docs/05, docs/08, audited in docs/06); the exact-linear engine, DMD/DMDc identification, V1 reduced-order layers, V3 spectral diagnostics, surrogate-gradient training, and the SHD demo shipped and passed the Phase 5 benchmark gate (docs/09). **Remaining loose ends (owner):** the `koopman-dmd` `dmdc` branch is local-only (unpushed) and 0.2.0 is unpublished, so fresh clones build only via the sibling-checkout `[patch.crates-io]` (see README "Building"), and CI — which strips the patch — cannot resolve `koopman-dmd = "0.2"` until release. *(Originally approved 2026-08-14 — Q1–Q8 answered inline (bottom); key decisions: DMDc on a `rust-dmd` branch → PR → v0.2.0 (Q1); nonlinear-first phase ordering (Q2); N-MNIST/SHD demo (Q3); 10k routine / 100k supported, no checkpointing in v1 (Q4); subtractive-only fast path (Q5); hand-rolled BPTT (Q6); kill criteria accepted (Q8).)*
**Inputs:** `SNN-project.md` (premise) · `docs/01-scientific-foundations.md` (scientist) · `docs/02-architecture.md` (architect) · `docs/03-skeptic-review.md` (skeptic)

---

## Overview

Build `kdmd-snn`: a Rust library for Spiking Neural Networks whose sub-threshold dynamics advance through a linear-plus-control step `x_{t+1} = A·x_t + B·u_t`, with the threshold `Θ` as the single isolated nonlinearity, identification of `A` via DMD/DMDc from the `koopman-dmd` crate, and training via surrogate-gradient BPTT.

**Architecture**: Cargo workspace, one library crate `crates/kdmd-snn`, depending on `koopman-dmd` (faer 0.22 pinned, types cross the boundary). A DMDc solver is added to `koopman-dmd` on a `dmdc` branch of `rust-dmd`, PR'd and merged as v0.2.0 once tested (Q1 — decided).

### The honest value proposition (re-scoped from the premise)

All three review documents converge on the same finding, and the plan adopts it:

**LIF sub-threshold dynamics are already exactly linear.** The discrete propagator is known in closed form (`α = exp(−dt/τ_m)`, diagonal per neuron, O(N) per step). DMD applied to plain LIF can only re-estimate known constants, noisily, at O(N²) cost. Therefore plain LIF is the **validation oracle and performance baseline**, not the product. The Koopman-DMD machinery earns its place in four re-scoped claims, each with a falsifiable test:

| # | Claim | Where DMD wins | Test that would falsify it |
|---|-------|----------------|---------------------------|
| V1 | **Reduced-order recurrent layers** | Rank-r surrogate of an N-neuron recurrent layer: O(Nr) vs O(N²) per step | `inference_e2e` bench vs dense recurrent baseline at matched accuracy |
| V2 | **Lifted surrogates for nonlinear neurons** | EDMD(-c) linear stepping for AdEx/Izhikevich/conductance models with no closed-form integrator | Rollout accuracy-vs-speed against RK4 reference sim |
| V3 | **Spectral diagnostics** | Post-fit eigenvalues give per-mode timescales, stability margins, gradient-decay depth `T_half = ln2/|ln|λ||` | Cheap; built on existing `dmd_spectrum`/`dmd_stability` |
| V4 | **Spectral transparency for training** (restated gradient claim) | Gradient decay through the linear part is `~ρ(A)^T` — *diagnosable and regularizable*, not prevented | Log ‖∂L/∂x_t‖ vs t against a plain-LIF surrogate-gradient baseline; premise predicts a difference, skeptic predicts none |

The premise's original claims are corrected as follows (tracked in `docs/03-skeptic-review.md` checklist):
- "Prevents exploding/vanishing gradients" → **restated** as spectral transparency/regularization (V4). Gradients through a leaky `A` (ρ<1) still vanish geometrically; what changes is that the spectrum is *known* post-fit.
- "Solve for both A and B" → **B is never estimated in the training path.** `B = [E_i·W, −θ·E_v]` is known by construction (W is the learned weights; the subtractive-reset coefficient is −θ). DMDc's estimate of B is a validation check. This also sidesteps the closed-loop identification bias (spikes are state feedback, not exogenous control).
- Hard reset (`v ← V_reset`) is **bilinear**, not representable as `A x + B u` → **subtractive (soft) reset is canonical** (`v ← v − θ·s`), making the step *exactly* linear-plus-control. Hard reset is supported in the reference simulator with masked identification (Q5).
- The premise's `Ã = UᵀYVΣ⁻¹` is the r×r *reduced* operator, not the full A — the library keeps the two distinct (`Operator::LowRank {P, A_r}` vs `Operator::Dense`).

### Novelty position

The scientist's survey (Aug 2026) found **no published work using Koopman/DMD to model SNN layer dynamics as a linear-plus-control system for inference or surrogate-gradient training** — the combination is open territory. The nearest occupied design point is the 2023–2026 spiking state-space-model literature (PSN, SpikingSSMs, SPikE-SSM, SiLIF), which exploits LIF linearity for parallel training without Koopman; novelty claims must rest on V1–V4, not on "linearizing LIF."

---

## Project Structure

```
kdmd-SNN/
├── Cargo.toml                  # [workspace] members = ["crates/kdmd-snn"]
│                               # [patch.crates-io] koopman-dmd → ../rust-dmd (dev only)
├── IMPLEMENTATION_PLAN.md      # this file
├── docs/
│   ├── 01-scientific-foundations.md
│   ├── 02-architecture.md
│   └── 03-skeptic-review.md
└── crates/
    └── kdmd-snn/
        ├── Cargo.toml
        ├── src/
        │   ├── lib.rs
        │   ├── error.rs        # SnnError (thiserror), wraps DmdError via #[from]
        │   ├── neuron/         # ground-truth reference simulators
        │   │   ├── lif.rs      #   LIF + exp synapse (closed-form A_local = test oracle)
        │   │   ├── adlif.rs    #   adaptive LIF (3-state)
        │   │   └── izhikevich.rs # nonlinear ground truth (EDMD target, Phases 1+4)
        │   ├── state.rs        # LayerState: variable-major (k·N)×batch faer matrix
        │   ├── operator.rs     # Operator enum {Dense, PerVariable, PerNeuron, LowRank}
        │   ├── spikes.rs       # SpikeVec (sparse idx, inference), SpikeBatch (dense, training)
        │   ├── layer.rs        # KoopmanLayer: step() / step_batch()
        │   ├── network.rs      # Network + Readout
        │   ├── identify/       # snapshots.rs, dmdc.rs (fallback), validate.rs
        │   ├── train/          # surrogate.rs, tape.rs, loss.rs, optim.rs
        │   ├── encoding.rs     # Poisson, latency, current-injection encoders
        │   ├── data.rs         # synthetic tasks; N-MNIST/SHD behind `datasets` feature
        │   └── metrics.rs      # firing rates, trajectory RMSE, spike coincidence, accuracy
        ├── benches/            # criterion: step_kernels, spike_drive, inference_e2e,
        │                       #            train_epoch, identify
        ├── examples/           # lif_identify.rs, synthetic_task.rs, shd.rs (feature-gated)
        └── tests/              # oracle test, sim-equivalence, gradient checks
```

Full data-structure and API sketches (all faer-0.22-idiomatic) are in `docs/02-architecture.md` §2–§5.

## Dependency Stack

| Crate | Purpose |
|-------|---------|
| `koopman-dmd` | DMD/EDMD identification, lifting, spectrum/stability/residual analysis (0.1 → 0.2 via the `dmdc` branch, Q1) |
| `faer` = 0.22 | Matrices; **must track koopman-dmd's minor version exactly** (types cross the boundary) |
| `rayon` | Batch-shard parallel training; large-N matmuls |
| `thiserror` | Error types |
| `rand`, `rand_distr` | Encoders, weight init, probe inputs |
| `criterion`, `approx` (dev) | Benchmarks, tolerance assertions |

Reused `koopman-dmd` API (confirmed public in 0.1.0): `dmd`, `lift_data`/`LiftingConfig`, `determine_rank`, `pinv`, `validate_matrix`, `dmd_stability`, `dmd_spectrum`, `dmd_residual`, `dmd_error`, `dmd_convergence`, `dmd_dominant_modes`, `predict_matrix`/`predict_modes` (tests), `hankel_dmd` (partial-state option).

Documented traps: never use `DmdConfig { center: true }` for identification (bakes the regime mean into the model); `dmd()` needs one contiguous trajectory (the `SnapshotSet` builder handles concatenation/masking and feeds explicit pair matrices to `dmdc`); `DmdResult.a_matrix: Vec<Vec<C64>>` never touches the hot path — `operator.rs` extracts real `faer::Mat<f64>` once at build time.

---

## Core design decisions (from the architecture review)

1. **Canonical step** (subtractive reset folded into control — *exactly* `x_{t+1} = A x_t + B u_t`):
   ```
   1. drive:      d = W · s_in              (sparse column-accum or dense matmul)
   2. advance:    y = A · x_t + E_i · d     (Operator::apply, zero-alloc)
   3. threshold:  s = Θ(v(y) − θ)           (the ONLY nonlinearity)
   4. reset:      x_{t+1} = y − θ ⊙ E_v · s
   ```
2. **Operator enum**: `PerVariable` (A_local ⊗ I_N — exact for homogeneous LIF, O(k²N)); `PerNeuron` (heterogeneous τ); `LowRank {P, A_r}` (reduced-order, O(Nr)); `Dense` (general). Real `f64` storage only on the hot path.
3. **Identified vs learned**: `A` identified (or closed-form for LIF); `W` and readout `R` learned by surrogate BPTT; `B` structural, never fitted; `θ` fixed in v1.
4. **Identification hygiene** (skeptic checklist): fit on continuous observables (potentials/currents, never binary spikes); known-B pinning or input subtraction before any fit; excise reset-straddling snapshot pairs in hard-reset mode; explicit rank policy with held-out `dmd_residual` validation; reject fits with ρ(A) ≥ 1+tol via `dmd_stability`; cross-input-distribution generalization test.
5. **No refits inside the training loop.** Plain LIF: `ReidentifyPolicy::Never` (A is exactly input- and weight-independent). Lifted/low-rank operators: residual-triggered alternating optimization, with eigenvalue-drift monitoring. Network-level DMD only on frozen post-training weights.
6. **Training**: hand-rolled BPTT tape (the per-step graph is fixed: linear → Θ → linear; backward is ~40 lines of transposed matmuls) — no autograd dependency (Q6). Surrogates: FastSigmoid/SuperSpike, Atan, Boxcar.
7. **Performance rules**: zero allocation per step; `faer::linalg::matmul` into preallocated scratch (never `*` on hot paths); batch-as-columns (BLAS-3); rayon across batch shards; sparse spike path below ~25% activity.

---

## Phases

Each phase ends green (`cargo fmt`, `clippy`, `test`) and demo-able. Exit criteria are pre-registered; the subagent team reviews at phase boundaries (see Team section).

**Phase 0 — Skeleton.** Workspace + crate + CI wiring, `error.rs`, `state.rs`, `spikes.rs`, `[patch.crates-io]` to local rust-dmd.
*Exit:* `cargo test` green; `LayerState`/`SpikeVec` unit-tested.

**Phase 1 — Ground-truth simulators (LIF + nonlinear, per Q2).** `neuron::lif` reference simulator (both reset modes) with closed-form discrete `A_local`; `neuron::izhikevich` and `neuron::adlif` reference simulators (RK4 / exact-exponential steppers); encoders, one synthetic task, metrics.
*Exit:* LIF f–I curve and membrane-decay tests against analytic solutions pass; Izhikevich reproduces canonical firing patterns (tonic, bursting, chattering per the 2003 paper's parameter table).

**Phase 2 — DMD identification (autonomous).** `SnapshotSet`, `operator.rs` extraction bridges, fit via `koopman_dmd::dmd` on sub-threshold trajectories, validation gates.
*Exit:* **oracle test** — identified A matches closed-form LIF A to ≤ 1e-8 relative error; `dmd_spectrum` recovers τ_m, τ_s.

**Phase 3 — DMDc (on the `dmdc` branch of rust-dmd, per Q1).** `dmdc()` with `known_b` pinning developed on a `dmdc` branch of `rust-dmd`, together with the two upstream bug fixes (see work items below); PR'd and merged as v0.2.0 when tested. `kdmd-snn` consumes the branch via `[patch.crates-io]`. `fit_layer_operator` end-to-end with `u = [s_in; s_self]`, reset-column consistency check (identified reset columns ≈ −θ·e_v).
*Exit:* A+B identified from spiking trajectories (R1, no masking) match structure to tolerance; masked-R2 path tested; the branch is green under rust-dmd's own test suite.

**Phase 4 — Nonlinear lifted surrogates (V2 — the scientific go/no-go, per Q2).** Dictionary design for Izhikevich/AdEx (monomials in (V, u); the AdEx exponential term as an observable; delay embeddings), per-neuron EDMD(-c) surrogates, hybrid stepping (lifted linear advance sub-threshold; explicit threshold/reset in original coordinates; lifted-state re-initialization after spikes), rollout accuracy-vs-speed vs the reference simulators.
*Exit (numbers pre-registered with the scientist agent at phase start):* surrogate rollout matches reference spike times/state trajectories to a stated horizon and tolerance at a stated speedup; results published in `docs/`. If the test fails, work pauses for an owner decision informed by a skeptic re-review (the V2 analog of the kill gate).

> **Phase 4 outcome (2026-08-14):** the pre-registered V2 experiment **FAILED** — spike counts and first-spike latency were close, but per-ISI timing bias accumulated as phase drift (coincidence at the de-phased chance floor); skeptic audit (docs/06) classified the failure REAL and STRUCTURAL (one-step EDMD cannot deliver ±2 ms phase over 1000 ms; zero training data on the masked upstroke). Cost predictions held (coarse Euler fails accuracy; frontier open). **Owner decision:** one bounded rescue — a new pre-registered experiment (docs/07) fitting the **ISI/Poincaré return-map** surrogate (predict spike-to-spike, so phase error cannot accumulate per step); if it fails, pivot to V1/V3/V4 and proceed to Phase 5 without further pause.

**Phase 5 — Fast inference network.** `KoopmanLayer`/`Network`, PerVariable (LIF) + PerNeuron (lifted) + Dense kernels, sparse spike path, scratch buffers; `inference_e2e` + `spike_drive` benches.
*Exit:* bit-comparable spikes vs reference LIF sim over T=1000; measured speedup reported. **Benchmark gate:** the fitted-operator path must be within ~2× of the plain-LIF baseline per step at equal accuracy, or it is demoted (kill criterion below — accepted, Q8).

**Phase 6 — Training + reduced order + demo + polish.** Surrogate kernels, tape, backward pass, SGD/Adam, readout, shard-parallel batching (hand-rolled per Q6); `LowRank` path with `resolve_rank` and basis augmentation (reset-projection residual reported); `ResidualTriggered` re-identification for lifted operators; N-MNIST/SHD demo behind a `datasets` feature (Q3); full bench suite, docs pass, README doctests.
*Exit:* training reproducibly reaches target accuracy on the synthetic task and the N-MNIST/SHD demo runs; gradient check vs finite differences on a tiny network; V4 experiment (gradient-norm curves vs plain-LIF baseline) logged; documentarian sign-off.

### Pre-registered evaluation & kill criteria (skeptic M4)

- **Tasks:** one nonlinear-neuron accuracy/speed rollout task (V2, Phase 4); synthetic Poisson-pattern classification + temporal XOR (Phase 6); N-MNIST/SHD classification demo (Phase 6, per Q3).
- **Baselines:** in-house plain-LIF Rust sim (exact, O(N)); snnTorch LIF on CPU for external reference.
- **Metrics:** accuracy, wall-clock/step, held-out one-step fit residual, gradient-norm decay curves.
- **Kill criterion:** if fitted-operator inference loses to plain LIF at equal accuracy (feedforward homogeneous case), that path is demoted to the documented negative result, and the library's product becomes: exact-linear fast LIF engine + nonlinear-neuron surrogates (V2) + reduced-order recurrent layers (V1) + spectral analysis (V3) — which is where all three review docs expect the defensible value to live anyway.

### Upstream (koopman-dmd) work items surfaced by this project

Per Q1, all of these land on the `dmdc` branch of `rust-dmd`, PR'd and merged as v0.2.0 after testing:

1. `dmdc()` + real-`Mat<f64>` result type; generalize `dmd_stability`/`dmd_spectrum` over an eigenvalue slice (per Q1).
2. Bug: `compute_full_a`'s pinv fallback normalizes every column by ‖mode₀‖² — should be an error, not a fallback (skeptic m4).
3. Dead `if config.center` branch around `x0` in `dmd.rs`; centered-prediction round-trip deserves a test.

---

## The subagent team across phases

| Agent | Phase 0–1 | Phase 2–3 | Phase 4–5 | Phase 6 |
|---|---|---|---|---|
| **Scientist** | done (docs/01) | consulted on dictionary design (V2) | pre-registers Phase 4 V2 exit numbers; reviews rollout results | reviews V4 experiment + claims in final docs |
| **Architect** | done (docs/02) | reviews DMDc API before merge | reviews perf results vs design | triages test-agent findings |
| **Main context (me)** | writes all code | writes all code | writes all code | writes all code |
| **Code quality** | — | review after each phase: docs, tests, clippy-level idiom, API hygiene | same | final pass |
| **Test agents** | — | property/oracle tests review | equivalence + gradient-check adversarial testing; issues documented for architect | full-suite audit |
| **Skeptic** | done (docs/03) | — | mid-project re-review at end of Phase 5 (checklist audit) | final re-review before any publish |
| **Documentarian** | — | — | — | README, rustdoc coverage, examples, usable-by-others audit |

---

## Questions for you (answer inline)

> **Q1 — Where does DMDc live?** Recommended: add `dmdc()` to your `koopman-dmd` crate as v0.2.0 (real `Mat<f64>` A/B outputs, explicit `(X, X', U)` snapshot-pair inputs, `known_b` option), since DMDc is squarely in that crate's charter and fixes the `Vec<Vec<C64>>` hot-path problem at the source. I'd develop it against your local checkout via `[patch.crates-io]`, and you publish 0.2.0 when ready. Alternative: keep koopman-dmd frozen and host `dmdc` inside `kdmd-snn::identify` (designed as a drop-in fallback either way). Also: OK to fix the two upstream bugs listed above while in there?
>
> **A:** Add to the koopman-dmd crate as a separate branch which we can PR and merge as 0.2.0 when done testing

> **Q2 — Priority of the nonlinear/lifted regime.** The re-scoped value cases are V1 (reduced-order recurrent), V2 (nonlinear neurons via EDMD), V3/V4 (diagnostics/spectral training). The plan schedules LIF+training first (Phases 1–5) and V1/V2 in Phase 6. If nonlinear neuron surrogates (V2) are the point of the project for you, I'd pull Izhikevich/AdEx forward to Phase 4–5 and push the training stack later. Which ordering do you want?
>
> **A:** "Nonlinear first"

> **Q3 — Demo task & dataset.** Options: (a) fully synthetic (zero external deps, weaker headline), (b) spiking/rate-encoded MNIST (~11 MB download, needs a loader dep or vendoring), (c) N-MNIST/SHD (event-based, more credible for SNN claims, bigger data). Plan assumes synthetic in Phase 5 + MNIST behind a cargo feature in Phase 6. OK?
>
> **A:** N-MNIST/SHD

> **Q4 — Target scale.** What should we optimize for: ~1k neurons/layer (laptop realtime), ~100k, or larger? This sets how much the LowRank path and tape checkpointing matter in v1.
>
> **A:** 10k routine / 100k supported, laptop, no checkpointing in v1

> **Q5 — Reset convention.** Canonical = subtractive reset (exactly linear, unbiased identification, clean gradients); hard reset supported in the reference simulator with masked identification and a documented accuracy caveat. Do you need hard-reset parity in the *fast* path too?
>
> **A:** Subtractive only in the fast path; hard reset in the reference simulator is enough.

> **Q6 — Training stack.** Hand-rolled BPTT tape, no burn/candle dependency (the step graph is fixed and small). Comfortable, or do you want optional autograd-framework integration on the roadmap?
>
> **A:** Hand-rolled is fine; revisit only if GPU training becomes a goal.

> **Q7 — Assumptions to confirm (one-word answers fine).** (a) f64 everywhere in v1, f32 fast path later if benches justify; (b) no `no_std`/embedded target; (c) no GPU in scope (Operator enum keeps a seam for a later backend); (d) `serde` save/load as an optional feature, format TBD; (e) MIT license, MSRV 1.85, matching koopman-dmd.
>
> **A:** Confirming all your assumptions

> **Q8 — Kill criteria sign-off.** Do you accept the pre-registered benchmark gate and kill criterion above (fitted-operator inference must be within ~2× of plain LIF at equal accuracy or it's demoted, pivoting the library to V1/V2/V3)? This is the skeptic's main structural demand, and I think it's right.
>
> **A:** Yes
