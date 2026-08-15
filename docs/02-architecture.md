# kdmd-SNN — System Architecture

**Status:** Draft for review · **Author:** System Architect subagent · **Date:** 2026-08-14
**Depends on:** `koopman-dmd` v0.1.0 (local: `/Users/jimharris/Documents/rust-dmd/koopman-dmd`, crates.io: `koopman-dmd`)
**Companion docs:** `01-foundations.md` (scientist subagent, mathematical basis), `SNN-project.md` (premise)

This document translates the Koopman/DMD SNN premise into a concrete Rust system design:
crate layout, data structures, the linear-advance + nonlinear-threshold split, DMDc
identification, surrogate-gradient training, the reduced-order fast path, and a phased
delivery plan. API sketches target **faer 0.22** (the version `koopman-dmd` pins) and
Rust edition 2021, MSRV 1.85 (inherited from `koopman-dmd`'s faer dependency).

---

## 0. Executive summary of decisions

| Decision | Choice | Rationale |
|---|---|---|
| Repo layout | Cargo **virtual workspace**, one library member `kdmd-snn` to start | Cheap now; leaves room for `xtask`, dataset crates, CLI later |
| DMDc location | **Extend `koopman-dmd` with a `dmdc` module** (v0.2.0); mirror implementation in `kdmd-snn::identify` as fallback if crate changes are ruled out | DMDc is a textbook DMD variant (Proctor, Brunton, Kutz 2016) squarely inside that crate's charter; its result type should return real `faer::Mat<f64>` directly, fixing the `Vec<Vec<C64>>` hot-path problem at the source |
| Operator storage | `Operator` enum: `Dense(Mat<f64>)`, `PerVariable` (k×k local matrix ⊗ I_N — covers LIF exactly), `LowRank {P, A_r}` | LIF sub-threshold dynamics are block-structured; O(kN) or O(Nr) step instead of O(N²) |
| Spike representation | Sparse `Vec<u32>` indices on the inference path; dense `Mat<f64>` (neurons × batch) on the training path | Column-accumulation against column-major `W` is contiguous and beats dense matvec below ~25% activity; surrogate BPTT needs dense reals anyway |
| Reset convention | **Subtractive reset folded into the control input** (`v ← v − θ·s`), i.e. `x_{t+1} = A x_t + B u_t` exactly; hard reset supported in the reference simulator only, with masked identification | Keeps the step *exactly* linear-plus-control, so DMDc identification is unbiased and BPTT Jacobians are clean |
| Training | Hand-rolled BPTT tape with surrogate gradients (no autograd dependency); `W`, readout `R`, optionally `θ` learned; `A` (and reset/injection structure of `B`) identified | The per-step graph is fixed (linear + threshold); backward pass is ~40 lines of matmuls |
| Re-identification | Event-triggered by `dmd_residual` relative error on fresh trajectories, inside an alternating optimization loop; **plain LIF never needs re-identification** (its A is exactly input-independent) | See §4.4 |
| Hot path | faer matmuls into preallocated buffers, batch-as-columns for BLAS-3, rayon across batch shards; criterion benches from Phase 1 | See §6 |

---

## 1. Crate layout

### 1.1 Workspace vs single crate

**Recommendation: virtual workspace with one member.** The marginal cost is one extra
`Cargo.toml`, and it gives us a natural home for future members (`kdmd-snn-datasets`
for MNIST loading behind a feature-free boundary, an `examples`/`xtask` crate, a CLI)
without a later restructuring commit.

```
kdmd-SNN/
├── Cargo.toml              # [workspace] members = ["crates/kdmd-snn"]
├── docs/
│   ├── 01-foundations.md
│   ├── 02-architecture.md  # this file
│   └── ...
└── crates/
    └── kdmd-snn/
        ├── Cargo.toml
        ├── src/
        │   ├── lib.rs
        │   ├── error.rs        # SnnError (thiserror), wraps koopman_dmd::DmdError via #[from]
        │   ├── neuron/         # neuron model definitions + reference (ground-truth) simulators
        │   │   ├── mod.rs      #   trait NeuronModel, ResetMode
        │   │   ├── lif.rs      #   LIF with exponential synapse (has CLOSED-FORM discrete A — test oracle)
        │   │   ├── adlif.rs    #   adaptive LIF (3-state, still linear sub-threshold)
        │   │   └── izhikevich.rs # nonlinear ground truth → exercises EDMD lifting (later phase)
        │   ├── state.rs        # LayerState: variable-major state block (potentials, currents, adaptation)
        │   ├── operator.rs     # Operator enum + apply kernels; extraction from DmdResult/DmdcResult
        │   ├── spikes.rs       # SpikeVec (sparse), SpikeBatch (dense), conversions
        │   ├── layer.rs        # KoopmanLayer: A, W, reset, threshold; step() / step_batch()
        │   ├── network.rs      # Network: layers + Readout; run(), run_batch()
        │   ├── identify/       # DMD/DMDc identification pipeline
        │   │   ├── mod.rs      #   fit_layer_operator(), IdentifyConfig, RankPolicy
        │   │   ├── snapshots.rs#   SnapshotSet: collect (x_t, x_{t+1}, u_t) triples across trajectories, masking
        │   │   ├── dmdc.rs     #   fallback dmdc() impl (compiled out if koopman-dmd >= 0.2 provides it)
        │   │   └── validate.rs #   stability gate, residual checks, spectrum report
        │   ├── train/          # surrogate-gradient BPTT
        │   │   ├── mod.rs      #   Trainer, TrainConfig
        │   │   ├── surrogate.rs#   SurrogateKind (FastSigmoid, Atan, SuperSpike) + derivative kernels
        │   │   ├── tape.rs     #   per-step saved tensors (v_pre, spikes), backward pass
        │   │   ├── loss.rs     #   CrossEntropySpikeCount, VanRossum (later)
        │   │   └── optim.rs    #   Sgd, Adam (plain, no deps)
        │   ├── encoding.rs     # PoissonRate, Latency, CurrentInjection encoders
        │   ├── data.rs         # synthetic tasks; `mnist` behind a cargo feature
        │   └── metrics.rs      # firing rates, spike-train distances, accuracy, state-trajectory RMSE
        ├── benches/            # criterion: step kernels, end-to-end inference, ref-sim vs koopman path
        ├── examples/           # lif_identify.rs, synthetic_task.rs, mnist.rs (feature-gated)
        └── tests/              # integration: identified-A vs closed-form LIF A, sim equivalence
```

`lib.rs` re-exports the user-facing surface: `KoopmanLayer`, `Network`, `NeuronModel`,
`LifParams`, `identify::fit_layer_operator`, `Trainer`, `SurrogateKind`, `SnnError`.

### 1.2 Dependency policy

```toml
# crates/kdmd-snn/Cargo.toml
[dependencies]
koopman-dmd = "0.1"          # bump to "0.2" once dmdc lands there
faer = "0.22"                # MUST track koopman-dmd's faer minor version exactly:
                             # faer::Mat<f64> crosses the API boundary; two semver-distinct
                             # faer versions would produce incompatible types.
rayon = "1.10"
thiserror = "2"
rand = "0.9"                 # encoders, weight init, ground-truth input generation
rand_distr = "0.5"           # Poisson, Normal

[dev-dependencies]
criterion = "0.5"
approx = "0.5"

[features]
default = []
mnist = []                   # pulls a loader only when the demo needs it (dep TBD, see §8)
serde = ["dep:serde"]        # model save/load, optional
```

Development against the local `koopman-dmd` checkout uses a workspace-level patch, so
the manifest stays publishable while iterating on `dmdc`:

```toml
# kdmd-SNN/Cargo.toml (workspace root)
[patch.crates-io]
koopman-dmd = { path = "../rust-dmd/koopman-dmd" }   # remove before publishing kdmd-snn
```

**Reused `koopman-dmd` public API** (all confirmed public in v0.1.0's `lib.rs`):

| Function/type | Where used in kdmd-snn |
|---|---|
| `dmd(&Mat<f64>, &DmdConfig)` | `identify`: fit autonomous `A` from a contiguous sub-threshold trajectory |
| `lift_data`, `LiftingConfig`, `LiftingInfo` | `identify`: EDMD lifting for nonlinear ground-truth models (poly for Izhikevich's quadratic term, `Delay` for hidden adaptation states) |
| `utils::determine_rank(&[f64], Option<usize>, f64)` | `RankPolicy::Energy` resolution (§5) |
| `utils::pinv(&Mat<f64>, Option<f64>)` | `dmdc` fallback; least-squares readout initialization |
| `utils::validate_matrix` | snapshot sanity checks before fitting |
| `dmd_stability(&DmdResult, tol)` / `Stability`, `StabilityResult` | `identify::validate`: reject/clamp identified `A` with spectral radius > 1 + tol |
| `dmd_spectrum(&DmdResult, dt)` / `ModeInfo` | validation report: recovered time constants vs ground-truth τ_m, τ_s |
| `dmd_residual`, `dmd_error` | re-identification trigger metric (§4.4); fit quality gates |
| `dmd_convergence` | tooling: "how many simulation steps are enough" analysis |
| `dmd_dominant_modes`, `DominantCriterion` | mode selection for the reduced-order operator (§5) |
| `predict_matrix`, `predict_modes` | integration tests: cross-check our `Operator::apply` against the reference predictor |
| `hankel_dmd`, `HankelConfig` | optional: identification from membrane-potential-only observations (partial state) |
| `DmdError` | wrapped into `SnnError` via `#[from]` |

**Known traps documented for all contributors:**
- `DmdConfig { center: true }` must **not** be used for SNN identification: the state mean
  is regime-dependent, and centering bakes it into the model.
- `dmd()` takes one **contiguous** trajectory and forms the shift pairs internally. It
  cannot consume concatenated trajectories or reset-masked snapshot pairs. That is why the
  `dmdc` API below takes explicit `(X, X', U)` matrices (§4.2) — the `SnapshotSet` builder
  handles concatenation and masking, then hands over pair matrices.
- `DmdResult.a_matrix` is `Vec<Vec<C64>>` — analysis-only. The inference path never touches
  it directly; `operator.rs` extracts a real `Mat<f64>` once at build time (§2.2).

---

## 2. Core data structures

### 2.1 Layer state — variable-major struct-of-arrays

A layer of N neurons with a k-variable neuron model (LIF: k = 2, `v` and `i_syn`;
adaptive LIF: k = 3, adds `w_adapt`) stores its state as one faer matrix in
**variable-major** order: rows `[0, N)` are all potentials, rows `[N, 2N)` all synaptic
currents, rows `[2N, 3N)` all adaptation variables. Batch samples are columns.

```rust
use faer::{Mat, MatRef, MatMut};

/// State of one layer: (k * n_neurons) rows × batch columns, variable-major.
/// Variable-major ordering makes the PerVariable operator (§2.2) apply as
/// contiguous axpy passes, and makes "the v block" a contiguous row range.
pub struct LayerState {
    x: Mat<f64>,          // (k*n) × batch
    n_neurons: usize,
    n_state_vars: usize,  // k
}

impl LayerState {
    pub fn zeros(n_neurons: usize, n_state_vars: usize, batch: usize) -> Self;
    pub fn n_neurons(&self) -> usize;
    pub fn batch(&self) -> usize;

    /// Full state block (for operator application).
    pub fn as_mat(&self) -> MatRef<'_, f64>;
    pub fn as_mat_mut(&mut self) -> MatMut<'_, f64>;

    /// View of state variable `var` (0 = membrane potential by convention):
    /// rows [var*n, (var+1)*n).
    pub fn var(&self, var: usize) -> MatRef<'_, f64>;
    pub fn var_mut(&mut self, var: usize) -> MatMut<'_, f64>;

    /// Convenience: membrane potentials (var 0).
    pub fn potentials(&self) -> MatRef<'_, f64> { self.var(0) }
}
```

Design notes:
- One allocation per layer; `subrows`-style views (`mat.as_ref().subrows(a, len)`) give
  zero-copy access to each variable block.
- Batch = 1 inference is the same code path with a 1-column matrix; no separate scalar path.
- The state carries no history. The training tape (§4.3) owns per-step saved tensors.

### 2.2 Operator storage

The premise's key performance insight: for LIF (and any neuron model whose sub-threshold
dynamics are linear and **identical across neurons in a layer**), the discrete-time
operator is not a dense N×N matrix — it is a small k×k matrix replicated per neuron,
`A = A_local ⊗ I_N` in variable-major ordering. For LIF with exponential synapse and
exact ZOH discretization:

```
A_local = [ α  γ ]      α = exp(-dt/τ_m),  β = exp(-dt/τ_s),
          [ 0  β ]      γ = (R/ (τ_m - τ_s)) · τ_s · (α − β)   (γ → dt·α·R/τ_m as τ_s → τ_m)
```

so the leak really is diagonal-per-variable and the whole advance is O(k²·N), not O(N²).
Dense A only becomes necessary when (a) DMD identifies cross-neuron coupling we choose to
keep (e.g. lifted observables that mix neurons, recurrent sub-threshold coupling), or
(b) we deliberately fold slow recurrent structure into A. Low-rank covers the middle
ground (§5).

```rust
use faer::{Mat, Col, MatRef, MatMut};

/// Discrete-time state-advance operator for one layer, x_{t+1} = A x_t (+ inputs).
pub enum Operator {
    /// Full dense A: (k*n) × (k*n). O(k²n²·batch) apply via faer matmul.
    Dense(Mat<f64>),

    /// A = A_local ⊗ I_n (variable-major). Exact for homogeneous linear neuron
    /// models (LIF, adaptive LIF). O(k²·n·batch) apply as axpy passes.
    PerVariable { a_local: Mat<f64> /* k × k */ },

    /// Per-neuron heterogeneous variant: neuron j uses its own k×k block
    /// (heterogeneous time constants). blocks[(p, q)] is a Col of length n:
    /// coefficient (p,q) for every neuron. Still O(k²·n·batch).
    PerNeuron { blocks: Mat<f64> /* (k*k) × n, column j = vec(A_j) */ },

    /// Rank-r factorization A ≈ P · A_r · Pᵀ with P orthonormal (n_state × r),
    /// A_r = Ã (r × r) from the DMD reduced operator. Used by the reduced-order
    /// path (§5); apply is O(n_state·r·batch).
    LowRank { p: Mat<f64>, a_r: Mat<f64> },
}

impl Operator {
    /// y = A · x (replace) or y += A · x (accumulate), into a caller-owned buffer.
    /// No allocation on the hot path.
    pub fn apply(&self, x: MatRef<'_, f64>, y: MatMut<'_, f64>, accumulate: bool);

    /// Spectral radius (for the stability gate). Cheap for PerVariable/LowRank.
    pub fn spectral_radius(&self) -> f64;
}
```

**Extraction from `koopman-dmd` results** (the `Vec<Vec<C64>>` → `Mat<f64>` bridge —
built once at identification time, never on the hot path):

```rust
/// Copy DmdResult.a_matrix into a real dense operator.
/// Errors if any |Im| > tol — a correctly identified operator from real data is
/// real up to conjugate-pair roundoff (A = Φ Λ Φ⁺ with λ's in conjugate pairs).
pub fn dense_from_dmd(res: &koopman_dmd::DmdResult, tol: f64) -> Result<Operator, SnnError>;

/// Build the LowRank variant from DmdResult: P = res.svd.u (n × r, orthonormal),
/// A_r = real part of res.a_tilde (r × r; a_tilde is real-valued by construction
/// in dmd.rs — it is Ã = Uᵀ X₂ V Σ⁻¹ before eigendecomposition).
pub fn low_rank_from_dmd(res: &koopman_dmd::DmdResult) -> Result<Operator, SnnError>;

/// Try to detect PerVariable structure in a dense identified A (block-pattern fit
/// within tolerance) and compress. Returns Dense unchanged if the fit is poor.
pub fn compress(op: Operator, tol: f64) -> Operator;
```

### 2.3 Spike representation

The consumer of spikes is the multiply `W·s` (drive into the next layer) plus the reset
term. Trade-off analysis for `W: n_out × n_in`, activity fraction `a` (fraction of
neurons spiking per step; typical SNN operating points are 1–10%):

| Representation | `W·s` cost | Notes |
|---|---|---|
| Dense `Col<f64>` | O(n_out·n_in) FMA | BLAS-2 matvec; SIMD-friendly; cost independent of activity. n_in = 4096 → 16.8M mul-adds/step |
| Sparse indices `Vec<u32>` | O(n_out·a·n_in) | `W` stored **column-major** (faer's native layout): `W·s = Σ_{j active} W[:, j]` is a sum of *contiguous* columns — pure axpy, fully SIMD. At 2% activity: ~0.34M adds, ~50× less work |
| Bitset (u64 words) | as sparse + decode | 64× memory saving over `Vec<u32>`; must be decoded to indices to accumulate columns anyway. Worth it only for spike *storage/transport* (tape, datasets), not compute |

Crossover in practice sits around 20–30% activity (dense matvec has better constants);
SNNs rarely operate there. **Decision:** sparse indices are the canonical inference
representation; dense is the canonical training representation (the tape needs dense
`f64` spike matrices for the backward matmuls regardless, and batched training turns
`W·S` into a BLAS-3 matmul that amortizes density); bitsets are a Phase-6 storage
optimization if the tape's memory becomes a problem.

```rust
/// Spikes from one layer at one time step, batch = 1: active neuron indices, sorted.
pub struct SpikeVec {
    pub active: Vec<u32>,
    pub n_neurons: u32,
}

/// Batched spikes for training: dense n_neurons × batch matrix of {0.0, 1.0}.
pub struct SpikeBatch(pub Mat<f64>);

impl SpikeVec {
    pub fn to_dense(&self) -> Col<f64>;
    pub fn activity(&self) -> f64;
}
impl From<&SpikeBatch> for Vec<SpikeVec> { /* per-column sparsify */ }
```

### 2.4 Synaptic weights

`W: Mat<f64>` (n_out × n_in), column-major (faer default) so sparse accumulation walks
contiguous columns. `W` is owned by the *receiving* layer together with a fixed
**injection map**: input drive enters the synaptic-current block (variable 1), not the
potential block. Formally `B_in = E_i · W` where `E_i` injects an n-vector into rows
`[N, 2N)`. We never materialize `E_i`; injection is "write into `state.var_mut(1)`".

---

## 3. The two-regime step: linear advance + nonlinear threshold

### 3.1 Canonical step (subtractive reset as control input)

Per layer, per time step:

```
1. drive:      d_t = W · s_in,t                       (sparse column-accum or dense matmul)
2. advance:    y   = A · x_t  +  E_i · d_t            (linear; A per §2.2)
3. threshold:  s_t = Θ( v(y) − θ )                    (v(y) = rows [0, N) of y; elementwise)
4. reset:      x_{t+1} = y − θ ⊙ E_v · s_t            (subtractive; E_v = injection into v rows)
```

Steps 2 and 4 together are **exactly** `x_{t+1} = A x_t + B u_t` with
`u_t = [s_in,t ; s_t]` and `B = [E_i W , −diag(θ) E_v]`. The only nonlinearity in the
whole network is the elementwise `Θ` in step 3. This is the load-bearing property of the
design; both identification and training lean on it.

### 3.2 Where the reset lives — analysis of the two choices

**R1 — subtractive reset folded into `u_t`** (`v ← v − θ·s`; canonical):
- *DMDc identification:* every snapshot triple `(x_t, u_t, x_{t+1})` satisfies the same
  linear relation **exactly** (for LIF, up to discretization of the ground-truth ODE).
  DMDc recovers `A` and both blocks of `B` without bias, including the reset column
  block — a strong internal consistency check: the identified reset columns should be
  `−θ·e_v` to high precision.
- *Gradients:* the step is differentiable everywhere except through `Θ`, which gets the
  surrogate. `∂x_{t+1}/∂x_t = A + (∂reset/∂s)(∂s/∂v)(∂v/∂x_t)` — all terms are
  well-defined matmuls (§4.3). No gradient is silently destroyed at reset; the membrane
  retains its overshoot information, which is also the biologically/numerically preferred
  behavior for rate fidelity.

**R2 — hard assignment** (`v ← V_reset` where spiked):
- *DMDc identification:* the assignment is state-dependent (the subtracted amount is
  `v − V_reset`, which varies with overshoot), so it is **not** representable as
  `B u_t` with binary `u_t`. Fitting DMDc on such data smears reset events into `A`,
  biasing the leak estimate. Mitigation if R2 is required: identify `A` only from
  **masked snapshot pairs** — drop every column `t` where any neuron in the layer spiked
  between `t` and `t+1` (the `SnapshotSet` builder supports this mask), and apply the
  reset mechanically outside the identified model.
- *Gradients:* the assignment either (a) passes a zeroed Jacobian through `v` at reset
  ("reset detach", standard in snnTorch/Norse practice — stable but discards overshoot
  gradient), or (b) requires a second surrogate for the reset indicator. Both are messier
  than R1.

**Decision:** R1 is the canonical mode for `KoopmanLayer`. The reference simulator
(`neuron::lif`) implements both (`ResetMode::Subtractive | ResetMode::HardTo(f64)`) so we
can quantify the behavioral difference, and R2 identification goes through the masked
path with a documented accuracy caveat.

### 3.3 Layer and network API sketch

```rust
pub struct ThresholdSpec {
    pub theta: Col<f64>,          // per-neuron threshold (broadcast scalar at construction)
    pub reset: ResetMode,         // Subtractive is canonical
}

pub enum ResetMode { Subtractive, HardTo(f64) }

pub struct KoopmanLayer {
    a: Operator,                  // identified (§4); Dense | PerVariable | LowRank
    w_in: Mat<f64>,               // learned synaptic weights, n_neurons × n_inputs
    thresh: ThresholdSpec,
    n_neurons: usize,
    n_state_vars: usize,
    // preallocated scratch: y buffer, drive buffer (see §6)
    scratch: LayerScratch,
}

impl KoopmanLayer {
    /// Build from an identified operator + freshly initialized weights.
    pub fn new(a: Operator, w_in: Mat<f64>, thresh: ThresholdSpec) -> Result<Self, SnnError>;

    /// Inference step, batch = 1, sparse spikes. Returns output spikes.
    pub fn step(&mut self, state: &mut LayerState, s_in: &SpikeVec) -> SpikeVec;

    /// Training/batched step: dense spikes in, dense spikes out.
    /// If `tape` is Some, saves (v_pre, s_out) for the backward pass.
    pub fn step_batch(
        &mut self,
        state: &mut LayerState,
        s_in: &SpikeBatch,
        tape: Option<&mut LayerTapeStep>,
    ) -> SpikeBatch;
}

pub struct Readout {
    pub r: Mat<f64>,              // n_classes × n_neurons, learned
    pub mode: ReadoutMode,        // SpikeCount | MembraneOfNonSpiking | LowPass
}

pub struct Network {
    pub layers: Vec<KoopmanLayer>,
    pub readout: Readout,
}

impl Network {
    /// Run T steps on one encoded input; returns readout logits.
    pub fn run(&mut self, input: &dyn SpikeSource, t_steps: usize) -> Col<f64>;
    /// Batched, optionally taped (training) version.
    pub fn run_batch(&mut self, input: &SpikeBatchSequence, tape: Option<&mut Tape>) -> Mat<f64>;
    pub fn reset_state(&mut self);
}
```

---

## 4. Identification and training

### 4.1 What is identified vs what is learned

| Object | Origin | Notes |
|---|---|---|
| `A` (state advance) | **Identified** by DMD/DMDc from ground-truth simulation | For LIF also available in closed form — the permanent test oracle |
| Reset block of `B` | Structurally known (`−θ E_v`); DMDc's estimate of it is a **validation check**, not the source of truth | |
| Injection structure `E_i` | Fixed by the neuron model | |
| `W` (synaptic weights) | **Learned** by surrogate BPTT | The `B_in = E_i W` factorization means DMDc never needs to see `W`: identification runs with known/teacher weights, training swaps them freely without touching `A` |
| Readout `R` | **Learned** (init via `utils::pinv` least-squares on collected features) | |
| `θ` (thresholds) | Fixed in v1; per-neuron learnable is a Phase-5 stretch | |

### 4.2 DMDc — API and placement

**Preferred: add to `koopman-dmd` as `src/dmdc.rs`, publish 0.2.0.** The algorithm
(Proctor, Brunton & Kutz, *SIAM J. Appl. Dyn. Syst.* 2016) is: stack `Ω = [X; U]`,
truncated SVD `Ω ≈ Ũ Σ̃ Ṽᵀ` (rank p), split `Ũᵀ = [Ũ₁ᵀ Ũ₂ᵀ]`, then
`A = X' Ṽ Σ̃⁻¹ Ũ₁ᵀ`, `B = X' Ṽ Σ̃⁻¹ Ũ₂ᵀ`; optionally project through a second SVD of
`X'` (rank r) for the reduced form. It reuses the crate's existing `determine_rank`,
`validate_matrix`, SVD idioms, and `DmdError` — a clean fit.

```rust
// koopman-dmd 0.2 addition — note: real Mat<f64> outputs, learning from the
// Vec<Vec<C64>> pain point. Explicit (x1, x2) pairs, NOT one contiguous matrix,
// so callers can concatenate trajectories and mask columns.
pub struct DmdcConfig {
    /// Truncation rank p for the [X; U] input SVD. None → determine_rank(_, None, 0.99).
    pub rank_input: Option<usize>,
    /// Truncation rank r for the output/X' SVD (reduced-order form). None → no second projection.
    pub rank_output: Option<usize>,
    pub dt: f64,
    /// If Some, B is known a priori: solve only for A on residual X' − B U
    /// (plain DMD on the residual). Used to pin B = [E_i W_teacher, −θ E_v].
    pub known_b: Option<Mat<f64>>,
}

pub struct DmdcResult {
    pub a: Mat<f64>,              // n × n, real
    pub b: Mat<f64>,              // n × q, real
    pub a_tilde: Mat<f64>,        // r × r reduced operator (r = rank_output or rank_input)
    pub b_tilde: Mat<f64>,        // r × q
    pub basis: Mat<f64>,          // n × r orthonormal output basis (for LowRank operators)
    pub eigenvalues: Vec<C64>,    // eig(a_tilde), for spectrum/stability reuse
    pub svd_input: SvdComponents, // of [X; U]
    pub rank_input: usize,
    pub rank_output: usize,
    pub dt: f64,
}

/// x1: n × m (states at t), x2: n × m (states at t+1), u: q × m (controls at t).
pub fn dmdc(x1: &Mat<f64>, x2: &Mat<f64>, u: &Mat<f64>, config: &DmdcConfig)
    -> Result<DmdcResult, DmdError>;
```

**Fallback:** if changing `koopman-dmd` is out of scope (open question §8), the identical
function lives in `kdmd_snn::identify::dmdc`, importing `koopman_dmd::utils::{determine_rank,
pinv, validate_matrix}` — all public in 0.1.0. The kdmd-snn code calls through one
internal alias either way, so the decision is reversible.

To make `dmd_stability` / `dmd_spectrum` reusable on DMDc results, either (a) koopman-dmd
0.2 generalizes them over an `eigenvalues: &[C64]` slice input, or (b) kdmd-snn applies
the same |λ| classification locally (10 lines). Prefer (a) if 0.2 is in scope.

### 4.3 Snapshot collection and the identification pipeline

```rust
pub struct SnapshotSet {
    x1: Mat<f64>,      // n_state × m  (states at t, possibly lifted)
    x2: Mat<f64>,      // n_state × m  (states at t+1)
    u:  Mat<f64>,      // q × m        (controls: [s_in; s_self] at t)
    lifting: Option<LiftingInfo>,
}

impl SnapshotSet {
    pub fn builder(n_state: usize, n_controls: usize) -> SnapshotSetBuilder;
}
impl SnapshotSetBuilder {
    /// Append one simulated trajectory; pairs are formed within the trajectory only.
    pub fn push_trajectory(&mut self, states: &Mat<f64>, controls: &Mat<f64>);
    /// Drop pairs where `mask[t]` is true (e.g. hard-reset steps, R2 mode).
    pub fn set_pair_mask(&mut self, mask: &[bool]);
    /// Optional EDMD lifting applied to x1/x2 (koopman_dmd::lift_data), not to u.
    pub fn lift(&mut self, cfg: LiftingConfig);
    pub fn build(self) -> Result<SnapshotSet, SnnError>;
}

pub enum RankPolicy {
    Fixed(usize),
    /// Energy threshold via koopman_dmd::utils::determine_rank(σ, None, threshold).
    Energy(f64),
    /// Scan candidate ranks; pick smallest whose held-out one-step relative
    /// residual (dmd_residual-style) is below max_rel.
    ValidatedResidual { max_rel: f64, candidates: Vec<usize> },
}

pub struct IdentifyConfig {
    pub rank: RankPolicy,
    pub dt: f64,
    pub pin_known_b: bool,        // use DmdcConfig::known_b with the structural B
    pub stability_tol: f64,       // gate: reject if spectral radius > 1 + tol
    pub max_imag: f64,            // gate for dense_from_dmd extraction
}

/// The main entry point: ground-truth sim → SnapshotSet → dmdc → validated Operator.
pub fn fit_layer_operator(
    snapshots: &SnapshotSet,
    cfg: &IdentifyConfig,
) -> Result<IdentifiedLayer, SnnError>;

pub struct IdentifiedLayer {
    pub a: Operator,                       // Dense, or LowRank when rank < n_state
    pub b_estimated: Mat<f64>,             // for validation against structural B
    pub report: IdentificationReport,      // spectrum (ModeInfo), stability, residuals
}
```

Validation gates inside `fit_layer_operator` (all via reused koopman-dmd analysis):
1. **Stability:** spectral radius ≤ 1 + tol (`dmd_stability` semantics). A leaky neuron
   model must identify as `Decaying`; a `Growing` result means bad data or rank.
2. **Realness:** `dense_from_dmd` imaginary-part gate.
3. **Residual:** one-step relative residual on a held-out trajectory below threshold.
4. **Structure check (LIF):** compare against the closed-form `A_local ⊗ I` to machine
   precision — this is a permanent integration test, not just a dev-time check.
5. **B consistency (R1):** identified reset columns ≈ `−θ e_v`.

### 4.4 Training: surrogate BPTT through the linear steps

Because each step is `linear map → elementwise Θ → linear map`, the backward pass is
hand-derivable; we do not need an autograd framework. Forward saves per step, per layer:
`v_pre` (n × batch) and `s_out` (n × batch). Memory: T·L·2·n·batch·8 B — e.g.
T=100, L=3, n=1024, batch=64 → ~315 MB; acceptable, with recompute-from-checkpoints as a
later option (§6).

Backward recursion per layer per step (λ = ∂L/∂x_{t+1} flowing backward):

```
g_s   = Wᵀ_next · (∂L/∂d_next,t)  +  Rᵀ · (∂L/∂logits contribution at t)   // uses of s_t
        − θ ⊙ (E_vᵀ λ)                                                     // reset path (R1)
g_v   = σ'(v_pre − θ) ⊙ g_s        // surrogate derivative — the ONLY approximation
∂L/∂y = λ + E_v · g_v
∂L/∂x_t = Aᵀ · (∂L/∂y)             // Operator::apply_transpose — exact Jacobian, the
                                    // Koopman "stable gradient highway" from the premise
∂L/∂W  += (E_iᵀ · ∂L/∂y) · s_in,tᵀ
```

`Operator::apply_transpose` mirrors `apply` for each variant (PerVariable transposes the
k×k block; LowRank transposes the factors — still O(Nr)).

```rust
pub enum SurrogateKind {
    /// σ'(x) = 1 / (β|x| + 1)²  (fast sigmoid / SuperSpike)
    FastSigmoid { beta: f64 },
    /// σ'(x) = 1 / (1 + (πβx)²) · β   (atan)
    Atan { beta: f64 },
    /// Boxcar/straight-through window
    Boxcar { half_width: f64 },
}

pub struct TrainConfig {
    pub surrogate: SurrogateKind,
    pub optimizer: OptimConfig,           // Sgd { lr, momentum } | Adam { lr, betas, eps }
    pub t_steps: usize,
    pub batch: usize,
    pub grad_clip: Option<f64>,
    pub reid: ReidentifyPolicy,           // §4.5
}

pub struct Trainer { /* holds Tape, optimizer state (m, v moments per param) */ }

impl Trainer {
    pub fn new(cfg: TrainConfig) -> Self;
    /// One minibatch: taped forward, backward, optimizer step on {W_l}, R.
    pub fn train_step(
        &mut self,
        net: &mut Network,
        batch: &SpikeBatchSequence,
        targets: &Targets,
    ) -> Result<StepStats, SnnError>;
}
```

### 4.5 The chicken-and-egg: A is fit under a regime, training changes the regime

Statement of the problem: `A` (especially a *lifted/EDMD* `A`, or a rank-truncated one)
is a best-fit under the state distribution induced by a particular input regime; training
`W` shifts that distribution, degrading the fit.

Two-part answer:

**(a) For plain LIF/adaptive-LIF with subtractive reset, there is no chicken-and-egg.**
The sub-threshold dynamics are *exactly* linear and input-independent; `A` is a property
of (τ_m, τ_s, dt) alone. Identification recovers a regime-independent object, `W` can be
retrained arbitrarily, and `A` never needs refitting. This is the v1 configuration and
should be stated loudly in docs: the DMD machinery is being used where it is exact.

**(b) For lifted or rank-truncated operators (nonlinear neurons, Phase 5+), use
event-triggered alternating optimization:**

```rust
pub enum ReidentifyPolicy {
    /// v1 / plain LIF: A is exact, never refit.
    Never,
    /// Every k epochs: freeze W, run the CURRENT network on training inputs,
    /// collect SnapshotSet, refit A, resume training. (Simple, predictable.)
    EveryEpochs(usize),
    /// Preferred: monitor the one-step relative residual of the current A on a
    /// rolling buffer of fresh trajectories (dmd_residual semantics); refit only
    /// when it exceeds `max_rel`. Cheap check, refit only when the regime has
    /// actually drifted. `hysteresis` prevents refit thrash.
    ResidualTriggered { max_rel: f64, check_every: usize, hysteresis: f64 },
}
```

The alternating loop (outer: identify, inner: train) converges in the same sense as any
EM-flavored scheme — each half-step reduces its own objective; we additionally monitor
eigenvalue drift between successive `A` fits (`dmd_convergence`-style max-|λ| change) and
surface it in training logs. Divergence of this metric is the skeptic's early-warning
signal that the lifted dictionary is inadequate, not that training is broken.

**Full workflow loop** (the premise's pipeline, made concrete):

```
1. Simulate ground truth:  neuron::lif reference sim, teacher W, encoded inputs
2. Collect snapshots:      SnapshotSetBuilder (states, controls; mask if R2)
3. Identify:               fit_layer_operator → validated Operator (+ report)
4. Assemble fast network:  KoopmanLayer::new(A, W_init, θ) per layer
5. Verify equivalence:     koopman path vs reference sim, same seeds (metrics::trajectory_rmse,
                           spike-coincidence rate) — CI gate
6. Train:                  Trainer::train_step loop; ReidentifyPolicy governs refits
7. Evaluate + benchmark:   metrics, criterion
```

---

## 5. Reduced-order fast path

Goal: an N-neuron layer (state dim n_s = k·N) steps in O(n_s·r) instead of O(n_s²).

State is kept in reduced coordinates `z = Pᵀ x` (P = `DmdResult.svd.u` or
`DmdcResult.basis`, orthonormal n_s × r). Per step:

```
1. z' = Ã z + Ŵ ·_sparse s_in         Ŵ = Pᵀ E_i W   (r × n_in, precomputed;
                                        sparse column-accum: O(r·n_active))
2. v  = P_v z'                          P_v = v-rows of P (N × r): O(N·r) — the only
                                        full-width work; needed because Θ is elementwise in v
3. s  = Θ(v − θ)
4. z_{t+1} = z' + B̂_r ·_sparse s       B̂_r = −Pᵀ (θ ⊙ E_v)  (r × N, precomputed): O(r·n_active)
```

Total: **O(N·r + r² + r·n_active)** per step. At N = 8192, k = 2, r = 64 that is ~0.5M
mul-adds vs ~268M for dense n_s² — before considering that PerVariable already gives
O(k²N) for homogeneous LIF. The LowRank path matters when A is *not* structured: lifted
observables, heterogeneous or recurrent sub-threshold coupling.

**Accuracy caveat (documented + tested):** step 4 projects the reset direction into
span(P); the discarded component `(I − PPᵀ)(−θ e_v s)` is a per-spike model error.
Mitigations, in order: (i) **basis augmentation** — orthonormalize `[P | e_v-block]` so
reset directions are represented exactly (adds ≤ N columns in the worst case; in practice
the v-block adds rank ≤ N but for homogeneous θ a single direction per variable often
suffices after averaging — measure); (ii) hybrid state: keep `v` explicit (N) and reduce
only the remaining variables. `identify::validate` reports the reset-projection residual
`‖(I − PPᵀ) E_v‖` so the choice is data-driven.

**Rank selection API** — thin wrapper over the crate:

```rust
/// Resolve a RankPolicy against identification data.
/// - Fixed(r): passed straight into DmdConfig { rank: Some(r), .. } / DmdcConfig.
/// - Energy(th): koopman_dmd::utils::determine_rank(&svd_sigmas, None, th).
/// - ValidatedResidual: fit at each candidate (dmd/dmdc), score with a
///   dmd_residual-style held-out one-step error, pick the knee.
/// Also runs koopman_dmd::dmd_convergence on the snapshot matrix to warn when
/// the data itself hasn't converged (changing-eigenvalue regime) — rank chosen
/// from unconverged data is reported as provisional.
pub fn resolve_rank(snapshots: &SnapshotSet, policy: &RankPolicy, cfg: &IdentifyConfig)
    -> Result<RankReport, SnnError>;

pub struct RankReport {
    pub rank: usize,
    pub energy_captured: f64,
    pub heldout_residual: f64,
    pub dominant_modes: Vec<usize>,   // koopman_dmd::dmd_dominant_modes(_, r, Energy)
    pub provisional: bool,
}
```

---

## 6. Performance plan

### 6.1 Hot path inventory

Per network step (L layers): for each layer — (1) spike drive `W·s` (sparse accum or
dense matmul), (2) `A·x` via `Operator::apply`, (3) threshold scan + spike emit, (4)
reset accumulate. Training adds the backward mirror (transposed matmuls) and tape writes.

Rules for the hot path:
- **Zero allocation per step.** `LayerScratch` preallocates `y`, drive, and spike
  buffers; `SpikeVec::active` reuses capacity across steps.
- **faer matmul, not operator `*`, on hot paths.** `&a * &b` allocates its output; use
  `faer::linalg::matmul::matmul(dst.as_mut(), Accum::Replace | Accum::Add, a.as_ref(),
  b.as_ref(), 1.0f64, par)` into scratch. (koopman-dmd itself uses `*` — fine there,
  identification is cold.)
- **Batch-as-columns.** Batched inference/training turns every per-step matvec into a
  BLAS-3 matmul `(n×n)·(n×batch)`; this is the single biggest lever and is why
  `LayerState`/`SpikeBatch` are matrices, not vectors.
- **Hand-rolled kernels only where structure beats BLAS:** `PerVariable::apply` (k² axpy
  passes over contiguous N-strips) and sparse column accumulation. Both are trivially
  SIMD-vectorizable by the compiler over contiguous slices; verify with criterion before
  any `unsafe`/intrinsics work (expectation: none needed).

### 6.2 Parallelism (rayon)

- **Across batch shards** for training (data-parallel gradient accumulation): split the
  batch into `num_threads` column blocks, each thread runs taped forward/backward on its
  shard with its own scratch, reduce gradients. Clean because shards share only `&` refs
  to parameters.
- **Within large matmuls:** faer's own `Par::rayon(0)` for Dense operators at large N.
  Do not nest with shard parallelism — choose per size at runtime (shard-parallel when
  batch ≥ threads; matmul-parallel for batch=1, huge N).
- **Not across layers per step** — layers are sequentially dependent within a time step.
  (Layer pipelining across time steps is possible but adds latency/complexity; noted as
  future work, not planned.)
- Threshold scan: `rayon::par_chunks` only above a measured N cutoff (~100k); below that
  the scan is memory-bound and single-thread SIMD wins.

### 6.3 Benchmark strategy (criterion, from Phase 1 onward)

Benches live in `crates/kdmd-snn/benches/`, mirroring koopman-dmd's harness setup:

| Bench | Axes | Question answered |
|---|---|---|
| `step_kernels` | Operator ∈ {Dense, PerVariable, LowRank(r∈{16,64,256})} × N ∈ {128, 1k, 8k, 64k} | Crossovers; validates §2.2/§5 claims |
| `spike_drive` | dense matvec vs sparse column-accum × activity ∈ {1%, 5%, 20%, 50%} × N | Validates §2.3 crossover; sets the runtime dense/sparse switch |
| `inference_e2e` | reference LIF sim vs koopman path, same network, T=100 | The headline speedup claim of the whole project |
| `train_epoch` | batch ∈ {16, 64, 256} × shard-parallel on/off | BPTT throughput; rayon scaling |
| `identify` | dmdc on m ∈ {1k, 10k, 100k} snapshots | Cold-path sanity (must stay "minutes, not hours") |

Every performance claim made in docs must cite a bench name. Regression tracking via
criterion's saved baselines in CI (compare against `main`).

---

## 7. Phased delivery plan

Each phase ends green (fmt, clippy, tests) and demo-able. Test/skeptic subagents review
at phase boundaries.

**Phase 0 — Skeleton** (foundation)
- Workspace + `kdmd-snn` crate, `error.rs`, `state.rs`, `spikes.rs`, CI (fmt, clippy,
  test), `[patch.crates-io]` wiring to the local koopman-dmd checkout.
- Exit: `cargo test` green; `LayerState`/`SpikeVec` unit-tested.

**Phase 1 — LIF ground truth**
- `neuron::lif` reference simulator (both reset modes), closed-form discrete `A_local`,
  `encoding.rs` (Poisson, current injection), one synthetic task in `data.rs`,
  `metrics.rs` basics. First criterion bench (`step_kernels` scaffold on the ref sim).
- Exit: reference sim reproduces textbook LIF behavior (f–I curve test, membrane decay
  test against analytic solution).

**Phase 2 — DMD identification (autonomous)**
- `identify::snapshots`, `operator.rs` with `dense_from_dmd` / `low_rank_from_dmd` /
  `compress`; fit `A` with `koopman_dmd::dmd` on contiguous sub-threshold LIF
  trajectories; validation gates (`dmd_stability`, `dmd_spectrum`, `dmd_residual`).
- Exit: **identified A matches closed-form LIF A to ≤ 1e-8 relative error** (the oracle
  test); recovered τ_m, τ_s from `dmd_spectrum` match ground truth.

**Phase 3 — DMDc**
- `dmdc()` (in koopman-dmd 0.2 if approved — see §8 — else `identify::dmdc`),
  `fit_layer_operator` end-to-end with control inputs `[s_in; s_self]`, `known_b` pinning,
  reset-column consistency check.
- Exit: A + B identified from *spiking* trajectories (no sub-threshold masking needed in
  R1) match structure to tolerance; masked-R2 path tested.

**Phase 4 — Fast inference network**
- `layer.rs`, `network.rs`, PerVariable + Dense apply kernels, sparse spike path,
  `LayerScratch`; equivalence test: koopman path vs reference sim, same seeds, per-step
  state RMSE and spike-coincidence gates. `inference_e2e` + `spike_drive` benches.
- Exit: bit-comparable spikes vs reference LIF sim over T=1000 (R1, PerVariable);
  measured speedup reported.

**Phase 5 — Training**
- `train/` (surrogate kernels, tape, backward, Sgd/Adam, loss), `Readout`,
  shard-parallel batching, `ReidentifyPolicy::Never` (LIF). Learn a synthetic task
  (Poisson pattern classification or temporal XOR) to target accuracy.
- Exit: training curve reproducibly reaches target on the synthetic task; gradient check
  vs finite differences on a tiny network (surrogate path validated where differentiable).

**Phase 6 — Reduced order, nonlinear ground truth, demo, polish**
- `LowRank` path + `resolve_rank` + basis-augmentation option; Izhikevich ground truth
  with `LiftingConfig::{PolynomialCross, Delay}` and `ResidualTriggered` re-identification;
  demo example (spiking MNIST behind `mnist` feature, or the synthetic task if dataset
  scope is cut — §8); full bench suite, docs pass, README examples as doctests
  (pattern already proven in koopman-dmd).
- Exit: `cargo run --example ...` demo; benches published in docs; documentarian sign-off.

Dependency note: Phases 2–3 are where the koopman-dmd 0.2 decision bites; the fallback
keeps Phase 3 unblocked either way.

---

## 8. Open questions for the user

Decisions that genuinely need the project owner. Answer inline.

1. **koopman-dmd changes in scope?** Preferred plan adds `dmdc()` (+ real-`Mat` result
   type, and generalizing `dmd_stability`/`dmd_spectrum` to accept an eigenvalue slice)
   to koopman-dmd as v0.2.0, published to crates.io by you. Alternative keeps koopman-dmd
   frozen at 0.1 and hosts `dmdc` inside kdmd-snn. Which do you want? (Affects Phase 3;
   both are designed for, §4.2.)

2. **Target scale.** What N (neurons/layer), layer count, and time horizon T should we
   optimize for — order 1k neurons (laptop realtime), 100k, or beyond? This sets how much
   effort the LowRank/basis-augmentation path (§5) deserves vs the PerVariable fast path,
   and whether the tape needs checkpoint/recompute in v1.

3. **Demo task/dataset.** Spiking MNIST (needs a dataset download/loader dependency and
   ~11 MB of data — any licensing/vendoring preference?), N-MNIST, or a fully synthetic
   task (zero external data, weaker headline)? Plan assumes: synthetic task in Phase 5,
   MNIST behind a feature in Phase 6.

4. **Ground-truth neuron models.** Is LIF + adaptive LIF enough for v1, with Izhikevich
   (nonlinear, exercises EDMD lifting and re-identification) in Phase 6 — or is the
   nonlinear/lifted regime the point of the project for you, deserving earlier priority?

5. **Reset convention.** The design commits to subtractive reset as canonical (§3.2) for
   exact linearity; hard-reset is reference-sim-only with masked identification. OK, or
   do you need hard-reset parity in the fast path?

6. **`no_std` / embedded targets?** Assumed **no** (faer + rayon + std). If neuromorphic
   or embedded deployment is an ambition, it changes the operator/spike storage design
   and should be said now even if deferred.

7. **GPU ambitions.** None planned; the `Operator` enum keeps a seam where a
   `wgpu`/`cudarc` backend could slot in later. If GPU is on your roadmap within ~6
   months, the apply-kernel trait boundary should be designed slightly wider now — say
   the word.

8. **Precision.** koopman-dmd is `f64`-only, so identification is f64. The inference/
   training hot path could be `f32` for ~2× bandwidth. v1 plan: f64 everywhere for
   simplicity and exact oracle tests. Do you want an `f32` fast path on the roadmap?

9. **Training stack.** Plan is a hand-rolled tape (§4.3) — no burn/candle dependency.
   Comfortable with that, or do you want optional integration with an autograd framework
   for extensibility (custom losses, deeper heads)?

10. **Serialization.** Do trained/identified models need save/load in v1 (`serde`
    feature, §1.2), and if so any format preference (postcard/bincode vs JSON)?

11. **License/MSRV.** Match koopman-dmd: MIT, rust-version 1.85? (Assumed yes.)
