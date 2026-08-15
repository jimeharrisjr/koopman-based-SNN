# 11 — Phase 6 Test Report

Final test pass (Phase 6). Scope: full suite in both feature configurations,
plus 39 adversarial probes written as a temporary integration-test file and a
temporary SHD-loader example (both deleted after the run, per protocol — the
reproductions below are self-contained). No library code or existing tests
were modified.

Environment: macOS (Darwin 25.4.0), `cargo test --release`, workspace
`koopman-dmd` patched to the local `rust-dmd` checkout.

## 1. Suite results

| Configuration | Result |
|---|---|
| `cargo test --release` | **101/101 pass** (87 unit + 4 dmdc_layer + 2 layer_equivalence + 5 oracle + 1 reduced_order + 2 training; 0 doc-tests) |
| `cargo test --release --features datasets` | **101/101 pass** (identical set; static HDF5 builds cleanly) |

Note: the `datasets` configuration compiles the HDF5 stack but the suite
contains **no test that opens an H5 file** — the loader is only reachable via
`examples/shd_demo.rs`. It was exercised here via a scratch example (§2.7);
see §4 for a suggested permanent smoke test.

## 2. Adversarial probes

39 probes total: 29 scratch integration tests plus ~10 assertions in a scratch
SHD example. 33 behaved well; 6 findings (§3). Passing probes are listed as
coverage evidence.

### 2.1 Extreme LIF parameters — PASS (1 finding, §3.3)
- `dt = 1000, tau_m = 10` (alpha/beta underflow to 0): propagator and
  `b_local` finite; one step from rest lands exactly on `v = R·u` (≤1e-12).
- `tau_m == tau_s` exactly (7.3/7.3, dt 0.31): degenerate gamma limit finite
  and positive; fast layer vs reference simulator agree to 1e-12 over 500
  spiking steps.
- Non-finite/invalid params (`tau_m = NaN`, `dt = inf`, `theta = NaN`,
  `tau_s < 0`, `theta == v_rest`): all rejected with clean
  `InvalidParameter` errors.
- Massive drive (`W = 1e6`, one subtraction cannot bring `v` below threshold):
  reference and fast layer both fire every step and agree to 1e-9; states stay
  finite. The one-spike-per-step / single-subtraction convention is consistent
  across both paths.
- Near-degenerate taus: **finding §3.3** (gamma branch discontinuity).

### 2.2 Operator / layer shape safety — 2 findings (§3.1, §3.2, §3.4)
- `Operator::LowRank` with `r = state_dim`, `P = I`: apply and
  apply_transpose reproduce `Operator::Dense` to 1e-14 (identity-projection
  exactness). PASS.
- `Operator::LowRank` with `r = 0` (`P` is n×0): applies as exact zero, no
  panic. PASS.
- `fit_controlled` with `rank_output = Some(state_dim)` on exactly linear data
  (`x2 = A·x1`, `known_b` pinned to zero): returned `Operator::LowRank`
  reproduces the true `A` to 1e-8. PASS.
- `KoopmanLayer::new` with a `PerVariable(k=2, N=3)` operator and a length-3
  `b_local`: **accepted** — finding §3.2.
- `KoopmanLayer::lif` with `v_rest != 0`: **silently wrong dynamics** —
  finding §3.1.
- `KoopmanLayer::new` with a 0-neuron operator: **accepted** — finding §3.4.

### 2.3 Empty / degenerate spikes — 1 finding (§3.5)
- Empty `SpikeVec` fed to a 2-layer network for 200 steps: no output spikes,
  every state entry stays exactly 0. PASS.
- `SpikeVec::from_indices` range/order/duplicate validation: clean errors
  (existing coverage confirmed). PASS.
- `SpikeVec::new(0).activity()`: **NaN** — finding §3.5.

### 2.4 Batched network at scale — PASS
- Single 128-neuron layer, batch = 3, 100 steps of independent random input
  per column: every batch column's spike train is identical to a separate
  batch = 1 run of the same layer. Columns are fully independent.

### 2.5 Trainer edge cases — ALL PASS
- `T = 1` sequences (5 train steps): loss finite, accuracy in [0, 1], all
  weights and readout finite afterward.
- All-silent inputs (T = 20, 3 train steps): loss exactly `ln(n_classes)`
  (uniform softmax on zero counts), no NaN anywhere in weights, readout, or
  predictions. The backward pass through 20 steps of zero spikes is
  numerically clean.
- Target index out of range: clean `InvalidParameter` error, no panic.
- Determinism: two identical nets/trainers given identical inputs produce
  **bitwise identical** weights and readout after 4 train steps (the trainer
  is RNG-free, as documented).

### 2.6 Identification with pathological snapshot sets — ALL PASS (clean errors)
- Constant states (`x1 == x2`, zero controls), rank 2 requested:
  `invalid input: requested rank 2 exceeds the numerical rank of the data
  (σ_2/σ_1 = 0.000e0)`. Clean error, no NaN operator.
- Single snapshot pair: `invalid input: matrix has 1 columns, need at least
  2`. Clean error.
- All-zero trajectory into `fit_autonomous`: `eigendecomposition failed:
  NoConvergence`. Clean error (message could be friendlier, but it cannot be
  mistaken for success).
- `SnapshotSetBuilder` masking all but one pair: builds correctly with
  `n_pairs == 1` and the right column; masking *every* pair yields a clean
  "no pairs" error at `build()`.

### 2.7 Surrogate / return-map adversarial configs — ALL PASS
- `IzhikevichSurrogate::fit` with only 2 training currents: the lifted
  regression is genuinely rank-deficient (I-dependent observables take two
  values) and fails **cleanly**: `requested rank 10 exceeds the numerical
  rank of the data (σ_10/σ_1 = 9.7e-17)`. No NaN operator escapes.
- Surrogate config validation (degree 1, t_steps 2, n_holdout ≥
  n_trajectories, empty/non-finite currents): all clean errors (existing
  coverage confirmed).
- Return-map config validation: degree 0 and 7, negative horizon, zero
  sample interval, `n_holdout ≥ n_trajectories`, `h_ref = NaN`, empty
  currents, all-sub-rheobase currents — all clean errors.
- Determinism: same seed → `IzhikevichSurrogate::fit` twice gives **bitwise
  identical** lifted operators and identical rollouts;
  `IzhikevichReturnMap::fit` twice gives identical rollout spike times;
  `PoissonEncoder` with the same seeded `StdRng` gives identical spike
  trains. `rate·dt > 1` clamps to p = 1 (all bins spike, no panic); negative
  rates are a clean error.

### 2.8 SHD loader on real data (`--features datasets`) — PASS (1 note, §3.6)
- `load_shd(data/shd/shd_test.h5)`: 2264 samples, all 20 classes present,
  unit ids ≤ 699, event times finite and non-negative (max 1.170 s),
  times/units lengths consistent.
- `bin_events` output is strictly sorted, deduplicated, in-range, and
  accepted verbatim by `SpikeVec::from_indices`.
- Invalid arguments (`n_pooled` 0 and 701, `bin_s` 0 / negative / NaN,
  `t_steps` 0): all clean errors.
- Corrupt-input tolerance: **note §3.6**.

## 3. Issues found

### 3.1 `KoopmanLayer::lif` silently drops the `v_rest` affine term
**Severity: silent corruption** (worst class — wrong numbers, no error).

The fast layer is purely linear (`x ← A·x + B·u`), so the reference
simulator's affine rest term `v_rest·(1 − alpha)` has nowhere to live.
`lif_structural_b` guards its own path by rejecting `v_rest != 0`
(src/identify/mod.rs:293), but `KoopmanLayer::lif` (src/layer.rs:103) accepts
any valid `Lif` and silently produces different dynamics: with `v_rest = 0.5`,
`theta = 1.0`, zero input, and `v` initialized to `v_rest`, the reference
holds `v = 0.5` forever while the layer decays toward 0 (`v = 0.303` after 50
steps). Every crate default and test uses `v_rest = 0`, so nothing currently
trips this — but any user porting textbook parameters (`v_rest = −65 mV`)
gets silently wrong spike trains (or, for negative thresholds, a confusing
"threshold must be positive" rejection).

Reproduction:
```rust
let lif = Lif::new(LifParams { v_rest: 0.5, theta: 1.0, ..Default::default() })?;
let mut layer = KoopmanLayer::lif(&lif, 1, Mat::zeros(1, 1), 1)?; // accepted!
// init state to v_rest, step both with zero input 50x:
// reference v = 0.5 (exact fixed point), layer v = 0.5·alpha^50 = 0.303.
```
**Expected:** `KoopmanLayer::lif` rejects `v_rest != 0` with the same
shift-coordinates message `lif_structural_b` already uses.
**Actual:** accepted; silently divergent dynamics.

### 3.2 `KoopmanLayer::new` accepts a `b_local` whose length contradicts the operator's factorization
**Severity: silent corruption** (constructor validation gap).

The constructor only checks `dim % b_local.len() == 0` (src/layer.rs:66). A
`PerVariable { k = 2, N = 3 }` operator (dim 6) combined with a length-3
`b_local` passes, and the layer concludes `n_state_vars = 3, n_neurons = 2` —
a different factorization than the operator encodes. The threshold/reset then
treats rows 0–1 as "the potential block" while the operator's potentials
occupy rows 0–2: spikes are read from a row range that is half potentials,
half garbage. `PerVariable` and `PerNeuron` operators carry their own `k`, so
the mismatch is fully detectable at construction time.

Reproduction:
```rust
let a = Operator::PerVariable { a_local: /* 2x2 */, n_neurons: 3 }; // dim 6
let l = KoopmanLayer::new(a, Mat::zeros(2, 2), vec![0.1, 0.2, 0.3], 1.0, 1)?;
assert_eq!(l.n_neurons(), 2); // operator meant N = 3 — accepted silently
```
**Expected:** `DimensionMismatch` when the operator is `PerVariable`/
`PerNeuron` and `b_local.len() != k` (Dense/LowRank carry no `k` and cannot
be checked — document that).
**Actual:** accepted.

### 3.3 Near-degenerate time constants: gamma loses ~6 digits just outside the degeneracy switch
**Severity: silent numeric inaccuracy, small magnitude** (low).

`exp_input_coupling` (src/neuron/lif.rs:149) switches to the degenerate limit
only when `|tau_m − tau_s| ≤ 1e-9·max`. Just outside that window the analytic
branch computes `(alpha − beta)/(tau_m − tau_s)` with catastrophic
cancellation. Measured branch disagreement (tau_m = 10, dt = 0.1):

| rel tau diff | branch mismatch (relative) |
|---|---|
| 2e-9 | 2.9e-6 |
| 1e-8 | 6.7e-7 |
| 1e-7 | 2.7e-8 |
| 1e-6 | 7.6e-10 |

The true gamma deviates from the degenerate limit by only O(rel diff), so at
rel diff 2e-9 the analytic branch is ~1000× less accurate than the limit it
was switched away from. This quietly breaks the "exact propagator / 1e-12
oracle" contract in the band `1e-9 < rel diff < ~1e-6`.
**Expected:** error ≈ machine precision everywhere.
**Actual:** up to ~3e-6 relative error in `a_local[(0,1)]` near the switch.
Fix direction: widen `TAU_DEGENERATE_RTOL` to ~1e-6 (crossover point above),
or rewrite the difference via `exp_m1` to kill the cancellation.

### 3.4 0-neuron layers construct, then fail far away with an unrelated message
**Severity: delayed/confusing error** (low).

`KoopmanLayer::new` accepts a `PerVariable { n_neurons: 0 }` operator (dim 0,
`0 % k == 0`, W is 0×m). The failure surfaces only in `Network::new` as
`LayerState dimensions must be nonzero (n_neurons = 0, ...)` — correct but
far from the mistake, and a bare 0-neuron `KoopmanLayer` used directly never
errors at all. **Expected:** rejected at layer construction like every other
zero dimension in the crate. **Actual:** accepted.

### 3.5 `SpikeVec::new(0).activity()` returns NaN
**Severity: edge-case NaN** (low).

`SpikeVec` is the only spike/state type without a nonzero-dimension guard
(`SpikeBatch::zeros` and `LayerState::zeros` both reject 0).
`SpikeVec::new(0)` constructs and `activity()` returns `0.0 / 0.0 = NaN`,
which propagates through any statistics computed over it. **Expected:**
either reject `n_neurons = 0` at construction (consistent with the rest of
the crate) or define `activity() = 0.0` for the empty layer. **Actual:** NaN.

### 3.6 SHD `bin_events` silently normalizes corrupt events (note, not a defect)
**Severity: documented-behavior note** (informational).

For a corrupt sample, an out-of-range unit id (e.g. 5000 ≥ 700) is silently
clamped into the last pooled channel via `.min(n_pooled − 1)`
(src/data/shd.rs:132), and a negative event time lands in bin 0 (`as usize`
saturates). Real SHD files never contain either (verified across all 2264
test samples), so this is corruption tolerance rather than a bug — but a
range check in `load_shd` (units < 700, times ≥ 0) would turn a corrupted
download into a clean `Dataset` error instead of subtly wrong spike trains.

## 4. Tests worth adding permanently

1. **Layer/reference equivalence at parameter extremes** — the §2.1 probes
   (`tau_m == tau_s` exact, `dt >> tau`, massive-drive persistent firing) are
   cheap and pin the oracle contract where it is numerically most fragile.
2. **`Operator::LowRank` exactness and degeneracy** — full-rank identity
   projection reproduces `Dense` to 1e-14; rank-0 applies as zero. Guards the
   reduced-order path's algebra.
3. **Trainer determinism (bitwise)** — two identical runs must produce
   bitwise-identical weights; catches any accidental RNG or iteration-order
   nondeterminism introduced later.
4. **Trainer degenerate inputs** — T = 1 and all-silent sequences stay
   finite, silent-input loss equals `ln(n_classes)` exactly.
5. **Batch-column independence at width** — batch = k equals k independent
   batch = 1 runs for a wide layer (the existing test covers only 3 neurons,
   batch 2, one drive pattern).
6. **Pathological identification is loud** — constant states, single pair,
   and all-zero trajectories must keep producing errors, never NaN operators
   (pins the current good behavior of the `koopman-dmd` boundary).
7. **SHD loader smoke test** (`#[ignore]` or feature-gated on the data files
   existing) — load the real test file, assert 2264 samples / 20 classes /
   units < 700, and that `bin_events` output feeds `SpikeVec::from_indices`.
   Today the `datasets` feature has zero executed test coverage.
8. **After fixing §3.1/§3.2/§3.4/§3.5** — constructor-rejection tests for
   `v_rest != 0`, mismatched `b_local` length, 0-neuron operators, and
   0-neuron `SpikeVec`.
