# 10 — Code Quality Review (Phase 6 final pass)

Scope: every module in `crates/kdmd-snn/src`, plus `tests/`, `benches/`, and the
examples, read in full. Clippy/fmt-level issues are excluded by instruction; every
correctness claim below was verified by tracing a concrete input, not by pattern
matching. Paths are absolute; line numbers refer to the current tree.

**Verified sound (worth recording, since these were the designated risk areas):**

- The BPTT backward pass in `src/train/mod.rs:278-359` implements its documented
  recursion **exactly**. Expanding the code's `g_s`/`g_v`/`dldy` sequence gives
  `∂L/∂y_v = λ_v·(1 − θ·σ′) + σ′·(downstream)`, i.e. the reset's dependence on
  `y_v` through the spike is retained through the surrogate, as the module doc
  claims. Layer ordering (l descending inside t descending) makes `dldd[l+1]`
  available at the right time step; `λ ← Aᵀ·∂L/∂y` is the exact linear Jacobian;
  `∂L/∂W = ∂L/∂d · s_inᵀ` indexes agree with `W: n_neurons × n_inputs`.
- Variable-major index arithmetic is consistent across `state.rs` (`var·N + j`),
  `operator.rs` (all four variants), `layer.rs` (drive coupling `p·n + j`), and
  the train loop (`p·n + j`). `PerNeuron` transpose (`idx = q·k + p`,
  operator.rs:143) is the correct per-block transpose; `LowRank` transpose
  correctly transposes only `A_r` (since `(P·A_r·Pᵀ)ᵀ = P·A_rᵀ·Pᵀ`). Each
  variant is tested against a hand-built dense reference.
- Boundary conditions in `metrics::spike_coincidence` (usize subtraction around
  `binary_search`) cannot underflow: `ins` partitions strictly.
- The LIF closed form is validated three independent ways (free decay, steady
  state `R·u`, continuous-time f–I curve), not just self-consistently.

---

## CRITICAL

### C1. `KoopmanLayer::lif` silently produces wrong dynamics for `v_rest ≠ 0` (and silently ignores a hard-reset `Lif`)

`src/layer.rs:100-114` builds the fast layer from any valid `Lif`, but the fast
path computes `y = A·x` with **no affine rest term**, thresholds at absolute
`θ`, and subtracts exactly `θ` on reset. The reference simulator
(`src/neuron/lif.rs:192-201`) computes `v_rest + α(v − v_rest) + …` and
subtracts `θ − v_rest`. These coincide **only when `v_rest = 0`**.

Concrete divergence: `LifParams { v_rest: 0.5, theta: 1.0, dt: 0.1, .. }` is
accepted by both `Lif::new` (θ > v_rest ✓) and `KoopmanLayer::lif` (θ > 0 ✓).
From `init_state` (v = 0.5) with zero input, the reference holds v = 0.5
forever; the fast layer decays v ← α·0.5 → 0.495 → … → 0. Wrong from the very
first step, with no error, while the module doc (`src/layer.rs:19-23`) claims
the layer is "**exactly** the reference simulator's update". The same
constructor also accepts a `Lif` built with `ResetMode::HardTo(..)` and
silently substitutes subtractive semantics.

The codebase already knows about this constraint: `lif_structural_b`
(`src/identify/mod.rs:293-300`) rejects `v_rest != 0` with a precise message.
`KoopmanLayer::lif` simply lacks the same guard. All shipped tests use the
default `v_rest = 0`, so nothing catches it.

**Fix:** in `KoopmanLayer::lif`, return `SnnError::InvalidParameter` when
`lif.params().v_rest != 0.0` (reuse the `lif_structural_b` wording: "shift
coordinates to v − v_rest first") and when
`lif.params().reset != ResetMode::Subtractive`. Add one line to the
`layer.rs` module doc stating the normalized-coordinates convention
(`v_rest = 0`, subtractive reset), and a constructor test for both rejections.

---

## MAJOR

### M1. `shd::bin_events` pooling leaves permanently-dead channels whenever `n_pooled` does not divide 700

`src/data/shd.rs:125` computes `group = ceil(700 / n_pooled)` and then
`pooled = (u / group).min(n_pooled − 1)` (line 132). The number of pooled
channels actually reachable is `ceil(700 / group)`, which is **less than**
`n_pooled` whenever 700 % n_pooled ≠ 0. Concretely: `n_pooled = 300` →
`group = 3` → pooled ids span 0..=233, so channels 234–299 of the user's
300-input layer can never fire (22 % dead inputs); `n_pooled = 640` → `group = 2`
→ only 0..=349 used (45 % dead). No error, no doc warning — a user gets a
silently degraded SHD network. The shipped demo dodges this only because
`N_POOLED = 100` divides 700 exactly (`examples/shd_demo.rs:20`).

**Fix:** map proportionally — `pooled = (u as usize * n_pooled) / 700` (with
`u < 700` validated, see m6) — which uses every pooled channel and balances
group sizes to ⌊700/n⌋/⌈700/n⌉. Add unit tests for `bin_events` (it is pure
logic; nothing in the feature-gated module is tested today), including a
non-divisor `n_pooled` asserting every pooled index is reachable.

### M2. The multi-layer backward path has zero test coverage (and k > 2 / taped-`v_pre` are only indirectly tested)

The inter-layer gradient term `g_s += W_{l+1}ᵀ · ∂L/∂d_{l+1}` at
`src/train/mod.rs:292-305` — the part of the recursion that makes this BPTT
rather than single-layer regression — is never executed by any test:
`tests/training.rs:21-31` builds a **one-layer** network, and no other test
constructs a `Trainer`. A sign error, a transpose slip, or an off-by-one in
that block would pass the entire suite. Similarly untested: the
`for p in 1..k` state-variable loop with k = 3 (no adLIF layer ever trains),
and the content of `v_pre` from `step_batch_taped` (validated only through the
end-to-end learning curve).

**Fix:** add a two-layer variant of `surrogate_bptt_learns_the_poisson_pattern_task`
(same task, `vec![l0, l1]`) with the same loss-halving assertion — it directly
exercises train/mod.rs:292-305 — plus a small unit test asserting
`step_batch_taped`'s `v_pre` equals the post-advance pre-reset potential
(`v_pre[(j,b)] == state_v + θ·s` on spiking entries after one step).

### M3. `Trainer::train_step` / `predict` panic instead of returning `SnnError` when the network doesn't match the trainer's construction shape

`Trainer::new` (`src/train/mod.rs:89-111`) sizes the readout and per-layer Adam
states from `net`, but `train_step`/`predict` accept any `&mut Network` and
never re-validate. Pass a net whose top layer has more neurons than at
construction and `self.r[(i, j)]` at `src/train/mod.rs:224` (or `:398` in
`predict`) panics inside faer's bounds check; a net with more layers panics
indexing `self.opt_w`. `Adam::update` guards shapes only with `debug_assert`
(`src/train/optim.rs:40-41`), so in release the mismatch surfaces as a faer
index panic mid-update. Everywhere else in the crate a dimension mismatch is a
`SnnError::DimensionMismatch` — this is the one Result-returning API family
that violates the crate's own error policy on a plausible misuse (rebuilding a
network between epochs).

**Fix:** store `(n_layers, per-layer W shapes, n_out)` in `Trainer` and check
them at the top of `train_step`/`predict`, returning
`SnnError::DimensionMismatch`. Document on `Trainer` that one trainer is bound
to one network shape.

---

## MINOR

### m1. `Network::new` cannot enforce its documented batch-capacity contract

`src/network.rs:27-63`: the doc says "every layer must share the same batch
capacity", but the constructor never checks the layers' scratch width (there is
no accessor), so a layer built with batch 4 inside `Network::new(.., 2)`
constructs fine and fails only at the first `step_batch` with the layer-level
message from `src/layer.rs:161-167`. **Fix:** add `KoopmanLayer::batch()`
(returns `self.y.ncols()`) and validate in `Network::new`.

### m2. `low_rank_from_dmd` is dead public API and skips the realness gate its sibling enforces

`src/operator.rs:227-243` is used by no crate code, test, or example
(`fit_controlled` builds `LowRank` directly from `DmdcResult`), and it copies
`a_tilde` taking `.re` with no imaginary-residue check, unlike
`dense_from_dmd` immediately above. Either delete it or give it the same
`max_imag` gate; if kept, add a test.

### m3. Needless full-matrix copies at the identification boundary

`src/identify/mod.rs:138` (`trajectory.to_owned()`) and `:204-209` (three
`snapshots.*().to_owned()` calls) clone entire snapshot matrices on every fit
because `SnapshotSet` exposes only `MatRef` while `koopman_dmd::dmd/dmdc` take
`&Mat`. Add `pub(crate)` accessors returning `&Mat<f64>` to `SnapshotSet` and
pass those. Likewise `IdentifiedControlled` stores `b: res.b.clone()` alongside
`dmdc: res` which already owns the same matrix (`:229-234`).

### m4. Borrow-checker workarounds that are not actually needed

`src/train/mod.rs:327` re-allocates `b_local().to_vec()` on every (t, l)
iteration of the backward loop — the slice borrow does not conflict with the
`dldy`/`grad_w` writes; hoist one borrow per layer (or per step) out of the
loop. `src/train/mod.rs:375-377` does a `mem::replace` dance to update `self.r`,
but `self.opt_r.update(&mut self.r, …)` borrows disjoint fields and compiles
directly.

### m5. `Operator`'s public fields carry unenforced invariants; malformed variants panic without a documented contract

`src/operator.rs:21-43`: users can construct `Dense` (non-square),
`PerNeuron` (`blocks.nrows() != k²`), or `LowRank` (`p.ncols() != a_r.nrows()`)
directly; `apply` then panics deep in faer indexing (or, for a wide `Dense`,
silently ignores trailing columns since the loop runs to `nrows`). Either add
smart constructors + `debug_assert`s in `apply_impl`, or document the shape
invariants and a `# Panics` section on the enum.

### m6. SHD loader trusts channel ids it never validates

`src/data/shd.rs:82-87` casts `i64 → u32` with `as` (negative values wrap to
huge ids); `load_shd` validates labels (`:60-62`) but never `units < 700`
although the type's doc promises "channel ids (0..700)". `bin_events` then
clamps garbage into the last pooled channel (`:132`), and a negative event time
in `(t / bin_s) as usize` saturates to bin 0. Validate `units` (and
non-negative times) in `load_shd` with a `Dataset` error, mirroring the label
check.

### m7. Zero-neuron `SpikeVec` is constructible and makes `activity()` NaN

`src/spikes.rs:27-32` accepts `SpikeVec::new(0)` while `SpikeBatch::zeros`
rejects zero dims (`:113-122`); `activity()` (`:73-75`) then returns NaN.
Either reject 0 in `new`/`from_indices` for consistency, or document
`activity()`'s NaN case.

### m8. `SurrogateConfig::rank: Option<RankPolicy>` is a double optional

`src/surrogate.rs:55-62`: `None` means "Fixed(d+1)" while
`Some(RankPolicy::Auto)` means the energy heuristic the same doc says is "not
to be trusted here". The distinction is documented but easy to misread; a
dedicated three-variant enum (`FullLifted | Fixed(n) | Auto`) would make the
dangerous option explicit. Related: `SnapshotSet::from_parts`
(`src/identify/snapshots.rs:46`) skips the column-count check entirely when
`u.nrows() == 0`, so a `0 × 17` control matrix pairs silently with 5 snapshot
columns.

### m9. Root re-export surface is inconsistent

`src/lib.rs:43-58` re-exports nearly everything a user touches, but not
`identify::IdentificationReport` (a public field type of the re-exported
`IdentifiedOperator`/`IdentifiedControlled`), nor anything from `metrics`,
`encoding`, or `data` (`spike_coincidence` backs a documented phase gate;
`PoissonEncoder`/`PoissonPatternTask` appear in the training test's public
workflow). Add `IdentificationReport` at minimum; consider the metrics trio.
Also several pub getters (`KoopmanLayer::n_neurons/n_inputs/n_state_vars/operator`,
`Network::n_layers/layer/state/batch`) have no doc comments — the only
undocumented pub items in an otherwise thoroughly documented crate; a
`#![warn(missing_docs)]` would lock the standard in.
