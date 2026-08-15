# Skeptic Review: Koopman-DMD SNN Premise

**Reviewer:** Skeptic subagent
**Date:** 2026-08-14
**Inputs reviewed:** `/Users/jimharris/Documents/kdmd-SNN/SNN-project.md` (premise); `koopman-dmd` v0.1.0 source (`lib.rs`, `dmd.rs`, `types.rs`, `lifting.rs`, plus grep of the full crate).

This review is adversarial by design. Each concern gives (a) the technical argument, (b) the impact on the project, and (c) the mitigation or experiment that would resolve it. The premise is not unsalvageable — but as written, its central claims are either false for the stated neuron model (LIF) or unsupported by the dependency it builds on. There is a defensible, narrower project inside this one; the concerns below mark the boundary.

---

## CRITICAL

### C1. For standard LIF, the premise solves a problem that does not exist

**Argument.** The premise's core pitch — "extract a finite-dimensional linear operator from the non-linear SNN dynamics, drastically simplifying both forward inference and backpropagation" — fails for its own chosen neuron model. LIF sub-threshold dynamics are *already linear*:

```
dV/dt = -(V - V_rest)/tau + I/C
```

The exact discrete update is known in closed form: `V_{t+1} = alpha * V_t + (1 - alpha) * V_rest + beta * I_t` with `alpha = exp(-dt/tau)`. That is a *diagonal*, *exact*, *analytically known* `A`. DMD applied to LIF trajectories can only (approximately, with estimation noise) recover the matrix you can write down in one line. The only "non-linearity" in LIF is the threshold/reset — which the premise explicitly leaves outside the linear model ("non-linearity explicitly isolated to the thresholding function"). So the Koopman machinery linearizes the part that was never nonlinear, and does nothing for the part that is. The claim that complexity "scales up rapidly" from one neuron to a multi-layer LIF network is also false in the relevant sense: LIF membrane dynamics are uncoupled per neuron; the coupling is entirely through `W s_t`, i.e., through the input term, and cost scales linearly in neurons.

**Impact.** The project's headline justification is void for the model in the pseudocode. Every downstream claim (efficiency, gradients) inherits this: DMD replaces an exact scalar exponential with a fitted matrix, i.e., strictly worse accuracy at strictly higher cost, for plain LIF.

**Mitigation / where DMD genuinely earns its place.** Re-scope the *fitted-operator* part of the library to regimes where the sub-threshold flow is genuinely nonlinear or unknown:

1. **Nonlinear neuron models**: Izhikevich, AdEx, quadratic IF, conductance-based synapses with reversal potentials (`g(t)(V - E_rev)` makes the current state-dependent), adaptation variables. Here no closed-form exponential integrator exists, and a per-neuron EDMD surrogate (small lifted state, ~5–15 observables) replacing an RK4 step is a real, testable value proposition.
2. **Model-order reduction of a *trained, frozen* network** for cheap inference (fit `A` once, post-training — see M2).
3. **Diagnostics**: DMD spectrum of a trained network for stability/timescale analysis (the crate's `dmd_stability`, `dmd_spectrum`, `dmd_dominant_modes` are genuinely useful here, at zero risk).
4. **Black-box dynamics**: identifying dynamics of neuromorphic hardware or recorded biological data where the model is unknown (cf. Brunton et al. 2016, DMD on large-scale neural recordings).

The implementation plan must pick one of these as the primary target and demote plain LIF to a *validation case* (DMD must recover `A ≈ diag(exp(-dt/tau))` — a good unit test, not a product).

### C2. The efficiency claim is inverted: dense DMD is 1–3 orders of magnitude *slower* than plain LIF

**Argument.** Per-layer, per-step costs:

- **Plain LIF**: leak is diagonal → `N` FMAs. Synaptic input `W s_t` is `O(N · k)` where `k` = number of *active* presynaptic spikes (event-driven sparsity; spike rates in practice are a few percent). Total ≈ `N + N·k_active`.
- **Dense DMD `A` (N×N)**, as the pseudocode does (`koopman_A.multiply(potentials)`): `N²` FMAs. For `N = 1024`: ~10⁶ vs ~10³ — **~1000× more work** to replace a diagonal decay.
- **Rank-r truncated** (`x → U^T x`, step `r×r`, project back): `2Nr + r²` per step. But you *cannot stay in the reduced space*, because thresholding and reset need original per-neuron coordinates **every step** (see C4). With `r = 32`, `N = 1024`: ~66K FMAs — still **~66× worse** than the diagonal update. The only regime where reduced stepping wins is against a *dense non-diagonal* baseline `A`, which plain LIF does not have.
- **Crate representation makes it worse**: `DmdResult::a_matrix` is `Vec<Vec<C64>>` — complex-valued (4 real multiplies + 4 adds per element, pointer-chasing nested Vecs, no `faer::Mat`, no BLAS path). Using it as the inner-loop stepping matrix as-is is a non-starter; `A` for a real system should be real (conjugate-pair symmetric), and the crate's `compute_full_a` fallback path (see m4) can even break that symmetry.

**Impact.** "Performant SNN" is the stated goal; the proposed mechanism makes the forward pass dramatically slower than the naive implementation the project must build anyway (to generate training snapshots — see M3). If this ships as designed, the honest benchmark result is a large regression.

**Mitigation.** (1) For per-neuron dynamics, `A` is **block-diagonal** (one small `r×r` block per neuron, identical across a homogeneous population → store *one* block): cost `O(N r_neuron²)` with `r_neuron ≈ 3–10`, which is competitive and is the only architecture consistent with both efficiency and C4. (2) Convert any fitted operator to a real, contiguous `faer::Mat<f64>` before it enters the inner loop. (3) Put a hard benchmark gate in the plan: KDMD forward step must be within ~2× of the plain-LIF Rust baseline per step at equal accuracy, or the fitted-operator inference path is cut.

### C3. Treating resets as "control inputs" breaks the identification: u_t is state feedback, not exogenous — and the crate has no DMDc at all

**Argument.** Two independent problems.

*(a) Closed-loop identification bias.* The premise fits `x_{t+1} = A x_t + B u_t` with `u_t` containing the layer's *own resets*. But the reset is a deterministic function of the state: `u_reset,t = Θ(x_t − V_th) · (something)`. The regressor `u_t` is therefore perfectly correlated with (a nonlinear function of) `x_t`. Least squares on `[X; U]` then cannot separate `A` from `B`: the stacked data matrix is (near-)rank-deficient along the feedback direction, and the estimates of `A` and `B` are individually biased/non-unique even when their combination fits the data. This is the classic closed-loop system-identification problem (direct LS identification under output feedback is biased without an accurate noise model; see Ljung, *System Identification*, ch. 13; and note that DMDc as formulated by Proctor, Brunton & Kutz 2016, SIAM J. Appl. Dyn. Syst., assumes `u_t` is a known *exogenous* input). Incoming spikes from the previous layer are *less* pathological (they're exogenous *to this layer*) but still correlated with the layer's state through shared upstream drive, so `B` estimates inherit correlation bias too.

*(b) No implementation exists.* `grep -rni control` over `koopman-dmd/src` returns nothing. The crate's `dmd()` fits only the unforced `Y ≈ A X`. The premise's central equation `x_{t+1} = A x_t + B u_t` has **no supporting code in the dependency the project is built on**. Feeding forced (spiking-input-driven) trajectories to plain `dmd()` will fold the input response into `A`, corrupting it with the input statistics of the particular dataset used for fitting.

**Impact.** The identified `(A, B)` will be wrong in a data-dependent way; the model will not generalize across input distributions; and the first implementation task is not "use the crate" but "extend the crate with DMDc," which is unplanned work on the critical path.

**Mitigation.** (1) **Do not identify the reset at all.** The reset is *known exactly* — implement it explicitly (hard or soft) outside the linear model, and fit `A` only on snapshot pairs where no reset occurred in the interval (see M3). (2) For input coupling, `B = W` is *known by construction* in the premise's own architecture ("W maps to B") — so don't fit `B` either; subtract the known input contribution (`Y' = Y − B U`) and fit `A` on the residual, which is ordinary DMD and *is* supported by the crate. (3) If genuine DMDc is ever needed (unknown input coupling), implement Proctor et al.'s `G = [A B] = Y [X; U]⁺` in the crate, and identify from data with persistently exciting, exogenous probe inputs, not from closed-loop operation. (4) Add an identifiability test: fit on one input distribution, validate one-step prediction error on a different one.

### C4. Lifted (EDMD) linear evolution is incompatible with per-step threshold/reset — the premise's Koopman story and its own pseudocode contradict each other

**Argument.** The premise invokes EDMD/observables `g(x)` and the crate's value-add is its lifting machinery (`Polynomial`, `Trigonometric`, `Rbf`, `Delay`). But the whole point of lifting — lift once, evolve linearly in observable space — requires the dynamics to stay inside the (approximately Koopman-invariant) subspace spanned by the dictionary. The SNN loop violates this **every step**: you must read out `V` in original coordinates, apply `Θ(V − V_th)`, reset, and then the lifted state is stale (e.g., the `V²` observable no longer equals the square of the reset `V`). So the loop becomes lift → step → project → threshold/reset → **re-lift**, per step, per layer. Re-lifting per step (i) costs more than simulating the original dynamics, (ii) destroys the "purely linear algebraic stepping" claim, and (iii) is mathematically a *different* dynamical system from the one EDMD approximated. Worse, LIF-with-reset is a *hybrid* (piecewise-smooth, impulsive) system: its Koopman operator does not act nicely on the smooth dictionaries the crate provides — smooth observables approximate a jump map with slowly decaying (Gibbs-type) error, so the EDMD residual near threshold is irreducibly large regardless of dictionary size (the crate's smooth liftings — polynomial/trig/RBF — are exactly the wrong basis for a discontinuity). Notably, the premise's own pseudocode quietly abandons the Koopman story: `koopman_A` multiplies raw `potentials` — no observables, no lifting, no projection. Either the pseudocode is the design (then this is plain linear DMD on an already-linear system, i.e., C1) or the EDMD text is the design (then the pseudocode is wrong and the real loop is far more expensive, i.e., C2).

**Impact.** The single biggest conceptual hole. The mathematical framing (infinite-dimensional Koopman, EDMD) and the computational architecture (per-step threshold in state space) cannot both be true as written.

**Mitigation.** Restrict Koopman surrogates to the **continuous flow between spike events**, per neuron: lifted linear stepping *while sub-threshold*, with events handled explicitly in original coordinates and the (small, per-neuron) lifted state re-initialized after each reset. Because the block is per-neuron and small (C2 mitigation), re-lifting after a reset is O(r_neuron) and only happens at that neuron's spikes — this is coherent hybrid-systems practice, and it is a *much* narrower claim than the premise makes. State this architecture explicitly in the plan.

---

## MAJOR

### M1. The gradient claim is false as stated: linearity prevents neither vanishing nor exploding gradients

**Argument.** "The Koopman matrix A provides a stable, mathematically rigorous Jacobian… preventing exploding or vanishing gradients over long time windows." Backpropagating through `T` linear steps multiplies by `A^T` (transpose powers); gradient norms scale as `~ρ(A)^T` (more precisely, bounded by `‖A^T‖`, with transient growth possible for non-normal `A` — and DMD-fitted matrices are typically non-normal). A leaky system fitted by DMD *necessarily* has `ρ(A) = exp(−dt/τ) < 1` — that is what "leaky" means — so gradients **vanish** geometrically over long windows, exactly like any leaky RNN/plain LIF. If the fit ever yields `ρ(A) ≥ 1` (noise, spiking transients — see M3), gradients explode and the forward pass diverges. Linearity per se does neither harm nor good: the deep state-space model literature (S4, Gu et al. 2022; LRU, Orvieto et al. 2023) uses linear recurrences *and still* needs eigenvalues parameterized near the unit circle, careful initialization, and — crucially — a **trainable** `A`. A frozen DMD-fitted `A` has whatever spectrum the physics gave it. Additionally, the "Jacobian is A" claim silently drops the reset path: if resets are in `u_t`, then `∂x_{t+1}/∂x_t = A + B ∂u_t/∂x_t`, and the surrogate-gradient term re-enters *every* step. Ignoring it is a known trick (snnTorch's reset-detach), but that is a modeling choice with bias consequences, not "gradient flows perfectly." Finally, the framing misidentifies the problem BPTT has with SNNs: BPTT does not "fail" because of the sub-threshold dynamics (linear already, in plain LIF); it fails at the threshold — and the fix is the surrogate gradient (Neftci, Mostafa & Zenke 2019), which the premise *also* uses. The Koopman layer adds nothing to the gradient story that plain LIF + surrogate does not already have.

**Impact.** A reviewer or user will falsify this claim with one experiment. If long-horizon credit assignment is a project goal, the proposed mechanism does not deliver it.

**What is actually true (salvageable claim).** Post-DMD, the spectrum of `A` is *known explicitly* (the crate exposes `eigenvalues`, `dmd_stability`, per-mode `growth_rate`/`half_life`). That makes gradient decay **diagnosable and predictable per mode** — you can compute exactly how many steps each mode's gradient survives (`T_half = ln 2 / |ln|λ||`) and report/condition on it. Reframe the claim as *spectral transparency*, not gradient immunity. **Experiment:** log `‖∂L/∂x_t‖` vs `t` for KDMD-SNN and a snnTorch LIF baseline on the same task; the premise predicts a difference, this review predicts none (beyond noise).

### M2. Identification/training circularity: the fitted A is only weight-independent in exactly the regime where it is trivial

**Argument.** `A` is fit from trajectories generated under some weight matrix `W` and input distribution. Training changes `W`. When does that invalidate `A`?

- **Feedforward, per-neuron membrane dynamics, state = own membrane variables only:** the flow `x_{t+1} = A x_t + B (W s_t)` has `W` entering *only* through the input term. `A` is weight-independent. **This genuinely rescues the premise for feedforward nets** — but note the catch: in this regime, for LIF, `A` is the analytically known diagonal decay (C1), so DMD is unnecessary; and for nonlinear neurons the per-neuron `A` is small and shared, which is the C2/C4 architecture. So the rescue works, but it delivers a per-neuron surrogate library, not a network-level Koopman operator.
- **Recurrent layers, or lifted observables that mix units (PolynomialCross, RBF over the joint state), or network-level `A` fit on full-layer trajectories:** the closed-loop dynamics depend on `W`, so `A(W)` is stale after every optimizer step. Refitting requires fresh trajectory collection plus an SVD of an `N×T` matrix (`O(N² T)`) inside the training loop — and the fit is most stale precisely when learning is making the fastest progress (early training), which is when you need gradients to be right. There is also a subtler statistical circularity: even the weight-independent `A` is fit under an input distribution; DMD least squares weights the fit by where the data lives, so `A` is biased toward the operating regime of the *initial* network (firing rates, mean depolarization), which training will shift.

**Impact.** Without this distinction stated up front, the design will drift into fitting network-level operators (it's the natural reading of "record a time-series of network states") and hit a refit-cost wall or silently train against a stale model.

**Mitigation.** The plan must draw this boundary explicitly: (1) *fitted* operators are per-neuron (or per-neuron-type), weight-independent, computed once before training; (2) `W` lives only in the explicit input path `B u_t = W s_t` and is trained normally; (3) network-level DMD is permitted only *post-training* on frozen weights (model-order reduction for inference, or analysis/diagnostics); (4) if online refitting is ever attempted, use a streaming/incremental DMD variant and measure the staleness–accuracy trade-off — do not put full SVD refits in the training loop.

### M3. Fitting A on full spiking trajectories vs sub-threshold segments — the premise conflates two very different estimation problems

**Argument.** Snapshot pairs `(x_t, x_{t+1})` that straddle a reset event teach the least-squares fit a mixture of the smooth leak map and a near-discontinuous jump (`V_th → V_reset`). LS returns the input-distribution-weighted average of these incompatible maps, so `A` is neither the correct sub-threshold operator nor the correct reset map: eigenvalues shift (typically toward faster apparent decay, since resets look like large negative "dynamics"), and prediction is wrong everywhere. Statistically, spike/reset events are impulsive and broadband: the snapshot matrix of a spiking trajectory is *not* low-rank-plus-noise, its singular value spectrum decays slowly, and the crate's default rank rule (99% cumulative variance in `determine_rank`) will either keep a large rank (defeating truncation) or discard the very components that carry spike-timing information — and SNNs compute with spike timing. This is the same reason DMD practitioners pre-process shocks/discontinuities out of fluid data before fitting.

**Impact.** Silent model corruption: everything will run, `dmd()` will return a matrix, and the matrix will be wrong. This is the most likely "it doesn't work and we don't know why" failure mode.

**Mitigation.** (1) A data-conditioning pipeline is a *required deliverable*: excise snapshot pairs whose interval contains a spike/reset of any neuron in the fitted block (trivial when blocks are per-neuron, another argument for C2/C4's architecture; nearly impossible for network-level fits at realistic firing rates — at population rate `f`, the fraction of clean network snapshots is `≈ exp(−N f dt)`, which vanishes for large `N`). (2) Subtract the known input contribution before fitting (C3 mitigation). (3) Acceptance tests using the crate's own analysis tools: `dmd_error`/`dmd_residual` on held-out sub-threshold data, and the ground-truth check that for LIF the fit recovers `λ ≈ exp(−dt/τ)` within tolerance. (4) `ρ(A) < 1` asserted at fit time via `dmd_stability` before any operator is accepted for inference or BPTT.

### M4. Evaluation and ecosystem risk: no defined benchmark can currently demonstrate the claimed benefit — and the strongest baseline is a subproject of this project

**Argument.** To fit `A` you need membrane-potential trajectories, which come from… a plain LIF/nonlinear-neuron simulator that the project must build first. That simulator *is* the baseline, and per C1/C2 it is simpler, exact (for LIF), and faster than the KDMD path. So the burden of proof is unusually concrete: on what task, against that in-house baseline plus an established framework (snnTorch/Norse on CPU for fairness, given no GPU story), does KDMD win, on what metric? The premise names none. Meanwhile the training claim requires BPTT with surrogate gradients *in Rust*: there is no mature autograd here — either hand-roll reverse-mode through the custom recurrence (substantial, error-prone work that the premise doesn't mention) or take a `burn`/`candle` dependency and integrate a custom recurrent op with surrogate gradients (also real work). Established frameworks get this for free from PyTorch. Single developer, `koopman-dmd` at v0.1.0 (unstable API, `Vec<Vec<C64>>` interfaces, hand-rolled complex solves), no GPU story vs. CUDA-backed competitors and Lava on neuromorphic hardware: the systemic risk is a large engineering spend with no demonstrable advantage at the end.

**Impact.** Without pre-registered success criteria, the project cannot fail visibly — it will just accumulate code. That is the worst outcome for a research library.

**Mitigation.** The implementation plan must pre-register: (1) tasks (e.g., SHD or N-MNIST classification; a long-horizon temporal-credit task where the gradient claim would matter if true; a nonlinear-neuron simulation-accuracy/speed task where DMD *should* win per C1's re-scope); (2) baselines (in-house plain LIF in Rust; snnTorch LIF, CPU); (3) metrics (accuracy, wall-clock per step, fit quality on held-out data, gradient-norm decay curves); (4) kill criteria — if the fitted-operator inference path loses to plain LIF at equal accuracy, it is demoted, and the library's value proposition becomes nonlinear-neuron surrogates + post-hoc DMD analysis/MOR, which per C1 is where the defensible product lives anyway. Milestone 1 should be the plain baseline + the LIF-recovery validation test, before any KDMD forward path is built.

### M5. The premise's mathematics never actually produces B, and conflates the reduced and full operators

**Argument.** The text says "We solve for both the system dynamics matrix A and the control/input matrix B" — but the only formula given is plain unforced DMD (`Ã = UᵀYVΣ⁻¹`, correct as far as it goes for real data, matching `dmd.rs` step 3). DMDc requires the augmented regression `[A B] = Y [X; U]⁺` with its own SVD structure (Proctor et al. 2016); none of that appears in the premise or the crate (C3b). Separately, `Ã` is the **r×r reduced** operator (projected onto POD coordinates), while the pseudocode's `koopman_A` multiplies the **N-dimensional** potential vector — these are different objects; using the full reconstructed `A = ΦΛΦ⁺` is the O(N²), complex-valued object of C2. The premise never says which one the layer stores, and the answer determines the entire performance profile.

**Impact.** Whoever implements from this document will discover mid-build that the key quantity (`B`) has no estimator and the key matrix (`A`) has two incompatible candidate definitions.

**Mitigation.** The plan must specify: `B := W` known by construction, never estimated (C3); the stored operator is either the small per-neuron block (preferred) or the pair `(Uᵀ, Ã, U)` for reduced stepping — with the projection-cost caveat of C2 — and it must be converted to real `faer::Mat<f64>` storage.

---

## MINOR

### m1. Notation and typos
- "Jacobin" (line 57) → "Jacobian". Cosmetic, but it is in the load-bearing sentence of the training claim.
- `Ã = UᵀYVΣ⁻¹` should in general be `U*` (conjugate transpose); harmless for real snapshot data, worth writing correctly in docs.

### m2. Hard reset vs "reset penalty via control input" — the text and pseudocode disagree, and the difference has gradient consequences
The text (§Forward Pass) says the reset is "applied via the control input"; the pseudocode does an in-place hard assignment `next_potentials[i] = reset_val`. These are different models: hard reset destroys the pre-spike potential (and, in BPTT, truncates the gradient path through the membrane state at every spike unless a surrogate/detach policy is chosen), while soft/subtractive reset (`V ← V − V_th`, expressible as `−V_th · s_t` through `B u_t`) preserves residual charge and keeps a differentiable path. The SNN literature treats this as a first-class design choice (it measurably changes accuracy and gradient behavior). Pick one, state the gradient policy at the reset, and make the other a config option.

### m3. Pseudocode nits
- `surrogate_gradient` (fast-sigmoid derivative, `1/(1+k|v−θ|)²`) is standard and fine — but nothing in the forward pass or structures connects to it; there is no tape/graph. This is where the hidden autograd dependency (M4) surfaces.
- `.pow(2)` on `f64` isn't Rust (`powi(2)`); trivial, but signals the pseudocode was not checked against the language.
- The premise says spikes *and* resets form `u_t`, but the pseudocode's forcing term is only `W · input_spikes`; the reset never appears in `u_t`. Consistent with the C3 mitigation (good), inconsistent with the text (bad).

### m4. Crate-quality items that will bite this project specifically (koopman-dmd v0.1.0)
- `DmdResult.a_matrix: Vec<Vec<C64>>` — complex nested-Vec for a matrix that is mathematically real for real data; must be converted before inner-loop use (C2).
- `compute_full_a`'s pseudo-inverse fallback (on singular Gram) normalizes *every* column by `‖mode₀‖²` (`modes[k][0]` only) — if that path ever triggers, the reconstructed `A` is silently wrong. Should be an error, not a fallback.
- Dead `if config.center {…} else {…}` around `x0` in `dmd.rs` (both branches identical) — suggests the centered-amplitude path wasn't finished; centering + prediction round-trips deserve a test before this project relies on them.
- Automatic rank via 99% cumulative variance is a poor default for spiking data (M3); the SNN library must always pass explicit rank and validate with `dmd_residual`.
- v0.1.0: no API stability guarantee; the SNN crate should pin the version and wrap the DMD interface behind its own trait so crate churn doesn't propagate.

### m5. Scope
Single developer, two-crate stack, no GPU story, and a task list that also spawns six subagent roles: the coordination overhead is nontrivial. Nothing here is fatal, but the milestone plan should be ruthlessly serial: baseline → validation → one narrow KDMD claim → benchmark, before any breadth (multiple neuron models, multiple lifting dictionaries, analysis suites).

---

## What the implementation plan MUST address

**Scope and claims**
- [ ] Re-scope the fitted-operator value proposition away from plain LIF: pick the primary target (nonlinear neuron surrogates / post-training MOR / diagnostics) and state it (C1).
- [ ] Demote plain LIF to a validation case with the explicit acceptance test: DMD on sub-threshold LIF data recovers `λ ≈ exp(−dt/τ)` (C1, M3).
- [ ] Rewrite the gradient claim as "spectral transparency" (known, diagnosable decay per mode), not prevention of vanishing/exploding gradients; state the Jacobian including the reset path and the chosen detach policy (M1, m2).

**Architecture**
- [ ] Specify the operator layout: per-neuron (block-diagonal, shared across homogeneous populations) fitted operators; no network-level fitted `A` in the training path (C2, C4, M2).
- [ ] Specify the hybrid stepping scheme: lifted linear evolution only between spike events; explicit threshold/reset in original coordinates; lifted-state re-initialization after reset (C4).
- [ ] `B := W` by construction — never estimated; input contribution subtracted before any fit (C3, M5).
- [ ] Decide and document hard vs soft reset, and its gradient policy (m2).
- [ ] All inner-loop operators stored as real `faer::Mat<f64>` (or scalar blocks); no `Vec<Vec<C64>>` in the hot path (C2, m4).
- [ ] Name the autograd strategy for BPTT in Rust (hand-rolled reverse-mode vs `burn`/`candle` integration) with a cost estimate (M4, m3).

**Identification pipeline**
- [ ] Data conditioning: excise snapshot pairs straddling reset events; fit only sub-threshold segments (M3, C3).
- [ ] Explicit rank selection + held-out validation via `dmd_error`/`dmd_residual`; reject fits with `ρ(A) ≥ 1` via `dmd_stability` (M3, M1).
- [ ] Cross-input-distribution generalization test for any fitted operator (C3).
- [ ] If DMDc is ever truly needed, it is a new `koopman-dmd` feature (Proctor et al. formulation) with exogenous excitation — scheduled as such, not assumed (C3).
- [ ] No operator refits inside the training loop; network-level DMD only on frozen post-training weights (M2).

**Evaluation (pre-registered)**
- [ ] Milestone 1 = plain LIF Rust baseline + LIF-recovery validation, before any KDMD forward path (M4).
- [ ] Benchmarks: named tasks (e.g., SHD/N-MNIST + one long-horizon temporal task + one nonlinear-neuron accuracy/speed task), named baselines (in-house LIF, snnTorch CPU), named metrics (accuracy, wall-clock/step, held-out fit error, gradient-norm curves) (M4).
- [ ] Kill criterion: if KDMD inference loses to plain LIF at equal accuracy, the fitted-operator inference path is demoted and the library pivots to surrogate-modeling + analysis (M4).

**Hygiene**
- [ ] Fix premise doc: "Jacobin" typo, `U*` notation, reconcile reset text vs pseudocode, add the missing `B` estimator discussion or remove the DMDc claim (m1, m2, M5).
- [ ] Upstream fixes or workarounds for `koopman-dmd`: `compute_full_a` fallback made an error; centered-amplitude path tested; SNN crate pins the version and wraps DMD behind its own trait (m4).
