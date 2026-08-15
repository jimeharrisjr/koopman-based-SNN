# Skeptic Final Review — End-of-Project Systemic Audit

**Document:** 12-skeptic-final-review.md · **Author:** Skeptic subagent · **Date:** 2026-08-15
**Mandate:** the plan's final gate — "review the entire project skeptically to determine and
document any systemic or reasoning weaknesses which must be addressed" before any publish.
**Method:** re-read README, IMPLEMENTATION_PLAN, docs/01–09; read every exit-gate test
(`crates/kdmd-snn/tests/*.rs`), the bench suite, `lib.rs` and module docs, `Cargo.toml`/lockfile,
`.github/workflows/ci.yml`; re-ran `cargo test --workspace` (green, 97 s dominated by
`reduced_order`) and the koopman-dmd `dmdc`-branch suite (green); inspected both git repositories.

**Bottom line.** The scientific record (docs/04–08) is unusually honest: two pre-registered
experiments failed and were documented as failures. But the *engineering shell around that record
does not meet the standard the record sets*. The README's claim table contains six identifiable
overclaims, the rustdoc surface still asserts the value case the experiments refuted, the
pre-registrations are "frozen" only by convention because **the repository has zero commits**, CI
has never run and cannot pass as written, and the V2b experiment — unlike V2 — was never audited.
None of this is fatal; all of it is fixable in hours; publishing without fixing it would convert an
honest project into an overclaiming one at the moment of maximum visibility.

---

## Part I — Audit of the original demands (docs/03 "What the implementation plan MUST address")

| # | Demand (docs/03) | Verdict | Evidence |
|---|---|---|---|
| 1 | Re-scope fitted-operator value away from plain LIF; pick primary target | **Honored, then overtaken** | Plan adopted V1–V4, nonlinear-first (Q2). V2 then failed twice; the shipped centerpiece is now the exact-linear LIF engine — the object C1 called "a validation case, not a product." The pivot was per protocol (Q8), but see S3 below: the consequence is not stated plainly anywhere. |
| 2 | Demote plain LIF to a validation case with the λ ≈ exp(−dt/τ) acceptance test | **Honored** | `tests/oracle.rs`: ≤ 1e-8 propagator recovery, τ_m/τ_s recovery to 1e-6, rank-collapse identifiability test, stability-rejection test. Good work. |
| 3 | Rewrite gradient claim as spectral transparency; state Jacobian incl. reset path + detach policy | **Half-honored** | Reset-path gradient is exact and documented (`train/mod.rs`: "no detach trick") — better than demanded. But the V4 *experiment* (gradient-norm curves vs plain-LIF baseline, a named Phase 6 exit criterion) **never ran**: no gradient-norm logging exists anywhere (`StepStats` = loss + accuracy only). See Part II, F4. |
| 4 | Per-neuron operator layout; no network-level fitted A in the training path | **Honored** | `Operator::PerVariable`; `single_trajectory_cannot_identify_a_network_operator` documents the trap; no refit machinery exists in the trainer at all. |
| 5 | Hybrid stepping scheme for lifted operators | **Honored, now dead code** | Implemented in `src/surrogate.rs` exactly as demanded (C4 resolution) — for a method that failed its gate. Ships as public API with no failure notice (F2). |
| 6 | `B := W` known by construction, never estimated in the training path | **Honored** | `lif_structural_b`; joint (A,B) fit exists only as a consistency check (`dmdc_layer.rs`), recovering the −θ reset columns. Clean. |
| 7 | Hard vs soft reset decided, with gradient policy | **Honored** | Subtractive canonical (Q5); hard reset in reference sim with masked identification *and* a control experiment showing the unmasked fit corrupts (`hard_reset_requires_masked_subthreshold_fitting`) — one of the best tests in the repo. |
| 8 | Real `Mat<f64>` in the hot path | **Honored** | `operator.rs` extracts once at identification time; upstream `DmdcResult` is real-typed. |
| 9 | Autograd strategy named with cost estimate | **Honored** | Hand-rolled BPTT (Q6); FD check on the readout with the piecewise-constancy limitation honestly documented. |
| 10 | Data conditioning: excise reset-straddling pairs | **Honored and exceeded** | The subtractive-reset-as-control formulation makes masking unnecessary for the canonical path (fit on *full* spiking trajectories at 1e-8 — a genuinely elegant result); masking retained and tested for hard reset. |
| 11 | Explicit rank + held-out validation; reject ρ(A) ≥ 1 | **Honored, with a scar** | `identify/validate.rs` + `growing_dynamics_are_rejected_by_the_stability_gate`. The scar: docs/06 F2 showed this LIF-era gate, transplanted into V2's pre-registration, made PASS unreachable — a protocol-design failure the project itself caught and recorded. |
| 12 | Cross-input-distribution generalization test | **NOT delivered** | The demand was: fit on one input distribution, validate on a *different* one. `dmdc_layer.rs` held-out data is the same sinusoid family (0.8× scale, +13 phase); `reduced_order.rs` held-out is the same 10 % Bernoulli process. Benign for exact-linear LIF (A is input-independent), **material for V1**, whose reduced basis is input-distribution-dependent by construction. See F7. |
| 13 | DMDc as a scheduled koopman-dmd feature | **Honored, unshipped** | `dmdc` branch exists, suite green locally, both upstream bug fixes included (commit 59231c0). But: one commit, **never pushed** (no remote `dmdc` branch), no PR, no second pair of eyes, not on crates.io. See F1. |
| 14 | No operator refits inside the training loop | **Honored** | Trivially — the machinery was never built (the lifted path it existed for died with V2). |
| 15 | Milestone 1 = baseline + LIF-recovery before any KDMD path | **Honored** | Phase ordering followed the demand. |
| 16 | Pre-registered benchmarks: named tasks, baselines, metrics | **Half-delivered** | Delivered: Poisson-pattern task (gated ≥ 90 %), SHD demo (built, data downloaded, **no accuracy recorded anywhere in the repo**), kill-criterion bench (docs/09). Silently dropped: **temporal XOR** (named in the plan, no trace in `data.rs`), **snnTorch CPU external baseline** (named in the plan and in my checklist; never run), **gradient-norm curves** (F4). Roughly half the registered evaluation matrix executed, with no deviation note — in a project that scrupulously logs experimental deviations, infrastructure deviations went unrecorded (F5). |
| 17 | Kill criterion pre-registered and enforced | **Honored, exemplary** | Owner accepted (Q8); executed literally (docs/09): dense fitted path measured at 274×, demoted exactly as C2 predicted. This is the process working. |
| 18 | Fix premise doc (Jacobin typo, gradient claim, reset inconsistency) | **NOT done** | `SNN-project.md` line 57 still reads "Jacobin" and still claims A "prevent[s] exploding or vanishing gradients" — a claim this project *refuted*. The repo ships its own refuted premise with no errata pointer. |
| 19 | Upstream bug fixes; pin versions; wrap DMD behind own trait | **Mostly honored** | Both bugs fixed on the branch; faer pinned. Leak: `koopman_dmd` types (`ModeInfo`, `DmdResult` in `dense_from_dmd`) sit in kdmd-snn's public API, so upstream churn still propagates. Acceptable given the version pin; note it. |

**Score: 12 honored (several exceeded), 4 partially, 3 not delivered (12, 16-partial, 18).**
The dropped items cluster tellingly: everything dropped is *external calibration* — the different
input distribution, the external baseline, the gradient-norm comparison, the premise correction.
Everything the project could grade on its own terms was done; everything that would grade it
against the outside world was deferred.

---

## Part II — Systemic findings, severity-ordered

### F1 (CRITICAL): There is no version-control history. The pre-registrations are frozen by narrative, not by evidence — and CI has never run and cannot pass.

- `git log` on kdmd-SNN: **"your current branch 'main' does not have any commits yet."** Every
  file — code, plan, both "frozen" pre-registrations, both results docs — is an untracked working-tree
  file. There is no commit hash proving docs/04 predates docs/05, or docs/07 predates docs/08. The
  entire epistemic architecture of this project (frozen thresholds, tighten-only amendments,
  predictions registered "before data") currently rests on file-modification timestamps and trust.
  For a repository whose *stated contribution* is its pre-registration discipline, this is the single
  worst finding: **the discipline is real but unverifiable.**
- `.github/workflows/ci.yml` has therefore never executed. Worse, it cannot pass as written: it
  strips `[patch.crates-io]` and then builds against crates.io, where `koopman-dmd = "0.2"` does
  **not exist** (only 0.1.0 is published; 0.2.0 lives on the local, unpushed `dmdc` branch).
  Dependency resolution fails before a single test compiles. The file's own comment says "revisit
  in Phase 3 — while kdmd-snn depends on unreleased dmdc work…"; Phase 3 came and went; nobody
  revisited. Every README "✅ gated by tests" claim is gated only by `cargo test` on one laptop.
- The koopman-dmd `dmdc` branch: one commit (59231c0) containing the DMDc solver *and* both
  upstream bug fixes, unpushed, un-PR'd, reviewed by nobody but its author. The README says
  "to be released as 0.2.0" — until that release, `kdmd-snn` is unbuildable by anyone else on earth.

### F2 (CRITICAL): The claim surface contradicts the negative results it advertises. Six precise overclaims (S1).

1. **`lib.rs` crate docs (the rustdoc front page) still assert the refuted claim.** Lines 18–20:
   "The fitted-operator machinery **earns its place on nonlinear neuron models (Izhikevich, AdEx)
   via EDMD lifting**…" — written before Phase 4, never updated. The project's own docs/05/06/08
   prove this sentence false as stated. Similarly `src/surrogate.rs`'s module doc ends "whether the
   surrogate meets the … exit criteria **is decided by** the V2 experiment" — the experiment decided
   it: FAIL — and neither `surrogate.rs` nor `return_map.rs` records the verdict. Both failed-method
   modules are re-exported at crate root (`IzhikevichSurrogate`, `IzhikevichReturnMap`) at the same
   API prominence as the supported engine, with no deprecation, no failure notice. A rustdoc-first
   user will discover the negative results only if they read the README's table.
2. **"LIF/adLIF layers … with spike-for-spike agreement against the reference simulator"** — the
   spike-for-spike equivalence gate (`layer_equivalence.rs`) exists **only for LIF**. adLIF has four
   unit tests inside `neuron/adlif.rs` (analytic checks) and appears in *zero* integration tests;
   there is not even a `KoopmanLayer::adlif` constructor. The adLIF half of the headline claim is
   ungated.
3. **"gated by tests + benches"** — criterion benches cannot gate anything; they have no
   assertions. The 1.02× number is one measurement, on one machine (Apple Silicon, single thread,
   N = 1024, 8 % activity), read once by a human and recorded in docs/09. No automated check would
   catch a regression to 5×. And the equivalence test itself is one regime: one weight distribution
   (U[−0.2, 0.8]), one activity level (~8 %), one seed, batch = 1, `SpikeVec` path only — the
   `step_batch` path has no equivalence test at all (it is validated only implicitly through
   training behavior). The identity claim is strong *by construction* (same accumulation order,
   documented); the *gate breadth* is one point.
4. **V1 "matched sub-threshold accuracy"** — the actual tolerances in `tests/reduced_order.rs` are
   **≤ 5 % one-step and ≤ 10 % free-rollout relative RMSE**. Ten percent state error is not
   "matched" in any reader's natural sense. Moreover the test runs at θ = 1e12 and *asserts the
   layer never spikes*: the sole surviving fitted-operator value case of a spiking-network library
   is validated exclusively in a regime with no spikes, and the behavior of a `LowRank` layer at
   threshold (the reset direction −θ·e_v generally lies outside span(P)) is unspecified and
   untested. "Sub-threshold" in the README row is technically present and easy to read past. The
   plan's original V1 falsifier — "`inference_e2e` bench vs dense recurrent baseline at matched
   accuracy" — was replaced by an in-test 50-repetition `Instant` timing comparison (flake-prone)
   plus RMSE tolerances; the bench has no recurrent/LowRank case.
5. **"learns synthetic tasks to ≥ 90 %"** — plural. Exactly one synthetic task exists
   (`PoissonPatternTask`, 2 classes, 50 % chance). Temporal XOR, named in the plan's pre-registered
   evaluation, was never built. And "runs the SHD demo": the data was downloaded and the example
   compiles, but **no SHD accuracy is recorded anywhere in the repository** — a claims-table row
   whose evidence is an unrecorded run.
6. **"Both negative results were … adversarial audits (docs/06)"** — docs/06 audits **V2 only**.
   V2b has had no audit of any kind (see F3). The sentence as written claims audit coverage the
   record does not contain.

Also in scope for S1: the DMDc rows are accurate for what is tested, but all identification tests
run on noiseless f64 trajectories generated by the very simulator being identified, at N = 3.
"Recovers the propagator from data" is true of oracle data only; no test adds noise or realistic
conditions. The claim should say so.

### F3 (HIGH): Audit asymmetry — the second failure was never attacked (S2, S5).

- V2's FAIL got a full adversarial audit (docs/06) that found a real protocol defect (F2: an
  unpassable precondition) and independently re-derived the failure arithmetic. Genuinely excellent.
- V2b's FAIL got **nothing**. The asymmetry has a direction: auditing V2 could have *rescued* the
  method (and did produce V2b); auditing V2b could only disturb the by-then-decided pivot. A
  hostile reviewer will say the project audited the failure it wanted to overturn and skipped the
  one it wanted to accept.
- The V2b positive that most needs adversarial calibration is the one the results doc celebrates:
  **perfect I = 10 timing**. Context the doc omits: I = 10 is one of *two* gated interior currents,
  bracketed by training currents 9 and 11 (a ±10 % bracket, on the flattest part of T(I), farthest
  from the saddle-node singularity at I = 4). The other interior current, I = 6, **failed** (0.714
  vs 0.80). "Event-level return-map regression achieves perfect ±2 ms reproduction *inside the
  training current envelope*" (docs/08's consequence paragraph) generalizes from the easier one of
  two interior points while the harder one missed. The honest statement is narrower: *perfect at
  the interpolation point far from rheobase; failed at the one near it.* To docs/08's credit, the
  per-current numbers are all printed and the Γ statistic (registered after docs/06 F6.2) makes the
  FS 1.000 meaningful against a 0.78 chance floor — the raw material for the correction is all
  there; the summary sentence just outruns it.
- S5 generally: the same mind (the orchestrator) designed the method, wrote the harness, wrote both
  pre-registrations, ran the experiments, and evaluated the gates — twice in one day. The existing
  independent checks are real but thin: frozen protocol documents (unverifiable — F1), shipped raw
  outputs (docs/05-raw, 08-raw), shipped harnesses (`examples/v2_rollout.rs`, `v2b_return_map.rs`),
  and one genuinely adversarial audit with independent re-derivation. What does *not* exist:
  commit-hash provenance for the freeze, any independent reimplementation of even one metric, any
  run on a second machine, any human review of the upstream DMDc math. The one protocol the skeptic
  audited turned out to contain an unpassable gate (docs/06 F2) — the base rate for protocol defects
  in this project is therefore 1 for 1 audited, and the second protocol is unaudited.

### F4 (HIGH): V4 is vaporware that the pivot documents still sell (S4).

Every pivot statement — plan Phase 4 outcome note, docs/05, docs/07 §6, docs/08 — names the
surviving value cases as "**V1/V3/V4**." V4 (spectral transparency: gradient-norm curves vs a
plain-LIF baseline, the restatement of the premise's falsified gradient claim) was a named Phase 6
exit criterion: "V4 experiment (gradient-norm curves vs plain-LIF baseline) logged." **It never
happened.** There is no gradient-norm logging in the trainer (`StepStats` carries loss and accuracy
only), no experiment, no doc. The README quietly omits V4 from the claim table — the *omission* is
honest, but the repository now contains four documents advertising a pivot to a value case that was
never built, and no deviation note anywhere records the drop. V3's status is only marginally
better: "shipped" means the fit report passes through koopman-dmd's spectrum/stability/timescales
(exercised by exactly one happy-path oracle test recovering two known τ's from a 2-mode LIF).
There is no diagnostics example on a trained network, no per-mode gradient-depth (`T_half`)
reporting — the feature the V4 story needed. V3 as shipped is a re-export with a report field, and
the claim table row should not imply more.

### F5 (MEDIUM): The honest positioning sentence exists nowhere (S3).

The Phase 5 gate showed the product's fast path is **1.02× the reference simulator** — i.e., for
homogeneous LIF, the Koopman machinery adds *approximately nothing* over a hand-rolled LIF loop,
exactly as docs/03 C1/C2 predicted before a line was written. docs/09 comes closest ("at parity
cost, with the layer abstraction, batching, and the training seam on top") but no document says the
plain version: **this library's value for the homogeneous case is software engineering (a tested
abstraction with an identification story and exact training Jacobians), not performance; and its
scientific value is the negative results and the upstream DMDc contribution, not the engine.**
Per-audience honest answers, which the README should state rather than imply:
- *Someone who wants to train SNNs* → snnTorch/Norse (GPU, ecosystem, more neuron models). The
  pre-registered snnTorch baseline that would have quantified this was silently dropped (Part I #16).
- *Someone who wants a fast Rust LIF sim* → a hand-rolled loop is 1.02× — i.e., the same.
- *Someone who wants DMD spectral analysis* → koopman-dmd directly.
- *The real user*: someone who wants the linear-plus-control formulation itself — exact
  identification, exact linear Jacobians, reduced-order recurrent layers, in Rust — a research
  artifact for a research niche. That phrase ("research artifact, not a competitive framework")
  appears nowhere and should appear in the README.
Also unmet under S3: the Q4 scale promise ("10k routine / 100k supported") was never exercised —
benches stop at N = 4096 (kernel) / 1024 (e2e); no test or bench touches 10k, let alone 100k.

### F6 (MEDIUM): Negative-result hygiene is genuinely good — with two wording debts (S2).

Credit where due: docs/05 and docs/08 state their failures as scoped facts ("one-step EDMD fitting
cannot deliver ±2 ms phase over 1000 ms of tonic Izhikevich firing at Δt ≥ 0.5 ms"; "this method,
this protocol"), publish raw outputs, record deviations under a tighten-only policy, and score
their own registered predictions including the refuted ones. The README's two negative rows are
scoped correctly. The two debts: (a) the docs/08 "training envelope" summary sentence (F3 above);
(b) the README's plural "audits" (F2.6). Fix both and S2 is fully clean.

### F7 (MEDIUM): The V1 gate's missing generalization test is the one that matters scientifically.

Unlike LIF's input-independent A, the V1 reduced basis P is estimated *from the training drive's
excitation pattern*. The held-out trajectory in `reduced_order.rs` is drawn from the identical
process (10 % Bernoulli on the same 8 channels, same RNG stream). A drive with different rate,
correlation, or channel weighting would exercise state-space directions the basis truncated — this
is precisely the C3/M2 "biased toward the operating regime" failure mode my original review flagged,
the checklist item (#12) written to prevent it was not delivered, and the one place it could bite
in the shipped library is the one place it is untested.

### F8 (LOW): Hygiene residue.

- `SNN-project.md` uncorrected in-repo (Part I #18): still says "Jacobin," still claims gradient
  prevention. Add an errata header pointing at docs/03 and the results, or move it to docs/00 with
  a "historical premise — superseded" banner.
- **Zero doctests in the crate** (`Doc-tests kdmd_snn: running 0 tests`). The README quick-start
  is a ```rust block with doctest-style hidden `Ok::<…>` lines that *looks* compile-checked and is
  not — no `#[doc = include_str!("README.md")]` exists (upstream koopman-dmd, by contrast, runs
  its README as doctests). The plan's Phase 6 exit named "README doctests." Unmet.
- `Operator::PerNeuron` (the heterogeneous-τ path) is unit-tested in `operator.rs` but reaches no
  integration test and no layer constructor exercises it.
- koopman-dmd types (`ModeInfo`, `DmdResult`) in kdmd-snn's public API (Part I #19).
- docs numbering: 01–09 then this document as 12; if 10/11 (code-quality, documentarian sign-off —
  both promised in the plan's team table) are pending, they are pending; no sign-off exists in
  `docs/` as of this review.

---

## Part III — What must be said in the project's favor

A skeptical review that only lists faults would misrepresent this project. The kill criterion was
pre-registered, owner-signed, and *executed against the project's own hopes* — twice for V2 and
once for the dense path. The oracle/masking/identifiability tests encode real failure modes as
permanent regressions (the unmasked-corruption control experiment and the rank-collapse test are
better than most published code). The subtractive-reset-as-control formulation validated on full
spiking trajectories at 1e-8 is a clean, small, real result. The B-never-fitted discipline held
everywhere. The V2 audit (docs/06) independently re-derived the failure arithmetic and caught a
protocol defect. The failure modes docs/03 predicted (C1, C2, M3) are the ones that occurred, and
the project *recorded that* rather than burying it. The weaknesses above are almost all in the
claim-wording and infrastructure shell, not in the science or the code.

---

## MUST-FIX-BEFORE-PUBLISH

Ordered; 1–3 are blocking for *any* public artifact, 4–9 for the README/crate claims as worded.

1. **Create the git history and push it.** Commit the repo (ideally: pre-registrations and results
   in commits whose ordering reflects the actual sequence, with a note that initial development was
   uncommitted — do not fabricate retroactive dates). Push koopman-dmd's `dmdc` branch, open the
   PR, and publish 0.2.0 **before** publishing kdmd-snn; until then `kdmd-snn` is unbuildable by
   anyone else and `koopman-dmd = "0.2"` is a phantom dependency.
2. **Make CI real.** As written it fails at dependency resolution (crates.io has no 0.2). Use a git
   dependency on the `dmdc` branch (or publish first), then get one green run — no "gated by tests"
   claim before CI has ever been green.
3. **Fix the rustdoc surface to match the negative results:** rewrite the `lib.rs` "earns its
   place on nonlinear neuron models via EDMD lifting" sentence; add FAIL-verdict notices (with
   docs/05/08 links) to the `surrogate` and `return_map` module docs and their crate-root
   re-exports (or gate them behind an `experiments` feature).
4. **Correct the README claim table:** restrict the spike-for-spike row to LIF (or add the adLIF
   equivalence test and a `KoopmanLayer::adlif` constructor); replace "gated by tests + benches"
   with wording that separates the test-gated equivalence from the once-measured 1.02×
   (machine/config stated); change V1's "matched sub-threshold accuracy" to the actual numbers
   ("≤ 5 % one-step / ≤ 10 % rollout relative state error, non-spiking regime"); singular "task";
   record the SHD demo's actual accuracy or delete "runs the SHD demo" from the claims row;
   fix "adversarial audits (docs/06)" → "an adversarial audit of the first (docs/06)".
5. **Resolve V4 one way or the other:** either run the gradient-norm experiment (the logging is
   ~20 lines in the trainer) and file the result, or excise V4 from the pivot language in the plan
   and docs and record the drop as a deviation. Do not publish documents advertising a pivot to an
   unbuilt value case.
6. **Add the positioning paragraph** (README): research artifact, not a competitive framework;
   fast path is parity with a hand-rolled LIF sim by design; who should use snnTorch/Norse or
   koopman-dmd directly instead; the negative results and the DMDc contribution are first-class
   outputs. State that the 10k/100k scale target was not benchmarked, or bench it.
7. **Audit V2b or label it unaudited.** Either commission the docs/06-style audit (its first
   targets: the I = 10 vs I = 6 asymmetry, the envelope-claim wording, the P-A collinearity as a
   protocol defect), or state explicitly in README and docs/08 that V2b's FAIL was accepted
   without adversarial audit because it confirmed the registered pivot. Fix the docs/08 summary
   sentence to "perfect at the interpolation current away from rheobase (I = 10); failed near it
   (I = 6)".
8. **Add the cross-input-distribution held-out test for V1** (fit under the Bernoulli drive,
   validate under a different rate/correlation pattern; report the degradation) — checklist item
   #12, still owed, and the only owed item that can silently mislead a *user* of the shipped
   library.
9. **Wire the README into doctests** (`#![doc = include_str!…]` as upstream does) so the
   quick-start actually compiles, or mark the blocks `ignore` — currently the crate has zero
   doctests and a README that cosplays as one.

Should-fix (non-blocking): errata banner on `SNN-project.md`; `step_batch` equivalence test;
`PerNeuron` integration test; keep koopman-dmd types out of the public API or document the
coupling; a criterion bench entry for the LowRank path replacing the in-test `Instant` timing.
