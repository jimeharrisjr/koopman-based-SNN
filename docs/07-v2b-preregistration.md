# Pre-Registration: V2b — ISI / Poincaré Return-Map Surrogate for the Izhikevich Neuron

**Document:** 07-v2b-preregistration.md
**Author:** Scientist subagent · **Date registered:** 2026-08-14 (before any V2b fitting or rollout code was run)
**Status:** V2b is the owner-approved single bounded rescue attempt after the V2 FAIL (docs/05, docs/06 F8). It is a NEW experiment with a NEW architecture, not a V2 re-run; it answers skeptic demands F8(a) (replace ρ(K) with a satisfiable stability criterion), F8(b) (a per-cycle bias budget derived from the phase gate), F8(c) (compute Γ), F8(d) (moot — no v-trace, no A4 segments).
**Amendment policy:** thresholds below are frozen; deviations must be recorded in the results doc and may only *tighten*. Every metric shared with V2 keeps its V2 threshold (docs/04 §6 allows tightening only); new metrics are registered here, before data.
**Owner's standing decision:** FAIL → immediate pivot to V1/V3/V4. There is no conditional row and no pause.

---

## 1. Experiment under test (design fixed by the orchestrator; restated, not decided here)

The V2 autopsy showed one-step lifted stepping cannot hold ±2 ms phase over 1000 ms (F3: needs ~1e-5 relative one-step bias in a region with zero training data, F4), while CH's burst re-anchoring (F5) showed phase predicted *at reset events* survives. V2b therefore fits the discrete-time system induced on a Poincaré section — the Koopman/EDMD machinery applied to the *section map* rather than the flow (Mauroy–Mezić–Moehlis 2013 give the operator-theoretic frame; foundations §3.3). Per-period error is now the regression residual itself, the fix F3 named.

- **Section:** the post-reset state. v is pinned at c, so the section coordinate is (u₊, I) with u₊ = u(t_spike) + d. On this section u₊ is a complete state for the deterministic constant-I flow, so the return map is exactly well-defined — including for CH (see §2 note).
- **ISI map** T ≈ w_T·Ψ(u₊, I) and **next-section map** u₊′ ≈ w_u·Ψ(u₊, I): ordinary least squares (QR/SVD, cutoff 1e-10, features z-scored on training stats) over dictionary Ψ, trained on crossing-interpolated spike data from the self-converged reference (§2).
- **First-spike map** T₀ ≈ w₀·Ψ₀(v, u, I) and u₊-at-first-spike ≈ w₀u·Ψ₀(v, u, I). *Registered training refinement:* one sample per trajectory (32 total) cannot fit d₀ ≥ 20 dictionaries, so training pairs are states sampled every 0.5 ms along each training trajectory from t = 0 to its first crossing, target = time remaining to that crossing (and its u₊). This stays within "trained from the randomized-IC trajectories," yields ≥10³ samples, and teaches the approach flow that the I = 2 quiescence gate probes.
- **Dictionaries (registered NOW; post-hoc additions forbidden).** Two families per degree p ∈ {2, 3, 4}:
  - **Family A:** PolynomialCross degree p over (u₊, I); dim d = 6 / 10 / 15. Ψ₀: degree p over (v, u, I); d₀ = 10 / 20 / 35.
  - **Family B (physics-informed):** Family A plus g(I) and u₊·g(I) in Ψ (d = 8 / 12 / 17), plus {g, v·g, u·g} in Ψ₀ (d₀ = 13 / 23 / 38), where **g(I) := min(10, (I − 4)^(−1/2)) for I > 4 and g(I) := 10 for I ≤ 4** (clamp = value at I − 4 = 0.01; frozen). *Justification:* I = 4 is the exact saddle-node (docs/04 §2), and ghost passage time is π/√(0.04·(I−4)) ≈ 15.7/11.1/6.4/5.2 ms at I = 5/6/10/13 — the leading singular part of T(I) that no low-degree polynomial in I represents; registering it now is the only honest way to have it (the CRITICAL I-interpolation risk, threat T1). The clamp keeps the feature finite, monotone, and physics-signed (T increasing as I ↓ 4) at the reported I = 4 and sub-rheobase currents.
- **Quiescence rule (frozen):** the surrogate emits no (further) spike when the predicted interval exceeds **T_max := min(1000 ms, 2 × max over the type's training split of {all ISIs, all first-spike latencies})**, computed per type from training data only. *Justification:* T ∝ (I−4)^(−1/2) means doubling the slowest trained interval corresponds to currents at I ≈ 4 + (I_min−4)/4 ≈ 4.25 — the misclassification band is confined to a narrow strip above rheobase that no gated current occupies (tonic gates ≥ 6; sub-rheo gate at 2). A predicted T ≤ 0 or non-finite is NOT quiescence: it aborts the rollout as an invalid configuration (an insane fit must not pass I = 2 by accident). Runaway guard: > 5000 emitted spikes aborts as invalid.
- **Rollout:** t₁ = T₀(v₀, u₀, I); u₊ from the first-spike u-map; then iterate (T, u₊) ← maps(u₊, I), accumulating spike times until t > 1000 ms or quiescence fires. No per-step state — **the surrogate is a spike-timing model.**

**SCOPE LIMITATION (prominent, honest):** V2b produces spike times only — no v(t) trace, no sub-threshold waveform, constant I per trajectory, ICs inside the registered box. It is NOT a drop-in simulator replacement; every cost claim in §5 is for spike-timing workloads only. A PASS narrows V2's ambition, it does not restore it.

## 2. Test protocol (carried from docs/04 unless stated)

| Item | Registered value | Rationale |
|---|---|---|
| Ground truth | self-converged fine Euler per type: h = 0.0015625 ms (RS), 0.000195 ms (FS), 0.00039 ms (CH) — the docs/05 deviation-1 values; convergence re-verified once, thresholds never adjusted | carried |
| Spike time | linearly interpolated v = 30 mV crossing, both sides | carried (docs/04 §3) |
| Coincidence | greedy one-to-one matching, ±2 ms | carried |
| Training currents | I ∈ {3, 5, 7, 9, 11}; **I = 3 contributes no regression samples** (no crossings) — it serves as reported in-sample quiescence evidence (fraction of I = 3 states with predicted T₀ > T_max; expectation ≈ 1) | sub-rheobase has no ISI targets by construction |
| Trajectories | 10 × 1000 ms per current, v₀ ~ U[−80, −50], u₀ = b·v₀ + U[−2, 2]; 80/20 split by whole trajectory | carried |
| Data floor | ≥ 50 training ISI pairs per spiking current per type, ≥ 500 first-spike pairs per type; else extend trajectories (contingency, recorded) | least-squares sanity for d ≤ 17, d₀ ≤ 38 |
| Test currents | interp I ∈ {6, 10} gated; extrap I = 13 gated (relaxed); sub-rheo I = 2 gated; I = 4 reported only | carried |
| Horizon | 1000 ms | carried; the drift-exposing horizon |
| Types | **RS and FS gated (both mandatory). CH reported, NOT gated** | carried |
| Config grid | {deg 2, 3, 4} × {Family A, B} per type = 6 fits per type | 2-D section makes this cheap |

**CH note (validity boundary, registered):** CH is evaluated with the *identical* machinery. The section is formally valid (u₊ separates intra-burst from inter-burst crossings), but T(u₊) is nearly discontinuous at burst termination (2.55 ms ↔ 75.8 ms across a narrow u₊ interval; docs/06 F5) — registered expectation: degree ≤ 4 polynomials cannot represent this and CH fails A2 (prediction R4). CH's outcome does not affect PASS/FAIL.

## 3. Accuracy metrics and PASS thresholds

Per (type × test current × config), surrogate vs reference over 1000 ms.

| Metric | Definition | Interp (6, 10) | Extrap (13) | Sub-rheo (2) |
|---|---|---|---|---|
| A1. Spike-count rel. error | \|N_sur − N_ref\|/N_ref (abs. slack \|ΔN\| ≤ 1 iff N_ref < 10) | ≤ 10 % | ≤ 15 % | N_sur = 0 required |
| A2. Coincidence (raw) | fraction of reference spikes matched within ±2 ms | ≥ 0.80 | ≥ 0.70 | n/a |
| A2b. Chance-corrected Γ **(NEW)** | Γ := (c − f)/(1 − f), c = raw A2 fraction, f = min(1, 2Δ·N_sur/T), Δ = 2 ms, T = 1000 ms (the docs/06 F1 floor convention); Γ := 0 if f ≥ 1 | ≥ 0.60 | ≥ 0.50 | n/a |
| A3. First-spike latency error | \|t₁_sur − t₁_ref\| | ≤ 2 ms | ≤ 4 ms | n/a |
| A5. Per-ISI bias **(NEW, replaces A4)** | teacher-forced on the reference: at every reference crossing i take the true (u₊ᵢ, I), eᵢ = T̂(u₊ᵢ, I) − Tᵢ; bias = mean(eᵢ) over all 10 test trajectories, relative to that current's mean reference ISI. RMS(eᵢ) and held-out u₊′ error reported, not gated | \|bias\| ≤ 0.25 % | \|bias\| ≤ 0.30 % | n/a |

**Justifications.**
- *A1/A2/A3 thresholds are V2's unchanged* (shared metrics keep V2 thresholds; docs/04 §3 justifications carry). A4 (sub-threshold v-RMSE) is **dropped as inapplicable** — the surrogate has no v-trace; its F7 double-counting defect dies with it.
- *A2b (Γ):* docs/06 F1 showed raw FS coincidence of 0.63 was pure chance (floor 0.62). The registered Γ gate makes chance-level raw scores fail: at RS currents the floor is ~0.06–0.09 so Γ ≥ 0.60 needs raw ≈ 0.63 (non-binding below A2's 0.80), but at FS I = 13 (floor 0.624) Γ ≥ 0.50 needs raw ≈ 0.81 — binding exactly where V2's raw metric was blind. A new metric, so freely registrable; it only tightens.
- *A5 bound — the F3 arithmetic in reverse.* A pure per-ISI bias b drifts as k·b; matching a fraction q of spikes in a T = 1000 ms train requires k·b ≤ 2 ms for k up to q·N, i.e. relative bias ≤ 2/(q·1000) — **0.25 % at q = 0.80, 0.29 % at q = 0.70 (registered as 0.30 %), independent of type and rate**. Zero-mean jitter of std σ instead drifts as σ√k, allowing σ up to ~2/√19 ≈ 0.47 ms ≈ 1 % for RS — 4× looser. **Decision: gate bias, report RMS, and keep A2 as the end-to-end arbiter.** Why: (i) bias is exactly the quantity F3 proved V2 left ungated while the gate depended on it; (ii) gating RMS would double-gate what A2 already scores end-to-end, and near-zero-bias jitter that A2 tolerates should not fail a per-cycle screen; (iii) A5 is teacher-forced, so it cleanly localizes map error, while closed-loop u₊-feedback effects remain A2's job (prediction R5 tests whether they diverge).
- *Calibration honesty:* at steady state on a training current the map interpolates memorized ISIs, so A5 at held-out *currents* is the real test — the I-interpolation across the steep T(I) — which is precisely why Family B's g(I) is registered up front and why every gate rides on currents never seen in training (§8, T5).

## 4. Preconditions (per fit, before rollout — each one satisfiable by a faithful model; the F2 lesson)

| # | Precondition | Why it is satisfiable |
|---|---|---|
| P-A | least-squares problems full-rank at the SVD cutoff; condition numbers reported | data floors in §2 guarantee overdetermined fits |
| P-B | at each ISI-training current: iterating the fitted u-map converges (≤ 200 iters) to a fixed point u* inside the training u₊-range with **\|dg/du(u*)\| < 1** (central difference) | the true return map of a stable limit cycle is Floquet-contracting at its fixed point — \|dg/du\| < 1 is a property the *true* system has, so a faithful fit inherits it. This replaces V2's ρ(K) gate, which demanded contraction of a genuinely expanding flow (docs/06 F2) and made PASS unreachable. NO spectral-radius condition appears anywhere in V2b. For CH (reported) the analogue — a stable period-k orbit with \|∏ g′\| < 1 — is reported, not required |
| P-C | held-out (20 % split, training currents): per-ISI relative RMS ≤ 5 % and time-to-first-spike relative RMS ≤ 5 % | deliberately loose hygiene screens (the strict work belongs to §3 gates); same-current holdout shares the limit cycle, so a faithful fit passes with an order of magnitude to spare |

A fit failing any precondition is rejected before rollout; if every config of a type fails preconditions, that type FAILs. Preconditions and gates were calibrated together this time: P-C (5 %) is 20× looser than the A5 gate (0.25 %) on purpose — the precondition screens broken fits, the gate decides, and no precondition demands anything the true dynamics violate.

## 5. Cost dimension (honest framing; spike-timing workloads ONLY)

**Flop convention (fixed):** cost per emitted spike = 2·d² (ISI map + u-map, each charged d² to stay conservative and comparable with V2's matvec convention); first-spike evaluation d₀², once per trajectory (< 1.5 flops/ms amortized — negligible). **flops/ms = rate[spikes/ms] × 2·d².** Reference cost at converged h (15 flops/substep): **RS 9600 flops/ms, FS ≈ 76,900 flops/ms.**

| Config (worst case, deg 4 Family B, d = 17) | rate (sp/ms) | flops/ms | reference |
|---|---|---|---|
| RS @ I = 10 | 0.023 | ≈ 13 | 9600 |
| FS @ I = 13 | 0.194 | ≈ 112 | ≈ 76,900 |

- **C1 (vs reference):** surrogate flops/ms ≤ 0.5 × the type's converged-reference flops/ms (V2's "≥ 2× cheaper" convention). Expected to pass by 2–3 orders of magnitude for every config.
- **C2 (frontier):** surrogate flops/ms strictly below the cheapest Euler h that passes every §3 accuracy gate. **Carried finding: V2 measured that no Euler h ∈ {0.5, 0.25, 0.1, 0.05} passes accuracy for RS or FS (P2 CONFIRMED, docs/05) — the standing frontier is the converged reference itself. Euler baselines are NOT re-run;** the V2 measurement stands.
- **C3 (wall-clock): reported, not gated** — single-rollout timing for context. Registered reason: per-spike work is microscopic and flop gates already decide by orders of magnitude; a N = 1024 bench belongs to Phase 5 (V2's C3 was already deviated to report-only, docs/05 deviation 2).

**Caveat, stated once more because it is the price of this rescue:** these cost numbers buy spike trains, not membrane traces. Any workload needing v(t) still requires the reference integrator; C1/C2 claims must never be quoted without this restriction.

## 6. Decision rule — PASS / FAIL only

Evaluated per type at some single config (degree, family) — the same config must satisfy §3 (A1, A2, A2b, A3, A5 at all gated currents, including sub-rheo A1) and §4–§5 for that type; types may differ in their passing config. **PASS requires RS AND FS both.**

| Outcome | Condition | Consequence |
|---|---|---|
| **PASS** | RS and FS each pass all gated metrics + preconditions + C1–C2 at some config | Phase 5 proceeds with the event-level spike-timing surrogate as the (re-scoped) V2b value case, claims restricted to: constant I ∈ [5, 13], registered IC box, spike-timing workloads, no v-trace. The V2 per-step negative result stands unamended. CH scope statement published |
| **FAIL** | anything else — any gate missed by every config for either type, or invalid rollouts across the grid | **Immediate pivot** to V1/V3/V4 per the owner's standing decision — no pause, no re-review gate, no conditional row. V2 and V2b are both recorded as negative results; the nonlinear-surrogate track closes |

No middle path exists by construction. CH's outcome, all reported metrics, and I = 4 affect nothing.

## 7. Registered predictions (falsifiable in both directions)

| # | Prediction | If it holds | If it fails |
|---|---|---|---|
| R1 | RS passes all interpolation gates for some Family B config — per-period error is now the regression residual, and g(I) makes T(I) near-polynomial between training currents | the F5 re-anchoring mechanism generalizes; event-level surrogacy validated | even direct return-map regression cannot reach 0.25 % bias at held-out currents; the event-level route dies with the per-step route; pivot |
| R2 | Family B beats Family A on \|extrap I = 13 per-ISI bias\| at matched degree, both types | physics-informed registration vindicated | g(I) unnecessary; polynomials sufficed — reported honestly |
| R3 | quiescence: I = 2 passes (zero spikes) for all Family B configs; at least one Family A deg-4 config emits a spurious spike at I = 2 (unsigned polynomial extrapolation) | the T_max rule + monotone g mechanism works as designed | if Family A also always passes, the quiescence risk was overstated; if Family B fails, the clamp design is wrong — either way informative |
| R4 | CH fails A2 at I = 6 with the same machinery (burst-termination stiffness beyond deg-4 capacity) | validity boundary mapped as registered | architecture generalizes beyond tonic firing; CH promotion candidate for a future registration |
| R5 | A5 (teacher-forced) and A2 (closed-loop) agree in sign and order — closed-loop u₊ feedback does not dominate | per-cycle gate is a faithful localizer | closed-loop amplification through the u-map is a new structural obstacle, itself worth recording |

## 8. Threats to validity (registered before data)

1. **T1 — I-interpolation near rheobase.** T(I) steepens as (I−4)^(−1/2); I = 6 is the nearest gated current to the singularity. Family B is the registered mitigation; A5 at I = 6 is the designed detector. No post-hoc observable may be added if it fails.
2. **T2 — u₊-transient coverage.** Each trajectory contributes only ~1–3 pre-limit-cycle cycles, so the map is data-rich only near u*(I); early rollout cycles carry the worst u₊ error. Detectors: A3 and per-spike-index match curves (reported).
3. **T3 — quiescence extrapolation.** There are no supervised "no-spike" targets; T_max is a decision boundary in an extrapolated region, and sub-rheobase I contributes no ISI data by construction. The clamped g bounds feature blow-up; the I = 2 gate and the I = 3 in-sample report are the detectors.
4. **T4 — CH section stiffness.** Registered expectation of failure (R4); reported only.
5. **T5 — memorization vs interpolation.** Same-current holdout is near-memorization at steady state (all ICs share one limit cycle), so P-C numbers must never be quoted as generalization evidence. What distinguishes genuine skill: the gated currents {6, 10, 13, 2} never appear in training, so their steady-state ISIs cannot be memorized — only interpolated in I. Every gate rides on them; that is the design.
6. **T6 — reference truth.** Crossing-interpolated spike times at the converged h; residual reference bias is shared by both sides of every comparison. The one-time convergence re-check (§2) must pass before any surrogate is scored.
7. **T7 — scope.** Constant current, registered IC box, no v-trace. A PASS licenses nothing outside that sentence.

## 9. References

- Mauroy, A., Mezić, I., & Moehlis, J. (2013). Isostables, isochrons, and Koopman spectrum for the action–angle representation of stable fixed point dynamics. *Physica D* 261, 19–30. (operator-theoretic frame for section/return-map coordinates)
- Izhikevich, E. M. (2003). *IEEE Trans. Neural Networks* 14(6), 1569–1572; and (2007) *Dynamical Systems in Neuroscience*, MIT Press, Ch. 6 (saddle-node ghost passage-time scaling T ∝ (I − I_c)^(−1/2)).
- Jolivet, R., et al. (2006) *J. Comput. Neurosci.* 21, 35–49; (2008) *J. Neurosci. Methods* 169, 417–424. (±2 ms window; Γ chance-correction conventions)
- Project documents: docs/04-v2-preregistration.md (carried conventions and thresholds); docs/05-v2-results.md (converged h values; P2 Euler frontier); docs/06-skeptic-v2-review.md (F1 floor convention, F2 precondition lesson, F3 bias budget, F5 mechanism, F8 follow-up demands); docs/01-scientific-foundations.md §3.3.
