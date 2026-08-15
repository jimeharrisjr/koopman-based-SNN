# Pre-Registration: Phase 4 / V2 — EDMD Lifted Surrogates for the Izhikevich Neuron

**Document:** 04-v2-preregistration.md
**Author:** Scientist subagent · **Date registered:** 2026-08-14 (before any Phase 4 fitting or rollout code was run)
**Gates:** Phase 4 exit (IMPLEMENTATION_PLAN.md, V2); answers skeptic demands C1 (value must be shown where dynamics are genuinely nonlinear), C4 (hybrid stepping architecture), M4 (pre-registered tasks/baselines/metrics/kill criteria).
**Amendment policy:** thresholds below are frozen. Any deviation during the experiment must be recorded in the results document with a reason, and a threshold may only be *tightened* post-hoc, never loosened.

---

## 1. Experiment under test (design fixed by docs/02-architecture.md; restated, not decided here)

Per-neuron EDMD surrogate of the Izhikevich model (`crates/kdmd-snn/src/neuron/izhikevich.rs`): state (v, u) with constant injected current I folded in as a third, augmented state variable; dictionary `LiftingConfig::PolynomialCross` over (v, u, I), candidate degrees 2 (lifted dim d ≈ 9–10) and 3 (d ≈ 19–20). Fit on sub-threshold snapshot pairs only — every pair whose interval contains a reference-simulator spike (v ≥ 30 mV at any substep) is masked out. Rollout is hybrid (skeptic C4 architecture): lifted linear stepping between spikes; when readout v ≥ 30 mV, explicit reset in original coordinates (v←c, u←u+d) and re-lift Ψ(c, u+d, I).

- **Ground truth:** the same simulator at forward-Euler sub-step h = 0.025 ms. *Convergence check (precondition):* halving to h = 0.0125 ms must move every reference spike time by < 0.1 ms over 1000 ms at the highest gated current; if not, h is reduced until it does. Thresholds are never adjusted to compensate.
- **Practical baseline:** paper-standard coarse Euler, h = 0.5 ms (Izhikevich 2003's two half-steps per ms) and h = 0.25 ms; additionally h ∈ {0.1, 0.05} to trace the Euler cost–accuracy frontier for the cost gate (§4).

## 2. Test protocol

For b = 0.2 the RS/FS u-nullcline gives a saddle-node (rheobase) at exactly I = 4 (0.04v² + 4.8v + 144 + (I−4) = 0 has a double root at I = 4), so the grid brackets it deliberately.

| Item | Registered value | Rationale |
|---|---|---|
| Training currents | I ∈ {3, 5, 7, 9, 11} | one sub-rheobase (pure relaxation data), four tonic; spans the operating range |
| Held-out interpolation (gated) | I ∈ {6, 10} | strictly between training points |
| Held-out extrapolation (gated, relaxed) | I = 13 | mild: 18 % above training max |
| Sub-rheobase probe (gated) | I = 2 | below training min; isolates dictionary quality with no resets at all |
| Hard interpolation (reported, NOT gated) | I = 4 | at the saddle-node ghost, spike latency is singular in I; an unfair gate but a valuable stress report |
| Trajectories per training current | 10 × 1000 ms, v₀ ~ U[−80, −50], u₀ = b·v₀ + U[−2, 2] | ≥ 5,000 clean masked pairs per current per candidate Δt required (else extend trajectories; pre-registered contingency) |
| Validation split | 80/20 by whole trajectory (never by pair) | avoids within-trajectory leakage |
| Rollout horizon | 1000 ms (primary); 200 ms error-growth curves also reported | matches Izhikevich (2003) demonstration protocol; long enough that a 1 % ISI bias accumulates ~10 ms of phase drift and is caught by the ±2 ms window — cumulative drift is the failure mode that must not be hidden |
| Surrogate step grid | Δt* ∈ {0.5, 1.0, 2.0} ms (one EDMD fit per Δt, data sub-sampled from the h = 0.025 reference) | the value proposition is accuracy at LARGE Δt; each Δt is a different operator |
| Neuron types | **RS gated (mandatory). FS gated (mandatory)** — same dictionary structure, coefficients refit per type. **CH is a stretch goal: reported, not gated** | RS is the canonical cortical cell; FS (a = 0.1) tests the same dictionary across a 5× change in recovery timescale at near-zero extra cost. CH's bursting re-enters the upstroke region at high intra-burst rates — exactly where foundations open question 7 predicts degradation — so it maps the validity boundary rather than gating v1 |
| Fit-hygiene preconditions | held-out one-step relative residual ≤ 1e-3; ρ(K) < 1 + 1e-6 (`dmd_stability`); I-row of K ≈ identity (I is constant) | a fit failing these is rejected before rollout; per plan §Core-decisions 4 |

## 3. Accuracy metrics and PASS thresholds

All computed per (neuron type × test current × Δt*), surrogate vs h = 0.025 reference, over the full 1000 ms unless stated. Spike matching is greedy one-to-one within the window.

**Spike-time definition (fixed now, because it interacts with the window):** for both reference and surrogate, the spike time is the v = 30 mV crossing *linearly interpolated within the step that crosses* — not the step index. Without interpolation, a Δt = 2 ms surrogate carries up to ±1 ms of pure quantization error, half the coincidence window; with it, timing error measures the model, not the clock. The same rule is applied to the coarse-Euler baselines.

| Metric | Definition | Interp (I=6,10) | Extrap (I=13) | Sub-rheo (I=2) |
|---|---|---|---|---|
| A1. Spike-count relative error | \|N_sur − N_ref\| / N_ref (absolute slack: pass if \|ΔN\| ≤ 1 when N_ref < 10) | ≤ 10 % | ≤ 15 % | N_sur = 0 required |
| A2. Spike-time coincidence | fraction of reference spikes matched within **±2 ms**; chance-corrected Γ (Jolivet et al.) reported alongside | ≥ 0.80 | ≥ 0.70 | n/a |
| A3. First-spike latency error | \|t₁_sur − t₁_ref\| | ≤ 2 ms | ≤ 4 ms | n/a |
| A4. Sub-threshold v RMSE | per matched inter-spike segment, each segment re-indexed from its own spike time, excluding ±5 ms around spikes | ≤ 1.0 mV | ≤ 2.0 mV | ≤ 0.5 mV (full trace) |

**Justifications.**
- *±2 ms window (A2):* the standard temporal resolution of the Γ coincidence factor in the predicting-spike-timing literature — Jolivet et al. (2006) use Δ = 2 ms; the INCF quantitative single-neuron-modeling benchmark (Jolivet et al. 2008, *J. Neurosci. Methods* 169:417–424) is built on the same Γ with windows in the 2–4 ms range; Victor–Purpura cost parameters in that literature likewise correspond to few-ms timescales. We take the strict end (2 ms) because our target is a noiseless deterministic simulator, not a noisy neuron.
- *0.80 fraction (A2):* winning INCF-competition models predicted ~75–80 % of a real neuron's spikes at this window against an intrinsic reliability ceiling of Γ_int ≈ 0.75–0.85; with no biological noise there is no ceiling excuse, so we require ≥ 0.80 outright and treat failure by phase drift as genuine failure (spike timing *is* the computation), not a metric artifact. Extrapolation relaxes to 0.70, consistent with the accepted degradation of data-driven surrogates outside the sampled regime (Korda & Mezić 2018a: EDMD is an L²(ρ)-projection — it is only guaranteed where the data measure ρ lives).
- *10 % count error (A1):* rate error of this order is what separates simulators run at different dt in the Brette et al. (2007) benchmark comparisons; it is also well inside the ISI-CV ≈ 10 % tonic-regularity band already asserted in the crate's own RS test.
- *1.0 mV RMSE (A4):* ~1 % of the model's ~95 mV dynamic range (−65 → +30) and at or below the 1–2 mV membrane-noise floor that makes sub-threshold agreement "excellent" in the fit-to-real-neuron literature; halved to 0.5 mV at I = 2 where no reset ever intervenes and the dictionary is on home turf.
- *Segment re-indexing (A4):* without per-segment alignment, phase drift double-counts as voltage error; A2 owns timing, A4 owns shape.

## 4. Cost dimension (honest framing)

**Flop convention (fixed):** one Izhikevich Euler substep = 15 flops; one d-dimensional lifted matvec = d² flops; a re-lift = ~3d flops, amortized over the inter-spike interval (< 1 flop/ms below 30 Hz — negligible). Analytic cost in flops per simulated ms per neuron:

| Stepper | flops/ms |
|---|---|
| Reference Euler h = 0.025 | 600 |
| Euler h = 0.05 / 0.1 | 300 / 150 |
| Euler h = 0.25 / 0.5 (paper standard) | 60 / 30 |
| Surrogate deg 2 (d = 10) at Δt = 0.5 / 1 / 2 | 200 / 100 / 50 |
| Surrogate deg 3 (d = 20) at Δt = 0.5 / 1 / 2 | 800 / 400 / 200 |

A degree-2 lifted matvec is ~7× the arithmetic of ONE substep — the surrogate can never win per-substep. Its only honest value proposition is **fewer, bigger steps**: reference-grade accuracy at a fraction of reference cost. The gates:

- **C1 (vs reference):** at the accuracy-passing Δt*, surrogate flops/ms ≤ 300 (≥ 2× cheaper than the h = 0.025 reference). Note the registered arithmetic: degree 3 at Δt = 0.5 (800) fails C1 outright, and degree 3 at Δt = 1 (400) fails too — **the go decision realistically rides on degree 2 (or degree 3 only at Δt = 2)**; degree 3 elsewhere is an accuracy diagnostic, not a product.
- **C2 (frontier — the decisive gate):** surrogate flops/ms at Δt* must be **strictly less than the cheapest Euler h that itself passes every §3 accuracy gate** against the reference. Euler accuracy at h ∈ {0.5, 0.25, 0.1, 0.05} is measured with the identical metrics. **Explicit NO-GO clause: if paper-standard h = 0.5 Euler (30 flops/ms) already passes all accuracy gates, no surrogate on our grid can pass C2, and V2 fails on cost even with perfect surrogate accuracy.** Registered prediction: coarse Euler will *fail* the ±2 ms gate over 1000 ms (O(h) spike-time bias on the quadratic upstroke, cumulative — the dt-sensitivity of clock-driven spike times documented in Brette et al. 2007); this prediction is falsifiable and its failure is a legitimate way for V2 to die.
- **C3 (wall-clock):** criterion bench, N = 1024 neurons, f64, zero-alloc steppers for both sides; median ns per simulated ms per neuron. Surrogate ≤ 0.7× reference wall-clock at Δt*, and the measured ratio within 3× of the flop-predicted ratio (guards against a memory-bound implementation quietly voiding the analytic story).

## 5. Decision rule

Evaluated per neuron type; **PASS requires RS and FS both**, each at some (degree, Δt*) with Δt* ≥ 1.0 ms — the same (degree, Δt*) must satisfy §3 and §4 simultaneously for that type (types may differ in their passing Δt*).

| Outcome | Condition | Consequence (per plan kill-criteria section) |
|---|---|---|
| **PASS** | RS and FS: all gated accuracy metrics (interp + extrap + sub-rheo) AND C1–C3 pass at some (degree, Δt* ≥ 1 ms) | V2 confirmed. Phase 5 proceeds with `PerNeuron` lifted operators; numbers published in the results doc; CH result reported as scope statement |
| **CONDITIONAL PASS** | Interpolation + sub-rheo + cost pass on RS and FS, but (a) extrapolation (I = 13) fails, or (b) only Δt* = 0.5 ms passes accuracy while C1/C2 still hold there, or (c) FS fails but RS fully passes | Proceed to Phase 5 with the V2 claim *narrowed in writing* (operating range restricted to the training current envelope / RS-class cells / stated Δt); owner notified; a dictionary-augmentation follow-up (delay embedding or exponential observable per foundations §5-Q2) is filed; skeptic audits the narrowed claim at the already-scheduled end-of-Phase-5 review |
| **FAIL** | Any RS interpolation gate missed at every (degree ∈ {2,3}, Δt* ∈ {0.5, 1, 2}), OR the C2 NO-GO clause fires (an Euler h dominates the surrogate on both cost and accuracy) | Work pauses for an owner decision informed by a skeptic re-review (plan Phase 4 exit text). V2 becomes a documented negative result; the library's value cases reduce to V1 (reduced-order recurrent) + V3 (spectral diagnostics) + V4 (spectral transparency), which docs/01 §3 and docs/03 C1 identified as independently defensible |

No middle path is left undefined: any result not matching PASS or CONDITIONAL PASS rows is a FAIL.

## 6. Registered predictions (so the result is informative whichever way it falls)

| # | Prediction | If it holds | If it fails |
|---|---|---|---|
| P1 (scientist) | Degree-2 PolynomialCross passes all interpolation gates at Δt* = 1 ms for RS — the quadratic nonlinearity is *inside* the degree-2 dictionary span, so closure error is dominated by the reset re-entry, not the flow | V2's core mechanism is validated | The masking/data-measure bias (threat 2) is worse than theory suggests; dictionary work (Q2) precedes any Phase 5 lifted path |
| P2 (scientist) | Coarse Euler h = 0.5 fails A2 (±2 ms over 1000 ms) via cumulative phase drift | C2 frontier is winnable | The C2 NO-GO clause fires; V2 dies honestly on cost |
| P3 (scientist, from foundations Q7) | Accuracy degrades monotonically with firing rate; CH fails at least one gate-equivalent metric | Validity boundary documented as predicted | Pleasant surprise; CH promoted to gated in a v2 amendment |
| P4 (skeptic's position, carried from docs/03) | Even a passing surrogate wins only the large-Δt regime, never per-flop-per-substep | Framing of §4 confirmed | (not falsifiable in the direction of surrogate per-substep victory; arithmetic forbids it) |

## 7. Threats to validity (registered before data)

1. **Dictionary closure error at the upstroke.** Masking removes spike-straddling pairs, so the region just below v = 30 is data-sparse and the surrogate *generates* each spike by extrapolating the polynomial fit into its worst-constrained region; A2/A3 are the designed detectors. Per-segment error-vs-time-since-reset curves will be reported to localize this (foundations §5-Q7 predicts error grows with firing rate; CH is the sentinel).
2. **Mask-induced sampling bias.** Least squares weights the fit by data density (Korda & Mezić 2018a), which concentrates near the slow manifold / rest; eigenvalues may bias toward slow relaxation, under-fitting the fast upstroke. Same detectors as (1); the I = 2 gate isolates the slow regime so a slow-regime-only fit cannot hide.
3. **Extrapolation in I.** With I as an augmented state, the operator is only informed at five discrete currents; polynomial structure interpolates plausibly but extrapolates without guarantee — hence the extrapolation gate is mild (+18 %) and relaxed, and passing it does NOT license claims beyond I = 13.
4. **Δt sensitivity.** Each Δt* is a different discrete-time operator; conclusions at Δt = 1 ms do not transfer to other steps. The grid {0.5, 1, 2} is registered up front, and the decision rule names Δt* explicitly to prevent post-hoc step-shopping.
5. **Reference-truth error.** Forward Euler at h = 0.025 ms is itself approximate; the §1 self-convergence check must pass *before* any surrogate is evaluated, and failure lowers h rather than loosening thresholds. Residual reference bias is shared by surrogate and baseline alike (both are scored against the same trace), so it shifts no comparison.

## 8. References

- Izhikevich, E. M. (2003). Simple model of spiking neurons. *IEEE Trans. Neural Networks* 14(6), 1569–1572. (parameters, Euler practice, 1000 ms protocol)
- Jolivet, R., Rauch, A., Lüscher, H.-R., & Gerstner, W. (2006). Predicting spike timing of neocortical pyramidal neurons by simple threshold models. *J. Comput. Neurosci.* 21, 35–49. (Γ coincidence factor, Δ = 2 ms)
- Jolivet, R., Kobayashi, R., Rauch, A., Naud, R., Shinomoto, S., & Gerstner, W. (2008). A benchmark test for a quantitative assessment of simple neuron models. *J. Neurosci. Methods* 169(2), 417–424. (INCF benchmark; Γ conventions; attainable prediction levels)
- Brette, R., et al. (2007). Simulation of networks of spiking neurons: a review of tools and strategies. *J. Comput. Neurosci.* 23(3), 349–398. (dt-dependent spike-time accuracy in clock-driven simulation; benchmark practice)
- Williams, M. O., Kevrekidis, I. G., & Rowley, C. W. (2015). A data-driven approximation of the Koopman operator. *J. Nonlinear Sci.* 25(6), 1307–1346. (EDMD)
- Korda, M., & Mezić, I. (2018a). On convergence of EDMD to the Koopman operator. *J. Nonlinear Sci.* 28, 687–710. (projection w.r.t. the data measure; basis for extrapolation caution)
- Project documents: docs/01-scientific-foundations.md (§2.4, §3.3, §5 Q2/Q7); docs/02-architecture.md; docs/03-skeptic-review.md (C1, C4, M4); IMPLEMENTATION_PLAN.md (V2, Phase 4, kill criteria).
