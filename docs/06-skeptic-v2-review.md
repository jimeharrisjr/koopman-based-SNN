# Skeptic Re-Review of the V2 FAIL — Was the Negative Result Real?

**Document:** 06-skeptic-v2-review.md · **Author:** Skeptic subagent · **Date:** 2026-08-14
**Scope:** adversarial audit of the FAIL verdict in `docs/05-v2-results.md` under the frozen
protocol `docs/04-v2-preregistration.md`, per the Phase 4 exit rule. My duty here inverts:
the experiment produced the negative result I said was possible, so this review attacks the
NEGATIVE result — harness, fit pipeline, and protocol — with the same rigor a positive one
would have received. Arithmetic claims below were verified by direct re-simulation of the
same Euler equations (probe scripts run and discarded; all inputs are in the cited files).

**Bottom line first: the failure is REAL and, for the registered architecture, STRUCTURAL
(classification 3). But the protocol also contained a precondition no faithful surrogate
could meet, so PASS was unreachable as coded — an important defect to record even though it
did not cause, and cannot excuse, the accuracy failure.**

---

## F1 (CRITICAL — verdict-confirming): the phase failure is real; coincidence sits at the chance floor

For a surrogate spike train phase-decorrelated from the reference, the expected raw
coincidence is the Γ chance term: `min(1, 2Δ·N_sur/T)` with Δ = 2 ms, T = 1000 ms.
Computing that floor for every non-diverged configuration in `docs/05-v2-results-raw.txt`:

| Config | floor | observed | ratio |
|---|---|---|---|
| RS d2 Δt0.5 I=10 | 0.088 | 0.087 | **0.99** |
| RS d2 Δt1 I=10 | 0.084 | 0.130 | 1.55 |
| FS d2 Δt0.5 I=13 | 0.624 | 0.629 | **1.01** |
| FS d2 Δt0.5 I=10 | 0.444 | 0.504 | 1.14 |
| CH d2 Δt1 I=6 | 0.184 | 0.184 | **1.00** |
| CH d3 Δt0.5 I=6 | 0.148 | 0.755 | **5.10** |

Every RS and FS configuration is within 1.0–1.6× of chance; the seemingly respectable FS
numbers (0.50–0.67) are an artifact of short FS ISIs, not retained phase. Exactly ONE
configuration in the entire grid (CH degree 3, Δt 0.5, I=6) carries phase information.
The internal consistency check (audit question B) closes the case: RS I=10 reference has
N=23, post-transient ISI 44.8 ms (verified). Surrogate N=21 at Δt=1 implies ISI bias
+4.14 ms/ISI (+9.5%); drift exceeds the ±2 ms window inside ONE ISI, predicting ~3
matches (first spike + two accidental re-locks) = coincidence 0.13 — exactly as reported.
At Δt=0.5, N=22 implies +2.0 ms/ISI, predicting ~2/23 = 0.087 — exactly as reported.
Count error, coincidence, and first-spike latency are three consistent views of one
phase-drift process. **No metric bug can produce this coherence; audit question A's
"could a metric bug alone explain it" is answered no.** The holdout residual is also
consistent: 1.34e-2 in unit-RMS-normalized space ≈ 0.1–0.9 mV per-step v error; a small
systematic fraction of that, accumulated over a 44-step ISI at slow-region dv/dt
(~0.7–2 mV/ms), yields exactly the observed few-ms/ISI drift.

The decisive metric code is correct: greedy one-to-one ±2 ms matching
(`examples/v2_rollout.rs:134-154`), crossing-interpolated spike times with identical
conventions on both sides (`v2_rollout.rs:83-96` reference, `:112-124` surrogate, using the
stored `pre_reset_v` from `src/surrogate.rs:364-371`), reference h honestly driven to
self-convergence per the §1 contingency (registered h=0.025 failed by 1.48 ms; harness
halved to 0.0015625 without touching thresholds, `v2_rollout.rs:263-284`).

## F2 (CRITICAL — protocol defect): the ρ(K) precondition was unreachable; PASS was impossible as coded

The registered gate ρ(K) < 1+1e-6 (`04-v2-preregistration.md` §2) gates a quantity the
sub-threshold Izhikevich flow genuinely violates. Measured on the true flow at I=10:
∂(dv/dt)/∂v = 0.08v+5 gives local expansion e^1.8 ≈ 6.0/ms at v=−40, e^2.6 ≈ 13.5/ms at
v=−30; direct two-trajectory measurement gives a 3.7× tangent amplification over 0.5 ms at
v=−40 (from −45 upward the flow reaches the 30 mV peak in under 1 ms). Any surrogate that
reproduces the sampled data in this region must exhibit one-step gains of that size, and
the registered estimator — unconstrained least squares, `enforce_stability: false`
(`src/surrogate.rs:245-254`) — realizes such gain spectrally: fitted degree-2 ρ = 5.2, 8.8,
12.0 at Δt = 0.5, 1, 2 tracks the measured physical expansion, it is not noise. Only an
explicitly stability-constrained estimator (never registered) could hide the gain in
non-normal transient growth; and ρ is in any case the wrong object for a rollout that
re-lifts every step, where one-step gain governs. The gate's provenance is the LIF-era
leaky-linear validator ("a leaky neural system must identify as decaying",
`src/identify/validate.rs:5-7`) transplanted into a regime with genuine expansion — a
transplant `surrogate.rs`'s own comment disavows while the registration froze it.
Because the decision logic requires `preconditions_ok` inside `cost_ok`
(`v2_rollout.rs:421,527-539`), **every configuration was disqualified before any accuracy
number was consulted; the experiment could never return PASS.** This does not rescue the
method — F1's accuracy failure is independent and real — but it means the FAIL was
overdetermined, and any follow-up registration must replace this gate (e.g., bounded
1000 ms rollout + mode-wise growth budget), not merely re-run the experiment.

## F3 (HIGH — the structural cause): one-step least squares cannot deliver a ±2 ms / 1000 ms phase gate here

Audit question C(iv), quantified. The 0.80 coincidence gate at ±2 ms requires ≥19 of 23 RS
spikes matched, i.e. systematic per-ISI period error ≤ 2/19 ≈ 0.105 ms = **0.24% of the
ISI**. Achieved: +4.5% (Δt=0.5) and +9.5% (Δt=1) — a 19–40× gap. Propagating backwards:
0.105 ms/ISI over a 44-step ISI at slow-region dv/dt allows a systematic one-step v bias of
order 1e-3 mV (~1e-5 relative) — roughly two orders below the registered 1e-3 residual
precondition and three below the achieved 1e-2. **Even a fit passing every registered
precondition could fail the phase gate by an order of magnitude if its residual is coherent
bias** — the precondition and the gate were never mutually calibrated. One-step EDMD
minimizes mean residual under the data measure; nothing in the objective penalizes bias
accumulation, which is the gated quantity. Degree 3 shows the futility of dictionary
escalation inside this objective: holdout improved only 1.6× (8.2e-3 vs 1.34e-2) while
ρ exploded to 23–350 and every gated rollout diverged to NaN. This was foreseeable (it is
threat 2 plus the Korda–Mezić projection caveat the registration itself cites) and it is
not fixable by any dictionary within one-step fitting: the fix must change the objective
(multi-step/shooting loss, or fitting the ISI return map, where per-period error IS the
regression residual).

## F4 (HIGH): the surrogate had literally zero training data where it generates spikes

Audit question C(iii). The upstroke from v=−40 to +30 takes 0.741 ms at I=10 (measured), so
at Δt=1 the entire upstroke lies inside the single masked pair. Direct count on training
trajectories at I=9 and I=11, Δt=1: **kept pairs with x1 v > −40: 0 of ~975 (0.00%)**; at
Δt=0.5 only a thin sliver (~1 pair per spike, ~2%) survives. Every spike the surrogate
emits is polynomial extrapolation into a region with no data — registered threat 1,
confirmed quantitatively, and unfixable by adding trajectories: it is forced by the masking
rule × Δt geometry. Post-reset coverage is better than feared: limit-cycle post-reset u
reaches +0.5 (vs the IC band u₀ ∈ [−18,−8]), but training trajectories spike and their
retained post-reset pairs cover the re-entry orbit at training currents; consistently, A3
first-spike latency (data-dense early trajectory) passed nearly everywhere while A2 failed.
Fairness note (C(i)): the per-step re-lift rollout (`surrogate.rs:377-384`) deviates from
the registered "lifted linear stepping between spikes" — in the method's FAVOR (free-running
lifted-linear at ρ ≈ 9 diverges within an ISI; the stronger EDMD-as-nonlinear-predictor
variant was tested and still failed). C(ii): row normalization is algebraically clean
(similarity transform, spectrum preserved, `surrogate.rs:206-259`); no conditioning
pathology found; degree-3 divergence is overfit instability, not a linear-algebra bug.

## F5 (MEDIUM): the CH anomaly is explained — burst-gated phase re-anchoring — and it maps the rescue route

Audit question E. CH at I=6 (verified): 12 bursts of ~4 spikes; intra-burst ISI 2.55 ms,
inter-burst gap 75.8 ms. Burst onsets are timed by the slow inter-burst recovery of u —
sub-threshold, data-dense, the dictionary's home turf (every sub-rheobase I=2 gate passed,
v-RMSE down to 0.28 mV) — and each burst re-anchors phase: a burst onset within 2 ms
matches ~4 spikes at once, and drift accumulated across 3 intra-burst ISIs of 2.55 ms stays
inside the window even at ~10% relative error. Tonic RS is the OPPOSITE regime: 23
uninterrupted 44 ms ISIs accumulating upstroke bias coherently with no re-anchoring. P3
assumed firing rate was the difficulty axis; the data say the axis is *un-re-anchored
horizon × relative ISI error*. This is a genuine, mechanistically-understood signal (5.1×
chance floor) and it points exactly at the viable follow-up: predict phase at reset events
(ISI return map) rather than integrating phase through them.

## F6 (MEDIUM): unrecorded protocol deviations (none verdict-changing)

1. Per-step re-lift (F4 above) is absent from the results doc's deviation list — must be
   recorded; direction favorable to the method.
2. The registered chance-corrected Γ was never computed (`v2_rollout.rs` has no Γ code).
   F1's floor table shows why it mattered: raw 0.63 for FS is pure chance. Gates used the
   raw fraction as registered, so no outcome changes.
3. The ≥5000 clean-pairs floor is missed at Δt=2 (8 train trajectories × ~474 kept ≈ 3800)
   and the registered extend-trajectories contingency was not invoked. Δt=2 failed for
   independent reasons.
4. Preconditions failing should have "rejected before rollout"; rollouts ran anyway. The
   decision logic still honored the preconditions, so this is extra information, not a leak.
5. Cosmetic: results table justifies the RS Δt1 I=6 count fail as "|ΔN| = 2 > 1"; with
   N_ref = 14 ≥ 10 the relative rule (14.3% > 10%) is what applies. Same outcome.

## F7 (LOW-MEDIUM): the A4 v-RMSE implementation double-counts phase drift; do not quote 11–13 mV as shape error

`v2_rollout.rs:200-212` walks segments between *consecutive matched pairs* without checking
adjacency. With 3 of 23 spikes matched, a "segment" spans ~10 unmatched reference spikes,
so `v_at` sweeps full spike waveforms against a de-phased model — the 11–13 mV values
measure phase drift again, exactly what §3's "A2 owns timing, A4 owns shape" separation was
designed to prevent. The fix is to keep only segments where no unmatched reference spike
lies between the matched endpoints. Not verdict-relevant (A2 fails at chance level
regardless; where matching was dense, CH d3 Δt0.5, A4 reads a plausible 1.66 mV), but the
raw-table RMSE column is unusable as shape evidence wherever coincidence is low.

## F8 — VERDICT: (3) REAL and STRUCTURAL. Recommend the registered pivot.

Not (1) ARTIFACT: the decisive metric (A2) is correctly implemented, the drift arithmetic is
internally consistent to three decimal places across independent configurations, and the
failure mechanism is physically explained. Not (2) rescuable-in-place: the 19–40× phase gap
traces to the objective (F3) compounded by the data vacuum (F4); no dictionary inside
one-step EDMD fitting closes a gap that requires ~1e-5 relative systematic one-step
accuracy in an unsampled, genuinely expanding region. **One-step EDMD fitting cannot
deliver ±2 ms phase over 1000 ms of tonic Izhikevich firing at Δt ≥ 0.5 ms. V2 should be
recorded as a real negative result and Phase 5 should proceed on V1/V3/V4 per the
registered consequence.** If the owner wants one bounded follow-up, the only route this
data supports is a NEW experiment, not a V2 re-run: fit the spike-to-spike (ISI/Poincaré)
return map (u_reset, I) → (ISI, u'_reset) — where per-period error is the training
residual, CH's re-anchoring mechanism becomes the architecture — with a fresh
pre-registration that (a) replaces the ρ(K) gate with a rollout-boundedness criterion (F2),
(b) states a one-step-bias budget derived from the phase gate (F3), (c) computes Γ (F6.2),
and (d) fixes A4 segment adjacency (F7). That is a different value proposition (event-level
surrogate, no intra-ISI v(t) trace) and must be scoped as such.
