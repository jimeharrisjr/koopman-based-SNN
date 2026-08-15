# V2b Experiment Results — Return-Map Surrogate (the Bounded Rescue)

**Date run:** 2026-08-14 · **Protocol:** `docs/07-v2b-preregistration.md` (frozen) ·
**Harness:** `crates/kdmd-snn/examples/v2b_return_map.rs` · **Raw output:** `docs/08-v2b-results-raw.txt`

## VERDICT: FAIL

No (degree, family) configuration passed all gated currents for either RS or FS. Per the
owner's standing decision (docs/07 §6): **immediate pivot to V1/V3/V4 — the
nonlinear-surrogate track closes.** V2 and V2b both stand as documented negative results.

## The one striking positive (recorded because honesty cuts both ways)

At the interior interpolation current I = 10, the ISI map delivered **perfect
spike-train prediction for both gated types**:

| Type / config | coincidence ±2 ms | Γ | A5 bias | first-spike err |
|---|---|---|---|---|
| RS, deg 3 Family B | **1.000** | **1.000** | +0.017 % | 0.26 ms |
| RS, deg 2 Family B | **1.000** | **1.000** | −0.098 % | 0.50 ms |
| FS, deg 4 Family B | **1.000** | **1.000** | +0.157 % | 0.08 ms |

Every reference spike matched within ±2 ms across a full second — something V2's
step-by-step surrogate never approached (its best was 0.13 at chance level). The core
V2b mechanism — regressing the inter-spike interval directly so per-cycle error is the
fitted quantity — **works where the data measure covers the dynamics.** The experiment
failed on its edges, not its center.

## Why it failed (three specific mechanisms, all pre-registered threats)

1. **The first-spike/approach map is bad everywhere (threat T2).** Held-out
   time-to-first-spike relative RMS ran 23–58 % across every configuration — two orders
   above the ISI map's error — so the P-C precondition (≤ 5 %) failed for **all 12 gated
   configurations**. The approach flow from an arbitrary initial state is genuinely
   harder than the on-cycle return map: transients cover a 2-D region (not a 1-D
   section), each trajectory contributes one segment, and the polynomial fit
   under-resolves it.
2. **Quiescence extrapolation failed completely (threat T3).** The surrogate emitted
   spurious spikes at I = 2 in every configuration (1–24 of them), and the in-sample
   I = 3 quiescence report was 0/10 everywhere: the first-spike map, trained only on
   spiking segments (a censored-data problem with no "no-spike" labels), happily
   predicts finite latencies below rheobase. The registered T_max mechanism never
   triggered because predicted latencies stayed small.
3. **The rheobase edge and extrapolation (threats T1, R5).** I = 6 (nearest the
   saddle-node) missed narrowly at the best configs (RS deg-3 B: coincidence 0.714 vs
   0.80, A5 bias −0.34 % vs 0.25 %); I = 13 failed via a new, instructive route: RS
   deg-3 B had teacher-forced bias of only −0.31 %, yet closed-loop coincidence
   collapsed to 0.13 — **closed-loop amplification through the u₊-feedback**, exactly
   prediction R5's registered failure clause. Several configs also produced negative
   predicted intervals at I = 13 (correctly treated as INVALID, not quiescence).

**Protocol finding (P-A collinearity).** With only 4 spiking training currents, any
dictionary holding ≥ 5 pure-I functions ({1, I, I², I³, g}) is exactly collinear on 4
nodes: degree-3/4 Family B and degree-4 Family A could never satisfy the full-rank
precondition. Foreseeable by counting; registered here so any future protocol pairs the
current grid with the dictionary size.

## Predictions scorecard

| # | Prediction | Outcome |
|---|---|---|
| R1 | RS passes all interpolation gates at some Family B config | **REFUTED** — I = 10 passed fully, I = 6 never did |
| R2 | Family B beats A on extrapolation bias at matched degree | **CONFIRMED** where comparable (deg 3: A invalid vs B −0.31 %) |
| R3 | I = 2 passes for Family B; some Family A config emits a spurious spike | **HALF-REFUTED** — Family A misfired as predicted, but Family B also emitted spurious spikes; the clamp design is wrong (informative per the registered clause) |
| R4 | CH fails with the same machinery | **CONFIRMED** — burst sections produced invalid/collapsed predictions across the board |
| R5 | teacher-forced and closed-loop errors agree | **REFUTED at I = 13** — closed-loop amplification is a real, newly recorded structural obstacle |

## Deviations (amendment policy)

- The registered data floors were unmet at the default trajectory budget; the §2
  contingency (extend trajectories) was applied — budget raised until floors met
  (printed per configuration in the raw output). Thresholds unchanged.
- C3 wall-clock: report-only per the registration; flop gates decided by orders of
  magnitude (all configs passed C1 with 0.3–80 flops/ms vs caps of 4,800–38,450).

## Consequence (executed immediately, per the standing decision)

- The nonlinear-surrogate track (V2 + V2b) closes as a pair of documented negative
  results with one genuine methodological finding: *event-level return-map regression
  achieves perfect ±2 ms spike-train reproduction inside the training current envelope,
  and fails at its edges via approach-flow error, censored quiescence data, and
  closed-loop feedback amplification.* Worth writing up on its own.
- Phase 5 proceeds now with the exact-linear LIF/adLIF engine; the library's fitted-
  operator value cases are V1 (reduced-order recurrent layers), V3 (spectral
  diagnostics), V4 (spectral transparency).
- If anyone revisits this later, the results point at exactly two missing ingredients:
  a dedicated approach-flow model (not a shared polynomial regression), and explicit
  censored/no-spike supervision for quiescence. Both are out of scope by the standing
  decision.
