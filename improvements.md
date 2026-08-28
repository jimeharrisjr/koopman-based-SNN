# Improvements — prioritized roadmap

**Status:** proposed 2026-08-27, after the round-5 verdict (`demo/RESULTS.md`)
and the draft paper (`paper/kdmd-snn-paper.md`). Round 5 measured every axis
the current harness exposes; each is exhausted or negative. Closing the
remaining gap is a **library-feature project, not a sweep away**. Items below
are ordered by expected value within each track; the tracks themselves are
ordered so that cheap hygiene lands first and speculative science last.

Ground rule inherited from the seed audit (±2.7 points across weight-init
seeds, `demo/RESULTS.md` round 5): **no new idea is accepted on a single
seed.** Every accuracy claim below gets ≥ 3 seeds (`seed_bump`) or an
ensemble, and margins under ~3 points are reported as inconclusive.

---

## P0 — Methodology debt (cheap; do before any new experiment)

1. **Hash-verifiable pre-registration.** ✅ *Adopted 2026-08-27: the
   learned-τ protocol (`docs/14`) was committed and pushed before its runs.*
   Commit each new protocol in its own commit *before* the results commit,
   so protocol-before-results ordering no longer rests on documents'
   internal dating. Flagged as the record's one weak point in `README.md`
   and the paper's limitations.
2. **Multi-seed reporting as the default.** ✅ *Done 2026-08-27:
   `shd_sweep --seeds N` runs seed_bump offsets 0/100/200/… and prints an
   AGGREGATE mean ± half-range line per tag.*

## P1 — Accuracy on SHD (closing ~3.5 points to the 0.90–0.94 band)

1. **Augmentation variety — now the top item** (was #3). The budget ceiling
   is real: 12,000 minibatches overfit even with the current augmentation
   (tag U). New corruptions to try cheaply in the harness: channel dropout,
   additive spike noise, mixup-style event-stream blends. More epochs is a
   dead lever. Promoted by the registered consequence of round 7 (docs/17).
2. **The combination round** (attention × learned τ × 5 ms bins): after four
   single-axis literature features came back null/negative *with engaged
   mechanisms* (adaptation r4, learned τ r6, static profile r7, attention
   r7), the standing attribution is that the ~4-point gap to 0.90+ lives in
   joint interactions. One registered round combining the engaged-but-null
   features is the designated test of that attribution.
3. **A trained temporal readout.** ⚫ *Run 2026-08-27 — AI NEGATIVE
   (−2.0), AJ NULL (+0.3)* (pre-registered docs/16, results docs/17). Both
   mechanisms engaged; a learned static profile is actively harmful, and
   spike attention buys no mean accuracy — though its seed spread was ~3×
   tighter than the control's (worst member 0.8504 vs 0.8165; post-hoc,
   n = 3). Both modes ship in the library (`ReadoutMode`, inert by
   default). Attention returns in the combination round.
4. **Learned per-neuron time constants.** ⚪ *Run 2026-08-27 — NULL*
   (pre-registered docs/14, results docs/15): mean(AG) − mean(X) = −0.0137
   over 3 seeds, inside the ±0.015 band, with the mechanism strongly engaged
   (τ distributions spread across the full clamp range). Implementation
   ships in the library (`learn_tau`, inert when off). Returns in the
   combination round (item 2).
5. **Depth's enabling ingredient.** A third layer lost ~2 pts at 1.5× cost
   (tag AE); the pattern all campaign is that each depth increment needs a
   new ingredient. One focused experiment each: per-layer learning rates,
   skip connections, a normalization analog. Not a sweep.
6. **Diverse ensembles.** Same-architecture members correlate too much (AF
   bought robustness, not a leap). Ensemble *different* recipes: one-layer,
   two-layer, attention-readout (its tight seed spread makes it a natural
   ensemble member), 5 ms bins.

## P2 — Library engineering

1. **Parallelism.** *Batch-level threading done 2026-08-27
   (`TrainConfig::threads` / `--threads N`): ~9× on the R recipe at 16
   threads (38 steps/s vs 4.1 serial); threads = 1 stays the bit-exact
   recorded path. Parallel-in-time training via the linear-state-space form
   (the spiking-SSM trick) remains open.*
2. **adLIF fast-path integration test.** The closed-form structure is shared
   but the fast path has no spike-for-spike integration gate yet
   (`README.md` claims table). Required before any learned-τ ALIF result is
   trusted.
3. **snnTorch baseline.** External calibration point recommended in
   `docs/10`; run the same SHD protocol on a mainstream framework to anchor
   our numbers.

## P3 — Science: identification and the Koopman side

1. **Reduced-order models in the spiking regime.** The rank-r compression
   (V1) is validated only sub-threshold (≤ 10 % rollout RMSE,
   `tests/reduced_order.rs`). Extending it across spikes is the best-posed
   open problem the project owns; the paper (§6) names it first.
2. **V4 spectral-regularization experiment.** Constrain fitted eigenvalues
   toward the unit circle and *measure* whether credit-assignment horizons
   lengthen. Converts the corrected gradient claim (paper §4.5: diagnosis,
   not prevention) into a positive, testable hypothesis. Pre-register it.
3. **Probe-richness quantification.** The paper's identification experiments
   (§3.2) show excitation dominates identifiability (constant drive →
   cond(X) ~ 1e19, unrecoverable). The minimal probe richness for a given
   network scale is unquantified; a small study would make the
   identification API's requirements concrete.
4. **Nonlinear-surrogate track — reopen only with the two named
   ingredients.** V2/V2b closed as pre-registered failures (`docs/05`,
   `docs/08`). The record points at exactly two missing pieces: a dedicated
   approach-flow model, and censored/no-spike supervision for quiescence.
   Reopening requires a new pre-registration per the standing decision;
   otherwise the track stays closed.
5. **Staleness schedule for fitted operators.** If any future work trains
   *through* a fitted Â of a recurrent layer, it needs a re-fit schedule
   (alternation or streaming DMD). No prior art; currently moot because
   training uses only the closed-form operator.

---

## Recommended first move

**Learnable τ on the two-layer recipe, 3 seeds** (P1.1), preceded by the P0
items and P2.1's batch threading so the runs are cheap and the result is
believable. It is the one feature that plausibly unlocks three negative
results at once: adaptation (round 4), finer bins (AB), and the gap to the
published state of the art.
