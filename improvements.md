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

1. **Augmentation variety.** ⚫ *Run 2026-08-28 — NEGATIVE at the tested
   operating point* (pre-registered docs/20, results docs/21): AK + channel
   blocks/time masks/noise events lost 1.6 pts with the manipulation check
   passed (train loss ~2×) — the untuned strengths destroy signal rather
   than regularize. Axis closed at this operating point; softer strengths
   would need a new registration. Companion result: **the AK ensemble ×3
   (AN) hit exactly 0.9000 — the campaign's first honest number in the
   published band.**
2. **The combination round** (attention × learned τ × 5 ms bins).
   ✅ *Run 2026-08-27 — POSITIVE and SUPERADDITIVE* (pre-registered
   docs/18, results docs/19): mean(AK) = 0.8888 ± 0.019, +3.3 over the X
   control and +5.1 over the additive expectation; AL (5 ms alone) NULL;
   mechanisms 3/3; predictions 4/4. First campaign run over 0.90 (seed
   200: 0.9116). **AK is the new default recipe.** The interaction
   attribution is confirmed: features that were individually null are
   jointly worth five points. Next registered rounds build on AK:
   item 1 (augmentation variety) and an AK-recipe ensemble.
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
5. **Depth's enabling ingredient.** ◑ *Run 2026-08-28 — ENABLER CONFIRMED,
   depth axis closed* (pre-registered docs/22, results docs/23): a plain
   third layer on AK loses 2.5 pts (AE replicated); zero-init skip
   connections recover 3.0 of them (W_skip grown to 3.6–4.4, gate 3/3) but
   enabled depth only ties two layers (+0.5, NULL). Skips ship validated
   in the library (`with_skip`) for any larger-scale attempt.
6. **Diverse ensembles.** ✅ *Run 2026-08-28 — DIVERSITY WINS, new campaign
   headline* (docs/22–23): {AK, AJ, X} summed logits = **0.9179**, three
   points above its best member and +1.8 over the homogeneous AN, from
   members averaging just 0.873. Decorrelation across readouts and bin
   widths beats member strength. Eligible registered follow-up: diversity
   WITH member strength.

**P1 status: CONCLUDED (2026-08-28).** Every item carries a registered
outcome; campaign scoreboard 0.502 → **0.9179**. Remaining open work: P2
(parallel-in-time, snnTorch baseline), P3 (identification science), and
the paper revision (draft still ends at round 5's 0.88).

## P2 — Library engineering

1. **Parallelism.** ✅ *CLOSED 2026-08-28 (docs/24).* Batch threading
   shipped 2026-08-27 (~9× at 16 threads; threads = 1 stays the bit-exact
   recorded path). Parallel-in-time is **structurally unavailable** for
   this model class: reset-as-control (plus recurrence and the attention
   readout) closes the state→threshold→input loop every step — the same
   structure that gives exactness forces sequentiality. The reset-free
   "PSN-mode" alternative is documented and deferred behind a registration
   requirement (docs/24 §4).
2. **adLIF fast-path integration test.** ✅ *CLOSED 2026-08-28.* The README
   claim was stale (800-step spike-for-spike reference gates existed); the
   genuine gap — the batched training path — is now gated
   (`adlif_sparse_and_batch_paths_agree`) and the claims table corrected.
3. **snnTorch baseline.** ✅ *CLOSED 2026-08-28* (`baselines/README.md`):
   direct transplant 0.60–0.76 (below the pre-stated 0.80–0.88 band — the
   miss is recorded), gain-corrected diagnostic 0.73–0.84. Calibration
   conclusions: the protocol transfers (our numbers aren't a harness
   artifact); hyperparameters do NOT transfer across sub-threshold
   integrator conventions (RSynaptic's coupling gain 1.0 vs the exact γ =
   0.239 — the sensitivity the exact propagator exists to remove); and no
   claim of engine superiority is made (home-field tuning advantage
   disclosed).

**P2 status: CONCLUDED (2026-08-28).** All three items closed (docs/24,
the adLIF gate, `baselines/`).

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
