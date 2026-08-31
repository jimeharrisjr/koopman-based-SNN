# PSN-Mode Time-Parallel Training — Engineering Plan (FOR FUTURE CONSIDERATION)

**Document:** 33-psn-parallel-plan.md · **Status: BACKLOG DRAFT — not
registered, no runs authorized** · **Date drafted:** 2026-08-31
**Origin:** docs/24 §4 (the deferral), priced by round 12 (docs/29–30).
This is a plan, not a protocol: per the P0.1 discipline, a frozen
registration (with the gates below calibrated and locked) must be committed
and pushed before any gated run, if and when this item is picked up.

## 1 Why this is worth keeping on the shelf

Round 12 measured the accuracy price of the time-parallelizable model class
(no reset, no recurrence, otherwise fully modern): **2.5 points** vs the AK
mean — five-fold less than predicted — decomposing as reset +2.3,
recurrence ≈ 0. The class already trains 2.4× faster *sequentially*
(feedforward steps), and its members reach 0.86 with the campaign's
standard budget. The trade is real but small, and the restructured
implementation below would compound the speedup substantially while
computing the *identical* model (equivalence-gated, not approximated).

What is traded away and what is kept (from the round-12/13 record and the
discussion in this repo's issue history): the reset-as-control formulation
goes (spike trains become duration-coded threshold readouts of the free
trajectory — amplitude above threshold is clipped, the 2.3-point cost);
the exact propagator, learned τ, attention readout, skip connections, and
the entire DMDc identification pillar stay — identification actually gets
*cleaner*, since the closed-loop own-spike hazard vanishes with the reset.

## 2 The restructuring

For a network that is (feedforward ∧ no_reset), nothing at step t depends
on the network's own spikes before t, so each layer's full trajectory is
computable without a time loop:

**Forward, per layer (layers stay sequential; time parallelizes within
each):**
1. Drive for all steps as one BLAS-3 matmul: `D = W · S_prev`
   (n × n_in)·(n_in × T·batch) — replaces T cache-hostile per-step loops
   and is where most current wall-clock lives (the docs/09-flagged naive
   kernels).
2. State trajectories as elementwise exponential IIR filters (A is
   diagonal-per-variable, including learned-τ variants):
   `i_t = β∘i_{t−1} + b₂∘d_t`, `v_t = α∘v_{t−1} + γ∘i_{t−1} + δ∘d_t` —
   trivially cheap sequentially; thread-parallel via chunked prefix scans
   with elementwise carry-ins (α^chunk precomputable); GPU-shaped by
   construction.
3. Threshold + tape: elementwise over the whole trajectory.
4. Skip drive: same-step, so `D += W_skip · S_{l−2}` is another single
   matmul — skips are parallelism-compatible.

**Backward, mirrored:** surrogate factors elementwise from the taped
v_pre trajectory; λ by the transposed filters (reverse scans); weight
gradients collapse from T rank-1 accumulations to one matmul `G · Sᵀ`;
learned-τ entry gradients accumulate from the same trajectories; the
attention/readout backward is unchanged (already whole-trajectory).

**Compatibility matrix:** learned τ ✓ · attention readout ✓ · static
profile / count / leaky-trace readouts ✓ · skips ✓ · augmentation ✓ ·
batch threading ✓ (composes: chunks × time) · recurrence ✗ · reset ✗
(both rejected loudly by the capability check).

## 3 Work items and effort

| item | effort |
|---|---|
| Capability check + `ParallelTrainer` fast path (forward) | ~half day |
| Backward restructure + tau/attention/skip grad plumbing | ~half day |
| Equivalence gates (see §4) + threading composition tests | ~2 h |
| Profiling pass to calibrate the speed gate before freezing | ~1 h |
| Benchmark run + docs | ~1 h |
| PSN-XL scale arm (registered runs) | ~2 h wall-clock |

## 4 Gates to freeze at registration time (placeholders, calibrate first)

1. **Equivalence:** parallel path reproduces the sequential PSN-mode
   path's spikes exactly and gradients to ≤ 1e-12 relative (bit-exactness
   is off the table — scan association order differs, same situation as
   batch threading, documented identically).
2. **Speed kill-criterion** (docs/09 style): ≥ **X×** end-to-end training
   speedup on the AT configuration (N = 256×2, T = 200, batch 32) versus
   the current 16-thread sequential path, or the parallel path is demoted.
   X to be set from the profiling pass; the drafter's uncalibrated guess
   is X = 4.
3. **The scale arm (the scientific question):** PSN-XL — capacity chosen
   so wall-clock matches today's AK runs (candidates: 512-wide × 3 layers
   with skips, or 2.5 ms × 400 bins) — vs the AK mean 0.8888, 3 seeds,
   ±0.015 bands. Hypothesis: the freed compute buys back the 2.5-point
   parallelism tax. Honest prior: a coin flip.

## 5 Risks

- Hand-rolled backward restructuring is exacting; the equivalence gate is
  the safety net, and the sequential path remains the reference forever.
- Memory: full drive trajectories join the tape (~T·n·batch doubles);
  bounded and known.
- The speedup claim is CPU-measured only; GPU portability is an argument,
  not a deliverable, of this plan.
- Opportunity cost: the accuracy program is concluded and saturated
  (round 13); this item's value is *training cost*, which is currently
  not binding.

## 6 Pick-up triggers

Revisit when any of these becomes true: (a) a longer-sequence target
enters scope (SSC, ms-bin audio, T ≳ 1000), where O(T) unrolls dominate;
(b) a GPU/SIMD port is desired; (c) sweep compute becomes the bottleneck
for a future campaign; (d) the spiking-SSM comparison in the paper needs
a same-library baseline at matched training cost.

Until then this document is the plan of record, and no part of it is
authorized to run.
