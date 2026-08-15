# Phase 5 Benchmark Gate — Results

**Date:** 2026-08-14 · **Bench:** `crates/kdmd-snn/benches/snn_bench.rs` (criterion,
Apple Silicon, f64, single thread) · **Gate:** IMPLEMENTATION_PLAN.md pre-registered
kill criterion (owner-accepted, Q8): *the fitted-operator inference path must be within
~2× of the plain-LIF baseline per step at equal accuracy, or it is demoted.*

## Equivalence (accuracy side of the gate)

`tests/layer_equivalence.rs`: the fast `KoopmanLayer` (PerVariable operator = the
closed-form LIF propagator) reproduces the reference simulator **spike-for-spike over
1000 steps** (N = 64, random weights, Poisson-like input, ~8 % activity), with state
agreement at ≤ 1e-12 relative. Dense and structured operator variants produce identical
spike trains. Equal accuracy is exact, not approximate.

## Wall-clock (N = 1024, T = 100 steps, ~8 % input activity, single thread)

| Path | time / trial | ratio vs reference | gate |
|---|---|---|---|
| Reference LIF simulator | 3.52 ms | 1.00× | — |
| **KoopmanLayer, PerVariable (product config)** | 3.61 ms | **1.02×** | **PASS (≤ 2×)** |
| KoopmanLayer, Dense fitted operator | 965 ms | 274× | **FAIL → demoted** |

**Consequences, per the pre-registered rule:**
- The **structured** fast path (`A_local ⊗ I_N`, sparse spike drive) is the supported
  inference engine: identical output to the reference at parity cost, with the layer
  abstraction, batching, and the training seam on top.
- The **dense fitted-operator** inference path is demoted, exactly as the skeptic's C2
  analysis predicted before any code existed: a dense identified A can never compete
  with a diagonal-structured update for homogeneous layers. Dense remains available for
  analysis and for the Phase 6 LowRank/reduced-order work (value case V1), where rank-r
  structure — not density — carries the claim.

## Supporting kernels

- `step_kernels`: PerVariable apply scales linearly — 106 ns (N=256), 404 ns (N=1024),
  1.51 µs (N=4096); Dense is 9.55 ms at N=1024 (~23,600×), growing quadratically.
- `spike_drive` (N=1024): sparse column accumulation beats this dense mat-vec at every
  measured activity (1 %: 2.5 µs vs 1.04 ms; 50 %: 293 µs vs 1.04 ms). *Caveat:* the
  dense baseline here is a naive row-major loop over a column-major matrix
  (cache-hostile); a faer-matmul dense path would move the predicted ~25 % crossover
  back into view. Logged as the target of a later optimization pass — it does not
  affect the gate, which compares end-to-end paths using their intended kernels.
