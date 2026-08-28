# Parallel-in-Time Training — Design Assessment (P2.1 close-out)

**Document:** 24-parallel-in-time-assessment.md · **Date:** 2026-08-28
**Origin:** improvements.md P2.1's second rung: after batch-level threading
shipped (~9× wall-clock), "parallel-in-time training via the
linear-state-space form — the spiking-SSM trick" remained open. This
document resolves that item as a **design decision with a negative
feasibility core and a scoped, deferred alternative** — recorded so nobody
re-derives it.

## 1 The question

The spiking-SSM literature (PSN, SpikingSSMs, SiLIF) trains linear-state
SNNs *parallel across time*: all T steps of the sub-threshold recursion
computed at once (parallel scan / FFT-style), instead of the T-sequential
BPTT unroll this library uses. Our layers are exactly linear between
spikes — so can we have that speedup?

## 2 The answer for THIS model class: no — and it is structural, not an
implementation gap

A parallel scan computes `x_{t+1} = A·x_t + B·u_t` for all t in O(log T)
depth **only when the inputs u_t are known in advance**. In this library's
formulation, `u_t` *contains the network's own thresholded output*:

- the **subtractive reset** feeds `s_t = Θ(v_t − θ)` back into the very
  next state — computing step t requires knowing every spike before t;
- **recurrence** (`W_rec`, the campaign's single largest accuracy
  ingredient, +12.8) feeds `s_{t−1}` into the drive at t;
- the **spike-driven attention readout** (round 7/8) needs the realized
  spike trains, which need the sequential rollout.

The dependency chain state → threshold → input → state closes at every
step. This is precisely why PSN **removes the reset** and why the spiking
SSM papers restructure the neuron so the nonlinearity sits *outside* the
recursion. The exactness-and-honesty premise of this library — the reset
folded into `u` so the dynamics stay exactly linear-plus-control — is the
same structure that makes the dynamics inherently sequential in time.
You can have "exact spiking dynamics with resets" or "parallel in time,"
not both. This is a theorem-shaped fact about the model class, not a
missing optimization.

## 3 What parallelism this library DOES support, and what it cost

| axis | status | measured |
|---|---|---|
| Batch (data-parallel chunks) | ✅ shipped (`TrainConfig::threads`) | ~9× on the R recipe at 16 threads; bit-exact serial path preserved |
| Ensemble members / seeds | ✅ trivially parallel at the process level | used throughout rounds 8–10 |
| Time | ❌ structurally unavailable for reset+recurrent dynamics | — |

The batch axis already saturates the available cores for every recipe in
the campaign; a time axis would add nothing until batch parallelism is
exhausted (it is not, on any hardware this project targets).

## 4 The scoped alternative, if anyone ever wants it

A **reset-free, feedforward "PSN-mode" layer** (no reset in `u`, no
`W_rec`, threshold applied to the parallel-computed sub-threshold
trajectory) WOULD scan in parallel. But it is a *different model class*:
- it abandons the exact reset-as-control formulation (the library's
  central claim), and
- the campaign's record says its ingredients cost accuracy here
  (recurrence is +12.8; the reset is load-bearing for LIF dynamics).

Adopting it would be a model-change study requiring its own
pre-registration (accuracy vs wall-clock trade), not an engineering patch.
Nothing in the current roadmap justifies that study; it is recorded here
as the known path should training cost ever become the binding constraint
(e.g., much longer sequences, where O(T) unroll dominates).

## 5 Minor engineering notes banked while closing this item

- The batched drive `D = W·S` is still the naive loop flagged in docs/09;
  a faer-matmul path remains the first micro-optimization candidate if
  per-step cost ever matters again. Not taken now: the measured bottleneck
  at 16 threads is core count, not kernel efficiency.
- P2.2 (adLIF fast-path gate) closed alongside this assessment: the
  README's "no fast-path integration test yet" claim was stale (800-step
  spike-for-spike reference gates existed), and the one real gap — the
  batched path — is now gated too (`adlif_sparse_and_batch_paths_agree`).

**P2.1 status: CLOSED** — batch axis shipped and measured; time axis
structurally unavailable for this model class; the reset-free alternative
documented and deferred behind a registration requirement.
