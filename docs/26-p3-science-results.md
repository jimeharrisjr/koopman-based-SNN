# P3 Science Studies — Results

**Protocol:** `docs/25-p3-science-preregistration.md` (frozen, committed and
pushed before the runs) · **Date run:** 2026-08-28 ·
**Raw logs:** `demo/probe-richness-log.txt`, `demo/grad-horizon-log.txt`,
`demo/spiking-rom-log.txt` · **Code:** `analysis/probe_richness.py`,
`analysis/spiking_rom.py`, `examples/grad_horizon.rs`.

Deviations from the frozen setups occurred in S-B and S-C and are recorded
in full below (§deviations); S-A ran exactly as registered.

---

## S-A: Probe richness — 4/4 predictions confirmed, with a sharper law

| N | m | full-Â err | excited-subspace err | cond(X) | v-rows | i-rows |
|---|---|---|---|---|---|---|
| 8 | 1 | 8.1e-1 | 2.4e-6 | 4e18 | 0.75 | 0.94 |
| 8 | 4 | 5.7e-1 | 3.7e-6 | 3e17 | 0.51 | 0.70 |
| 8 | 8 | **1.8e-5** | 1.8e-5 | 4e2 | 0.00 | 0.00 |
| 32 | 16 | 5.8e-1 | 5.5e-6 | 9e17 | 0.50 | 0.71 |
| 32 | 32 | **2.3e-4** | 2.3e-4 | 3e4 | 0.00 | 0.00 |

- **PA1 confirmed** (error > 0.1 for m < N/2) — and stronger: error stays
  ≈ 0.5–0.9 for *every* m < N; the transition at m = N is a **cliff**, four
  orders of magnitude in one doubling.
- **PA2 confirmed** — m = N recovers to 1.8e-5 (N=8) / 2.3e-4 (N=32).
- **PA3 confirmed spectacularly** — the excited-subspace error is ~1e-6
  even with a single driven channel. Under-probed identification fails
  *globally, never locally*: everything the probe reaches is learned
  essentially perfectly.
- **PA4 confirmed** — reset feedback partially rescues the voltage block
  (v-row error < i-row error at every m < N; e.g. 0.75 vs 0.94 at m = 1).

**The law, stated for the record:** full identification of a width-N LIF
layer requires all N input directions independently driven — spike-reset
feedback excites voltages but cannot substitute for missing synaptic-drive
rank — and a deficient probe yields an operator that is *trustworthy
precisely on the subspace the data visited* and garbage elsewhere.
Practical corollary for any user of `fit_controlled`: check cond(X), and
interpret Â only on X's numerical range.

---

## S-B: The gradient horizon — a two-regime law (PB1 refuted where it matters)

Instrumentation shipped: `TrainConfig::record_grad_norms` +
`Trainer::grad_norms` (per-layer, per-step batch-mean ‖λ_t‖; single-thread
path). Estimator note: the registered log-slope measured the
rise-to-saturation of the count readout's per-step injection (values ≈ 1.04
–1.13, meaningless as decay); the AR(1) fit `n(d+1) ≈ ρ·n(d) + g` is the
meaningful estimator and was reported alongside, as anticipated in the
protocol's estimator clause.

**Silent regime** (registered 0.6 input scale — which turned out to leave
every net sub-threshold; loss pinned at ln 2):

| τ_m | α | ρ̂ | ρ̂/α |
|---|---|---|---|
| 10 | 0.905 | 0.856 | 0.95 |
| 20 | 0.951 | 0.882 | 0.93 |
| 40 | 0.975 | 0.931 | 0.96 |
| 80 | 0.988 | 0.956 | 0.97 |

**Active regime** (per-τ calibrated drive; nets fire and losses fall):

| τ_m | α | ρ̂ | ρ̂/α | activity |
|---|---|---|---|---|
| 10 | 0.905 | **0.924** | **1.021** | 0.086 |
| 20 | 0.951 | 0.901 | 0.95 | 0.017 |
| 40 | 0.975 | 0.858 | 0.88 | 0.016 |
| 80 | 0.988 | 0.739 | 0.75 | 0.003 |

- **PB1 (ρ̂ ≤ α): REFUTED at τ = 10 in the active regime** — with a
  mechanism: after 30 steps of training, grown recurrent weights add
  backward credit paths (W_recᵀ terms) that push the effective decay
  *above* the leak bound. The refutation is the finding: **the propagator
  spectrum does not cap the credit horizon of a recurrent SNN — learned
  recurrence buys horizon back**, which is exactly why round 2's
  recurrence was worth +12.8 points. Confirmed everywhere else.
- **PB2 (τ80 horizon ≥ 3× τ10): CONFIRMED in the silent regime**
  (ρ̂ monotone in τ; horizon ratio 3.45) — **and inverted in the active
  regime**, where
  τ = 80's ρ̂ = 0.74 gives a *shorter* horizon than τ = 10's 0.92:
  once spiking, the surrogate and reset-path factors dominate the leak
  entirely.
- **PB3 (ρ̂ ∈ [0.5α, α]): confirmed in the silent regime and at
  τ ∈ {20, 40, 80} active; the τ = 10 exception exits through the top**
  (shared with PB1).

**The two-regime law, stated for the record:** sub-threshold, backward
gradients decay at 0.91–0.97 of the spectral prediction α (the small gap
is the reset-derivative path, which attenuates even without spikes). In
operation, the effective decay is set by activity-dependent surrogate
factors and learned recurrence, which can invert the τ-ordering and even
exceed the leak bound. The V4 idea — spectral regularization to extend
credit horizons — is therefore mis-aimed at the *neuron* spectrum: the
lever that actually controls horizons in operation is the *recurrent
weight* spectrum. This closes V4 as originally posed and re-derives, from
gradient measurements, why recurrence was the campaign's biggest single
win.

---

## S-C: Spiking-regime ROM — the rate law is INVERTED from the prediction

(Corrected library-convention propagator; see deviations.)

| rate | r=8 | r=16 | r=32 | r=64 (=N) | r=128 (=2N) |
|---|---|---|---|---|---|
| 3.6% | 0.12 | 0.13 | 0.17 | 0.18 | **1.000** |
| 8.0% | 0.30 | 0.34 | 0.37 | 0.41 | **1.000** |
| 20.3% | 0.71 | 0.77 | 0.83 | **0.95** | **1.000** |

(cells: spike coincidence ±2 bins; rate errors at r = N: 15%/7%/4%.)

- **PC1 confirmed** — r = 2N is exact (coincidence 1.0000, rate error 0).
- **PC2 REFUTED — inverted:** at every r < 2N, fidelity *improves* with
  firing rate. docs/01 §5-Q7 predicted degradation with rate; the truth is
  the opposite, and the mechanism is clear in hindsight: sparse spikes are
  rare threshold-grazing events whose timing flips under microscopic
  truncation error, while strong drive forces robust, reproducible
  crossings. **Reduced-order spiking models fail hardest exactly where
  SNNs are prized — the sparse regime.**
- **PC3 confirmed** at the medium drive (0.41 at r = N < 0.8) — the
  V2-style pessimism held there — but the high-drive column delivers the
  registered "publishable positive": **a half-state ROM reaches 0.95
  coincidence at 20% firing.**
- **PC4 confirmed at medium/high drives** (rate error 6.7%/4.1% at r = N
  while coincidence sat at 0.41/0.95) and **missed at low drive** (15.4%
  vs the predicted ≤ 10%): reduced models lose timing before statistics,
  except in the sparse regime where they lose both.

**Consequence for the V1 value case:** rank-r compression of spiking
recurrent layers is viable for high-rate networks and rate-level questions,
and unviable for sparse, timing-coded ones at useful ranks. The library's
sub-threshold-validated ROM claim stands unchanged; this maps its frontier.

---

## Deviations (recorded per the amendment policy)

1. **S-C propagator convention (bug-level):** the first run implemented a
   current→voltage coupling 20× the library's (γ ≈ 0.93 vs 0.047),
   saturating every drive at ~0.47 spikes/step (that run's table heads the
   raw log). The study was rerun with `Lif::a_local`'s exact convention
   and the harness-style [δ, 1−β] input coupling.
2. **S-C drives:** the registered targets (2%/8%/20%) were realized as
   measured rates 3.6%/8.0%/20.3% via drive probabilities 0.08/0.13/0.28
   ("reported as measured" per protocol).
3. **S-B input scale:** the registered 0.6 scale left all nets silent
   (loss = ln 2); that run is retained as the *silent-regime* table — a
   fortunate accident that produced the clean linear-law measurement —
   and the active regime uses per-τ calibrated scales (1.5/2.1/3.5/4.4),
   needed because the LIF's DC gain is τ-independent while its fluctuation
   variance low-passes with τ.
4. **S-B estimator:** the registered log-slope is reported but measures
   injection saturation; the AR(1) fit carries the conclusions (both in
   the log).

## P3 status

Items 1–3 measured and closed; item 4 remains closed (docs/08 standing
decision); item 5 remains moot. The three studies produced three
for-the-record laws: the m = N probe cliff with local trustworthiness
(S-A), the two-regime gradient-horizon law with recurrence beating the
leak bound (S-B), and the inverted rate law for spiking ROMs (S-C). Each
corrects or sharpens a claim in docs/01 — and each belongs in the paper
revision.
