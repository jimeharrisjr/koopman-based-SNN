# kdmd-SNN

A Rust library for **spiking neural networks built on the Koopman/DMD
linear-plus-control formulation**, developed on top of the
[`koopman-dmd`](https://crates.io/crates/koopman-dmd) crate.

Every layer advances through

```text
x_{t+1} = A·x_t + B·u_t        (linear state advance, exact for LIF/adLIF)
s_t     = Θ(v_t − θ)           (threshold — the network's only nonlinearity)
```

with the **subtractive reset folded into the control input**, which makes the
step exactly linear-plus-control: identification is unbiased, and the training
Jacobian through the linear part is exact.

## What this library actually claims (and what it does not)

This project pre-registered its scientific claims and killed the ones that
failed. The result is a smaller, honest set:

| Claim | Status |
|---|---|
| **Exact-linear fast engine** — LIF layers step through the closed-form propagator (`A_local ⊗ I_N`, O(N)); spike-for-spike agreement over 1000 steps is test-gated, and a single benchmark run measured 1.02× the reference simulator's cost (docs/09). adLIF shares the closed-form structure but has no fast-path integration test yet | ✅ shipped |
| **DMD/DMDc identification** — recovers the LIF propagator from data to ≤ 1e-8 (the permanent oracle test); known-B DMDc identifies A from *full spiking trajectories* with no masking | ✅ shipped |
| **V1: reduced-order recurrent layers** — DMDc's rank-r output basis compresses a dense recurrent layer to O(N·r) stepping, within 10 % rollout RMSE in the sub-threshold (non-spiking) regime — the exact gate is `tests/reduced_order.rs`; spiking-regime reduction is future work | ✅ shipped |
| **V3: spectral diagnostics** — identified operators come with spectrum, stability, per-mode timescales (via koopman-dmd's analysis suite, exercised by the identification gates) | ✅ shipped |
| **Surrogate-gradient training** — hand-rolled BPTT (no autograd dependency); learns the synthetic Poisson-pattern task to ≥ 90 % held-out accuracy; the SHD demo reached 50.2 % test accuracy (20 classes, 5 % chance) in one measured run | ✅ shipped |
| Per-step **lifted (EDMD) surrogates of nonlinear neurons** | ❌ **negative result** — fails ±2 ms spike timing via cumulative phase drift ([docs/05](docs/05-v2-results.md)) |
| **Spike-to-spike return-map surrogates** of nonlinear neurons | ❌ **negative result** — perfect timing *inside* the training envelope, fails at its edges ([docs/08](docs/08-v2b-results.md)) |

Both negative results were pre-registered experiments with frozen thresholds
(docs/04, docs/07); the first was additionally put through an adversarial
audit (docs/06). They are part of the repository's contribution. One honest
caveat: the repository currently has no commit history, so the
protocol-before-results ordering rests on the documents' internal dating
rather than commit-hash evidence — committing docs and results separately is
the first step toward making the discipline externally verifiable.

## Building

The crate builds standalone against crates.io
([`koopman-dmd` 0.2.0](https://crates.io/crates/koopman-dmd) carries the DMDc
solver this project contributed upstream).

Prerequisites:

- **Rust 1.85+** (MSRV, inherited via `koopman-dmd` → `faer`).
- **cmake** — only for `--features datasets` (HDF5 is compiled statically; no
  system libhdf5 is needed).

```sh
cargo test --release          # library + oracle/equivalence/gradient tests
cargo bench                   # criterion benches behind docs/09
```

## Quick start

```rust
use faer::Mat;
use kdmd_snn::neuron::{Lif, LifParams};
use kdmd_snn::{KoopmanLayer, Network, SnnError, SpikeVec};

fn main() -> Result<(), SnnError> {
    // A LIF layer whose operator IS the closed-form propagator.
    let lif = Lif::new(LifParams::default())?;
    let n = 128;
    let w = Mat::from_fn(n, 16, |_, _| 0.5);
    let layer = KoopmanLayer::lif(&lif, n, w, 1)?;
    let mut net = Network::new(vec![layer], 1)?;

    // Drive it with spikes.
    let input = SpikeVec::from_indices(vec![0, 3, 7], 16)?;
    for _ in 0..100 {
        let out = net.step(&input)?;
        println!("{} spikes", out.count());
    }
    Ok(())
}
```

Training (surrogate-gradient BPTT, batch mode):

```rust,ignore
use kdmd_snn::{TrainConfig, Trainer};

let mut trainer = Trainer::new(&net, n_classes, TrainConfig::default())?;
let stats = trainer.train_step(&mut net, &input_batches, &targets)?;
println!("loss {} accuracy {}", stats.loss, stats.accuracy);
```

Identification (fit A from data, validated):

```rust,ignore
use kdmd_snn::identify::{fit_controlled, lif_structural_b, IdentifyConfig, RankPolicy};

// B is known by construction — never fitted (spikes are state feedback,
// so joint estimation is biased; see docs/03 C3).
let b = lif_structural_b(&lif, n)?;
let fit = fit_controlled(&snapshots, Some(b), None, &IdentifyConfig::default())?;
```

## The SHD demo

From the repository root (the demo resolves `data/shd/` relative to the
current directory):

```sh
cargo run --release --features datasets --example shd_demo
```

Downloads the [Spiking Heidelberg Digits](https://zenkelab.org/datasets/)
(~200 MB, via `curl` + `gunzip`; skipped if `data/shd/shd_train.h5` and
`shd_test.h5` are already present), trains a pooled-channel LIF layer +
readout with surrogate BPTT, and reports test accuracy (20 classes, 5 %
chance). The `datasets` feature builds HDF5 statically (needs `cmake`).

## Repository map

- `crates/kdmd-snn/` — the library (neurons, layers, network, identification,
  training, surrogates, benches).
- `IMPLEMENTATION_PLAN.md` — the phased plan, including the owner decisions
  and the pre-registered kill criteria that shaped the scope above.
- `docs/` — the project story in reading order; see
  [docs/README.md](docs/README.md) for the index. Highlights:
  - `docs/01…03` — scientific foundations (with ~30 verified citations),
    system architecture, and the adversarial review that re-scoped the
    project.
  - `docs/04…08` — the two pre-registered experiments (protocols, results,
    and the skeptic audit of the first failure).
  - `docs/09` — the Phase 5 benchmark-gate results.
  - `docs/10+` — final review reports (Phase 6 close-out).

## Relationship to koopman-dmd

The identification layer reuses `koopman-dmd` throughout (`dmd`, lifting,
spectrum/stability/residual analysis, `pinv`). The **DMDc solver** this
project needed was contributed upstream and released as
[`koopman-dmd` 0.2.0](https://crates.io/crates/koopman-dmd)
([rust-dmd#9](https://github.com/jimeharrisjr/rust-dmd/pull/9)), together with
two upstream bug fixes found during review.

## License

MIT, matching `koopman-dmd`. MSRV 1.85.
