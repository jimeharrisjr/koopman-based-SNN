# SHD performance demo — architecture sweep

Goal: improve on the first-cut Spiking Heidelberg Digits result (50.2 % test
accuracy, 20 classes, 5 % chance — from `examples/shd_demo.rs`: 100 pooled
channels, one 128-neuron LIF layer, 600 minibatches, evaluated on a
512-sample subset) by varying **network width, depth, input pooling, and
training budget**, and keep an honest log of what worked and what didn't.

Harness: `crates/kdmd-snn/examples/shd_sweep.rs`
(`cargo run --release --features datasets --example shd_sweep [tags…]`).

## Method — what is controlled

- **Identical minibatch sequence for every run**: the data RNG (seed 7) is
  separate from the weight-init RNG (seed 42), so every architecture trains
  on exactly the same sample stream in the same order. Differences in outcome
  are attributable to the architecture, not the data draw.
- **Fixed everything else**: Adam (5e-3, clip 1.0), fast-sigmoid surrogate
  (β = 5), 100 × 10 ms bins (first second of audio), batch 32, LIF with
  τ_m = 20 ms / τ_s = 10 ms.
- **Fixed budget for the comparison runs**: 1500 minibatches (the original
  demo's 600 left the loss still falling, so all variations get the larger
  but equal budget). Long-budget probes (3000) run separately, after the
  equal-budget comparison picks a winner.
- **Full-test-set evaluation**: all 70 complete batches (2240 of 2264 test
  samples), fixed order — tighter than the original 512-sample estimate
  (±~1 % at these accuracies vs ±~2 %).

## The variations

| tag | architecture | pooled channels | budget | rationale |
|---|---|---|---|---|
| A | 1 × 128 | 100 | 1500 | the demo baseline, re-measured under sweep conditions |
| B | 1 × 256 | 100 | 1500 | width: more feature detectors |
| C | 1 × 512 | 100 | 1500 | width, pushed further |
| D | 256 → 128 | 100 | 1500 | depth with a wide first layer |
| E | 128 → 128 | 100 | 1500 | depth at constant layer width |
| F | 1 × 256 | 350 | 1500 | finer input resolution (2:1 pooling instead of 7:1) |
| G/H/I | winner variants | — | 3000 | budget: does more training keep paying? |

Initialization note (a "what didn't work" learned before this sweep, during
Phase 6 testing): hidden layers after the first see sparse spike *volleys*
rather than dense per-bin activity, and with input-layer-scaled weights they
are born dead — the network's loss pins at ln 20 ≈ 3.0 and nothing trains.
The harness therefore scales hidden-layer initial weights ~2.6× stronger
(uniform(0, 90/fan_in) vs uniform(0, 35/fan_in) for the input layer). Dead
configurations, if any remain, are reported as findings, not silently tuned
away.

## Findings log

Full comparison, curves, and caveats: **[RESULTS.md](RESULTS.md)**. Raw
output: `sweep-AF-log.txt`, `sweep-GI-log.txt`. Headline outcomes:

- **Best: 0.680 test accuracy** (350 channels, 1 × 256, 3000 minibatches,
  159 s of training) vs the original demo's 0.502 — a 17.8-point gain.
- **Worked**: more budget (to a point), finer input pooling (the winning
  axis — it also *unlocks* further budget scaling), moderate width.
- **Didn't work**: 512-wide (overfits: best train loss, no test gain);
  doubling budget at coarse 100-channel pooling (test *fell* 0.627 → 0.608
  while train loss improved — budget amplifies memorization when the input
  is information-poor); depth under this uniform training recipe (256→128
  merely matches width at 3.5× cost; 128→128 collapses to 0.484 with a
  near-dead second layer until step ~1100).
- **Interaction finding**: input resolution and training budget must be
  tuned together — the same budget increase that helped at 2:1 pooling hurt
  at 7:1.
