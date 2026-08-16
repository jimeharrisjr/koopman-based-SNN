//! SHD architecture sweep: vary width, depth, input pooling, and budget;
//! compare test accuracy under controlled conditions. Results and findings
//! are logged in the repository's `demo/` folder.
//!
//! Fairness controls: every experiment trains on the IDENTICAL minibatch
//! sequence (data RNG seeded separately from weight-init RNG), with the same
//! optimizer settings, time binning, and horizon; evaluation covers the full
//! test set (all complete batches).
//!
//! Run all:      cargo run --release --features datasets --example shd_sweep
//! Run a subset: cargo run --release --features datasets --example shd_sweep A C G
//!
//! Requires data/shd/*.h5 (run shd_demo once, or see its download logic).

use std::path::PathBuf;
use std::time::Instant;

use faer::Mat;
use kdmd_snn::data::shd::{bin_events, load_shd, ShdSample};
use kdmd_snn::neuron::{AdLif, AdLifParams, Lif, LifParams};
use kdmd_snn::{KoopmanLayer, Network, SpikeBatch, TrainConfig, Trainer};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const N_CLASSES: usize = 20;
const BIN_S: f64 = 0.010;
const T_STEPS: usize = 100;
const BATCH: usize = 32;
const DATA_SEED: u64 = 7; // shared: identical minibatch sequence for all runs
const INIT_SEED: u64 = 42;

struct ExpConfig {
    tag: &'static str,
    name: &'static str,
    n_pooled: usize,
    hidden: &'static [usize],
    minibatches: usize,
    /// Zero-initialized recurrent connections within each hidden layer.
    recurrent: bool,
    /// Label-uniform minibatch sampling instead of sample-uniform.
    balanced: bool,
    /// Multiply the learning rate by `.1` at step `.0`.
    lr_decay: Option<(usize, f64)>,
    /// Train-time data augmentation (event dropout + channel shift + time
    /// stretch on the raw event stream; test data is never augmented).
    augment: bool,
    /// Decoupled weight decay λ (0 disables).
    weight_decay: f64,
    /// Neuron model for the hidden layers.
    neuron: NeuronKind,
    /// Added to both the init and data seeds (multi-seed error-bar runs).
    seed_bump: u64,
    /// Bin width (s) and step count — the time resolution / duration axis.
    bin_s: f64,
    t_steps: usize,
    /// Leaky-trace readout decay κ (None = uniform count readout).
    readout_decay: Option<f64>,
    /// Number of independently seeded members trained; evaluation sums their
    /// logits (1 = a single model).
    ensemble: usize,
}

#[derive(Clone, Copy, PartialEq)]
enum NeuronKind {
    /// Plain LIF (k = 2), the rounds-1–3 model.
    Lif,
    /// Adaptive LIF (k = 3): spike-triggered adaptation, homogeneous τ.
    Alif,
    /// Adaptive LIF with per-neuron time constants
    /// (τ_m ~ U(10,40), τ_s ~ U(5,15), τ_w ~ logU(60,400) ms).
    AlifHetero,
}

/// Round-1 defaults (A–I ran with these).
const BASE: ExpConfig = ExpConfig {
    tag: "",
    name: "",
    n_pooled: 100,
    hidden: &[128],
    minibatches: 1500,
    recurrent: false,
    balanced: false,
    lr_decay: None,
    augment: false,
    weight_decay: 0.0,
    neuron: NeuronKind::Lif,
    seed_bump: 0,
    bin_s: BIN_S,
    t_steps: T_STEPS,
    readout_decay: None,
    ensemble: 1,
};

/// Adaptation defaults for the ALIF rounds (dt = 10 ms bins): τ_w = 150 ms
/// (15 bins), increment 0.1 of θ per spike.
const ALIF_TAU_W: f64 = 150.0;
const ALIF_B_JUMP: f64 = 0.1;

/// Augmentation strengths (fixed for the round; only presence/absence is
/// varied). Applied to raw events before pooling/binning.
const AUG_EVENT_DROP: f64 = 0.15;
const AUG_CHANNEL_SHIFT: i32 = 25; // uniform in [−25, 25] of 700 channels
const AUG_STRETCH: (f64, f64) = (0.9, 1.1); // uniform time-stretch factor

const EXPERIMENTS: &[ExpConfig] = &[
    ExpConfig {
        tag: "A",
        name: "baseline 1x128",
        ..BASE
    },
    ExpConfig {
        tag: "B",
        name: "wide 1x256",
        hidden: &[256],
        ..BASE
    },
    ExpConfig {
        tag: "C",
        name: "wide 1x512",
        hidden: &[512],
        ..BASE
    },
    ExpConfig {
        tag: "D",
        name: "deep 256-128",
        hidden: &[256, 128],
        ..BASE
    },
    ExpConfig {
        tag: "E",
        name: "deep 128-128",
        hidden: &[128, 128],
        ..BASE
    },
    ExpConfig {
        tag: "F",
        name: "fine channels 350, 1x256",
        n_pooled: 350,
        hidden: &[256],
        ..BASE
    },
    // Long-budget probes (round 1 follow-up).
    ExpConfig {
        tag: "G",
        name: "long 1x256 (3000)",
        hidden: &[256],
        minibatches: 3000,
        ..BASE
    },
    ExpConfig {
        tag: "H",
        name: "long 1x512 (3000)",
        hidden: &[512],
        minibatches: 3000,
        ..BASE
    },
    ExpConfig {
        tag: "I",
        name: "long fine 350, 1x256 (3000)",
        n_pooled: 350,
        hidden: &[256],
        minibatches: 3000,
        ..BASE
    },
    // Round 2: the untried next steps from demo/RESULTS.md.
    ExpConfig {
        tag: "J",
        name: "no pooling: 700ch, 1x256 (3000)",
        n_pooled: 700,
        hidden: &[256],
        minibatches: 3000,
        ..BASE
    },
    ExpConfig {
        tag: "K",
        name: "longer on winner: 350ch, 1x256 (6000)",
        n_pooled: 350,
        hidden: &[256],
        minibatches: 6000,
        ..BASE
    },
    ExpConfig {
        tag: "L",
        name: "RECURRENT 350ch, 1x256 (3000), W_rec zero-init",
        n_pooled: 350,
        hidden: &[256],
        minibatches: 3000,
        recurrent: true,
        ..BASE
    },
    ExpConfig {
        tag: "M",
        name: "balanced minibatches: 350ch, 1x256 (3000)",
        n_pooled: 350,
        hidden: &[256],
        minibatches: 3000,
        balanced: true,
        ..BASE
    },
    ExpConfig {
        tag: "N",
        name: "lr decay x0.3 @2000: 350ch, 1x256 (3000)",
        n_pooled: 350,
        hidden: &[256],
        minibatches: 3000,
        lr_decay: Some((2000, 0.3)),
        ..BASE
    },
    // Round-2 combo: assembled from the winners of J..N.
    ExpConfig {
        tag: "O",
        name: "combo: recurrent 350ch 1x256, 6000, lr decay @4000",
        n_pooled: 350,
        hidden: &[256],
        minibatches: 6000,
        recurrent: true,
        lr_decay: Some((4000, 0.3)),
        ..BASE
    },
    // Round 3: regularization and augmentation on the recurrent champion
    // (L = recurrent 350ch 1x256 @3000, 0.808), targeting > 0.83.
    ExpConfig {
        tag: "P",
        name: "L + weight decay 0.01",
        n_pooled: 350,
        hidden: &[256],
        minibatches: 3000,
        recurrent: true,
        weight_decay: 0.01,
        ..BASE
    },
    ExpConfig {
        tag: "Q",
        name: "L + augmentation",
        n_pooled: 350,
        hidden: &[256],
        minibatches: 3000,
        recurrent: true,
        augment: true,
        ..BASE
    },
    ExpConfig {
        tag: "R",
        name: "L + augmentation, 6000 (aug should unlock budget)",
        n_pooled: 350,
        hidden: &[256],
        minibatches: 6000,
        recurrent: true,
        augment: true,
        ..BASE
    },
    ExpConfig {
        tag: "S",
        name: "L + aug + wd 0.01, 6000",
        n_pooled: 350,
        hidden: &[256],
        minibatches: 6000,
        recurrent: true,
        augment: true,
        weight_decay: 0.01,
        ..BASE
    },
    ExpConfig {
        tag: "T",
        name: "capacity unlock: recurrent 1x512, aug + wd, 6000",
        n_pooled: 350,
        hidden: &[512],
        minibatches: 6000,
        recurrent: true,
        augment: true,
        weight_decay: 0.01,
        ..BASE
    },
    // Round 4: past 0.90 — R's recipe (recurrent 350ch 1x256, aug, no wd)
    // as the base, varying budget, neuron model, and depth.
    ExpConfig {
        tag: "U",
        name: "R + 12000 mb (losses were still falling)",
        n_pooled: 350,
        hidden: &[256],
        minibatches: 12000,
        recurrent: true,
        augment: true,
        ..BASE
    },
    ExpConfig {
        tag: "V",
        name: "ALIF (homogeneous adaptation), R recipe",
        n_pooled: 350,
        hidden: &[256],
        minibatches: 6000,
        recurrent: true,
        augment: true,
        neuron: NeuronKind::Alif,
        ..BASE
    },
    ExpConfig {
        tag: "W",
        name: "ALIF heterogeneous taus, R recipe",
        n_pooled: 350,
        hidden: &[256],
        minibatches: 6000,
        recurrent: true,
        augment: true,
        neuron: NeuronKind::AlifHetero,
        ..BASE
    },
    ExpConfig {
        tag: "X",
        name: "two recurrent layers 256-256, R recipe",
        n_pooled: 350,
        hidden: &[256, 256],
        minibatches: 6000,
        recurrent: true,
        augment: true,
        ..BASE
    },
    // Multi-seed error bars on the actual champion recipe (R: LIF).
    ExpConfig {
        tag: "Z1",
        name: "R recipe, seed repeat 1",
        n_pooled: 350,
        hidden: &[256],
        minibatches: 6000,
        recurrent: true,
        augment: true,
        seed_bump: 101,
        ..BASE
    },
    ExpConfig {
        tag: "Z2",
        name: "R recipe, seed repeat 2",
        n_pooled: 350,
        hidden: &[256],
        minibatches: 6000,
        recurrent: true,
        augment: true,
        seed_bump: 202,
        ..BASE
    },
    // Round 5 (target > 0.92): time axis, readout, capacity, ensemble.
    ExpConfig {
        tag: "AA",
        name: "full duration 1.4 s (140 bins), R recipe",
        n_pooled: 350,
        hidden: &[256],
        minibatches: 6000,
        recurrent: true,
        augment: true,
        t_steps: 140,
        ..BASE
    },
    ExpConfig {
        tag: "AB",
        name: "fine bins 5 ms x 200, R recipe",
        n_pooled: 350,
        hidden: &[256],
        minibatches: 6000,
        recurrent: true,
        augment: true,
        bin_s: 0.005,
        t_steps: 200,
        ..BASE
    },
    ExpConfig {
        tag: "AC",
        name: "leaky readout kappa 0.95, R recipe",
        n_pooled: 350,
        hidden: &[256],
        minibatches: 6000,
        recurrent: true,
        augment: true,
        readout_decay: Some(0.95),
        ..BASE
    },
    ExpConfig {
        tag: "AD",
        name: "1x512 recurrent, aug only, 6000",
        n_pooled: 350,
        hidden: &[512],
        minibatches: 6000,
        recurrent: true,
        augment: true,
        ..BASE
    },
    // AE: deeper — three recurrent layers (depth was round 4's winner; the
    // round-5 single-variation axes AA/AB/AC all came back neutral/negative).
    ExpConfig {
        tag: "AE",
        name: "three recurrent layers 256-256-256, aug, 6000",
        n_pooled: 350,
        hidden: &[256, 256, 256],
        minibatches: 6000,
        recurrent: true,
        augment: true,
        ..BASE
    },
    // AF: 3-member logit-ensemble of the X recipe (two recurrent layers) —
    // both an accuracy and a seed-variance play (Z runs showed ±2.7).
    ExpConfig {
        tag: "AF",
        name: "ensemble x3 of two recurrent layers 256-256",
        n_pooled: 350,
        hidden: &[256, 256],
        minibatches: 6000,
        recurrent: true,
        augment: true,
        ensemble: 3,
        ..BASE
    },
];

/// Train-time augmentation on the raw event stream: per-event dropout, a
/// whole-sample channel shift (spectral jitter), and a whole-sample time
/// stretch. Events pushed outside [0, 700) channels are dropped; times
/// outside the horizon are handled by binning's range check.
fn augment_sample(sample: &ShdSample, rng: &mut StdRng) -> ShdSample {
    let shift = rng.random_range(-AUG_CHANNEL_SHIFT..=AUG_CHANNEL_SHIFT);
    let stretch = rng.random_range(AUG_STRETCH.0..AUG_STRETCH.1);
    let mut times = Vec::with_capacity(sample.times_s.len());
    let mut units = Vec::with_capacity(sample.units.len());
    for (&t, &u) in sample.times_s.iter().zip(&sample.units) {
        if rng.random::<f64>() < AUG_EVENT_DROP {
            continue;
        }
        let shifted = u as i32 + shift;
        if !(0..700).contains(&shifted) {
            continue;
        }
        times.push(t * stretch);
        units.push(shifted as u32);
    }
    ShdSample {
        times_s: times,
        units,
        label: sample.label,
    }
}

fn write_sample(
    sample: &ShdSample,
    inputs: &mut [SpikeBatch],
    col: usize,
    cfg: &ExpConfig,
) -> Result<(), kdmd_snn::SnnError> {
    let steps = bin_events(sample, cfg.n_pooled, cfg.bin_s, cfg.t_steps)?;
    for (t, active) in steps.iter().enumerate() {
        for &c in active {
            inputs[t].as_mat_mut()[(c as usize, col)] = 1.0;
        }
    }
    Ok(())
}

fn zero_inputs(cfg: &ExpConfig) -> Vec<SpikeBatch> {
    (0..cfg.t_steps)
        .map(|_| SpikeBatch::zeros(cfg.n_pooled, BATCH).unwrap())
        .collect()
}

fn build_network(cfg: &ExpConfig, seed_bump: u64) -> Network {
    let dt_ms = cfg.bin_s * 1e3;
    let lif = Lif::new(LifParams {
        tau_m: 20.0,
        tau_s: 10.0,
        dt: dt_ms,
        ..LifParams::default()
    })
    .unwrap();
    let mut rng = StdRng::seed_from_u64(INIT_SEED + seed_bump);
    let mut layers = Vec::new();
    let mut fan_in = cfg.n_pooled;
    for (l, &n) in cfg.hidden.iter().enumerate() {
        // Input layers see dense per-bin activity; hidden layers see sparse
        // volleys and need proportionally stronger weights or they are born
        // dead (see demo/README.md, finding on initialization).
        let numerator = if l == 0 { 35.0 } else { 90.0 };
        let gain = numerator / fan_in as f64;
        let w = Mat::from_fn(n, fan_in, |_, _| rng.random_range(0.0..gain));
        let mut layer = match cfg.neuron {
            NeuronKind::Lif => KoopmanLayer::lif(&lif, n, w, BATCH).unwrap(),
            NeuronKind::Alif => {
                let cell = AdLif::new(AdLifParams {
                    tau_m: 20.0,
                    tau_s: 10.0,
                    tau_w: ALIF_TAU_W,
                    b_jump: ALIF_B_JUMP,
                    dt: dt_ms,
                    ..AdLifParams::default()
                })
                .unwrap();
                KoopmanLayer::adlif(&cell, n, w, BATCH).unwrap()
            }
            NeuronKind::AlifHetero => {
                let cells: Vec<AdLif> = (0..n)
                    .map(|_| {
                        let tau_m = rng.random_range(10.0..40.0);
                        let tau_s = rng.random_range(5.0..15.0);
                        let tau_w = (rng.random_range(60.0f64.ln()..400.0f64.ln())).exp();
                        AdLif::new(AdLifParams {
                            tau_m,
                            tau_s,
                            tau_w,
                            b_jump: ALIF_B_JUMP,
                            dt: dt_ms,
                            ..AdLifParams::default()
                        })
                        .unwrap()
                    })
                    .collect();
                KoopmanLayer::adlif_hetero(&cells, w, BATCH).unwrap()
            }
        };
        if cfg.recurrent {
            // Zero-init: exactly feedforward at step 0; training grows the
            // recurrence from nothing (clean attribution).
            layer = layer.with_recurrent(Mat::zeros(n, n)).unwrap();
        }
        layers.push(layer);
        fan_in = n;
    }
    Network::new(layers, BATCH).unwrap()
}

struct RunResult {
    tag: &'static str,
    test_accuracy: f64,
    final_train_loss: f64,
    train_secs: f64,
    loss_curve: Vec<(usize, f64)>,
}

fn run_experiment(cfg: &ExpConfig, train: &[ShdSample], test: &[ShdSample]) -> RunResult {
    println!(
        "\n### [{}] {} — pooled {}, hidden {:?}, {} minibatches, {}x{}s bins, ensemble {}",
        cfg.tag,
        cfg.name,
        cfg.n_pooled,
        cfg.hidden,
        cfg.minibatches,
        cfg.t_steps,
        cfg.bin_s,
        cfg.ensemble
    );
    // Per-class index for the balanced-sampling variation.
    let mut by_class: Vec<Vec<usize>> = vec![Vec::new(); N_CLASSES];
    for (i, s) in train.iter().enumerate() {
        by_class[s.label].push(i);
    }

    let start = Instant::now();
    let mut loss_curve = Vec::new();
    let mut members: Vec<(Network, Trainer)> = Vec::new();
    for member in 0..cfg.ensemble {
        let bump = cfg.seed_bump + 1000 * member as u64;
        let mut net = build_network(cfg, bump);
        let mut trainer = Trainer::new(
            &net,
            N_CLASSES,
            TrainConfig {
                weight_decay: cfg.weight_decay,
                readout_decay: cfg.readout_decay,
                ..TrainConfig::default()
            },
        )
        .unwrap();
        let mut data_rng = StdRng::seed_from_u64(DATA_SEED + bump);
        let mut recent = Vec::new();
        for step in 0..cfg.minibatches {
            if let Some((at, factor)) = cfg.lr_decay {
                if step == at {
                    trainer.set_learning_rate(5e-3 * factor);
                    println!("  step {step:4}: learning rate → {:.1e}", 5e-3 * factor);
                }
            }
            let mut inputs = zero_inputs(cfg);
            let mut targets = Vec::with_capacity(BATCH);
            for b in 0..BATCH {
                let sample = if cfg.balanced {
                    let class = data_rng.random_range(0..N_CLASSES);
                    let list = &by_class[class];
                    &train[list[data_rng.random_range(0..list.len())]]
                } else {
                    &train[data_rng.random_range(0..train.len())]
                };
                targets.push(sample.label);
                if cfg.augment {
                    let augmented = augment_sample(sample, &mut data_rng);
                    write_sample(&augmented, &mut inputs, b, cfg).unwrap();
                } else {
                    write_sample(sample, &mut inputs, b, cfg).unwrap();
                }
            }
            let stats = trainer.train_step(&mut net, &inputs, &targets).unwrap();
            recent.push(stats.loss);
            if recent.len() == 50 {
                let mean = recent.iter().sum::<f64>() / recent.len() as f64;
                if member == 0 {
                    loss_curve.push((step + 1, mean));
                    println!("  step {:4}: mean loss {mean:.4}", step + 1);
                }
                recent.clear();
            }
        }
        members.push((net, trainer));
    }
    let train_secs = start.elapsed().as_secs_f64();

    // Full test set, all complete batches, fixed order; ensemble members'
    // logits are summed before the argmax.
    let (mut correct, mut total) = (0usize, 0usize);
    for chunk in test.chunks(BATCH) {
        if chunk.len() < BATCH {
            break;
        }
        let mut inputs = zero_inputs(cfg);
        let mut targets = Vec::with_capacity(BATCH);
        for (b, sample) in chunk.iter().enumerate() {
            targets.push(sample.label);
            write_sample(sample, &mut inputs, b, cfg).unwrap();
        }
        let mut sum_logits = Mat::<f64>::zeros(N_CLASSES, BATCH);
        for (net, trainer) in members.iter_mut() {
            let logits = trainer.logits(net, &inputs).unwrap();
            for b in 0..BATCH {
                for i in 0..N_CLASSES {
                    sum_logits[(i, b)] += logits[(i, b)];
                }
            }
        }
        for (b, &t) in targets.iter().enumerate() {
            let mut best = 0usize;
            for i in 1..N_CLASSES {
                if sum_logits[(i, b)] > sum_logits[(best, b)] {
                    best = i;
                }
            }
            if best == t {
                correct += 1;
            }
            total += 1;
        }
    }
    let test_accuracy = correct as f64 / total as f64;
    let final_train_loss = loss_curve.last().map_or(f64::NAN, |&(_, l)| l);
    println!(
        "  RESULT [{}]: test accuracy {:.4} ({correct}/{total}), final loss {final_train_loss:.4}, \
         train {train_secs:.1}s",
        cfg.tag, test_accuracy
    );
    RunResult {
        tag: cfg.tag,
        test_accuracy,
        final_train_loss,
        train_secs,
        loss_curve,
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let selected: Vec<&ExpConfig> = if args.is_empty() {
        // Default: the six controlled-budget variations A-F.
        EXPERIMENTS.iter().filter(|e| e.tag < "G").collect()
    } else {
        EXPERIMENTS
            .iter()
            .filter(|e| args.iter().any(|a| a == e.tag))
            .collect()
    };
    if selected.is_empty() {
        eprintln!("no experiments selected; tags: A..I");
        std::process::exit(1);
    }

    let data_dir = PathBuf::from("data/shd");
    if std::env::var_os("HDF5_PLUGIN_PATH").is_none() {
        std::env::set_var("HDF5_PLUGIN_PATH", &data_dir);
    }
    println!("loading SHD …");
    let train = load_shd(&data_dir.join("shd_train.h5")).expect("train load (run shd_demo first)");
    let test = load_shd(&data_dir.join("shd_test.h5")).expect("test load");
    println!("{} train / {} test samples", train.len(), test.len());

    let mut results = Vec::new();
    for cfg in &selected {
        results.push(run_experiment(cfg, &train, &test));
    }

    println!("\n## Summary (chance = {:.3})", 1.0 / N_CLASSES as f64);
    println!("| tag | test acc | final loss | train (s) |");
    println!("|---|---|---|---|");
    for r in &results {
        println!(
            "| {} | {:.4} | {:.4} | {:.1} |",
            r.tag, r.test_accuracy, r.final_train_loss, r.train_secs
        );
    }
    // Loss curves for the demo log.
    for r in &results {
        let pts: Vec<String> = r
            .loss_curve
            .iter()
            .step_by(3)
            .map(|(s, l)| format!("{s}:{l:.3}"))
            .collect();
        println!("curve {}: {}", r.tag, pts.join(" "));
    }
}
