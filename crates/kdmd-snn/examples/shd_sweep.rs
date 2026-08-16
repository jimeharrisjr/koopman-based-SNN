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
use kdmd_snn::neuron::{Lif, LifParams};
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
};

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
    n_pooled: usize,
) -> Result<(), kdmd_snn::SnnError> {
    let steps = bin_events(sample, n_pooled, BIN_S, T_STEPS)?;
    for (t, active) in steps.iter().enumerate() {
        for &c in active {
            inputs[t].as_mat_mut()[(c as usize, col)] = 1.0;
        }
    }
    Ok(())
}

fn zero_inputs(n_pooled: usize) -> Vec<SpikeBatch> {
    (0..T_STEPS)
        .map(|_| SpikeBatch::zeros(n_pooled, BATCH).unwrap())
        .collect()
}

fn build_network(cfg: &ExpConfig) -> Network {
    let lif = Lif::new(LifParams {
        tau_m: 20.0,
        tau_s: 10.0,
        dt: BIN_S * 1e3,
        ..LifParams::default()
    })
    .unwrap();
    let mut rng = StdRng::seed_from_u64(INIT_SEED);
    let mut layers = Vec::new();
    let mut fan_in = cfg.n_pooled;
    for (l, &n) in cfg.hidden.iter().enumerate() {
        // Input layers see dense per-bin activity; hidden layers see sparse
        // volleys and need proportionally stronger weights or they are born
        // dead (see demo/README.md, finding on initialization).
        let numerator = if l == 0 { 35.0 } else { 90.0 };
        let gain = numerator / fan_in as f64;
        let w = Mat::from_fn(n, fan_in, |_, _| rng.random_range(0.0..gain));
        let mut layer = KoopmanLayer::lif(&lif, n, w, BATCH).unwrap();
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
        "\n### [{}] {} — pooled {}, hidden {:?}, {} minibatches",
        cfg.tag, cfg.name, cfg.n_pooled, cfg.hidden, cfg.minibatches
    );
    let mut net = build_network(cfg);
    let mut trainer = Trainer::new(
        &net,
        N_CLASSES,
        TrainConfig {
            weight_decay: cfg.weight_decay,
            ..TrainConfig::default()
        },
    )
    .unwrap();
    let mut data_rng = StdRng::seed_from_u64(DATA_SEED);

    // Per-class index for the balanced-sampling variation.
    let mut by_class: Vec<Vec<usize>> = vec![Vec::new(); N_CLASSES];
    for (i, s) in train.iter().enumerate() {
        by_class[s.label].push(i);
    }

    let start = Instant::now();
    let mut loss_curve = Vec::new();
    let mut recent = Vec::new();
    for step in 0..cfg.minibatches {
        if let Some((at, factor)) = cfg.lr_decay {
            if step == at {
                trainer.set_learning_rate(5e-3 * factor);
                println!("  step {step:4}: learning rate → {:.1e}", 5e-3 * factor);
            }
        }
        let mut inputs = zero_inputs(cfg.n_pooled);
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
                write_sample(&augmented, &mut inputs, b, cfg.n_pooled).unwrap();
            } else {
                write_sample(sample, &mut inputs, b, cfg.n_pooled).unwrap();
            }
        }
        let stats = trainer.train_step(&mut net, &inputs, &targets).unwrap();
        recent.push(stats.loss);
        if recent.len() == 50 {
            let mean = recent.iter().sum::<f64>() / recent.len() as f64;
            loss_curve.push((step + 1, mean));
            println!("  step {:4}: mean loss {mean:.4}", step + 1);
            recent.clear();
        }
    }
    let train_secs = start.elapsed().as_secs_f64();

    // Full test set, all complete batches, fixed order.
    let (mut correct, mut total) = (0usize, 0usize);
    for chunk in test.chunks(BATCH) {
        if chunk.len() < BATCH {
            break;
        }
        let mut inputs = zero_inputs(cfg.n_pooled);
        let mut targets = Vec::with_capacity(BATCH);
        for (b, sample) in chunk.iter().enumerate() {
            targets.push(sample.label);
            write_sample(sample, &mut inputs, b, cfg.n_pooled).unwrap();
        }
        let predictions = trainer.predict(&mut net, &inputs).unwrap();
        for (p, t) in predictions.iter().zip(&targets) {
            if p == t {
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
