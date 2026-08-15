//! Spiking Heidelberg Digits demo (Phase 6, owner decision Q3).
//!
//! End-to-end: download SHD (if absent) → bin events → train a LIF
//! KoopmanLayer + readout with surrogate BPTT → report test accuracy.
//! 20 classes, chance = 5 %. This is a demo configuration (pooled channels,
//! single hidden layer, short budget), not a benchmark entry.
//!
//! Run: `cargo run --release --features datasets --example shd_demo`

use std::path::{Path, PathBuf};
use std::process::Command;

use faer::Mat;
use kdmd_snn::data::shd::{bin_events, load_shd, ShdSample};
use kdmd_snn::neuron::{Lif, LifParams};
use kdmd_snn::{KoopmanLayer, Network, SpikeBatch, TrainConfig, Trainer};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const N_POOLED: usize = 100; // 700 cochlear channels pooled 7:1
const N_HIDDEN: usize = 128;
const N_CLASSES: usize = 20;
const BIN_S: f64 = 0.010; // 10 ms bins
const T_STEPS: usize = 100; // first second of audio
const BATCH: usize = 32;
const TRAIN_MINIBATCHES: usize = 600;
const TEST_SAMPLES: usize = 512;
const URLS: &[(&str, &str)] = &[
    (
        "shd_train.h5",
        "https://zenkelab.org/datasets/shd_train.h5.gz",
    ),
    (
        "shd_test.h5",
        "https://zenkelab.org/datasets/shd_test.h5.gz",
    ),
];

fn ensure_dataset(dir: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)?;
    for (name, url) in URLS {
        let target = dir.join(name);
        if target.exists() {
            continue;
        }
        let gz = dir.join(format!("{name}.gz"));
        println!("downloading {url} …");
        let status = Command::new("curl")
            .args(["-L", "--fail", "-o"])
            .arg(&gz)
            .arg(url)
            .status()?;
        if !status.success() {
            return Err(std::io::Error::other(format!("curl failed for {url}")));
        }
        println!("decompressing {} …", gz.display());
        let status = Command::new("gunzip").arg(&gz).status()?;
        if !status.success() {
            return Err(std::io::Error::other("gunzip failed"));
        }
    }
    Ok(())
}

/// Bin a sample into dense per-step spike batches for one batch column.
fn write_sample(
    sample: &ShdSample,
    inputs: &mut [SpikeBatch],
    col: usize,
) -> Result<(), kdmd_snn::SnnError> {
    let steps = bin_events(sample, N_POOLED, BIN_S, T_STEPS)?;
    for (t, active) in steps.iter().enumerate() {
        for &c in active {
            inputs[t].as_mat_mut()[(c as usize, col)] = 1.0;
        }
    }
    Ok(())
}

fn zero_inputs() -> Vec<SpikeBatch> {
    (0..T_STEPS)
        .map(|_| SpikeBatch::zeros(N_POOLED, BATCH).unwrap())
        .collect()
}

fn main() {
    let data_dir = PathBuf::from("data/shd");
    // The statically-built HDF5 insists its filter-plugin directory exists;
    // point it at the data dir before the library initializes.
    if std::env::var_os("HDF5_PLUGIN_PATH").is_none() {
        std::env::set_var("HDF5_PLUGIN_PATH", &data_dir);
    }
    ensure_dataset(&data_dir).expect("dataset download failed");

    println!("loading SHD …");
    let train = load_shd(&data_dir.join("shd_train.h5")).expect("train load");
    let test = load_shd(&data_dir.join("shd_test.h5")).expect("test load");
    println!("{} train / {} test samples", train.len(), test.len());

    let lif = Lif::new(LifParams {
        tau_m: 20.0,
        tau_s: 10.0,
        dt: BIN_S * 1e3, // ms
        ..LifParams::default()
    })
    .unwrap();
    let mut rng = StdRng::seed_from_u64(42);
    let w = Mat::from_fn(N_HIDDEN, N_POOLED, |_, _| rng.random_range(0.0..0.35));
    let layer = KoopmanLayer::lif(&lif, N_HIDDEN, w, BATCH).unwrap();
    let mut net = Network::new(vec![layer], BATCH).unwrap();
    let mut trainer = Trainer::new(&net, N_CLASSES, TrainConfig::default()).unwrap();

    println!("training {TRAIN_MINIBATCHES} minibatches of {BATCH} …");
    let start = std::time::Instant::now();
    for step in 0..TRAIN_MINIBATCHES {
        let mut inputs = zero_inputs();
        let mut targets = Vec::with_capacity(BATCH);
        for b in 0..BATCH {
            let sample = &train[rng.random_range(0..train.len())];
            targets.push(sample.label);
            write_sample(sample, &mut inputs, b).unwrap();
        }
        let stats = trainer.train_step(&mut net, &inputs, &targets).unwrap();
        if step % 50 == 0 {
            println!(
                "step {step:4}: loss {:.4}, batch accuracy {:.2}",
                stats.loss, stats.accuracy
            );
        }
    }
    println!("training took {:.1?}", start.elapsed());

    println!("evaluating on {TEST_SAMPLES} held-out test samples …");
    let (mut correct, mut total) = (0usize, 0usize);
    let mut idx: Vec<usize> = (0..test.len()).collect();
    // Deterministic subsample.
    for i in (1..idx.len()).rev() {
        idx.swap(i, rng.random_range(0..=i));
    }
    for chunk in idx[..TEST_SAMPLES.min(idx.len())].chunks(BATCH) {
        if chunk.len() < BATCH {
            break;
        }
        let mut inputs = zero_inputs();
        let mut targets = Vec::with_capacity(BATCH);
        for (b, &i) in chunk.iter().enumerate() {
            targets.push(test[i].label);
            write_sample(&test[i], &mut inputs, b).unwrap();
        }
        let predictions = trainer.predict(&mut net, &inputs).unwrap();
        for (p, t) in predictions.iter().zip(&targets) {
            if p == t {
                correct += 1;
            }
            total += 1;
        }
    }
    let accuracy = correct as f64 / total as f64;
    println!(
        "\nSHD test accuracy: {accuracy:.3} ({correct}/{total}; chance = {:.3})",
        1.0 / N_CLASSES as f64
    );
    println!(
        "demo configuration: {N_POOLED} pooled channels, {N_HIDDEN} LIF neurons, \
         {T_STEPS}×{BIN_S}s bins, {TRAIN_MINIBATCHES} minibatches"
    );
}
