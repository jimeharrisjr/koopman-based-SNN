//! Phase 6 training exit gates:
//! 1. The readout gradient is finite-difference-exact (the smooth part).
//! 2. Surrogate BPTT reproducibly learns the synthetic Poisson-pattern task
//!    to high held-out accuracy (the end-to-end validation of the W path —
//!    see train/mod.rs for why W cannot be FD-checked directly).

use faer::Mat;
use kdmd_snn::data::PoissonPatternTask;
use kdmd_snn::neuron::{Lif, LifParams};
use kdmd_snn::{KoopmanLayer, Network, SpikeBatch, TrainConfig, Trainer};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const N_IN: usize = 24;
const N_HIDDEN: usize = 32;
const N_CLASSES: usize = 2;
const T_STEPS: usize = 60;
const BATCH: usize = 8;
const DT: f64 = 1.0;

fn build_net(seed: u64) -> Network {
    let lif = Lif::new(LifParams {
        dt: DT,
        ..LifParams::default()
    })
    .unwrap();
    let mut rng = StdRng::seed_from_u64(seed);
    let w = Mat::from_fn(N_HIDDEN, N_IN, |_, _| rng.random_range(0.0..0.6));
    let layer = KoopmanLayer::lif(&lif, N_HIDDEN, w, BATCH).unwrap();
    Network::new(vec![layer], BATCH).unwrap()
}

/// Draw one minibatch: `T_STEPS` dense spike batches plus targets.
fn minibatch(task: &PoissonPatternTask, rng: &mut StdRng) -> (Vec<SpikeBatch>, Vec<usize>) {
    let mut inputs: Vec<SpikeBatch> = (0..T_STEPS)
        .map(|_| SpikeBatch::zeros(N_IN, BATCH).unwrap())
        .collect();
    let mut targets = Vec::with_capacity(BATCH);
    for b in 0..BATCH {
        let class = rng.random_range(0..N_CLASSES);
        targets.push(class);
        // Rates are per-ms; dt = 1 ms bins.
        let sample = task.sample(class, T_STEPS, DT, rng).unwrap();
        for (t, batch_t) in sample.iter().enumerate() {
            for i in 0..N_IN {
                if batch_t.as_mat()[(i, 0)] == 1.0 {
                    inputs[t].as_mat_mut()[(i, b)] = 1.0;
                }
            }
        }
    }
    (inputs, targets)
}

#[test]
fn readout_gradient_matches_finite_differences() {
    let mut rng = StdRng::seed_from_u64(5);
    let task = PoissonPatternTask::new(N_IN, N_CLASSES, 0.12, 0.01, 0.4, &mut rng).unwrap();
    let mut net = build_net(1);
    let trainer = Trainer::new(&net, N_CLASSES, TrainConfig::default()).unwrap();
    let (inputs, targets) = minibatch(&task, &mut rng);
    let worst = trainer
        .check_readout_gradient(&mut net, &inputs, &targets, 1e-6)
        .unwrap();
    assert!(
        worst < 1e-5,
        "readout gradient mismatch vs finite differences: {worst:.2e}"
    );
}

#[test]
fn multi_layer_bptt_reduces_loss_and_stays_finite() {
    // The W_{ℓ+1}ᵀ inter-layer gradient path (code-quality finding M2):
    // a two-layer network must train without NaNs and reduce its loss.
    let lif = Lif::new(LifParams {
        dt: DT,
        ..LifParams::default()
    })
    .unwrap();
    let mut rng = StdRng::seed_from_u64(29);
    // The second layer sees sparse volleys, not tonic drive: it needs
    // stronger initial weights than the input layer or it never fires and
    // the network is born dead (loss pinned at ln 2).
    let w0 = Mat::from_fn(N_HIDDEN, N_IN, |_, _| rng.random_range(0.0..0.6));
    let w1 = Mat::from_fn(16, N_HIDDEN, |_, _| rng.random_range(0.0..1.5));
    let l0 = KoopmanLayer::lif(&lif, N_HIDDEN, w0, BATCH).unwrap();
    let l1 = KoopmanLayer::lif(&lif, 16, w1, BATCH).unwrap();
    let mut net = Network::new(vec![l0, l1], BATCH).unwrap();
    let mut trainer = Trainer::new(&net, N_CLASSES, TrainConfig::default()).unwrap();

    let w0_before = net.layer(0).weights().clone();
    let task = PoissonPatternTask::new(N_IN, N_CLASSES, 0.12, 0.01, 0.4, &mut rng).unwrap();
    let mut losses = Vec::new();
    for step in 0..300 {
        let (inputs, targets) = minibatch(&task, &mut rng);
        let stats = trainer.train_step(&mut net, &inputs, &targets).unwrap();
        assert!(stats.loss.is_finite(), "loss diverged at step {step}");
        losses.push(stats.loss);
    }
    // Two-layer credit assignment is slower than single-layer; the gate here
    // is gradient FLOW, not task mastery (the single-layer test owns that):
    // the loss must clearly decrease…
    let early: f64 = losses[..20].iter().sum::<f64>() / 20.0;
    let late: f64 = losses[losses.len() - 20..].iter().sum::<f64>() / 20.0;
    assert!(
        late < 0.9 * early,
        "two-layer training did not reduce the loss: early {early:.4}, late {late:.4}"
    );
    // …and the FIRST layer's weights must have moved, proving the
    // W_{ℓ+1}ᵀ inter-layer gradient path reached it (the optimizer only
    // moves parameters with nonzero gradients).
    let w0_after = net.layer(0).weights();
    let mut max_delta = 0.0f64;
    for j in 0..w0_after.ncols() {
        for i in 0..w0_after.nrows() {
            max_delta = max_delta.max((w0_after[(i, j)] - w0_before[(i, j)]).abs());
        }
    }
    assert!(
        max_delta > 1e-6,
        "hidden-layer weights never moved — the inter-layer gradient path is dead"
    );
}

#[test]
fn recurrent_bptt_reduces_loss_and_grows_recurrence_from_zero() {
    // Zero-initialized W_rec makes the recurrent net exactly the feedforward
    // net at step 0; training must (a) stay finite, (b) reduce the loss, and
    // (c) actually move W_rec — proving the through-time gradient path works.
    let lif = Lif::new(LifParams {
        dt: DT,
        ..LifParams::default()
    })
    .unwrap();
    let mut rng = StdRng::seed_from_u64(37);
    let w = Mat::from_fn(N_HIDDEN, N_IN, |_, _| rng.random_range(0.0..0.6));
    let layer = KoopmanLayer::lif(&lif, N_HIDDEN, w, BATCH)
        .unwrap()
        .with_recurrent(Mat::zeros(N_HIDDEN, N_HIDDEN))
        .unwrap();
    let mut net = Network::new(vec![layer], BATCH).unwrap();
    let mut trainer = Trainer::new(&net, N_CLASSES, TrainConfig::default()).unwrap();

    let task = PoissonPatternTask::new(N_IN, N_CLASSES, 0.12, 0.01, 0.4, &mut rng).unwrap();
    let mut losses = Vec::new();
    for step in 0..200 {
        let (inputs, targets) = minibatch(&task, &mut rng);
        let stats = trainer.train_step(&mut net, &inputs, &targets).unwrap();
        assert!(stats.loss.is_finite(), "loss diverged at step {step}");
        losses.push(stats.loss);
    }
    let early: f64 = losses[..20].iter().sum::<f64>() / 20.0;
    let late: f64 = losses[losses.len() - 20..].iter().sum::<f64>() / 20.0;
    assert!(
        late < 0.7 * early,
        "recurrent training did not reduce the loss: early {early:.4}, late {late:.4}"
    );
    let w_rec = net.layer(0).recurrent_weights().unwrap();
    let mut max_abs = 0.0f64;
    for j in 0..w_rec.ncols() {
        for i in 0..w_rec.nrows() {
            max_abs = max_abs.max(w_rec[(i, j)].abs());
        }
    }
    assert!(
        max_abs > 1e-6,
        "W_rec never moved from zero — the recurrent gradient path is dead"
    );
}

#[test]
fn mismatched_trainer_and_network_error_instead_of_panicking() {
    let lif = Lif::new(LifParams {
        dt: DT,
        ..LifParams::default()
    })
    .unwrap();
    let mut rng = StdRng::seed_from_u64(31);
    let net_a = build_net(3);
    let trainer = Trainer::new(&net_a, N_CLASSES, TrainConfig::default()).unwrap();
    // A different network with a different output width.
    let w = Mat::from_fn(12, N_IN, |_, _| rng.random_range(0.0..0.5));
    let layer = KoopmanLayer::lif(&lif, 12, w, BATCH).unwrap();
    let mut net_b = Network::new(vec![layer], BATCH).unwrap();

    let task = PoissonPatternTask::new(N_IN, N_CLASSES, 0.12, 0.01, 0.4, &mut rng).unwrap();
    let (inputs, _) = minibatch(&task, &mut rng);
    assert!(trainer.predict(&mut net_b, &inputs).is_err());
}

#[test]
fn fast_path_rejects_nonzero_rest_and_hard_reset() {
    // Code-quality finding C1: these configurations were silently
    // mis-simulated before; now they must error.
    let lif = Lif::new(LifParams {
        v_rest: -65.0,
        theta: -50.0,
        ..LifParams::default()
    })
    .unwrap();
    assert!(KoopmanLayer::lif(&lif, 4, Mat::from_fn(4, 2, |_, _| 0.5), 1).is_err());

    let hard = Lif::new(LifParams {
        reset: kdmd_snn::ResetMode::HardTo(0.0),
        ..LifParams::default()
    })
    .unwrap();
    assert!(KoopmanLayer::lif(&hard, 4, Mat::from_fn(4, 2, |_, _| 0.5), 1).is_err());
}

#[test]
fn surrogate_bptt_learns_the_poisson_pattern_task() {
    let mut rng = StdRng::seed_from_u64(17);
    // Two random rate patterns over 24 channels: 0.12 spikes/ms on active
    // channels, 0.01 on the rest.
    let task = PoissonPatternTask::new(N_IN, N_CLASSES, 0.12, 0.01, 0.4, &mut rng).unwrap();
    let mut net = build_net(2);
    let mut trainer = Trainer::new(&net, N_CLASSES, TrainConfig::default()).unwrap();

    let mut last_losses = Vec::new();
    for step in 0..250 {
        let (inputs, targets) = minibatch(&task, &mut rng);
        let stats = trainer.train_step(&mut net, &inputs, &targets).unwrap();
        assert!(
            stats.loss.is_finite(),
            "loss diverged at step {step}: {}",
            stats.loss
        );
        last_losses.push(stats.loss);
    }
    // Loss must have come down substantially from its start.
    let early: f64 = last_losses[..20].iter().sum::<f64>() / 20.0;
    let late: f64 = last_losses[last_losses.len() - 20..].iter().sum::<f64>() / 20.0;
    assert!(
        late < 0.5 * early,
        "training did not reduce the loss: early {early:.4}, late {late:.4}"
    );

    // Held-out accuracy over 10 fresh minibatches (80 samples).
    let (mut correct, mut total) = (0usize, 0usize);
    for _ in 0..10 {
        let (inputs, targets) = minibatch(&task, &mut rng);
        let predictions = trainer.predict(&mut net, &inputs).unwrap();
        for (p, t) in predictions.iter().zip(&targets) {
            if p == t {
                correct += 1;
            }
            total += 1;
        }
    }
    let accuracy = correct as f64 / total as f64;
    assert!(
        accuracy >= 0.9,
        "held-out accuracy {accuracy:.3} below the 0.9 exit gate ({correct}/{total})"
    );
}
