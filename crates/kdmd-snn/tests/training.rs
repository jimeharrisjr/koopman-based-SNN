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
fn leaky_readout_learns_the_task_and_gradient_stays_fd_exact() {
    // The κ-trace readout must train comparably to the count readout, and
    // its readout gradient (the smooth part) must stay finite-difference
    // exact under the changed normalization and per-step scaling.
    let mut rng = StdRng::seed_from_u64(53);
    let task = PoissonPatternTask::new(N_IN, N_CLASSES, 0.12, 0.01, 0.4, &mut rng).unwrap();
    let cfg = TrainConfig {
        readout_decay: Some(0.95),
        ..TrainConfig::default()
    };
    let mut net = build_net(6);
    let trainer_probe = Trainer::new(&net, N_CLASSES, cfg.clone()).unwrap();
    let (inputs, targets) = minibatch(&task, &mut rng);
    let worst = trainer_probe
        .check_readout_gradient(&mut net, &inputs, &targets, 1e-6)
        .unwrap();
    assert!(
        worst < 1e-5,
        "leaky-readout gradient mismatch vs finite differences: {worst:.2e}"
    );

    let mut trainer = Trainer::new(&net, N_CLASSES, cfg).unwrap();
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
        late < 0.6 * early,
        "leaky-readout training did not reduce the loss: early {early:.4}, late {late:.4}"
    );
}

#[test]
fn weight_decay_shrinks_weights_under_silent_input() {
    // With no input spikes the network is silent, gradients are ~0, and
    // decoupled decay must shrink the layer weights geometrically.
    let mut net = build_net(41);
    let mut trainer = Trainer::new(
        &net,
        N_CLASSES,
        TrainConfig {
            weight_decay: 10.0, // lr 5e-3 → shrink factor 0.95 per step
            ..TrainConfig::default()
        },
    )
    .unwrap();
    let w_before = net.layer(0).weights().clone();
    let silent: Vec<SpikeBatch> = (0..10)
        .map(|_| SpikeBatch::zeros(N_IN, BATCH).unwrap())
        .collect();
    let targets = vec![0usize; BATCH];
    for _ in 0..20 {
        trainer.train_step(&mut net, &silent, &targets).unwrap();
    }
    let w_after = net.layer(0).weights();
    let expected = 0.95f64.powi(20);
    let mut checked = 0usize;
    for j in 0..w_after.ncols() {
        for i in 0..w_after.nrows() {
            if w_before[(i, j)].abs() > 0.1 {
                let ratio = w_after[(i, j)] / w_before[(i, j)];
                assert!(
                    (ratio - expected).abs() < 0.02,
                    "weight ({i},{j}) shrank by {ratio:.4}, expected ≈ {expected:.4}"
                );
                checked += 1;
            }
        }
    }
    assert!(checked > 100, "too few weights checked ({checked})");
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
fn threaded_train_step_matches_serial_gradients() {
    // Data-parallel chunking (improvements.md P2.1) must produce the same
    // update as the serial path up to floating-point summation order: after
    // one identical train_step, weights, recurrent weights, and the readout
    // agree to ~1e-12, and the forward-only logits agree exactly.
    let lif = Lif::new(LifParams {
        dt: DT,
        ..LifParams::default()
    })
    .unwrap();
    let build = || {
        let mut rng = StdRng::seed_from_u64(71);
        let w0 = Mat::from_fn(N_HIDDEN, N_IN, |_, _| rng.random_range(0.0..0.6));
        let w1 = Mat::from_fn(16, N_HIDDEN, |_, _| rng.random_range(0.0..1.5));
        let l0 = KoopmanLayer::lif(&lif, N_HIDDEN, w0, BATCH)
            .unwrap()
            .with_recurrent(Mat::from_fn(N_HIDDEN, N_HIDDEN, |i, j| {
                0.02 * ((i * 7 + j) % 5) as f64
            }))
            .unwrap();
        let l1 = KoopmanLayer::lif(&lif, 16, w1, BATCH).unwrap();
        Network::new(vec![l0, l1], BATCH).unwrap()
    };
    let mut rng = StdRng::seed_from_u64(73);
    let task = PoissonPatternTask::new(N_IN, N_CLASSES, 0.12, 0.01, 0.4, &mut rng).unwrap();
    let (inputs, targets) = minibatch(&task, &mut rng);

    let mut net_serial = build();
    let mut net_threaded = build();
    let mut tr_serial = Trainer::new(&net_serial, N_CLASSES, TrainConfig::default()).unwrap();
    let mut tr_threaded = Trainer::new(
        &net_threaded,
        N_CLASSES,
        TrainConfig {
            threads: 3, // batch 8 → uneven chunks 3/3/2
            ..TrainConfig::default()
        },
    )
    .unwrap();

    // Forward-only equivalence is exact: columns are independent and each is
    // computed with identical arithmetic.
    let lg_serial = tr_serial.logits(&mut net_serial, &inputs).unwrap();
    let lg_threaded = tr_threaded.logits(&mut net_threaded, &inputs).unwrap();
    for b in 0..BATCH {
        for i in 0..N_CLASSES {
            assert_eq!(
                lg_serial[(i, b)],
                lg_threaded[(i, b)],
                "threaded logits differ at ({i}, {b})"
            );
        }
    }

    let s1 = tr_serial
        .train_step(&mut net_serial, &inputs, &targets)
        .unwrap();
    let s2 = tr_threaded
        .train_step(&mut net_threaded, &inputs, &targets)
        .unwrap();
    assert!((s1.loss - s2.loss).abs() < 1e-12, "loss differs");
    assert_eq!(s1.accuracy, s2.accuracy, "accuracy differs");

    let close = |a: &Mat<f64>, b: &Mat<f64>, what: &str| {
        for c in 0..a.ncols() {
            for i in 0..a.nrows() {
                let denom = a[(i, c)].abs().max(1.0);
                assert!(
                    (a[(i, c)] - b[(i, c)]).abs() <= 1e-12 * denom,
                    "{what} differs at ({i},{c}): {} vs {}",
                    a[(i, c)],
                    b[(i, c)]
                );
            }
        }
    };
    for l in 0..net_serial.n_layers() {
        close(
            net_serial.layer(l).weights(),
            net_threaded.layer(l).weights(),
            "W",
        );
        if let (Some(a), Some(b)) = (
            net_serial.layer(l).recurrent_weights(),
            net_threaded.layer(l).recurrent_weights(),
        ) {
            close(a, b, "W_rec");
        }
    }
    close(tr_serial.readout(), tr_threaded.readout(), "readout");
}

#[test]
fn threaded_training_learns_the_task() {
    // End-to-end: the threaded path must actually train, not just match one
    // step.
    let mut rng = StdRng::seed_from_u64(59);
    let task = PoissonPatternTask::new(N_IN, N_CLASSES, 0.12, 0.01, 0.4, &mut rng).unwrap();
    let mut net = build_net(9);
    let mut trainer = Trainer::new(
        &net,
        N_CLASSES,
        TrainConfig {
            threads: 4,
            ..TrainConfig::default()
        },
    )
    .unwrap();
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
        late < 0.5 * early,
        "threaded training did not reduce the loss: early {early:.4}, late {late:.4}"
    );
}

#[test]
fn uniform_lif_hetero_matches_homogeneous_lif_exactly() {
    // The learnable-τ layer at its uniform starting point must be the fixed-τ
    // layer, spike for spike and bit for bit — so a learnable-τ run starts
    // exactly at the baseline it is compared against.
    let lif = Lif::new(LifParams {
        tau_m: 20.0,
        tau_s: 10.0,
        dt: 10.0,
        ..LifParams::default()
    })
    .unwrap();
    let n = 12;
    let mut rng = StdRng::seed_from_u64(97);
    let w = Mat::from_fn(n, N_IN, |_, _| rng.random_range(0.0..0.8));
    let mut homo = KoopmanLayer::lif(&lif, n, w.clone(), BATCH).unwrap();
    let mut hetero =
        KoopmanLayer::lif_hetero(&vec![20.0; n], &vec![10.0; n], 1.0, 10.0, 1.0, w, BATCH)
            .unwrap();
    let mut net_a = Network::new(vec![homo.clone_with_batch(BATCH).unwrap()], BATCH).unwrap();
    let mut net_b = Network::new(vec![hetero.clone_with_batch(BATCH).unwrap()], BATCH).unwrap();
    let _ = (&mut homo, &mut hetero);

    let mut rng = StdRng::seed_from_u64(11);
    for t in 0..300 {
        let mut s_in = SpikeBatch::zeros(N_IN, BATCH).unwrap();
        for b in 0..BATCH {
            for i in 0..N_IN {
                if rng.random::<f64>() < 0.15 {
                    s_in.as_mat_mut()[(i, b)] = 1.0;
                }
            }
        }
        let sa = net_a.step_batch(&s_in).unwrap().as_mat().to_owned();
        let sb = net_b.step_batch(&s_in).unwrap().as_mat();
        for b in 0..BATCH {
            for i in 0..n {
                assert_eq!(sa[(i, b)], sb[(i, b)], "spikes diverged at step {t}");
            }
        }
        for i in 0..2 * n {
            for b in 0..BATCH {
                assert_eq!(
                    net_a.state(0).as_mat()[(i, b)],
                    net_b.state(0).as_mat()[(i, b)],
                    "state diverged at step {t}, row {i}"
                );
            }
        }
    }
}

#[test]
fn learn_tau_off_leaves_hetero_training_identical_to_fixed() {
    // With learn_tau disabled, training a uniform lif_hetero network must
    // produce exactly the fixed-τ network's updates (the τ machinery is
    // provably inert when off).
    let lif = Lif::new(LifParams {
        tau_m: 20.0,
        tau_s: 10.0,
        dt: DT,
        ..LifParams::default()
    })
    .unwrap();
    let mut rng = StdRng::seed_from_u64(23);
    let w = Mat::from_fn(N_HIDDEN, N_IN, |_, _| rng.random_range(0.0..0.6));
    let fixed = KoopmanLayer::lif(&lif, N_HIDDEN, w.clone(), BATCH).unwrap();
    let hetero = KoopmanLayer::lif_hetero(
        &vec![20.0; N_HIDDEN],
        &vec![10.0; N_HIDDEN],
        1.0,
        DT,
        1.0,
        w,
        BATCH,
    )
    .unwrap();
    let mut net_a = Network::new(vec![fixed], BATCH).unwrap();
    let mut net_b = Network::new(vec![hetero], BATCH).unwrap();
    let mut tr_a = Trainer::new(&net_a, N_CLASSES, TrainConfig::default()).unwrap();
    let mut tr_b = Trainer::new(&net_b, N_CLASSES, TrainConfig::default()).unwrap();

    let mut rng = StdRng::seed_from_u64(29);
    let task = PoissonPatternTask::new(N_IN, N_CLASSES, 0.12, 0.01, 0.4, &mut rng).unwrap();
    for _ in 0..5 {
        let (inputs, targets) = minibatch(&task, &mut rng);
        let sa = tr_a.train_step(&mut net_a, &inputs, &targets).unwrap();
        let sb = tr_b.train_step(&mut net_b, &inputs, &targets).unwrap();
        assert_eq!(sa.loss, sb.loss, "losses diverged");
    }
    let wa = net_a.layer(0).weights();
    let wb = net_b.layer(0).weights();
    for j in 0..wa.ncols() {
        for i in 0..wa.nrows() {
            assert_eq!(wa[(i, j)], wb[(i, j)], "weights diverged at ({i},{j})");
        }
    }
}

#[test]
fn learn_tau_trains_and_moves_the_time_constants() {
    // End-to-end learnable τ: training must stay finite, reduce the loss,
    // actually move the time constants off their uniform start, and respect
    // the clamp region.
    let mut rng = StdRng::seed_from_u64(43);
    let w = Mat::from_fn(N_HIDDEN, N_IN, |_, _| rng.random_range(0.0..0.6));
    let layer = KoopmanLayer::lif_hetero(
        &vec![20.0; N_HIDDEN],
        &vec![10.0; N_HIDDEN],
        1.0,
        DT,
        1.0,
        w,
        BATCH,
    )
    .unwrap()
    .with_recurrent(Mat::zeros(N_HIDDEN, N_HIDDEN))
    .unwrap();
    let mut net = Network::new(vec![layer], BATCH).unwrap();
    let mut trainer = Trainer::new(
        &net,
        N_CLASSES,
        TrainConfig {
            learn_tau: true,
            threads: 2,
            ..TrainConfig::default()
        },
    )
    .unwrap();

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
        "learnable-τ training did not reduce the loss: early {early:.4}, late {late:.4}"
    );
    let meta = net.layer(0).lif_taus().expect("τ metadata");
    let mut moved = 0.0f64;
    for j in 0..N_HIDDEN {
        moved = moved
            .max((meta.taus_m[j] - 20.0).abs())
            .max((meta.taus_s[j] - 10.0).abs());
        assert!(
            (5.0..=100.0).contains(&meta.taus_m[j]),
            "tau_m[{j}] = {} escaped its clamp",
            meta.taus_m[j]
        );
        assert!(
            (2.0..=50.0).contains(&meta.taus_s[j]),
            "tau_s[{j}] = {} escaped its clamp",
            meta.taus_s[j]
        );
        assert!(
            meta.taus_m[j] >= 1.2 * meta.taus_s[j] - 1e-9,
            "separation violated at neuron {j}: {} vs {}",
            meta.taus_m[j],
            meta.taus_s[j]
        );
    }
    assert!(
        moved > 0.05,
        "time constants never moved off the uniform start (max |Δτ| = {moved:.4} ms)"
    );
}

#[test]
fn trained_readouts_init_identical_to_count() {
    // Both trained-readout modes must start AT the count-readout baseline:
    // the static profile (all-ones) bitwise, the zero-query attention to FP
    // roundoff (its uniform weights are accumulated per step rather than
    // summed then divided).
    use kdmd_snn::ReadoutMode;
    let mut rng = StdRng::seed_from_u64(83);
    let task = PoissonPatternTask::new(N_IN, N_CLASSES, 0.12, 0.01, 0.4, &mut rng).unwrap();
    let (inputs, targets) = minibatch(&task, &mut rng);

    let make = |mode: ReadoutMode| {
        let mut net = build_net(4);
        let trainer = Trainer::new(
            &net,
            N_CLASSES,
            TrainConfig {
                readout_mode: mode,
                ..TrainConfig::default()
            },
        )
        .unwrap();
        let lg = trainer.logits(&mut net, &inputs).unwrap();
        (net, trainer, lg)
    };
    let (_, _, lg_count) = make(ReadoutMode::Count);
    let (_, _, lg_static) = make(ReadoutMode::StaticProfile { t_steps: T_STEPS });
    let (_, _, lg_attn) = make(ReadoutMode::SpikeAttention);
    for b in 0..BATCH {
        for i in 0..N_CLASSES {
            assert_eq!(
                lg_count[(i, b)],
                lg_static[(i, b)],
                "static-profile logits differ at init ({i},{b})"
            );
            assert!(
                (lg_count[(i, b)] - lg_attn[(i, b)]).abs() <= 1e-12,
                "attention logits differ at init ({i},{b}): {} vs {}",
                lg_count[(i, b)],
                lg_attn[(i, b)]
            );
        }
    }

    // And the first training loss matches (identical forward).
    let mut net_c = build_net(4);
    let mut tr_c = Trainer::new(&net_c, N_CLASSES, TrainConfig::default()).unwrap();
    let mut net_s = build_net(4);
    let mut tr_s = Trainer::new(
        &net_s,
        N_CLASSES,
        TrainConfig {
            readout_mode: ReadoutMode::StaticProfile { t_steps: T_STEPS },
            ..TrainConfig::default()
        },
    )
    .unwrap();
    let sc = tr_c.train_step(&mut net_c, &inputs, &targets).unwrap();
    let ss = tr_s.train_step(&mut net_s, &inputs, &targets).unwrap();
    assert_eq!(sc.loss, ss.loss, "first-step loss differs (static profile)");
}

#[test]
fn trained_readouts_learn_and_engage() {
    // Both modes must train (finite, decreasing loss) and actually move
    // their temporal parameters off the identity.
    use kdmd_snn::ReadoutMode;
    for (mode, name) in [
        (ReadoutMode::StaticProfile { t_steps: T_STEPS }, "static"),
        (ReadoutMode::SpikeAttention, "attention"),
    ] {
        let mut rng = StdRng::seed_from_u64(61);
        let task = PoissonPatternTask::new(N_IN, N_CLASSES, 0.12, 0.01, 0.4, &mut rng).unwrap();
        let mut net = build_net(12);
        let mut trainer = Trainer::new(
            &net,
            N_CLASSES,
            TrainConfig {
                readout_mode: mode,
                threads: 2,
                ..TrainConfig::default()
            },
        )
        .unwrap();
        let mut losses = Vec::new();
        for step in 0..200 {
            let (inputs, targets) = minibatch(&task, &mut rng);
            let stats = trainer.train_step(&mut net, &inputs, &targets).unwrap();
            assert!(
                stats.loss.is_finite(),
                "{name}: loss diverged at step {step}"
            );
            losses.push(stats.loss);
        }
        let early: f64 = losses[..20].iter().sum::<f64>() / 20.0;
        let late: f64 = losses[losses.len() - 20..].iter().sum::<f64>() / 20.0;
        assert!(
            late < 0.6 * early,
            "{name}: training did not reduce the loss: early {early:.4}, late {late:.4}"
        );
        match mode {
            ReadoutMode::StaticProfile { .. } => {
                let w = trainer.temporal_profile().unwrap();
                let dev = (0..w.ncols())
                    .map(|t| (w[(0, t)] - 1.0).abs())
                    .fold(0.0f64, f64::max);
                assert!(dev > 0.01, "static profile never left the identity ({dev:.4})");
            }
            ReadoutMode::SpikeAttention => {
                let u = trainer.attention_query().unwrap();
                let norm: f64 = (0..u.nrows()).map(|j| u[(j, 0)].abs()).sum();
                assert!(norm > 1e-3, "attention query never left zero ({norm:.2e})");
            }
            ReadoutMode::Count => unreachable!(),
        }
    }
}

#[test]
fn attention_readout_threaded_matches_serial() {
    // The new readout gradients must survive the data-parallel chunking like
    // every other parameter group.
    use kdmd_snn::ReadoutMode;
    let mut rng = StdRng::seed_from_u64(101);
    let task = PoissonPatternTask::new(N_IN, N_CLASSES, 0.12, 0.01, 0.4, &mut rng).unwrap();
    let (inputs, targets) = minibatch(&task, &mut rng);
    let run = |threads: usize| {
        let mut net = build_net(13);
        let mut trainer = Trainer::new(
            &net,
            N_CLASSES,
            TrainConfig {
                readout_mode: ReadoutMode::SpikeAttention,
                threads,
                ..TrainConfig::default()
            },
        )
        .unwrap();
        let stats = trainer.train_step(&mut net, &inputs, &targets).unwrap();
        let u = trainer.attention_query().unwrap().clone();
        (stats.loss, u, net)
    };
    let (l1, u1, net1) = run(1);
    let (l3, u3, net3) = run(3);
    assert!((l1 - l3).abs() < 1e-12, "loss differs across threading");
    for j in 0..u1.nrows() {
        assert!(
            (u1[(j, 0)] - u3[(j, 0)]).abs() <= 1e-12,
            "attention-query update differs at {j}"
        );
    }
    let (wa, wb) = (net1.layer(0).weights(), net3.layer(0).weights());
    for j in 0..wa.ncols() {
        for i in 0..wa.nrows() {
            assert!(
                (wa[(i, j)] - wb[(i, j)]).abs() <= 1e-12 * wa[(i, j)].abs().max(1.0),
                "weights differ at ({i},{j})"
            );
        }
    }
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
