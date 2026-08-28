//! S-B: the V4 gradient-horizon measurement (docs/25).
//!
//! One recurrent LIF layer (N = 64) on the synthetic Poisson-pattern task,
//! T = 60 steps at dt = 1 ms. For each τ_m ∈ {10, 20, 40, 80} ms
//! (τ_s = τ_m/2): 30 train steps with gradient-norm recording, then estimate
//! the per-step backward decay factor of ‖λ_t‖ and report it against the
//! spectral prediction α = exp(−dt/τ_m).
//!
//! Estimators (both reported): the registered log-slope over the
//! pre-saturation window near the horizon, and an AR(1) fit
//! (n(d+1) ≈ ρ·n(d) + g, robust to the count readout's constant per-step
//! injection, which saturates ‖λ‖ in the bulk — deviation note in docs/26).
//!
//! Run: cargo run --release --example grad_horizon

use faer::Mat;
use kdmd_snn::data::PoissonPatternTask;
use kdmd_snn::neuron::{Lif, LifParams};
use kdmd_snn::{KoopmanLayer, Network, SpikeBatch, TrainConfig, Trainer};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const N_IN: usize = 24;
const N_HIDDEN: usize = 64;
const N_CLASSES: usize = 2;
const T_STEPS: usize = 60;
const BATCH: usize = 16;
const DT: f64 = 1.0;

fn minibatch(
    task: &PoissonPatternTask,
    rng: &mut StdRng,
) -> (Vec<SpikeBatch>, Vec<usize>) {
    let mut inputs: Vec<SpikeBatch> = (0..T_STEPS)
        .map(|_| SpikeBatch::zeros(N_IN, BATCH).unwrap())
        .collect();
    let mut targets = Vec::with_capacity(BATCH);
    for b in 0..BATCH {
        let class = rng.random_range(0..N_CLASSES);
        targets.push(class);
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

/// Least-squares slope of y over x.
fn slope(xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len() as f64;
    let mx = xs.iter().sum::<f64>() / n;
    let my = ys.iter().sum::<f64>() / n;
    let mut num = 0.0;
    let mut den = 0.0;
    for (x, y) in xs.iter().zip(ys) {
        num += (x - mx) * (y - my);
        den += (x - mx) * (x - mx);
    }
    num / den
}

fn main() {
    // `silent` argument: fix the registered 0.6 input scale (all nets stay
    // sub-threshold — the clean linear-law regime); default: per-tau
    // calibrated drive (the active regime). Both tables in docs/26.
    let silent = std::env::args().any(|a| a == "silent");
    println!(
        "S-B gradient-horizon measurement: N = {N_HIDDEN}, T = {T_STEPS}, \
         batch {BATCH}, 30 train steps per tau, regime: {}",
        if silent { "SILENT (0.6 scale)" } else { "ACTIVE (calibrated)" }
    );
    println!("tau_m | alpha (pred) | AR(1) rho | log-slope rho | rho/alpha | activity");
    for tau_m in [10.0, 20.0, 40.0, 80.0] {
        let tau_s = tau_m / 2.0;
        let lif = Lif::new(LifParams {
            tau_m,
            tau_s,
            dt: DT,
            ..LifParams::default()
        })
        .unwrap();
        let mut rng = StdRng::seed_from_u64(7);
        // Per-tau input-scale calibration (deviation from the registered
        // fixed 0.6 scale, noted in docs/26): the LIF DC gain is
        // tau-independent but the fluctuation variance low-passes as tau
        // grows, so a single scale cannot give comparable activity across
        // tau. Search the smallest scale whose UNTRAINED activity reaches
        // the 2-10% band.
        let task_probe = PoissonPatternTask::new(N_IN, N_CLASSES, 0.12, 0.01, 0.4, &mut rng).unwrap();
        let mut chosen_scale = if silent { 0.6 } else { 4.4 };
        for scale_i in 0..if silent { 0 } else { 30 } {
            let scale = 1.4 + 0.1 * scale_i as f64;
            let w = Mat::from_fn(N_HIDDEN, N_IN, |_, _| rng.random_range(0.0..scale));
            let layer = KoopmanLayer::lif(&lif, N_HIDDEN, w, BATCH).unwrap();
            let mut probe_net = Network::new(vec![layer], BATCH).unwrap();
            let mut rng_probe = StdRng::seed_from_u64(99);
            let (inputs, _) = minibatch(&task_probe, &mut rng_probe);
            let mut spikes = 0.0;
            for input in &inputs {
                let out = probe_net.step_batch(input).unwrap().as_mat().to_owned();
                for b in 0..out.ncols() {
                    for i in 0..out.nrows() {
                        spikes += out[(i, b)];
                    }
                }
            }
            let act = spikes / (T_STEPS * N_HIDDEN * BATCH) as f64;
            if act >= 0.02 {
                chosen_scale = scale;
                break;
            }
        }
        let w = Mat::from_fn(N_HIDDEN, N_IN, |_, _| rng.random_range(0.0..chosen_scale));
        let layer = KoopmanLayer::lif(&lif, N_HIDDEN, w, BATCH)
            .unwrap()
            .with_recurrent(Mat::zeros(N_HIDDEN, N_HIDDEN))
            .unwrap();
        let mut net = Network::new(vec![layer], BATCH).unwrap();
        eprintln!("  tau {tau_m}: calibrated input scale {chosen_scale:.1}");
        let mut trainer = Trainer::new(
            &net,
            N_CLASSES,
            TrainConfig {
                record_grad_norms: true,
                ..TrainConfig::default()
            },
        )
        .unwrap();
        let task = PoissonPatternTask::new(N_IN, N_CLASSES, 0.12, 0.01, 0.4, &mut rng).unwrap();

        // Train 30 steps (norms recorded each step; we analyze an average of
        // the last 5 tapes for stability), and measure spike activity.
        let mut tapes: Vec<Vec<f64>> = Vec::new();
        let mut first_loss = 0.0;
        let mut last_loss = 0.0;
        for step in 0..30 {
            let (inputs, targets) = minibatch(&task, &mut rng);
            let stats = trainer.train_step(&mut net, &inputs, &targets).unwrap();
            assert!(stats.loss.is_finite());
            if step == 0 {
                first_loss = stats.loss;
            }
            last_loss = stats.loss;
            if step >= 25 {
                let norms = trainer.grad_norms().expect("recording on")[0].clone();
                tapes.push(norms);
            }
        }
        eprintln!("  tau {tau_m}: loss {first_loss:.4} -> {last_loss:.4}");
        // Mean tape over the last 5 steps; index by distance from horizon
        // d = T-1-t (λ at the horizon is the freshest injection).
        let t_len = tapes[0].len();
        let mean_tape: Vec<f64> = (0..t_len)
            .map(|t| tapes.iter().map(|tp| tp[t]).sum::<f64>() / tapes.len() as f64)
            .collect();
        let n_of_d: Vec<f64> = (0..t_len).map(|d| mean_tape[t_len - 1 - d]).collect();

        // AR(1): n(d+1) ~ rho*n(d) + g over d in [1, T-6].
        let pairs: Vec<(f64, f64)> = (1..t_len - 5)
            .map(|d| (n_of_d[d], n_of_d[d + 1]))
            .collect();
        let xs: Vec<f64> = pairs.iter().map(|p| p.0).collect();
        let ys: Vec<f64> = pairs.iter().map(|p| p.1).collect();
        let rho_ar = slope(&xs, &ys);

        // Registered log-slope estimator over the pre-saturation window
        // d in [0, 15] (near the horizon, before the constant-injection
        // plateau).
        let d_max = 15.min(t_len - 1);
        let xs2: Vec<f64> = (0..=d_max).map(|d| d as f64).collect();
        let ys2: Vec<f64> = (0..=d_max).map(|d| n_of_d[d].max(1e-300).ln()).collect();
        let rho_log = slope(&xs2, &ys2).exp();

        // Rough spike activity of the trained net on one fresh batch.
        let (inputs, _) = minibatch(&task, &mut rng);
        let mut spikes = 0.0;
        net.reset_state();
        for input in &inputs {
            let out = net.step_batch(input).unwrap().as_mat().to_owned();
            for b in 0..out.ncols() {
                for i in 0..out.nrows() {
                    spikes += out[(i, b)];
                }
            }
        }
        let act = spikes / (T_STEPS * N_HIDDEN * BATCH) as f64;

        let alpha = (-DT / tau_m).exp();
        println!(
            "{tau_m:5.0} | {alpha:12.4} | {rho_ar:9.4} | {rho_log:13.4} | {:9.3} | {act:.4}",
            rho_ar / alpha
        );
    }
    println!(
        "\nPB1: measured rho <= alpha; PB2: horizon grows with tau_m; \
         PB3: rho within [0.5*alpha, alpha]."
    );
}
