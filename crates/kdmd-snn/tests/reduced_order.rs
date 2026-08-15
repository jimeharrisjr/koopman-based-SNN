//! Phase 6 / V1 gate: reduced-order identification of a *recurrent* layer.
//!
//! The one place a fitted operator genuinely beats structure (per the
//! re-scoped value cases): an N-neuron layer with dense recurrent
//! sub-threshold coupling steps in O(N²); when its trajectories live on a
//! low-dimensional subspace (here: driven through 8 input channels), DMDc's
//! rank-r output projection yields an `Operator::LowRank` stepping in O(N·r)
//! at matched accuracy.
//!
//! Demonstrated in the sub-threshold (reservoir) regime — the regime where
//! the compression claim lives; threshold-path basis augmentation is the
//! documented follow-up (architecture doc §5).

use faer::Mat;
use kdmd_snn::identify::{fit_controlled, IdentifyConfig, RankPolicy, SnapshotSet};
use kdmd_snn::neuron::{Lif, LifParams};
use kdmd_snn::{KoopmanLayer, LayerState, Operator, SpikeVec};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Instant;

const N: usize = 256;
const N_IN: usize = 8;
const T: usize = 400;
/// Upper bound on the reduced rank; the fit uses min(this, the data's
/// numerical rank at 1e-10) — data decides, not wishful thinking.
const RANK_CAP: usize = 64;
/// Effectively-infinite threshold: pure sub-threshold (reservoir) dynamics.
const SILENT: f64 = 1e12;

/// Dense recurrent operator: the LIF block structure plus weak all-to-all
/// v→v coupling (Gershgorin-bounded to stay stable).
fn recurrent_operator(lif: &Lif, rng: &mut StdRng) -> Mat<f64> {
    let a_local = lif.a_local();
    let mut a = Mat::<f64>::zeros(2 * N, 2 * N);
    for p in 0..2 {
        for q in 0..2 {
            for j in 0..N {
                a[(p * N + j, q * N + j)] = a_local[(p, q)];
            }
        }
    }
    let coupling_row_sum = 0.02;
    for i in 0..N {
        for j in 0..N {
            if i != j {
                a[(i, j)] += rng.random_range(-1.0..1.0) * coupling_row_sum / N as f64;
            }
        }
    }
    a
}

/// The structural input matrix: drive channel j injects `b_local ⊗ W[:, j]`.
fn structural_b(lif: &Lif, w: &Mat<f64>) -> Mat<f64> {
    let [delta, one_minus_beta] = lif.b_local();
    let mut b = Mat::<f64>::zeros(2 * N, N_IN);
    for j in 0..N_IN {
        for i in 0..N {
            b[(i, j)] = delta * w[(i, j)];
            b[(N + i, j)] = one_minus_beta * w[(i, j)];
        }
    }
    b
}

/// Run the dense recurrent layer, recording states and input indicators.
fn record_run(layer: &mut KoopmanLayer, rng: &mut StdRng, t_steps: usize) -> (Mat<f64>, Mat<f64>) {
    let mut state = LayerState::zeros(N, 2, 1).unwrap();
    let mut states = Mat::<f64>::zeros(2 * N, t_steps + 1);
    let mut controls = Mat::<f64>::zeros(N_IN, t_steps);
    let mut out = SpikeVec::new(N);
    for t in 0..t_steps {
        let active: Vec<u32> = (0..N_IN as u32)
            .filter(|_| rng.random::<f64>() < 0.10)
            .collect();
        for &j in &active {
            controls[(j as usize, t)] = 1.0;
        }
        let s_in = SpikeVec::from_indices(active, N_IN).unwrap();
        layer.step(&mut state, &s_in, &mut out).unwrap();
        assert!(out.is_empty(), "silent regime must not spike");
        for i in 0..2 * N {
            states[(i, t + 1)] = state.as_mat()[(i, 0)];
        }
    }
    (states, controls)
}

#[test]
fn low_rank_fit_matches_dense_recurrent_dynamics_at_a_fraction_of_the_cost() {
    let lif = Lif::new(LifParams {
        dt: 0.5,
        theta: SILENT,
        ..LifParams::default()
    })
    .unwrap();
    let mut rng = StdRng::seed_from_u64(23);
    let a_rec = recurrent_operator(&lif, &mut rng);
    let w = Mat::from_fn(N, N_IN, |_, _| rng.random_range(0.0..1.0));
    let b_structural = structural_b(&lif, &w);

    // Ground-truth runs (4 training + 1 held-out), all from rest.
    let mut builder = SnapshotSet::builder(2 * N, N_IN);
    for _ in 0..4 {
        let mut layer = KoopmanLayer::new(
            Operator::Dense(a_rec.clone()),
            w.clone(),
            lif.b_local().to_vec(),
            SILENT,
            1,
        )
        .unwrap();
        let (states, controls) = record_run(&mut layer, &mut rng, T);
        builder
            .push_trajectory(states.as_ref(), Some(controls.as_ref()))
            .unwrap();
    }
    let snapshots = builder.build().unwrap();

    let mut heldout_layer = KoopmanLayer::new(
        Operator::Dense(a_rec.clone()),
        w.clone(),
        lif.b_local().to_vec(),
        SILENT,
        1,
    )
    .unwrap();
    let (heldout_states, heldout_controls) = record_run(&mut heldout_layer, &mut rng, T);
    let mut heldout_builder = SnapshotSet::builder(2 * N, N_IN);
    heldout_builder
        .push_trajectory(heldout_states.as_ref(), Some(heldout_controls.as_ref()))
        .unwrap();
    let heldout = heldout_builder.build().unwrap();

    // Numerical rank of the data (relative σ cutoff 1e-10): the regression
    // and the reduced basis can only be as rich as what the drive explored.
    let numerical_rank = |m: &Mat<f64>| -> usize {
        let svd = m.svd().unwrap();
        let s = svd.S().column_vector();
        let s0 = s[0];
        (0..s.nrows()).filter(|&i| s[i] >= 1e-10 * s0).count()
    };
    let mut omega = Mat::<f64>::zeros(2 * N + N_IN, snapshots.n_pairs());
    for c in 0..snapshots.n_pairs() {
        for i in 0..2 * N {
            omega[(i, c)] = snapshots.x1()[(i, c)];
        }
        for i in 0..N_IN {
            omega[(2 * N + i, c)] = snapshots.u()[(i, c)];
        }
    }
    let rank_input = numerical_rank(&omega);
    let rank_output = RANK_CAP
        .min(numerical_rank(&snapshots.x2().to_owned()))
        .min(rank_input);
    assert!(
        rank_output >= 16,
        "drive explored implausibly few directions ({rank_output})"
    );

    // Reduced-order identification with the structural B pinned.
    let cfg = IdentifyConfig {
        rank: RankPolicy::Fixed(rank_input),
        rank_output: Some(rank_output),
        dt: 0.5,
        ..IdentifyConfig::default()
    };
    let fit = fit_controlled(&snapshots, Some(b_structural), Some(&heldout), &cfg).unwrap();
    let low_rank = match &fit.operator {
        op @ Operator::LowRank { p, a_r } => {
            assert_eq!(p.ncols(), rank_output);
            assert_eq!(a_r.nrows(), rank_output);
            assert!(
                rank_output * (2 * (2 * N) + rank_output) < (2 * N) * (2 * N),
                "reduced apply is not cheaper than dense at this rank"
            );
            op.clone()
        }
        other => panic!("expected LowRank, got {other:?}"),
    };

    // One-step held-out accuracy (scaled by the state's own magnitude).
    let state_rms = {
        let mut sum = 0.0;
        for c in 0..heldout_states.ncols() {
            for i in 0..2 * N {
                sum += heldout_states[(i, c)] * heldout_states[(i, c)];
            }
        }
        (sum / (heldout_states.ncols() * 2 * N) as f64).sqrt()
    };
    let one_step = fit.report.heldout_rmse.unwrap();
    assert!(
        one_step < 0.05 * state_rms,
        "one-step reduced error {one_step:.3e} vs state RMS {state_rms:.3e}"
    );

    // Free rollout on the held-out input sequence: reduced layer vs dense.
    let mut reduced_layer = KoopmanLayer::new(
        low_rank.clone(),
        w.clone(),
        lif.b_local().to_vec(),
        SILENT,
        1,
    )
    .unwrap();
    let mut state = LayerState::zeros(N, 2, 1).unwrap();
    let mut out = SpikeVec::new(N);
    let mut err_sum = 0.0;
    let mut n_samples = 0usize;
    for t in 0..T {
        let active: Vec<u32> = (0..N_IN as u32)
            .filter(|j| heldout_controls[(*j as usize, t)] == 1.0)
            .collect();
        let s_in = SpikeVec::from_indices(active, N_IN).unwrap();
        reduced_layer.step(&mut state, &s_in, &mut out).unwrap();
        for i in 0..2 * N {
            let d = state.as_mat()[(i, 0)] - heldout_states[(i, t + 1)];
            err_sum += d * d;
            n_samples += 1;
        }
    }
    let rollout_rmse = (err_sum / n_samples as f64).sqrt();
    assert!(
        rollout_rmse < 0.10 * state_rms,
        "rollout reduced error {rollout_rmse:.3e} vs state RMS {state_rms:.3e}"
    );

    // Cost: the rank-r apply must beat the dense apply outright
    // (O(N·r) = ~45k mul-adds vs O(N²) = ~262k at these sizes).
    let x = Mat::from_fn(2 * N, 1, |i, _| (i as f64 * 0.13).sin());
    let mut y = Mat::<f64>::zeros(2 * N, 1);
    let dense_op = Operator::Dense(a_rec);
    let reps = 50;
    let t0 = Instant::now();
    for _ in 0..reps {
        dense_op.apply(x.as_ref(), y.as_mut(), false);
    }
    let dense_time = t0.elapsed();
    let t1 = Instant::now();
    for _ in 0..reps {
        low_rank.apply(x.as_ref(), y.as_mut(), false);
    }
    let low_rank_time = t1.elapsed();
    assert!(
        low_rank_time < dense_time,
        "LowRank apply ({low_rank_time:?}) not faster than Dense ({dense_time:?})"
    );
}
