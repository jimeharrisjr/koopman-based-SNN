//! Phase 3 exit gates: DMDc identification of a *spiking* LIF layer.
//!
//! With the subtractive reset folded into the control input, the layer step
//! is exactly `x_{t+1} = A·x_t + B·u_t` with `u_t = [drive_t ; s_t]` — so
//! identification works on full spiking trajectories with **no masking**, and
//! the identified reset columns must equal `−θ·e_v` (the structural
//! consistency check). Hard reset breaks that linearity, and its masked
//! sub-threshold path is tested separately.

use faer::Mat;
use kdmd_snn::identify::{
    fit_controlled, lif_structural_b, record_input_sequence, IdentifyConfig, RankPolicy, Recording,
    SnapshotSet,
};
use kdmd_snn::neuron::{Lif, LifParams, NeuronModel, ResetMode};
use kdmd_snn::operator::{per_variable_from_dense, Operator};
use kdmd_snn::LayerState;

const N: usize = 3;
const T: usize = 800;
const DT: f64 = 0.5;

/// Per-neuron drive: distinct frequencies and offsets so the layer spikes
/// irregularly and the regressors are persistently excited.
fn drive_at(t: usize) -> Mat<f64> {
    let mut m = Mat::<f64>::zeros(N, 1);
    let time = t as f64 * DT;
    m[(0, 0)] = 1.6 + 0.9 * (0.13 * time).sin();
    m[(1, 0)] = 1.4 + 0.8 * (0.29 * time + 1.0).sin();
    m[(2, 0)] = 1.5 + 0.7 * (0.07 * time + 2.0).cos();
    m
}

/// Simulate a spiking layer and assemble the snapshot set with
/// `u_t = [drive_t ; s_t]`, plus the recording for masking decisions.
fn spiking_snapshots(lif: &Lif) -> (SnapshotSet, Recording, Vec<Mat<f64>>) {
    let mut state = LayerState::zeros(N, 2, 1).unwrap();
    lif.init_state(&mut state);
    let inputs: Vec<Mat<f64>> = (0..T).map(drive_at).collect();
    let rec = record_input_sequence(lif, &mut state, &inputs).unwrap();

    let mut controls = Mat::<f64>::zeros(2 * N, T);
    for t in 0..T {
        for j in 0..N {
            controls[(j, t)] = inputs[t][(j, 0)];
            controls[(N + j, t)] = rec.spikes[t].as_mat()[(j, 0)];
        }
    }
    let mut builder = SnapshotSet::builder(2 * N, 2 * N);
    builder
        .push_trajectory(rec.states.as_ref(), Some(controls.as_ref()))
        .unwrap();
    (builder.build().unwrap(), rec, inputs)
}

/// Dense `A_local ⊗ I_N` reference.
fn kron_oracle(lif: &Lif) -> Mat<f64> {
    let a_local = lif.a_local();
    let mut a = Mat::<f64>::zeros(2 * N, 2 * N);
    for p in 0..2 {
        for q in 0..2 {
            for j in 0..N {
                a[(p * N + j, q * N + j)] = a_local[(p, q)];
            }
        }
    }
    a
}

fn max_abs_diff(a: &Mat<f64>, b: &Mat<f64>) -> f64 {
    let mut worst = 0.0f64;
    for j in 0..a.ncols() {
        for i in 0..a.nrows() {
            worst = worst.max((a[(i, j)] - b[(i, j)]).abs());
        }
    }
    worst
}

#[test]
fn known_b_fit_on_spiking_data_recovers_a_without_masking() {
    // R1 exit gate: subtractive reset in u_t makes the step exactly linear,
    // so A is recovered from FULL spiking trajectories at oracle precision.
    let lif = Lif::new(LifParams {
        dt: DT,
        ..LifParams::default()
    })
    .unwrap();
    let (snapshots, rec, _) = spiking_snapshots(&lif);
    let n_spiking = rec.any_spike.iter().filter(|&&s| s).count();
    assert!(
        n_spiking > 30,
        "test needs a genuinely spiking regime (got {n_spiking} spiking transitions)"
    );

    let b_structural = lif_structural_b(&lif, N).unwrap();
    let cfg = IdentifyConfig {
        rank: RankPolicy::Fixed(2 * N),
        dt: DT,
        ..IdentifyConfig::default()
    };
    let fit = fit_controlled(&snapshots, Some(b_structural), None, &cfg).unwrap();

    let a = match &fit.operator {
        Operator::Dense(a) => a.clone(),
        other => panic!("expected Dense, got {other:?}"),
    };
    let oracle = kron_oracle(&lif);
    let err = max_abs_diff(&a, &oracle);
    assert!(err <= 1e-8, "A error {err:.2e} > 1e-8 on spiking data");

    // The recovered dense A carries the A_local ⊗ I structure.
    assert!(per_variable_from_dense(a.as_ref(), N, 1e-8).is_ok());
    assert!(fit.report.stability.is_stable);
}

#[test]
fn joint_fit_recovers_structural_b_including_reset_columns() {
    // Consistency check: with the reset expressed as control, a joint (A, B)
    // fit on this deterministic data must reproduce the structural B —
    // in particular the reset columns ≈ −θ·e_v.
    let lif = Lif::new(LifParams {
        dt: DT,
        ..LifParams::default()
    })
    .unwrap();
    let (snapshots, _, _) = spiking_snapshots(&lif);

    let cfg = IdentifyConfig {
        rank: RankPolicy::Fixed(4 * N), // state (2N) + controls (2N)
        dt: DT,
        ..IdentifyConfig::default()
    };
    let fit = fit_controlled(&snapshots, None, None, &cfg).unwrap();

    let b_structural = lif_structural_b(&lif, N).unwrap();
    let b_err = max_abs_diff(&fit.b, &b_structural);
    assert!(
        b_err <= 1e-6,
        "estimated B deviates from structural B by {b_err:.2e}"
    );
    // Reset columns specifically: −θ on the v rows.
    for j in 0..N {
        let got = fit.b[(j, N + j)];
        assert!(
            (got - (-1.0)).abs() <= 1e-6,
            "reset column {j}: got {got}, expected −θ = −1"
        );
    }

    let a = match &fit.operator {
        Operator::Dense(a) => a.clone(),
        other => panic!("expected Dense, got {other:?}"),
    };
    let a_err = max_abs_diff(&a, &kron_oracle(&lif));
    assert!(a_err <= 1e-6, "joint-fit A error {a_err:.2e}");
}

#[test]
fn hard_reset_requires_masked_subthreshold_fitting() {
    // R2 path: hard reset is bilinear — not representable as B·u — so
    // identification must excise every spike-straddling transition and pin
    // only the drive block of B.
    let lif = Lif::new(LifParams {
        dt: DT,
        reset: ResetMode::HardTo(0.0),
        ..LifParams::default()
    })
    .unwrap();
    let mut state = LayerState::zeros(N, 2, 1).unwrap();
    lif.init_state(&mut state);
    let inputs: Vec<Mat<f64>> = (0..T).map(drive_at).collect();
    let rec = record_input_sequence(&lif, &mut state, &inputs).unwrap();
    let n_spiking = rec.any_spike.iter().filter(|&&s| s).count();
    assert!(
        n_spiking > 30,
        "test needs a genuinely spiking regime (got {n_spiking} spiking transitions)"
    );

    // Controls: drive only (q = N) — there is no reset control in R2.
    let mut controls = Mat::<f64>::zeros(N, T);
    for t in 0..T {
        for j in 0..N {
            controls[(j, t)] = inputs[t][(j, 0)];
        }
    }
    let mut builder = SnapshotSet::builder(2 * N, N);
    builder
        .push_trajectory_masked(
            rec.states.as_ref(),
            Some(controls.as_ref()),
            &rec.subthreshold_mask(),
        )
        .unwrap();
    let snapshots = builder.build().unwrap();
    assert_eq!(snapshots.n_pairs(), T - n_spiking);

    // Drive block of the structural B (columns 0..N).
    let b_full = lif_structural_b(&lif, N).unwrap();
    let b_drive = b_full.as_ref().subcols(0, N).to_owned();

    let cfg = IdentifyConfig {
        rank: RankPolicy::Fixed(2 * N),
        dt: DT,
        ..IdentifyConfig::default()
    };
    let fit = fit_controlled(&snapshots, Some(b_drive), None, &cfg).unwrap();
    let a = match &fit.operator {
        Operator::Dense(a) => a.clone(),
        other => panic!("expected Dense, got {other:?}"),
    };
    let err = max_abs_diff(&a, &kron_oracle(&lif));
    assert!(err <= 1e-8, "masked R2 fit error {err:.2e} > 1e-8");

    // Control experiment: fitting the SAME hard-reset data WITHOUT the mask
    // must corrupt A — this is the failure mode the mask exists to prevent.
    let mut unmasked = SnapshotSet::builder(2 * N, N);
    unmasked
        .push_trajectory(rec.states.as_ref(), Some(controls.as_ref()))
        .unwrap();
    let b_drive2 = lif_structural_b(&lif, N)
        .unwrap()
        .as_ref()
        .subcols(0, N)
        .to_owned();
    let corrupted = fit_controlled(&unmasked.build().unwrap(), Some(b_drive2), None, &cfg);
    // Rejection by a validation gate is an equally acceptable outcome, so
    // only a *successful* unmasked fit is checked for corruption.
    if let Ok(fit) = corrupted {
        let a = match &fit.operator {
            Operator::Dense(a) => a.clone(),
            other => panic!("expected Dense, got {other:?}"),
        };
        let err = max_abs_diff(&a, &kron_oracle(&lif));
        assert!(
            err > 1e-3,
            "unmasked hard-reset fit should be visibly corrupted, error was {err:.2e}"
        );
    }
}

#[test]
fn heldout_pairs_report_near_zero_rmse_for_exact_fits() {
    let lif = Lif::new(LifParams {
        dt: DT,
        ..LifParams::default()
    })
    .unwrap();
    let (train, _, _) = spiking_snapshots(&lif);

    // A second, differently-driven run as held-out data.
    let mut state = LayerState::zeros(N, 2, 1).unwrap();
    lif.init_state(&mut state);
    let inputs: Vec<Mat<f64>> = (0..300)
        .map(|t| {
            let mut m = drive_at(t + 13);
            for j in 0..N {
                m[(j, 0)] *= 0.8;
            }
            m
        })
        .collect();
    let rec = record_input_sequence(&lif, &mut state, &inputs).unwrap();
    let mut controls = Mat::<f64>::zeros(2 * N, 300);
    for t in 0..300 {
        for j in 0..N {
            controls[(j, t)] = inputs[t][(j, 0)];
            controls[(N + j, t)] = rec.spikes[t].as_mat()[(j, 0)];
        }
    }
    let mut builder = SnapshotSet::builder(2 * N, 2 * N);
    builder
        .push_trajectory(rec.states.as_ref(), Some(controls.as_ref()))
        .unwrap();
    let heldout = builder.build().unwrap();

    let cfg = IdentifyConfig {
        rank: RankPolicy::Fixed(2 * N),
        dt: DT,
        ..IdentifyConfig::default()
    };
    let b = lif_structural_b(&lif, N).unwrap();
    let fit = fit_controlled(&train, Some(b), Some(&heldout), &cfg).unwrap();
    let rmse = fit.report.heldout_rmse.unwrap();
    assert!(rmse < 1e-9, "held-out pair RMSE {rmse:.2e} too large");
}
