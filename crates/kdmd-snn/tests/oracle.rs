//! The permanent LIF oracle: DMD identification must recover the closed-form
//! LIF propagator from sub-threshold data.
//!
//! For plain LIF the discrete propagator is known analytically
//! (`Lif::a_local`), so identification here can only ever re-estimate a known
//! matrix — which is exactly what makes it the perfect correctness gate for
//! the whole identification pipeline. This is the Phase 2 exit criterion:
//! **identified A matches the closed form to ≤ 1e-8 relative error.**

use faer::Mat;
use kdmd_snn::identify::{fit_autonomous, record_constant_input, IdentifyConfig, RankPolicy};
use kdmd_snn::neuron::{Lif, LifParams, NeuronModel};
use kdmd_snn::operator::{per_variable_from_dense, Operator};
use kdmd_snn::{LayerState, SpikeBatch};

/// A silent (never-firing) LIF for pure sub-threshold trajectories.
fn silent_lif(dt: f64) -> Lif {
    Lif::new(LifParams {
        theta: 1e12,
        dt,
        ..LifParams::default()
    })
    .unwrap()
}

/// Record a zero-input trajectory of a single neuron from `(v0, i0)`.
fn free_decay(lif: &Lif, v0: f64, i0: f64, t_steps: usize) -> Mat<f64> {
    let mut state = LayerState::zeros(1, 2, 1).unwrap();
    lif.init_state(&mut state);
    state.var_mut(0)[(0, 0)] = v0;
    state.var_mut(1)[(0, 0)] = i0;
    let input = Mat::<f64>::zeros(1, 1);
    record_constant_input(lif, &mut state, input.as_ref(), t_steps)
        .unwrap()
        .states
}

#[test]
fn identified_a_matches_closed_form_lif_oracle() {
    let dt = 0.1;
    let lif = silent_lif(dt);
    // Generic initial condition: both the α (membrane) and β (synapse) modes
    // are excited, so one trajectory spans the full 2-dimensional state.
    let trajectory = free_decay(&lif, 0.7, 1.3, 400);
    let heldout = free_decay(&lif, -0.4, 0.9, 200);

    let cfg = IdentifyConfig {
        rank: RankPolicy::Fixed(2),
        dt,
        ..IdentifyConfig::default()
    };
    let fit = fit_autonomous(trajectory.as_ref(), Some(heldout.as_ref()), &cfg).unwrap();

    // Exit criterion: entrywise agreement with the analytic propagator at
    // ≤ 1e-8 relative to the matrix scale.
    let oracle = lif.a_local();
    let a = match &fit.operator {
        Operator::Dense(a) => a,
        other => panic!("expected Dense operator, got {other:?}"),
    };
    let scale = (0..2)
        .flat_map(|i| (0..2).map(move |j| (i, j)))
        .map(|(i, j)| oracle[(i, j)].abs())
        .fold(0.0f64, f64::max);
    for i in 0..2 {
        for j in 0..2 {
            let err = (a[(i, j)] - oracle[(i, j)]).abs() / scale;
            assert!(
                err <= 1e-8,
                "A[({i}, {j})] = {} vs oracle {} — relative error {err:.2e} > 1e-8",
                a[(i, j)],
                oracle[(i, j)]
            );
        }
    }

    // The report must carry a clean bill of health.
    assert!(fit.report.stability.is_stable);
    assert_eq!(fit.report.rank, 2);
    let rmse = fit.report.heldout_rmse.unwrap();
    assert!(rmse < 1e-10, "held-out one-step RMSE {rmse:.2e} too large");
}

#[test]
fn spectrum_recovers_membrane_and_synapse_time_constants() {
    let dt = 0.1;
    let lif = silent_lif(dt);
    let trajectory = free_decay(&lif, 0.7, 1.3, 400);
    let cfg = IdentifyConfig {
        rank: RankPolicy::Fixed(2),
        dt,
        ..IdentifyConfig::default()
    };
    let fit = fit_autonomous(trajectory.as_ref(), None, &cfg).unwrap();

    // Eigenvalues are e^(−dt/τ_m) and e^(−dt/τ_s); recover the taus from the
    // spectrum and compare with the ground-truth parameters (10 ms, 5 ms).
    let mut taus: Vec<f64> = fit
        .report
        .spectrum
        .iter()
        .map(|m| -dt / m.magnitude.ln())
        .collect();
    taus.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(taus.len(), 2);
    assert!(
        (taus[0] - 5.0).abs() / 5.0 < 1e-6,
        "recovered τ_s = {} (expected 5)",
        taus[0]
    );
    assert!(
        (taus[1] - 10.0).abs() / 10.0 < 1e-6,
        "recovered τ_m = {} (expected 10)",
        taus[1]
    );
}

#[test]
fn per_neuron_fit_drives_a_whole_layer_via_per_variable_operator() {
    // The supported layer path: fit ONE neuron, replicate as A_local ⊗ I_N,
    // and verify the operator reproduces the reference simulator across a
    // 5-neuron layer to oracle precision.
    let dt = 0.1;
    let lif = silent_lif(dt);
    let trajectory = free_decay(&lif, 0.7, 1.3, 400);
    let cfg = IdentifyConfig {
        rank: RankPolicy::Fixed(2),
        dt,
        ..IdentifyConfig::default()
    };
    let fit = fit_autonomous(trajectory.as_ref(), None, &cfg).unwrap();
    let a_local = match fit.operator {
        Operator::Dense(a) => a,
        other => panic!("expected Dense, got {other:?}"),
    };
    let n = 5;
    let op = Operator::PerVariable {
        a_local,
        n_neurons: n,
    };

    // Layer with distinct per-neuron initial conditions, zero input.
    let mut state = LayerState::zeros(n, 2, 1).unwrap();
    lif.init_state(&mut state);
    for j in 0..n {
        state.var_mut(0)[(j, 0)] = 0.1 + 0.15 * j as f64;
        state.var_mut(1)[(j, 0)] = -0.5 + 0.3 * j as f64;
    }
    let mut op_state = Mat::<f64>::zeros(2 * n, 1);
    for i in 0..2 * n {
        op_state[(i, 0)] = state.as_mat()[(i, 0)];
    }

    let input = Mat::<f64>::zeros(n, 1);
    let mut spikes = SpikeBatch::zeros(n, 1).unwrap();
    let mut next = Mat::<f64>::zeros(2 * n, 1);
    for _ in 0..100 {
        lif.step(&mut state, input.as_ref(), &mut spikes);
        op.apply(op_state.as_ref(), next.as_mut(), false);
        std::mem::swap(&mut op_state, &mut next);
        for i in 0..2 * n {
            let err = (op_state[(i, 0)] - state.as_mat()[(i, 0)]).abs();
            assert!(
                err <= 1e-8,
                "operator and simulator diverged at row {i}: |Δ| = {err:.2e}"
            );
        }
    }

    // And the dense reconstruction of that operator compresses back to
    // PerVariable structure.
    let k = 2;
    let mut dense = Mat::<f64>::zeros(k * n, k * n);
    let mut basis = Mat::<f64>::zeros(k * n, k * n);
    for i in 0..k * n {
        basis[(i, i)] = 1.0;
    }
    op.apply(basis.as_ref(), dense.as_mut(), false);
    assert!(per_variable_from_dense(dense.as_ref(), n, 1e-10).is_ok());
}

#[test]
fn single_trajectory_cannot_identify_a_network_operator() {
    // Identifiability trap (documented in identify's module docs): one
    // trajectory of an N-neuron LIF layer spans only the 2-dimensional
    // {α-mode, β-mode} subspace, so DMD's rank collapses to 2 even though
    // the state dimension is 2N. Fitting a network-level A from a single
    // trajectory is structurally impossible — fit per neuron instead.
    let dt = 0.1;
    let lif = silent_lif(dt);
    let n = 2; // state_dim = 4
    let mut state = LayerState::zeros(n, 2, 1).unwrap();
    lif.init_state(&mut state);
    state.var_mut(0)[(0, 0)] = 0.7;
    state.var_mut(0)[(1, 0)] = -0.3;
    state.var_mut(1)[(0, 0)] = 1.1;
    state.var_mut(1)[(1, 0)] = 0.4;
    let input = Mat::<f64>::zeros(n, 1);
    let rec = record_constant_input(&lif, &mut state, input.as_ref(), 400).unwrap();

    let cfg = IdentifyConfig {
        rank: RankPolicy::Auto,
        dt,
        ..IdentifyConfig::default()
    };
    let fit = fit_autonomous(rec.states.as_ref(), None, &cfg).unwrap();
    assert!(
        fit.report.rank <= 2,
        "rank {} — a single trajectory should collapse to the 2-mode subspace",
        fit.report.rank
    );
}

#[test]
fn growing_dynamics_are_rejected_by_the_stability_gate() {
    // x_{t+1} = 1.05 · x_t: a leaky neuron model can never produce this; the
    // gate must refuse to release the operator.
    let mut traj = Mat::<f64>::zeros(2, 50);
    for t in 0..50 {
        let g = 1.05f64.powi(t as i32);
        traj[(0, t)] = g;
        traj[(1, t)] = 0.5 * g;
    }
    let cfg = IdentifyConfig {
        rank: RankPolicy::Fixed(1),
        dt: 1.0,
        ..IdentifyConfig::default()
    };
    let err = fit_autonomous(traj.as_ref(), None, &cfg).unwrap_err();
    assert!(
        err.to_string().contains("unstable"),
        "expected stability rejection, got: {err}"
    );
}
