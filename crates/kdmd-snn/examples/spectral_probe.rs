//! Spectral probe: `koopman-dmd`'s analysis suite applied to this library's
//! neurons. The hands-on companion to the "Read the equations back out of a
//! recording" section of blog post 4.
//!
//! Four short experiments, no dataset, about a second of runtime:
//!   1. free decay of a silent LIF → `fit_autonomous` → the spectrum reads the
//!      membrane and synapse time constants back out (truth: 10 ms / 5 ms);
//!   2. the same recording with measurement noise → `dmd_convergence`: how
//!      many steps a noisy recording needs before the spectrum settles;
//!   3. a driven, spiking 3-neuron LIF layer → known-B DMDc recovers `A` from
//!      the full spiking trajectory, and a joint (A, B) fit reproduces the
//!      reset column `−θ` as a consistency check;
//!   4. the same spiking recording fed to plain DMD with the drive and the
//!      reset ignored → the stability report and the per-step residual show
//!      exactly what was left out.
//!
//! Run: cargo run --release --example spectral_probe
//!
//! `--dump <dir>` additionally writes the recordings, fits and residuals as
//! CSV files, which `analysis/probe_figures.py` turns into the blog figures.

use faer::Mat;
use kdmd_snn::identify::{
    fit_autonomous, fit_controlled, heldout_pair_rmse, lif_structural_b, record_constant_input,
    record_input_sequence, IdentifyConfig, RankPolicy, Recording, SnapshotSet,
};
use kdmd_snn::neuron::{Lif, LifParams, NeuronModel};
use kdmd_snn::operator::Operator;
use kdmd_snn::LayerState;
use koopman_dmd::{
    dmd, dmd_convergence, dmd_error, dmd_reconstruct, dmd_residual, dmd_spectrum, dmd_stability,
    DmdConfig, ModeInfo,
};
use rand::{Rng, SeedableRng};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

/// Time constant implied by a discrete eigenvalue: λ = e^(−dt/τ).
fn tau_of(m: &ModeInfo, dt: f64) -> f64 {
    -dt / m.magnitude.ln()
}

/// Print a spectrum as a table, slowest mode first.
fn print_spectrum(spectrum: &[ModeInfo], dt: f64) {
    println!(
        "  {:>4} | {:>20} | {:>8} | {:>14} | {:>8} | class",
        "mode", "eigenvalue", "|lambda|", "half-life (ms)", "tau (ms)"
    );
    let mut modes: Vec<&ModeInfo> = spectrum.iter().collect();
    modes.sort_by(|a, b| b.magnitude.partial_cmp(&a.magnitude).unwrap());
    for m in modes {
        let hl = m.half_life.map_or("-".to_string(), |h| format!("{h:.3}"));
        println!(
            "  {:>4} | {:>+11.6}{:>+8.5}i | {:.6} | {:>14} | {:>8.4} | {}",
            m.index,
            m.eigenvalue.re,
            m.eigenvalue.im,
            m.magnitude,
            hl,
            tau_of(m, dt),
            m.stability
        );
    }
}

/// A LIF that never fires: pure sub-threshold dynamics.
fn silent_lif(dt: f64) -> Lif {
    Lif::new(LifParams {
        theta: 1e12,
        dt,
        ..LifParams::default()
    })
    .unwrap()
}

/// Record a zero-input trajectory of one neuron from `(v0, i0)`;
/// `state_dim × (t_steps + 1)`.
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

/// Standard normal via Box–Muller.
fn gauss(rng: &mut impl Rng) -> f64 {
    let u1: f64 = rng.random::<f64>().max(1e-300);
    let u2: f64 = rng.random();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

// ---- experiment 3/4: a driven, spiking layer ---------------------------

const N: usize = 3;
const T: usize = 800;
const DT: f64 = 0.5;

/// Per-neuron drive: distinct frequencies and offsets so the layer spikes
/// irregularly and the regressors are persistently excited. `phase` shifts
/// the whole pattern, which is how the held-out recording differs.
fn drive_at(t: usize, phase: f64) -> Mat<f64> {
    let mut m = Mat::<f64>::zeros(N, 1);
    let time = t as f64 * DT + phase;
    m[(0, 0)] = 1.6 + 0.9 * (0.13 * time).sin();
    m[(1, 0)] = 1.4 + 0.8 * (0.29 * time + 1.0).sin();
    m[(2, 0)] = 1.5 + 0.7 * (0.07 * time + 2.0).cos();
    m
}

/// Simulate the spiking layer and assemble snapshots with
/// `u_t = [drive_t ; s_t]` — the drive and the spikes as control inputs.
fn spiking_snapshots(lif: &Lif, phase: f64) -> (SnapshotSet, Recording) {
    let mut state = LayerState::zeros(N, 2, 1).unwrap();
    lif.init_state(&mut state);
    let inputs: Vec<Mat<f64>> = (0..T).map(|t| drive_at(t, phase)).collect();
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
    (builder.build().unwrap(), rec)
}

/// The closed-form layer propagator `A_local ⊗ I_N`.
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

/// `--dump <dir>` on the command line: where to write the CSV files.
fn dump_dir() -> Option<PathBuf> {
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        if a == "--dump" {
            return args.next().map(PathBuf::from);
        }
    }
    None
}

fn write_csv(dir: &Path, name: &str, header: &str, rows: &[String]) -> std::io::Result<()> {
    let mut s = String::with_capacity(rows.len() * 64 + header.len() + 1);
    s.push_str(header);
    s.push('\n');
    for r in rows {
        s.push_str(r);
        s.push('\n');
    }
    std::fs::write(dir.join(name), s)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dump = dump_dir();
    if let Some(d) = &dump {
        std::fs::create_dir_all(d)?;
    }

    // ---- 1. the spectrum reads the time constants back out -------------
    println!("== 1. Free decay of a silent LIF: fit_autonomous + the spectrum report ==");
    let dt = 0.1;
    let lif = silent_lif(dt);
    let traj = free_decay(&lif, 0.7, 1.3, 400);
    let heldout = free_decay(&lif, -0.4, 0.9, 200);
    let cfg = IdentifyConfig {
        rank: RankPolicy::Fixed(2),
        dt,
        ..IdentifyConfig::default()
    };
    let fit = fit_autonomous(traj.as_ref(), Some(heldout.as_ref()), &cfg)?;
    print_spectrum(&fit.report.spectrum, dt);
    println!(
        "  truth: tau_m = {} ms, tau_s = {} ms",
        lif.params().tau_m,
        lif.params().tau_s
    );
    let st = &fit.report.stability;
    println!(
        "  stability: spectral radius {:.6}, stable = {}",
        st.spectral_radius, st.is_stable
    );
    println!(
        "  held-out one-step RMSE: {:.2e}",
        fit.report.heldout_rmse.unwrap()
    );
    let res = dmd_residual(&fit.dmd, &traj)?;
    let err = dmd_error(&fit.dmd, &traj)?;
    println!(
        "  one-step residual, relative: {:.2e}; mode reconstruction RMSE over {} steps: {:.2e}",
        res.residual_relative,
        traj.ncols(),
        err.rmse
    );
    if let Some(d) = &dump {
        // The trajectory as the sum of its modes: one reconstruction per mode.
        let n_steps = traj.ncols();
        let recon = (0..fit.dmd.rank)
            .map(|k| dmd_reconstruct(&fit.dmd, n_steps, Some(&[k])))
            .collect::<Result<Vec<_>, _>>()?;
        let mut header = String::from("t_ms,v,i");
        for k in 0..fit.dmd.rank {
            write!(header, ",v_mode{k},i_mode{k}")?;
        }
        let rows: Vec<String> = (0..n_steps)
            .map(|t| {
                let mut r = format!("{},{},{}", t as f64 * dt, traj[(0, t)], traj[(1, t)]);
                for m in &recon {
                    write!(r, ",{},{}", m[(0, t)], m[(1, t)]).unwrap();
                }
                r
            })
            .collect();
        write_csv(d, "decay.csv", &header, &rows)?;
        let rows: Vec<String> = fit
            .report
            .spectrum
            .iter()
            .map(|m| {
                format!(
                    "{},{},{},{}",
                    m.index,
                    m.eigenvalue.re,
                    m.eigenvalue.im,
                    tau_of(m, dt)
                )
            })
            .collect();
        write_csv(d, "decay_modes.csv", "mode,re,im,tau_ms", &rows)?;
    }

    // ---- 2. how many steps does a noisy recording need? -----------------
    println!();
    println!("== 2. The same recording with measurement noise: dmd_convergence ==");
    let sigma = 1e-3;
    let mut rng = rand::rngs::StdRng::seed_from_u64(7);
    let mut noisy = traj.clone();
    for j in 0..noisy.ncols() {
        for i in 0..noisy.nrows() {
            noisy[(i, j)] += sigma * gauss(&mut rng);
        }
    }
    let dmd_cfg = DmdConfig {
        rank: Some(2),
        center: false,
        dt,
        lifting: None,
    };
    let fractions = [0.05, 0.1, 0.2, 0.4, 0.7, 1.0];
    let conv = dmd_convergence(&noisy, &fractions, &dmd_cfg)?;
    println!(
        "  noise sigma = {sigma:.0e} on a trajectory whose largest entry is {:.2}",
        (0..traj.ncols())
            .flat_map(|j| (0..traj.nrows()).map(move |i| (i, j)))
            .map(|(i, j)| traj[(i, j)].abs())
            .fold(0.0f64, f64::max)
    );
    println!(
        "  {:>5} | {:>10} | {:>10} | max |lambda| change vs previous",
        "steps", "tau_m (ms)", "tau_s (ms)"
    );
    for (k, (&m, eigs)) in conv.sample_sizes.iter().zip(&conv.eigenvalues).enumerate() {
        let mut taus: Vec<f64> = eigs.iter().map(|e| -dt / e.norm().ln()).collect();
        taus.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let change = if k == 0 {
            "-".to_string()
        } else {
            format!("{:.2e}", conv.eigenvalue_changes[k - 1])
        };
        println!(
            "  {:>5} | {:>10.4} | {:>10.4} | {}",
            m,
            taus.first().copied().unwrap_or(f64::NAN),
            taus.get(1).copied().unwrap_or(f64::NAN),
            change
        );
    }
    if let Some(alpha) = conv.convergence_estimate {
        println!("  fitted convergence exponent: successive changes shrink like 1/m^{alpha:.2}");
    }
    if let Some(d) = &dump {
        // 20 noise seeds (seed 7 is the one printed above) on a finer grid of
        // prefix lengths, for the convergence figure.
        let sizes = [10usize, 14, 20, 28, 40, 57, 80, 113, 160, 226, 320, 401];
        let mut rows = Vec::new();
        for seed in 0..20u64 {
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            let mut noisy = traj.clone();
            for j in 0..noisy.ncols() {
                for i in 0..noisy.nrows() {
                    noisy[(i, j)] += sigma * gauss(&mut rng);
                }
            }
            for &m in &sizes {
                let res = dmd(&noisy.subcols(0, m).to_owned(), &dmd_cfg)?;
                let mut taus: Vec<f64> = res
                    .eigenvalues
                    .iter()
                    .map(|e| -dt / e.norm().ln())
                    .collect();
                taus.sort_by(|a, b| b.partial_cmp(a).unwrap());
                rows.push(format!(
                    "{seed},{m},{},{}",
                    taus.first().copied().unwrap_or(f64::NAN),
                    taus.get(1).copied().unwrap_or(f64::NAN)
                ));
            }
        }
        write_csv(d, "convergence.csv", "seed,steps,tau_m,tau_s", &rows)?;
    }

    // ---- 3. a spiking layer, with the reset written as an input -----------
    println!();
    println!("== 3. A driven, spiking 3-neuron layer: known-B DMDc on the full trajectory ==");
    let lif = Lif::new(LifParams {
        dt: DT,
        ..LifParams::default()
    })?;
    let (snapshots, rec) = spiking_snapshots(&lif, 0.0);
    let (heldout_snaps, _) = spiking_snapshots(&lif, 137.0);
    let n_spiking = rec.any_spike.iter().filter(|&&s| s).count();
    let n_spikes: usize = rec
        .spikes
        .iter()
        .map(|s| (0..N).filter(|&j| s.as_mat()[(j, 0)] != 0.0).count())
        .sum();
    println!(
        "  recording: {N} neurons, {T} steps of {DT} ms, {n_spikes} spikes on {n_spiking} of the {T} transitions"
    );
    let b_structural = lif_structural_b(&lif, N)?;
    let cfg = IdentifyConfig {
        rank: RankPolicy::Fixed(2 * N),
        dt: DT,
        ..IdentifyConfig::default()
    };
    let fit = fit_controlled(
        &snapshots,
        Some(b_structural.clone()),
        Some(&heldout_snaps),
        &cfg,
    )?;
    print_spectrum(&fit.report.spectrum, DT);
    let a = match &fit.operator {
        Operator::Dense(a) => a.clone(),
        other => panic!("expected a dense operator, got {other:?}"),
    };
    println!(
        "  max |A_identified - A_closed_form| = {:.2e}; spectral radius {:.6}, stable = {}",
        max_abs_diff(&a, &kron_oracle(&lif)),
        fit.report.stability.spectral_radius,
        fit.report.stability.is_stable
    );
    println!(
        "  held-out one-step RMSE (different drive phase): {:.2e}",
        fit.report.heldout_rmse.unwrap()
    );
    let train_rmse = heldout_pair_rmse(&fit.operator, fit.b.as_ref(), &snapshots)?;
    println!("  one-step RMSE on the training recording itself: {train_rmse:.2e}");

    println!();
    println!("  -- joint (A, B) fit as a consistency check: is the reset column really -theta? --");
    let cfg_joint = IdentifyConfig {
        rank: RankPolicy::Fixed(4 * N),
        dt: DT,
        ..IdentifyConfig::default()
    };
    let joint = fit_controlled(&snapshots, None, None, &cfg_joint)?;
    let reset_cols: Vec<String> = (0..N)
        .map(|j| format!("{:+.6}", joint.b[(j, N + j)]))
        .collect();
    println!(
        "  estimated reset entries B[v_j, s_j] = [{}]  (theta = {})",
        reset_cols.join(", "),
        lif.params().theta
    );
    println!(
        "  max |B_estimated - B_structural| = {:.2e}",
        max_abs_diff(&joint.b, &b_structural)
    );

    // ---- 4. the same recording, drive and reset ignored -------------------
    println!();
    println!("== 4. The same spiking recording fed to plain DMD, drive and reset ignored ==");
    let naive_cfg = DmdConfig {
        rank: Some(2 * N),
        center: false,
        dt: DT,
        lifting: None,
    };
    let naive = dmd(&rec.states, &naive_cfg)?;
    print_spectrum(&dmd_spectrum(&naive, DT), DT);
    let st = dmd_stability(&naive, 1e-6);
    println!(
        "  stability: spectral radius {:.6}, stable = {}, marginal = {}, unstable = {}",
        st.spectral_radius, st.is_stable, st.is_marginal, st.is_unstable
    );
    let res = dmd_residual(&naive, &rec.states)?;
    let n_pairs = res.per_step_residual.len();
    let naive_rmse = res.residual_norm / ((2 * N * n_pairs) as f64).sqrt();
    let total: f64 = res.per_step_residual.iter().map(|r| r * r).sum();
    let on_spikes: f64 = res
        .per_step_residual
        .iter()
        .zip(&rec.any_spike)
        .filter(|(_, &s)| s)
        .map(|(r, _)| r * r)
        .sum();
    println!(
        "  one-step residual, relative: {:.3}; one-step RMSE {:.3e} (known-B fit above: {:.2e})",
        res.residual_relative, naive_rmse, train_rmse
    );
    println!(
        "  share of the residual energy on the {n_spiking} spiking transitions: {:.1}% (they are {:.1}% of the transitions)",
        100.0 * on_spikes / total,
        100.0 * n_spiking as f64 / n_pairs as f64
    );
    if let Some(d) = &dump {
        // Per-transition residual of the known-B fit on the same steps:
        // ‖A·x_t + B·u_t − x_{t+1}‖.
        let (x1, x2, u) = (snapshots.x1(), snapshots.x2(), snapshots.u());
        let m = snapshots.n_pairs();
        let mut pred = Mat::<f64>::zeros(2 * N, m);
        fit.operator.apply(x1, pred.as_mut(), false);
        let b = fit.b.as_ref();
        let resid_kb: Vec<f64> = (0..m)
            .map(|c| {
                (0..2 * N)
                    .map(|i| {
                        let mut p = pred[(i, c)];
                        for k in 0..b.ncols() {
                            p += b[(i, k)] * u[(k, c)];
                        }
                        (p - x2[(i, c)]).powi(2)
                    })
                    .sum::<f64>()
                    .sqrt()
            })
            .collect();
        // State layout is variable-major: rows 0..N are the voltages.
        let rows: Vec<String> = (0..m)
            .map(|t| {
                let s = rec.spikes[t].as_mat();
                format!(
                    "{},{},{},{},{},{},{},{},{}",
                    (t + 1) as f64 * DT,
                    rec.states[(0, t + 1)],
                    rec.states[(1, t + 1)],
                    rec.states[(2, t + 1)],
                    s[(0, 0)],
                    s[(1, 0)],
                    s[(2, 0)],
                    res.per_step_residual[t],
                    resid_kb[t]
                )
            })
            .collect();
        write_csv(
            d,
            "spiking.csv",
            "t_ms,v0,v1,v2,s0,s1,s2,resid_naive,resid_knownb",
            &rows,
        )?;
        let mut rows: Vec<String> = fit
            .dmdc
            .eigenvalues
            .iter()
            .map(|e| format!("knownb,{},{}", e.re, e.im))
            .collect();
        rows.extend(
            naive
                .eigenvalues
                .iter()
                .map(|e| format!("naive,{},{}", e.re, e.im)),
        );
        write_csv(d, "eigs.csv", "fit,re,im", &rows)?;
        println!("  (CSV dump written to {})", d.display());
    }
    Ok(())
}
