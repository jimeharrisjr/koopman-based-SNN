//! Operator identification: fitting the discrete-time propagator `A` from
//! simulated trajectories with DMD (autonomous, this module) and DMDc
//! (controlled, Phase 3).
//!
//! # What identification is for — and what it is not for
//!
//! For plain LIF the sub-threshold propagator is known in closed form
//! ([`Lif::a_local`](crate::neuron::Lif::a_local)); fitting it from data is a
//! **validation oracle**, not a feature — the permanent integration test
//! demands the fit recover the analytic matrix to ≤ 1e-8. Fitted operators
//! earn their keep on dynamics without closed forms: lifted nonlinear neuron
//! models and reduced-order recurrent layers.
//!
//! # Identifiability: one trajectory is not enough for a network
//!
//! A single trajectory of a linear system explores only the Krylov subspace
//! spanned by its initial condition — for an N-neuron LIF layer that span has
//! dimension ≤ 2 (one α-mode and one β-mode mixture), no matter how large N
//! is. A network-level `A` is therefore **not identifiable from one
//! trajectory**; fit per neuron (the supported path — homogeneous layers
//! share one `A_local`), or use multi-trajectory snapshot sets with DMDc
//! (Phase 3). A regression test documents this trap.
//!
//! # Rules enforced here
//!
//! - `center: false` always — the state mean is regime-dependent, and
//!   centering would bake the input regime into the model.
//! - Fits are gated on stability and realness before an operator is released.
//! - Fit on continuous observables (potentials, currents) — never on binary
//!   spike trains.

pub mod snapshots;
pub mod validate;

use faer::{Mat, MatRef};
use koopman_dmd::{dmd, dmdc, DmdConfig, DmdResult, DmdcConfig, DmdcResult};

use crate::error::SnnError;
use crate::metrics::trajectory_rmse;
use crate::neuron::Lif;
use crate::operator::{dense_from_dmd, Operator};
use crate::util;

pub use snapshots::{
    record_constant_input, record_input_sequence, Recording, SnapshotSet, SnapshotSetBuilder,
};
pub use validate::{
    heldout_one_step_rmse, stability_gate, stability_gate_from_eigenvalues, IdentificationReport,
};

/// How the truncation rank is chosen.
#[derive(Debug, Clone, Copy)]
pub enum RankPolicy {
    /// Use exactly this rank. Preferred: for a k-variable neuron model the
    /// sub-threshold rank is k, and passing it explicitly is both faster and
    /// safer than energy heuristics on spiking data.
    Fixed(usize),
    /// Defer to `koopman-dmd`'s automatic rule (99 % cumulative variance).
    /// Acceptable for smooth sub-threshold data; a poor default for anything
    /// containing spike transients.
    Auto,
}

/// Configuration for identification runs.
#[derive(Debug, Clone)]
pub struct IdentifyConfig {
    pub rank: RankPolicy,
    /// Optional reduced-order output projection for controlled fits
    /// (value case V1): `Some(r)` makes [`fit_controlled`] return an
    /// [`Operator::LowRank`] built from DMDc's rank-r output basis, stepping
    /// in O(N·r) instead of O(N²). `None` keeps the full operator.
    pub rank_output: Option<usize>,
    /// Time step of the recorded trajectories (for spectrum timescales).
    pub dt: f64,
    /// Stability gate: reject fits with spectral radius > 1 + tol.
    pub stability_tol: f64,
    /// Realness gate for operator extraction (relative to max |Re|).
    pub max_imag: f64,
    /// Whether an unstable fit is an error (default) or merely reported.
    ///
    /// Keep `true` for leaky models, whose identified dynamics must decay.
    /// Set `false` when fitting lifted surrogates of *excitable* dynamics:
    /// the regenerative upstroke between spikes is locally expanding, so a
    /// spectral radius above 1 can be legitimate within the inter-spike
    /// validity window. The spectrum is still computed into the report.
    pub enforce_stability: bool,
}

impl Default for IdentifyConfig {
    fn default() -> Self {
        Self {
            rank: RankPolicy::Auto,
            rank_output: None,
            dt: 1.0,
            stability_tol: 1e-6,
            max_imag: 1e-8,
            enforce_stability: true,
        }
    }
}

/// A validated identified operator plus its diagnostics.
#[derive(Debug, Clone)]
pub struct IdentifiedOperator {
    pub operator: Operator,
    pub report: IdentificationReport,
    /// The full DMD result, kept for downstream spectral analysis.
    pub dmd: DmdResult,
}

/// Fit an autonomous linear operator `x_{t+1} = A·x_t` to one contiguous
/// sub-threshold trajectory (`n_state × T` — see the module docs for why a
/// single trajectory identifies a per-neuron block, not a network).
///
/// `heldout`, if given, is a second trajectory used only for the one-step
/// prediction RMSE in the report.
pub fn fit_autonomous(
    trajectory: MatRef<'_, f64>,
    heldout: Option<MatRef<'_, f64>>,
    cfg: &IdentifyConfig,
) -> Result<IdentifiedOperator, SnnError> {
    if !util::is_positive(cfg.dt) {
        return Err(SnnError::InvalidParameter(format!(
            "dt must be positive (dt = {})",
            cfg.dt
        )));
    }
    let dmd_config = DmdConfig {
        rank: match cfg.rank {
            RankPolicy::Fixed(r) => Some(r),
            RankPolicy::Auto => None,
        },
        // Never center: the mean is input-regime-dependent (see module docs).
        center: false,
        dt: cfg.dt,
        lifting: None,
    };
    let res = dmd(&trajectory.to_owned(), &dmd_config)?;

    let stability = if cfg.enforce_stability {
        stability_gate(&res, cfg.stability_tol)?
    } else {
        koopman_dmd::dmd_stability(&res, cfg.stability_tol)
    };
    let operator = dense_from_dmd(&res, cfg.max_imag)?;
    let heldout_rmse = match heldout {
        Some(h) => Some(heldout_one_step_rmse(&operator, h)?),
        None => None,
    };
    let report = validate::build_report(&res, stability, heldout_rmse);
    Ok(IdentifiedOperator {
        operator,
        report,
        dmd: res,
    })
}

/// A validated controlled (DMDc) fit.
#[derive(Debug, Clone)]
pub struct IdentifiedControlled {
    /// The identified state-transition operator `A`.
    pub operator: Operator,
    /// The input matrix `B` — the pinned `known_b` when one was supplied,
    /// otherwise the jointly estimated matrix.
    pub b: Mat<f64>,
    pub report: IdentificationReport,
    /// The full DMDc result, kept for downstream analysis.
    pub dmdc: DmdcResult,
}

/// Fit `x_{t+1} = A·x_t + B·u_t` from a [`SnapshotSet`].
///
/// `known_b`, when given, pins `B` and fits only `A` on the input-subtracted
/// residual — the preferred mode for SNN layers, where
/// `B = [B_input | −θ·E_v]` is known by construction
/// ([`lif_structural_b`]). Joint estimation (`known_b: None`) is only sound
/// when `u` is exogenous and persistently exciting; the layer's own spikes
/// are state feedback, so joint fits on spiking data are a consistency
/// *check*, not a source of truth.
///
/// `heldout`, if given, contributes a one-step prediction RMSE
/// (`‖A·x₁ + B·u − x₂‖`) to the report.
pub fn fit_controlled(
    snapshots: &SnapshotSet,
    known_b: Option<Mat<f64>>,
    heldout: Option<&SnapshotSet>,
    cfg: &IdentifyConfig,
) -> Result<IdentifiedControlled, SnnError> {
    if !util::is_positive(cfg.dt) {
        return Err(SnnError::InvalidParameter(format!(
            "dt must be positive (dt = {})",
            cfg.dt
        )));
    }
    let dmdc_config = DmdcConfig {
        rank_input: match cfg.rank {
            RankPolicy::Fixed(r) => Some(r),
            RankPolicy::Auto => None,
        },
        rank_output: cfg.rank_output,
        dt: cfg.dt,
        known_b,
    };
    let res = dmdc(
        &snapshots.x1().to_owned(),
        &snapshots.x2().to_owned(),
        &snapshots.u().to_owned(),
        &dmdc_config,
    )?;

    let stability = if cfg.enforce_stability {
        stability_gate_from_eigenvalues(&res.eigenvalues, cfg.stability_tol)?
    } else {
        koopman_dmd::stability_from_eigenvalues(&res.eigenvalues, cfg.stability_tol)
    };
    let operator = match cfg.rank_output {
        // Reduced-order (V1): step through the rank-r factors, O(N·r).
        Some(_) => Operator::LowRank {
            p: res.basis.clone(),
            a_r: res.a_tilde.clone(),
        },
        None => Operator::Dense(res.a.clone()),
    };
    let heldout_rmse = match heldout {
        Some(h) => Some(heldout_pair_rmse(&operator, res.b.as_ref(), h)?),
        None => None,
    };
    let report = validate::build_report_controlled(&res, stability, heldout_rmse);
    Ok(IdentifiedControlled {
        operator,
        b: res.b.clone(),
        report,
        dmdc: res,
    })
}

/// One-step prediction RMSE of `(A, B)` on held-out snapshot pairs:
/// `rmse(A·x₁ + B·u, x₂)`.
pub fn heldout_pair_rmse(
    a: &Operator,
    b: MatRef<'_, f64>,
    heldout: &SnapshotSet,
) -> Result<f64, SnnError> {
    let n = heldout.n_state();
    let m = heldout.n_pairs();
    if a.state_dim() != n {
        return Err(SnnError::DimensionMismatch(format!(
            "operator acts on dim {}, held-out pairs have {n}",
            a.state_dim()
        )));
    }
    if b.nrows() != n || b.ncols() != heldout.n_controls() {
        return Err(SnnError::DimensionMismatch(format!(
            "B is {}×{}, expected {n}×{}",
            b.nrows(),
            b.ncols(),
            heldout.n_controls()
        )));
    }
    let mut pred = Mat::<f64>::zeros(n, m);
    a.apply(heldout.x1(), pred.as_mut(), false);
    let u = heldout.u();
    for c in 0..m {
        for i in 0..n {
            let mut acc = 0.0;
            for k in 0..b.ncols() {
                acc += b[(i, k)] * u[(k, c)];
            }
            pred[(i, c)] += acc;
        }
    }
    trajectory_rmse(pred.as_ref(), heldout.x2())
}

/// The structural input matrix of a subtractive-reset LIF layer driven by
/// per-neuron currents, for [`fit_controlled`]'s `known_b`:
///
/// ```text
/// B = [ δ·I_N | −θ·I_N ]      (v rows)      u_t = [ drive_t ; s_t ]
///     [ (1−β)·I_N | 0   ]      (i rows)
/// ```
///
/// where `[δ, 1−β]` is the exact ZOH input column ([`Lif::b_local`]) and the
/// `−θ` block is the subtractive reset written as a control input. Every
/// entry is known by construction — DMDc never needs to estimate it, and a
/// joint fit's estimate of these columns is a consistency check.
pub fn lif_structural_b(lif: &Lif, n_neurons: usize) -> Result<Mat<f64>, SnnError> {
    if n_neurons == 0 {
        return Err(SnnError::InvalidParameter(
            "n_neurons must be nonzero".into(),
        ));
    }
    if lif.params().v_rest != 0.0 {
        return Err(SnnError::InvalidParameter(format!(
            "structural B assumes v_rest = 0 (got {}): a nonzero rest adds an \
             affine term the linear-plus-control model cannot represent — \
             shift coordinates to v − v_rest first",
            lif.params().v_rest
        )));
    }
    let [delta, one_minus_beta] = lif.b_local();
    let theta = lif.params().theta;
    let n = n_neurons;
    let mut b = Mat::<f64>::zeros(2 * n, 2 * n);
    for j in 0..n {
        b[(j, j)] = delta; // drive → v
        b[(n + j, j)] = one_minus_beta; // drive → i
        b[(j, n + j)] = -theta; // self-spike → subtractive reset on v
    }
    Ok(b)
}
