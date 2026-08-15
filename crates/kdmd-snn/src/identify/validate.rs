//! Validation gates for identified operators.
//!
//! Every fitted operator passes through these before it may be used for
//! inference or training:
//! 1. **Stability** — a leaky neural system must identify as decaying;
//!    spectral radius above `1 + tol` means bad data, wrong rank, or reset
//!    contamination, and the fit is rejected outright.
//! 2. **Realness** — enforced during extraction
//!    ([`dense_from_dmd`](crate::operator::dense_from_dmd)).
//! 3. **Held-out residual** — one-step prediction error on data the fit never
//!    saw, reported for downstream thresholds.

use faer::{Mat, MatRef};
use koopman_dmd::{
    dmd_spectrum, spectrum_from_eigenvalues, stability_from_eigenvalues, DmdResult, DmdcResult,
    ModeInfo, StabilityResult, C64,
};

use crate::error::SnnError;
use crate::metrics::trajectory_rmse;
use crate::operator::Operator;

/// Everything a fit reports besides the operator itself.
#[derive(Debug, Clone)]
pub struct IdentificationReport {
    /// Per-mode spectrum (eigenvalue, timescale, stability) from
    /// `koopman_dmd::dmd_spectrum`.
    pub spectrum: Vec<ModeInfo>,
    /// System-level stability from `koopman_dmd::dmd_stability`.
    pub stability: StabilityResult,
    /// Truncation rank used.
    pub rank: usize,
    /// One-step RMSE on held-out data, if provided.
    pub heldout_rmse: Option<f64>,
}

/// Reject fits whose spectral radius exceeds `1 + tol`.
///
/// The identified dynamics of a leaky neuron model must decay; a growing mode
/// is diagnostic of reset-contaminated snapshots (fit on sub-threshold pairs
/// only), a rank that split a conjugate pair, or insufficient data.
pub fn stability_gate(res: &DmdResult, tol: f64) -> Result<StabilityResult, SnnError> {
    stability_gate_from_eigenvalues(&res.eigenvalues, tol)
}

/// [`stability_gate`] over a raw eigenvalue slice (e.g. a `DmdcResult`).
pub fn stability_gate_from_eigenvalues(
    eigenvalues: &[C64],
    tol: f64,
) -> Result<StabilityResult, SnnError> {
    let stability = stability_from_eigenvalues(eigenvalues, tol);
    if stability.spectral_radius > 1.0 + tol {
        return Err(SnnError::NumericalError(format!(
            "identified operator is unstable: spectral radius {:.6} > 1 + {tol:.1e}; \
             likely causes: reset events in the snapshot pairs, split conjugate \
             pair at the rank cut, or too little data",
            stability.spectral_radius
        )));
    }
    Ok(stability)
}

/// One-step prediction RMSE of `op` on a held-out trajectory
/// (`n_state × T`, `T ≥ 2`): apply `op` to columns `0..T−1` and compare with
/// columns `1..T`.
pub fn heldout_one_step_rmse(op: &Operator, heldout: MatRef<'_, f64>) -> Result<f64, SnnError> {
    let (n, t) = (heldout.nrows(), heldout.ncols());
    if n != op.state_dim() {
        return Err(SnnError::DimensionMismatch(format!(
            "held-out data has {n} state rows, operator expects {}",
            op.state_dim()
        )));
    }
    if t < 2 {
        return Err(SnnError::InvalidParameter(
            "held-out trajectory needs at least 2 snapshots".into(),
        ));
    }
    let x1 = heldout.subcols(0, t - 1);
    let x2 = heldout.subcols(1, t - 1);
    let mut pred = Mat::<f64>::zeros(n, t - 1);
    op.apply(x1, pred.as_mut(), false);
    trajectory_rmse(pred.as_ref(), x2)
}

/// Assemble the report for a validated fit.
pub fn build_report(
    res: &DmdResult,
    stability: StabilityResult,
    heldout_rmse: Option<f64>,
) -> IdentificationReport {
    IdentificationReport {
        spectrum: dmd_spectrum(res, res.dt),
        stability,
        rank: res.rank,
        heldout_rmse,
    }
}

/// Assemble the report for a validated DMDc fit (amplitudes are not defined
/// for pair-matrix fits, so per-mode amplitude reads 0).
pub fn build_report_controlled(
    res: &DmdcResult,
    stability: StabilityResult,
    heldout_rmse: Option<f64>,
) -> IdentificationReport {
    IdentificationReport {
        spectrum: spectrum_from_eigenvalues(&res.eigenvalues, None, res.dt),
        stability,
        rank: res.rank_input,
        heldout_rmse,
    }
}
