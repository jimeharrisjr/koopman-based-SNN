//! EDMD lifted linear surrogates for nonlinear neuron models (value case V2).
//!
//! The Izhikevich flow is genuinely nonlinear (quadratic in `v`), so no
//! closed-form discrete propagator exists. This module fits a **lifted linear
//! surrogate**: the augmented state `(v, u, I)` is pushed through a
//! polynomial dictionary and a linear operator is identified on the lifted
//! observables, with a constant control input (`u ≡ 1`) whose `B` column
//! absorbs the affine drift (the dictionary has no constant term).
//!
//! Stepping is **hybrid**, per the architecture's resolution of the
//! lifting-vs-reset incompatibility (skeptic C4): one lifted linear advance
//! per step, then a projection back to the observables `(v, u)`, threshold
//! and reset (`v ← c`, `u ← u + d`) in *original* coordinates, and a re-lift
//! — the standard EDMD-as-nonlinear-predictor form `x_{k+1} = C·K·Ψ(x_k)`.
//! The per-step re-lift is a consistency projection: left free-running, the
//! lifted state drifts off the dictionary manifold along the operator's
//! unstable directions (empirically diverging within ~300 steps). It costs
//! O(d) monomial evaluations, negligible next to the d² mat-vec.
//!
//! Fitting follows the pre-registered protocol (`docs/04-v2-preregistration.md`):
//! several randomized-initial-condition trajectories per constant training
//! current, sub-threshold pairs only (spike-straddling pairs masked), an
//! 80/20 train/holdout split **by whole trajectory**, and lifted observables
//! row-normalized to unit RMS before the regression. The held-out one-step
//! RMSE in the report is computed in that normalized space, which
//! operationalizes the pre-registration's *relative* residual.
//!
//! **Experiment outcome: the pre-registered V2 gates FAILED** (repository
//! docs/05, adversarially audited in docs/06): spike counts and first-spike
//! latency were close, but per-interval timing bias accumulated as phase
//! drift and coincidence sat at the de-phased chance floor. This module is
//! retained as the experimental record and harness support
//! (`examples/v2_rollout.rs`); it is **not** a supported simulation path.

use faer::{Mat, MatRef};
use koopman_dmd::{lift_data, LiftingConfig, LiftingInfo};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::error::SnnError;
use crate::identify::{
    fit_controlled, record_constant_input, IdentificationReport, IdentifyConfig, RankPolicy,
    SnapshotSet,
};
use crate::neuron::{Izhikevich, IzhikevichParams, NeuronModel};
use crate::state::LayerState;
use crate::util;

/// Spike cutoff of the Izhikevich model (mV).
const V_PEAK: f64 = 30.0;

/// Configuration for fitting a lifted surrogate.
#[derive(Debug, Clone)]
pub struct SurrogateConfig {
    /// Total degree of the polynomial-with-cross-terms dictionary over
    /// `(v, u, I)`. Degree 2 gives 9 observables; degree 3 gives 19.
    pub degree: usize,
    /// Rank policy for the lifted DMDc fit. `None` (default) uses the full
    /// lifted dimension plus the constant control — with row-normalized
    /// observables the regression is well-conditioned, and energy heuristics
    /// (which collapsed unnormalized monomial data to rank 1 in testing) are
    /// not to be trusted here.
    pub rank: Option<RankPolicy>,
    /// Steps recorded per trajectory.
    pub t_steps: usize,
    /// Trajectories per training current, with randomized initial conditions
    /// (`v₀ ~ U[−80, −50]`, `u₀ = b·v₀ + U[−2, 2]`).
    pub n_trajectories: usize,
    /// Of those, how many are held out (split by whole trajectory) for the
    /// one-step residual precondition.
    pub n_holdout: usize,
    /// Seed for the initial-condition draws (the experiment is deterministic
    /// given this seed).
    pub seed: u64,
}

impl Default for SurrogateConfig {
    fn default() -> Self {
        Self {
            degree: 2,
            rank: None,
            t_steps: 1000,
            n_trajectories: 10,
            n_holdout: 2,
            seed: 42,
        }
    }
}

/// A fitted lifted linear surrogate of one Izhikevich neuron.
#[derive(Debug, Clone)]
pub struct IzhikevichSurrogate {
    lifting: LiftingConfig,
    /// Lifted propagator (d × d).
    a: Mat<f64>,
    /// Constant-drift column (d × 1), identified as the `B` of a `u ≡ 1`
    /// control.
    b_const: Mat<f64>,
    info: LiftingInfo,
    /// Lifted rows of the original observables `v`, `u`, and `I`.
    iv: usize,
    iu: usize,
    ii: usize,
    params: IzhikevichParams,
}

/// One surrogate rollout.
#[derive(Debug, Clone)]
pub struct SurrogateRollout {
    /// `2 × (t_steps + 1)`: row 0 = v, row 1 = u; column 0 is the initial
    /// state. Stored values are post-reset.
    pub states: Mat<f64>,
    /// Pre-reset membrane readout after each step's lifted advance
    /// (`t_steps` entries). On a spiking step this holds the above-threshold
    /// value, enabling linearly interpolated crossing times as required by
    /// the pre-registration's spike-time definition.
    pub pre_reset_v: Vec<f64>,
    /// Spike flag per step.
    pub spikes: Vec<bool>,
}

impl IzhikevichSurrogate {
    /// Fit a surrogate to `model`'s sub-threshold dynamics across
    /// `train_currents`, following the pre-registered protocol. Returns the
    /// surrogate and the identification report (whose `heldout_rmse` is the
    /// normalized-space one-step residual).
    pub fn fit(
        model: &Izhikevich,
        train_currents: &[f64],
        cfg: &SurrogateConfig,
    ) -> Result<(Self, IdentificationReport), SnnError> {
        if train_currents.is_empty() {
            return Err(SnnError::InvalidParameter(
                "at least one training current is required".into(),
            ));
        }
        if cfg.degree < 2 {
            return Err(SnnError::InvalidParameter(format!(
                "dictionary degree must be ≥ 2 (got {})",
                cfg.degree
            )));
        }
        if cfg.t_steps < 3 {
            return Err(SnnError::InvalidParameter("t_steps must be ≥ 3".into()));
        }
        if cfg.n_trajectories == 0 || cfg.n_holdout >= cfg.n_trajectories {
            return Err(SnnError::InvalidParameter(format!(
                "need n_holdout < n_trajectories (got {} / {})",
                cfg.n_holdout, cfg.n_trajectories
            )));
        }
        for &i_inj in train_currents {
            if !i_inj.is_finite() {
                return Err(SnnError::InvalidParameter(format!(
                    "training current {i_inj} is not finite"
                )));
            }
        }

        // Collect masked sub-threshold pairs of the augmented state (v, u, I),
        // with a constant `1` control whose B column will hold the drift.
        // Split by whole trajectory: the last n_holdout per current are
        // held out.
        let mut rng = StdRng::seed_from_u64(cfg.seed);
        let mut train_builder = SnapshotSet::builder(3, 1);
        let mut heldout_builder = SnapshotSet::builder(3, 1);
        for &i_inj in train_currents {
            for r in 0..cfg.n_trajectories {
                let v0 = rng.random_range(-80.0..-50.0);
                let u0 = model.params().b * v0 + rng.random_range(-2.0..2.0);
                let mut state = LayerState::zeros(1, 2, 1)?;
                state.var_mut(0)[(0, 0)] = v0;
                state.var_mut(1)[(0, 0)] = u0;
                let input = crate::encoding::constant_current(1, 1, i_inj)?;
                let rec = record_constant_input(model, &mut state, input.as_ref(), cfg.t_steps)?;

                let t = rec.states.ncols();
                let mut augmented = Mat::<f64>::zeros(3, t);
                for c in 0..t {
                    augmented[(0, c)] = rec.states[(0, c)];
                    augmented[(1, c)] = rec.states[(1, c)];
                    augmented[(2, c)] = i_inj;
                }
                let ones = Mat::<f64>::from_fn(1, t - 1, |_, _| 1.0);
                let builder = if r < cfg.n_trajectories - cfg.n_holdout {
                    &mut train_builder
                } else {
                    &mut heldout_builder
                };
                builder.push_trajectory_masked(
                    augmented.as_ref(),
                    Some(ones.as_ref()),
                    &rec.subthreshold_mask(),
                )?;
            }
        }
        let raw = train_builder.build()?;
        let heldout_raw = if cfg.n_holdout > 0 {
            Some(heldout_builder.build()?)
        } else {
            None
        };

        let lifting = LiftingConfig::PolynomialCross { degree: cfg.degree };
        let (lifted, info) = raw.lifted(&lifting)?;
        let d = info.n_vars_lifted;

        // Row-normalize the lifted observables to unit RMS before the
        // regression: monomial rows differ by orders of magnitude (v² ~ 4·10³
        // vs u ~ 3), and on unnormalized data the energy-based rank rule
        // collapses the fit. The same diagonal scaling D (from the TRAIN
        // split) applies everywhere, so the fitted operator transforms back
        // by similarity (A = D·Ã·D⁻¹, b = D·b̃) and its spectrum is
        // unchanged.
        let m = lifted.n_pairs();
        let mut scales = vec![1.0f64; d];
        for (i, s) in scales.iter_mut().enumerate() {
            let mut sum_sq = 0.0;
            for c in 0..m {
                sum_sq += lifted.x1()[(i, c)] * lifted.x1()[(i, c)];
            }
            let rms = (sum_sq / m as f64).sqrt();
            if rms > 1e-300 {
                *s = rms;
            }
        }
        let scale_rows = |src: MatRef<'_, f64>| {
            Mat::from_fn(src.nrows(), src.ncols(), |i, c| src[(i, c)] / scales[i])
        };
        let scaled = SnapshotSet::from_parts(
            scale_rows(lifted.x1()),
            scale_rows(lifted.x2()),
            lifted.u().to_owned(),
        )?;
        let scaled_heldout = match &heldout_raw {
            Some(h) => {
                let (lifted_h, _) = h.lifted(&lifting)?;
                Some(SnapshotSet::from_parts(
                    scale_rows(lifted_h.x1()),
                    scale_rows(lifted_h.x2()),
                    lifted_h.u().to_owned(),
                )?)
            }
            None => None,
        };

        let fit_cfg = IdentifyConfig {
            // Full lifted rank (+1 for the constant control) unless overridden.
            rank: cfg.rank.unwrap_or(RankPolicy::Fixed(d + 1)),
            dt: model.dt(),
            // Excitable dynamics: transient growth between spikes is
            // legitimate; report the spectrum instead of gating on it (the
            // pre-registration applies its own ρ(K) precondition).
            enforce_stability: false,
            ..IdentifyConfig::default()
        };
        let fit = fit_controlled(&scaled, None, scaled_heldout.as_ref(), &fit_cfg)?;

        // Undo the scaling: A = D·Ã·D⁻¹, b = D·b̃.
        let a = Mat::from_fn(d, d, |i, j| fit.dmdc.a[(i, j)] * scales[i] / scales[j]);
        let b_const = Mat::from_fn(d, 1, |i, _| fit.dmdc.b[(i, 0)] * scales[i]);

        let iv = info.observables[0];
        let iu = info.observables[1];
        let ii = info.observables[2];
        Ok((
            Self {
                lifting,
                a,
                b_const,
                info,
                iv,
                iu,
                ii,
                params: model.params().clone(),
            },
            fit.report,
        ))
    }

    /// Dimension of the lifted state.
    pub fn lifted_dim(&self) -> usize {
        self.info.n_vars_lifted
    }

    /// The lifted propagator (d × d), for precondition checks and spectral
    /// analysis.
    pub fn a(&self) -> MatRef<'_, f64> {
        self.a.as_ref()
    }

    /// The constant-drift column (d × 1).
    pub fn b_const(&self) -> MatRef<'_, f64> {
        self.b_const.as_ref()
    }

    /// Lifted rows holding the original observables `(v, u, I)`.
    pub fn observable_rows(&self) -> (usize, usize, usize) {
        (self.iv, self.iu, self.ii)
    }

    /// Deviation of the fitted `I` dynamics from the identity: `I` is
    /// constant per trajectory, so row `i_I` of `A` should be the unit vector
    /// onto itself with zero drift. Returns
    /// `max(|A[i_I, ·] − e_{i_I}|, |b[i_I]|)` — one of the pre-registered
    /// fit-hygiene checks.
    pub fn i_row_deviation(&self) -> f64 {
        let d = self.lifted_dim();
        let mut worst = self.b_const[(self.ii, 0)].abs();
        for j in 0..d {
            let expected = if j == self.ii { 1.0 } else { 0.0 };
            worst = worst.max((self.a[(self.ii, j)] - expected).abs());
        }
        worst
    }

    /// Lift one raw `(v, u, I)` state into observable space.
    fn lift_state(&self, v: f64, u: f64, i_inj: f64) -> Result<Mat<f64>, SnnError> {
        let mut raw = Mat::<f64>::zeros(3, 1);
        raw[(0, 0)] = v;
        raw[(1, 0)] = u;
        raw[(2, 0)] = i_inj;
        let (lifted, _) = lift_data(&raw, &self.lifting)?;
        Ok(lifted)
    }

    /// Roll the surrogate forward `t_steps` from `(v0, u0)` under constant
    /// current `i_inj`.
    ///
    /// Hybrid stepping: one lifted linear advance per step, observable
    /// readout, threshold/reset in original coordinates, re-lift.
    pub fn rollout_from(
        &self,
        v0: f64,
        u0: f64,
        i_inj: f64,
        t_steps: usize,
    ) -> Result<SurrogateRollout, SnnError> {
        if !util::all_finite(&[v0, u0, i_inj]) {
            return Err(SnnError::InvalidParameter(format!(
                "initial state ({v0}, {u0}) or current {i_inj} is not finite"
            )));
        }
        if t_steps == 0 {
            return Err(SnnError::InvalidParameter(
                "t_steps must be positive".into(),
            ));
        }
        let d = self.lifted_dim();
        let mut z = self.lift_state(v0, u0, i_inj)?;

        let mut states = Mat::<f64>::zeros(2, t_steps + 1);
        states[(0, 0)] = v0;
        states[(1, 0)] = u0;
        let mut pre_reset_v = Vec::with_capacity(t_steps);
        let mut spikes = Vec::with_capacity(t_steps);
        let mut z_next = Mat::<f64>::zeros(d, 1);

        for t in 0..t_steps {
            // z' = A z + b_const.
            for i in 0..d {
                let mut acc = self.b_const[(i, 0)];
                for j in 0..d {
                    acc += self.a[(i, j)] * z[(j, 0)];
                }
                z_next[(i, 0)] = acc;
            }
            let mut v = z_next[(self.iv, 0)];
            let mut u = z_next[(self.iu, 0)];
            if !util::all_finite(&[v, u]) {
                return Err(SnnError::NumericalError(format!(
                    "surrogate rollout diverged at step {t} (v = {v}, u = {u})"
                )));
            }
            pre_reset_v.push(v);
            let fired = v >= V_PEAK;
            if fired {
                v = self.params.c;
                u += self.params.d;
            }
            // Consistency projection: re-lift from the observable readout
            // every step (the standard EDMD-as-nonlinear-predictor form,
            // x_{k+1} = C·K·Ψ(x_k)). Leaving the lifted state free-running
            // lets off-manifold components grow along the operator's
            // unstable directions — empirically the rollout diverged within
            // ~300 steps. The projection costs O(d) monomial evaluations,
            // negligible next to the d² mat-vec.
            z = self.lift_state(v, u, i_inj)?;
            states[(0, t + 1)] = v;
            states[(1, t + 1)] = u;
            spikes.push(fired);
        }
        Ok(SurrogateRollout {
            states,
            pre_reset_v,
            spikes,
        })
    }

    /// [`rollout_from`](Self::rollout_from) starting at the model's standard
    /// initial condition (`v = −65`, `u = b·v`).
    pub fn rollout(&self, i_inj: f64, t_steps: usize) -> Result<SurrogateRollout, SnnError> {
        let v0 = -65.0;
        self.rollout_from(v0, self.params.b * v0, i_inj, t_steps)
    }

    /// Arithmetic cost of one surrogate step, in multiply-adds (`d² + d` for
    /// the affine lifted advance) — the basis of the pre-registered cost
    /// comparison against Euler sub-stepping.
    pub fn step_flops(&self) -> usize {
        let d = self.lifted_dim();
        d * d + d
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spikes::SpikeBatch;

    fn quick_cfg() -> SurrogateConfig {
        // Smaller than the pre-registered protocol: unit tests check the
        // machinery, not the science (that's examples/v2_rollout.rs).
        SurrogateConfig {
            t_steps: 800,
            n_trajectories: 4,
            n_holdout: 1,
            ..SurrogateConfig::default()
        }
    }

    fn reference_spike_steps(model: &Izhikevich, i_inj: f64, t_steps: usize) -> Vec<usize> {
        let mut state = LayerState::zeros(1, 2, 1).unwrap();
        model.init_state(&mut state);
        let input = crate::encoding::constant_current(1, 1, i_inj).unwrap();
        let mut spikes = SpikeBatch::zeros(1, 1).unwrap();
        let mut out = Vec::new();
        for t in 0..t_steps {
            model.step(&mut state, input.as_ref(), &mut spikes);
            if spikes.as_mat()[(0, 0)] == 1.0 {
                out.push(t);
            }
        }
        out
    }

    #[test]
    fn fit_and_rollout_produce_plausible_regular_spiking() {
        // Infrastructure sanity — NOT the pre-registered V2 gate (that runs
        // in examples/v2_rollout.rs with the numbers from
        // docs/04-v2-preregistration.md). Here: the surrogate must run, stay
        // finite, and fire within a loose factor of the reference.
        let model = Izhikevich::new(IzhikevichParams::regular_spiking(1.0)).unwrap();
        let (surrogate, report) =
            IzhikevichSurrogate::fit(&model, &[6.0, 8.0, 12.0, 14.0], &quick_cfg()).unwrap();
        assert!(report.rank > 0);
        assert!(
            report.heldout_rmse.is_some(),
            "holdout split must produce a residual"
        );

        let horizon = 1000;
        let rollout = surrogate.rollout(10.0, horizon).unwrap();
        let n_surrogate = rollout.spikes.iter().filter(|&&s| s).count();
        let n_reference = reference_spike_steps(&model, 10.0, horizon).len();
        assert!(
            n_reference > 0,
            "reference must spike at I = 10 for this test to mean anything"
        );
        assert!(
            n_surrogate > 0,
            "surrogate never fired at a supra-threshold current"
        );
        let ratio = n_surrogate as f64 / n_reference as f64;
        assert!(
            (0.2..=5.0).contains(&ratio),
            "surrogate spike count {n_surrogate} vs reference {n_reference} \
             is outside the loose sanity band"
        );
        // Pre-reset readout exceeds the peak exactly on spiking steps.
        for (t, &fired) in rollout.spikes.iter().enumerate() {
            assert_eq!(rollout.pre_reset_v[t] >= 30.0, fired);
        }
    }

    #[test]
    fn quiescent_current_stays_quiescent() {
        let model = Izhikevich::new(IzhikevichParams::regular_spiking(1.0)).unwrap();
        let (surrogate, _) =
            IzhikevichSurrogate::fit(&model, &[0.0, 2.0, 6.0, 10.0], &quick_cfg()).unwrap();
        let rollout = surrogate.rollout(0.0, 500).unwrap();
        let n_spikes = rollout.spikes.iter().filter(|&&s| s).count();
        assert_eq!(n_spikes, 0, "surrogate fired with no input current");
        // And v stays in a physiological band rather than drifting away.
        for t in 0..=500 {
            let v = rollout.states[(0, t)];
            assert!(
                (-90.0..=-40.0).contains(&v),
                "quiescent v drifted to {v} at step {t}"
            );
        }
    }

    #[test]
    fn i_row_is_close_to_identity() {
        // I is constant per trajectory; the fit should discover I' = I. One
        // of the pre-registered fit-hygiene checks.
        let model = Izhikevich::new(IzhikevichParams::regular_spiking(1.0)).unwrap();
        let (surrogate, _) =
            IzhikevichSurrogate::fit(&model, &[6.0, 8.0, 12.0], &quick_cfg()).unwrap();
        assert!(
            surrogate.i_row_deviation() < 1e-6,
            "I-row deviates from identity by {:.2e}",
            surrogate.i_row_deviation()
        );
    }

    #[test]
    fn config_validation_rejects_bad_inputs() {
        let model = Izhikevich::new(IzhikevichParams::regular_spiking(1.0)).unwrap();
        assert!(IzhikevichSurrogate::fit(&model, &[], &quick_cfg()).is_err());
        assert!(IzhikevichSurrogate::fit(
            &model,
            &[10.0],
            &SurrogateConfig {
                degree: 1,
                ..quick_cfg()
            }
        )
        .is_err());
        assert!(IzhikevichSurrogate::fit(
            &model,
            &[10.0],
            &SurrogateConfig {
                n_trajectories: 2,
                n_holdout: 2,
                ..quick_cfg()
            }
        )
        .is_err());
        let (surrogate, _) =
            IzhikevichSurrogate::fit(&model, &[8.0, 10.0, 12.0], &quick_cfg()).unwrap();
        assert!(surrogate.rollout(f64::NAN, 10).is_err());
        assert!(surrogate.rollout(10.0, 0).is_err());
    }

    #[test]
    fn lifted_dim_matches_dictionary_size() {
        // PolynomialCross degree 2 over 3 variables: monomials of total
        // degree 1..=2 → 3 + 6 = 9.
        let model = Izhikevich::new(IzhikevichParams::regular_spiking(1.0)).unwrap();
        let (surrogate, _) =
            IzhikevichSurrogate::fit(&model, &[8.0, 10.0, 12.0], &quick_cfg()).unwrap();
        assert_eq!(surrogate.lifted_dim(), 9);
        assert_eq!(surrogate.step_flops(), 90);
        let (iv, iu, ii) = surrogate.observable_rows();
        assert!(iv != iu && iu != ii && iv != ii);
    }
}
