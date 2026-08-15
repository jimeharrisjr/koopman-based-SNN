//! Leaky Integrate-and-Fire neuron with a current-based exponential synapse.
//!
//! Sub-threshold dynamics (per neuron):
//!
//! ```text
//! τ_s · di/dt = −i + u(t)                    (synaptic current, driven by input u)
//! τ_m · dv/dt = −(v − v_rest) + R·i          (membrane potential)
//! ```
//!
//! These are **exactly linear**, and the discrete-time propagator under a
//! zero-order hold on `u` is known in closed form ([`Lif::a_local`]). That
//! matrix is the permanent test oracle for DMD identification: a correct fit
//! on sub-threshold LIF data must recover it to numerical precision, and any
//! fitted operator that beats it is a bug, not a feature.

use faer::{Mat, MatRef};

use crate::error::SnnError;
use crate::neuron::{assert_step_dims, NeuronModel, ResetMode};
use crate::spikes::SpikeBatch;
use crate::state::LayerState;
use crate::util;

/// Relative tolerance below which two time constants are treated as equal and
/// the degenerate closed-form limit is used. Chosen near the crossover where
/// the exact formula's catastrophic cancellation in (α − β) and the limit
/// formula's O(Δτ/τ) truncation are comparable (~1e-7 relative error each;
/// adversarial-test finding, docs/11 — at the previous 1e-9 switch the exact
/// branch lost ~6 digits just outside the window).
const TAU_DEGENERATE_RTOL: f64 = 1e-7;

/// LIF parameters. Time constants and `dt` share one time unit (ms by
/// convention).
#[derive(Debug, Clone)]
pub struct LifParams {
    /// Membrane time constant τ_m.
    pub tau_m: f64,
    /// Synaptic time constant τ_s.
    pub tau_s: f64,
    /// Membrane resistance R (v-units per i-unit).
    pub r: f64,
    /// Resting potential.
    pub v_rest: f64,
    /// Absolute firing threshold θ (must exceed `v_rest`).
    pub theta: f64,
    /// Simulation time step.
    pub dt: f64,
    /// Reset convention.
    pub reset: ResetMode,
}

impl Default for LifParams {
    /// Textbook defaults: τ_m = 10 ms, τ_s = 5 ms, R = 1, rest at 0,
    /// threshold 1, dt = 0.1 ms, subtractive reset.
    fn default() -> Self {
        Self {
            tau_m: 10.0,
            tau_s: 5.0,
            r: 1.0,
            v_rest: 0.0,
            theta: 1.0,
            dt: 0.1,
            reset: ResetMode::Subtractive,
        }
    }
}

/// LIF reference simulator with a precomputed exact (ZOH) discrete propagator.
///
/// State layout: variable 0 = membrane potential `v`, variable 1 = synaptic
/// current `i`.
#[derive(Debug, Clone)]
pub struct Lif {
    p: LifParams,
    /// v ← v_rest + α(v − v_rest) + γ·i + δ·u
    alpha: f64,
    /// i ← β·i + (1 − β)·u
    beta: f64,
    gamma: f64,
    delta: f64,
}

impl Lif {
    pub fn new(p: LifParams) -> Result<Self, SnnError> {
        if !(util::is_positive(p.tau_m) && util::is_positive(p.tau_s) && util::is_positive(p.dt)) {
            return Err(SnnError::InvalidParameter(format!(
                "tau_m, tau_s, dt must be positive (tau_m = {}, tau_s = {}, dt = {})",
                p.tau_m, p.tau_s, p.dt
            )));
        }
        if !util::exceeds(p.theta, p.v_rest) {
            return Err(SnnError::InvalidParameter(format!(
                "threshold must exceed v_rest (theta = {}, v_rest = {})",
                p.theta, p.v_rest
            )));
        }
        if let ResetMode::HardTo(v_reset) = p.reset {
            if v_reset >= p.theta {
                return Err(SnnError::InvalidParameter(format!(
                    "hard reset value {v_reset} must be below threshold {}",
                    p.theta
                )));
            }
        }
        let alpha = (-p.dt / p.tau_m).exp();
        let beta = (-p.dt / p.tau_s).exp();
        let gamma = exp_input_coupling(p.r, p.tau_m, p.tau_s, alpha, beta, p.dt);
        // Response of v to a constant u over one step, u entering through the
        // synapse: total R(1 − α) minus the transient already counted in γ.
        let delta = p.r * (1.0 - alpha) - gamma;
        Ok(Self {
            p,
            alpha,
            beta,
            gamma,
            delta,
        })
    }

    pub fn params(&self) -> &LifParams {
        &self.p
    }

    /// The exact discrete sub-threshold propagator for one neuron,
    ///
    /// ```text
    /// A_local = [ α  γ ]        α = e^(−dt/τ_m),  β = e^(−dt/τ_s),
    ///           [ 0  β ]        γ = R·τ_s·(α − β)/(τ_m − τ_s)
    /// ```
    ///
    /// (γ → R·dt·α/τ_m as τ_s → τ_m). The layer-level operator is
    /// `A_local ⊗ I_N` in variable-major ordering. This is the closed-form
    /// oracle that identified operators are validated against.
    pub fn a_local(&self) -> Mat<f64> {
        let mut a = Mat::zeros(2, 2);
        a[(0, 0)] = self.alpha;
        a[(0, 1)] = self.gamma;
        a[(1, 1)] = self.beta;
        a
    }

    /// The exact input column: contribution of a constant `u` over one step to
    /// `[v, i]`, i.e. `[δ, 1 − β]` with `δ = R(1 − α) − γ`.
    pub fn b_local(&self) -> [f64; 2] {
        [self.delta, 1.0 - self.beta]
    }
}

/// Closed-form coupling of an exponentially decaying input (time constant
/// `tau_in`, decay factor `decay_in` over `dt`) into the membrane equation
/// with gain `r/τ_m`: `r·τ_in·(α − decay_in)/(τ_m − τ_in)`, with the
/// degenerate limit `r·dt·α/τ_m` when the time constants coincide.
pub(crate) fn exp_input_coupling(
    r: f64,
    tau_m: f64,
    tau_in: f64,
    alpha: f64,
    decay_in: f64,
    dt: f64,
) -> f64 {
    if (tau_m - tau_in).abs() <= TAU_DEGENERATE_RTOL * tau_m.max(tau_in) {
        r * dt * alpha / tau_m
    } else {
        r * tau_in * (alpha - decay_in) / (tau_m - tau_in)
    }
}

impl NeuronModel for Lif {
    fn n_state_vars(&self) -> usize {
        2
    }

    fn dt(&self) -> f64 {
        self.p.dt
    }

    fn init_state(&self, state: &mut LayerState) {
        assert_eq!(state.n_state_vars(), 2, "LIF state has 2 variables");
        state.var_mut(0).fill(self.p.v_rest);
        state.var_mut(1).fill(0.0);
    }

    fn step(&self, state: &mut LayerState, input: MatRef<'_, f64>, spikes: &mut SpikeBatch) {
        assert_step_dims(2, state, input, spikes);
        let n = state.n_neurons();
        let batch = state.batch();
        let (p, alpha, beta, gamma, delta) =
            (&self.p, self.alpha, self.beta, self.gamma, self.delta);

        for b in 0..batch {
            for j in 0..n {
                let u = input[(j, b)];
                let v = state.var(0)[(j, b)];
                let i = state.var(1)[(j, b)];

                let mut v_next = p.v_rest + alpha * (v - p.v_rest) + gamma * i + delta * u;
                let i_next = beta * i + (1.0 - beta) * u;

                let fired = v_next >= p.theta;
                if fired {
                    match p.reset {
                        ResetMode::Subtractive => v_next -= p.theta - p.v_rest,
                        ResetMode::HardTo(v_reset) => v_next = v_reset,
                    }
                }
                state.var_mut(0)[(j, b)] = v_next;
                state.var_mut(1)[(j, b)] = i_next;
                spikes.as_mat_mut()[(j, b)] = if fired { 1.0 } else { 0.0 };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn silent_lif(dt: f64) -> Lif {
        // High threshold so nothing fires: pure sub-threshold dynamics.
        Lif::new(LifParams {
            theta: 1e12,
            dt,
            ..LifParams::default()
        })
        .unwrap()
    }

    #[test]
    fn membrane_decay_matches_analytic_solution_exactly() {
        // With zero input and zero synaptic current, v_t − v_rest must equal
        // (v_0 − v_rest)·α^t to machine precision: the propagator is exact,
        // not an integrator approximation.
        let lif = silent_lif(0.1);
        let mut state = LayerState::zeros(1, 2, 1).unwrap();
        lif.init_state(&mut state);
        let v0 = 0.8;
        state.var_mut(0)[(0, 0)] = v0;

        let input = Mat::<f64>::zeros(1, 1);
        let mut spikes = SpikeBatch::zeros(1, 1).unwrap();
        let alpha = (-0.1f64 / 10.0).exp();
        for t in 1..=500 {
            lif.step(&mut state, input.as_ref(), &mut spikes);
            let expected = v0 * alpha.powi(t);
            assert_relative_eq!(state.var(0)[(0, 0)], expected, max_relative = 1e-12);
        }
    }

    #[test]
    fn synaptic_current_decay_matches_analytic_solution_exactly() {
        let lif = silent_lif(0.1);
        let mut state = LayerState::zeros(1, 2, 1).unwrap();
        lif.init_state(&mut state);
        state.var_mut(1)[(0, 0)] = 2.0;

        let input = Mat::<f64>::zeros(1, 1);
        let mut spikes = SpikeBatch::zeros(1, 1).unwrap();
        let beta = (-0.1f64 / 5.0).exp();
        for t in 1..=300 {
            lif.step(&mut state, input.as_ref(), &mut spikes);
            assert_relative_eq!(
                state.var(1)[(0, 0)],
                2.0 * beta.powi(t),
                max_relative = 1e-12
            );
        }
    }

    #[test]
    fn constant_input_steady_state_is_r_times_u() {
        // v(∞) = v_rest + R·u under constant drive.
        let lif = silent_lif(0.1);
        let mut state = LayerState::zeros(1, 2, 1).unwrap();
        lif.init_state(&mut state);
        let mut input = Mat::<f64>::zeros(1, 1);
        input[(0, 0)] = 0.5;
        let mut spikes = SpikeBatch::zeros(1, 1).unwrap();
        for _ in 0..10_000 {
            lif.step(&mut state, input.as_ref(), &mut spikes);
        }
        assert_relative_eq!(state.var(0)[(0, 0)], 0.5, max_relative = 1e-9);
        assert_relative_eq!(state.var(1)[(0, 0)], 0.5, max_relative = 1e-9);
    }

    #[test]
    fn f_i_curve_matches_continuous_time_prediction() {
        // Constant suprathreshold drive with the synapse pre-equilibrated
        // (i_0 = u): the membrane sees constant current R·u, so the
        // continuous-time ISI is T = τ_m · ln(R·u / (R·u − θ)) and the firing
        // rate 1/T. Fine dt + subtractive reset should land within ~2 %.
        let dt = 0.01;
        let u = 2.0;
        let lif = Lif::new(LifParams {
            dt,
            ..LifParams::default()
        })
        .unwrap();
        let mut state = LayerState::zeros(1, 2, 1).unwrap();
        lif.init_state(&mut state);
        state.var_mut(1)[(0, 0)] = u; // pre-equilibrate the synapse

        let mut input = Mat::<f64>::zeros(1, 1);
        input[(0, 0)] = u;
        let mut spikes = SpikeBatch::zeros(1, 1).unwrap();

        let t_steps = 200_000; // 2000 ms
        let mut n_spikes = 0u32;
        for _ in 0..t_steps {
            lif.step(&mut state, input.as_ref(), &mut spikes);
            if spikes.as_mat()[(0, 0)] == 1.0 {
                n_spikes += 1;
            }
        }
        let simulated_rate = n_spikes as f64 / (t_steps as f64 * dt);
        let isi = 10.0 * (2.0f64 / (2.0 - 1.0)).ln(); // τ_m ln(Ru/(Ru−θ))
        let analytic_rate = 1.0 / isi;
        let rel_err = (simulated_rate - analytic_rate).abs() / analytic_rate;
        assert!(
            rel_err < 0.02,
            "f–I mismatch: simulated {simulated_rate:.5}/ms vs analytic {analytic_rate:.5}/ms"
        );
    }

    #[test]
    fn subtractive_reset_preserves_overshoot() {
        let lif = Lif::new(LifParams {
            dt: 1.0,
            ..LifParams::default()
        })
        .unwrap();
        let mut state = LayerState::zeros(1, 2, 1).unwrap();
        lif.init_state(&mut state);
        // Big instantaneous current forces a large overshoot this step.
        state.var_mut(0)[(0, 0)] = 0.99;
        state.var_mut(1)[(0, 0)] = 10.0;

        let input = Mat::<f64>::zeros(1, 1);
        let mut spikes = SpikeBatch::zeros(1, 1).unwrap();
        lif.step(&mut state, input.as_ref(), &mut spikes);
        assert_eq!(spikes.as_mat()[(0, 0)], 1.0);
        // v landed above rest: the overshoot survived the subtraction.
        assert!(state.var(0)[(0, 0)] > 0.0);
    }

    #[test]
    fn hard_reset_discards_overshoot() {
        let lif = Lif::new(LifParams {
            dt: 1.0,
            reset: ResetMode::HardTo(-0.2),
            ..LifParams::default()
        })
        .unwrap();
        let mut state = LayerState::zeros(1, 2, 1).unwrap();
        lif.init_state(&mut state);
        state.var_mut(0)[(0, 0)] = 0.99;
        state.var_mut(1)[(0, 0)] = 10.0;

        let input = Mat::<f64>::zeros(1, 1);
        let mut spikes = SpikeBatch::zeros(1, 1).unwrap();
        lif.step(&mut state, input.as_ref(), &mut spikes);
        assert_eq!(spikes.as_mat()[(0, 0)], 1.0);
        assert_eq!(state.var(0)[(0, 0)], -0.2);
    }

    #[test]
    fn a_local_matches_stepped_dynamics() {
        // One silent step from a probe state must equal A_local·x + affine
        // rest term — i.e. the simulator really applies the closed-form
        // propagator it advertises.
        let lif = silent_lif(0.37); // deliberately awkward dt
        let a = lif.a_local();
        let (v0, i0) = (0.3, 0.7);

        let mut state = LayerState::zeros(1, 2, 1).unwrap();
        lif.init_state(&mut state);
        state.var_mut(0)[(0, 0)] = v0;
        state.var_mut(1)[(0, 0)] = i0;
        let input = Mat::<f64>::zeros(1, 1);
        let mut spikes = SpikeBatch::zeros(1, 1).unwrap();
        lif.step(&mut state, input.as_ref(), &mut spikes);

        assert_relative_eq!(
            state.var(0)[(0, 0)],
            a[(0, 0)] * v0 + a[(0, 1)] * i0,
            max_relative = 1e-14
        );
        assert_relative_eq!(state.var(1)[(0, 0)], a[(1, 1)] * i0, max_relative = 1e-14);
    }

    #[test]
    fn b_local_matches_stepped_input_response() {
        // One step from rest with constant u: state must equal b_local·u.
        let lif = silent_lif(0.37);
        let b = lif.b_local();
        let u = 1.3;

        let mut state = LayerState::zeros(1, 2, 1).unwrap();
        lif.init_state(&mut state);
        let mut input = Mat::<f64>::zeros(1, 1);
        input[(0, 0)] = u;
        let mut spikes = SpikeBatch::zeros(1, 1).unwrap();
        lif.step(&mut state, input.as_ref(), &mut spikes);

        assert_relative_eq!(state.var(0)[(0, 0)], b[0] * u, max_relative = 1e-14);
        assert_relative_eq!(state.var(1)[(0, 0)], b[1] * u, max_relative = 1e-14);
    }

    #[test]
    fn degenerate_time_constants_use_the_limit() {
        let lif = Lif::new(LifParams {
            tau_m: 10.0,
            tau_s: 10.0,
            dt: 0.1,
            ..LifParams::default()
        })
        .unwrap();
        let a = lif.a_local();
        let alpha = (-0.1f64 / 10.0).exp();
        assert_relative_eq!(a[(0, 1)], 0.1 * alpha / 10.0, max_relative = 1e-12);
    }

    #[test]
    fn invalid_params_are_rejected() {
        assert!(Lif::new(LifParams {
            tau_m: 0.0,
            ..LifParams::default()
        })
        .is_err());
        assert!(Lif::new(LifParams {
            theta: -1.0,
            v_rest: 0.0,
            ..LifParams::default()
        })
        .is_err());
        assert!(Lif::new(LifParams {
            reset: ResetMode::HardTo(2.0),
            ..LifParams::default()
        })
        .is_err());
    }
}
