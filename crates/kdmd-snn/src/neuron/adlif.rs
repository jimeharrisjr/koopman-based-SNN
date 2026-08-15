//! Adaptive Leaky Integrate-and-Fire neuron (3-state).
//!
//! LIF plus a spike-triggered adaptation current `w`:
//!
//! ```text
//! τ_s · di/dt = −i + u(t)
//! τ_w · dw/dt = −w                       (w ← w + b_jump at each spike)
//! τ_m · dv/dt = −(v − v_rest) + R·i − R·w
//! ```
//!
//! Sub-threshold this is still exactly linear; the nonlinearity enters only
//! through the spike-triggered jump in `w` (which, like the reset, is linear
//! in the spike indicator — another known column of `B`). With `b_jump = 0`
//! the model reduces exactly to [`Lif`](crate::neuron::Lif), which the tests
//! assert.

use faer::{Mat, MatRef};

use crate::error::SnnError;
use crate::neuron::lif::exp_input_coupling;
use crate::neuron::{assert_step_dims, NeuronModel, ResetMode};
use crate::spikes::SpikeBatch;
use crate::state::LayerState;
use crate::util;

/// Adaptive LIF parameters. Time constants and `dt` share one time unit.
#[derive(Debug, Clone)]
pub struct AdLifParams {
    pub tau_m: f64,
    pub tau_s: f64,
    /// Adaptation time constant τ_w (typically ≫ τ_m).
    pub tau_w: f64,
    pub r: f64,
    pub v_rest: f64,
    pub theta: f64,
    /// Adaptation increment applied to `w` at each spike.
    pub b_jump: f64,
    pub dt: f64,
    pub reset: ResetMode,
}

impl Default for AdLifParams {
    /// LIF defaults plus τ_w = 100 ms, b_jump = 0.1.
    fn default() -> Self {
        Self {
            tau_m: 10.0,
            tau_s: 5.0,
            tau_w: 100.0,
            r: 1.0,
            v_rest: 0.0,
            theta: 1.0,
            b_jump: 0.1,
            dt: 0.1,
            reset: ResetMode::Subtractive,
        }
    }
}

/// Adaptive LIF reference simulator with an exact (ZOH) discrete propagator.
///
/// State layout: variable 0 = membrane potential `v`, variable 1 = synaptic
/// current `i`, variable 2 = adaptation current `w`.
#[derive(Debug, Clone)]
pub struct AdLif {
    p: AdLifParams,
    alpha: f64,
    beta: f64,
    /// Decay of `w` over one step.
    rho: f64,
    /// Coupling of `i` into `v` over one step.
    gamma_s: f64,
    /// Coupling of `w` into `v` over one step (negative: w opposes drive).
    gamma_w: f64,
    /// Coupling of constant `u` into `v` over one step.
    delta: f64,
}

impl AdLif {
    pub fn new(p: AdLifParams) -> Result<Self, SnnError> {
        if !(util::is_positive(p.tau_m)
            && util::is_positive(p.tau_s)
            && util::is_positive(p.tau_w)
            && util::is_positive(p.dt))
        {
            return Err(SnnError::InvalidParameter(format!(
                "tau_m, tau_s, tau_w, dt must be positive \
                 (tau_m = {}, tau_s = {}, tau_w = {}, dt = {})",
                p.tau_m, p.tau_s, p.tau_w, p.dt
            )));
        }
        if !util::exceeds(p.theta, p.v_rest) {
            return Err(SnnError::InvalidParameter(format!(
                "threshold must exceed v_rest (theta = {}, v_rest = {})",
                p.theta, p.v_rest
            )));
        }
        if p.b_jump < 0.0 {
            return Err(SnnError::InvalidParameter(format!(
                "adaptation increment must be non-negative (b_jump = {})",
                p.b_jump
            )));
        }
        let alpha = (-p.dt / p.tau_m).exp();
        let beta = (-p.dt / p.tau_s).exp();
        let rho = (-p.dt / p.tau_w).exp();
        let gamma_s = exp_input_coupling(p.r, p.tau_m, p.tau_s, alpha, beta, p.dt);
        let gamma_w = exp_input_coupling(-p.r, p.tau_m, p.tau_w, alpha, rho, p.dt);
        let delta = p.r * (1.0 - alpha) - gamma_s;
        Ok(Self {
            p,
            alpha,
            beta,
            rho,
            gamma_s,
            gamma_w,
            delta,
        })
    }

    pub fn params(&self) -> &AdLifParams {
        &self.p
    }

    /// The exact discrete sub-threshold propagator for one neuron, state order
    /// `(v, i, w)`:
    ///
    /// ```text
    /// A_local = [ α  γ_s  γ_w ]
    ///           [ 0  β    0   ]
    ///           [ 0  0    ρ   ]
    /// ```
    pub fn a_local(&self) -> Mat<f64> {
        let mut a = Mat::zeros(3, 3);
        a[(0, 0)] = self.alpha;
        a[(0, 1)] = self.gamma_s;
        a[(0, 2)] = self.gamma_w;
        a[(1, 1)] = self.beta;
        a[(2, 2)] = self.rho;
        a
    }

    /// The exact input column `[δ, 1 − β, 0]`.
    pub fn b_local(&self) -> [f64; 3] {
        [self.delta, 1.0 - self.beta, 0.0]
    }
}

impl NeuronModel for AdLif {
    fn n_state_vars(&self) -> usize {
        3
    }

    fn dt(&self) -> f64 {
        self.p.dt
    }

    fn init_state(&self, state: &mut LayerState) {
        assert_eq!(state.n_state_vars(), 3, "adLIF state has 3 variables");
        state.var_mut(0).fill(self.p.v_rest);
        state.var_mut(1).fill(0.0);
        state.var_mut(2).fill(0.0);
    }

    fn step(&self, state: &mut LayerState, input: MatRef<'_, f64>, spikes: &mut SpikeBatch) {
        assert_step_dims(3, state, input, spikes);
        let n = state.n_neurons();
        let batch = state.batch();
        let p = &self.p;

        for b in 0..batch {
            for j in 0..n {
                let u = input[(j, b)];
                let v = state.var(0)[(j, b)];
                let i = state.var(1)[(j, b)];
                let w = state.var(2)[(j, b)];

                let mut v_next = p.v_rest
                    + self.alpha * (v - p.v_rest)
                    + self.gamma_s * i
                    + self.gamma_w * w
                    + self.delta * u;
                let i_next = self.beta * i + (1.0 - self.beta) * u;
                let mut w_next = self.rho * w;

                let fired = v_next >= p.theta;
                if fired {
                    match p.reset {
                        ResetMode::Subtractive => v_next -= p.theta - p.v_rest,
                        ResetMode::HardTo(v_reset) => v_next = v_reset,
                    }
                    w_next += p.b_jump;
                }
                state.var_mut(0)[(j, b)] = v_next;
                state.var_mut(1)[(j, b)] = i_next;
                state.var_mut(2)[(j, b)] = w_next;
                spikes.as_mat_mut()[(j, b)] = if fired { 1.0 } else { 0.0 };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neuron::{Lif, LifParams};
    use approx::assert_relative_eq;

    #[test]
    fn zero_adaptation_reduces_exactly_to_lif() {
        let dt = 0.1;
        let adlif = AdLif::new(AdLifParams {
            b_jump: 0.0,
            dt,
            ..AdLifParams::default()
        })
        .unwrap();
        let lif = Lif::new(LifParams {
            dt,
            ..LifParams::default()
        })
        .unwrap();

        let mut s3 = LayerState::zeros(2, 3, 1).unwrap();
        let mut s2 = LayerState::zeros(2, 2, 1).unwrap();
        adlif.init_state(&mut s3);
        lif.init_state(&mut s2);

        let mut input = Mat::<f64>::zeros(2, 1);
        input[(0, 0)] = 2.0; // suprathreshold: exercises the reset path too
        input[(1, 0)] = 0.4;
        let mut sp3 = SpikeBatch::zeros(2, 1).unwrap();
        let mut sp2 = SpikeBatch::zeros(2, 1).unwrap();

        for _ in 0..5_000 {
            adlif.step(&mut s3, input.as_ref(), &mut sp3);
            lif.step(&mut s2, input.as_ref(), &mut sp2);
            for j in 0..2 {
                assert_relative_eq!(s3.var(0)[(j, 0)], s2.var(0)[(j, 0)], max_relative = 1e-12);
                assert_relative_eq!(s3.var(1)[(j, 0)], s2.var(1)[(j, 0)], max_relative = 1e-12);
                assert_eq!(sp3.as_mat()[(j, 0)], sp2.as_mat()[(j, 0)]);
            }
        }
    }

    #[test]
    fn adaptation_slows_firing_over_time() {
        // Constant suprathreshold drive: with spike-triggered adaptation the
        // inter-spike intervals must lengthen from the first ISI to the
        // steady state.
        let adlif = AdLif::new(AdLifParams {
            b_jump: 0.2,
            dt: 0.1,
            ..AdLifParams::default()
        })
        .unwrap();
        let mut state = LayerState::zeros(1, 3, 1).unwrap();
        adlif.init_state(&mut state);
        let mut input = Mat::<f64>::zeros(1, 1);
        input[(0, 0)] = 3.0;
        let mut spikes = SpikeBatch::zeros(1, 1).unwrap();

        let mut spike_steps = Vec::new();
        for t in 0..50_000 {
            adlif.step(&mut state, input.as_ref(), &mut spikes);
            if spikes.as_mat()[(0, 0)] == 1.0 {
                spike_steps.push(t);
            }
        }
        assert!(
            spike_steps.len() >= 5,
            "expected sustained firing, got {} spikes",
            spike_steps.len()
        );
        let first_isi = spike_steps[1] - spike_steps[0];
        let n = spike_steps.len();
        let last_isi = spike_steps[n - 1] - spike_steps[n - 2];
        assert!(
            last_isi > first_isi,
            "adaptation should lengthen ISIs: first {first_isi} steps, last {last_isi} steps"
        );
        // And the adaptation current must be positive after spiking.
        assert!(state.var(2)[(0, 0)] > 0.0);
    }

    #[test]
    fn a_local_matches_stepped_dynamics() {
        let adlif = AdLif::new(AdLifParams {
            theta: 1e12, // silent
            dt: 0.37,
            ..AdLifParams::default()
        })
        .unwrap();
        let a = adlif.a_local();
        let (v0, i0, w0) = (0.3, 0.7, 0.5);

        let mut state = LayerState::zeros(1, 3, 1).unwrap();
        adlif.init_state(&mut state);
        state.var_mut(0)[(0, 0)] = v0;
        state.var_mut(1)[(0, 0)] = i0;
        state.var_mut(2)[(0, 0)] = w0;
        let input = Mat::<f64>::zeros(1, 1);
        let mut spikes = SpikeBatch::zeros(1, 1).unwrap();
        adlif.step(&mut state, input.as_ref(), &mut spikes);

        assert_relative_eq!(
            state.var(0)[(0, 0)],
            a[(0, 0)] * v0 + a[(0, 1)] * i0 + a[(0, 2)] * w0,
            max_relative = 1e-14
        );
        assert_relative_eq!(state.var(1)[(0, 0)], a[(1, 1)] * i0, max_relative = 1e-14);
        assert_relative_eq!(state.var(2)[(0, 0)], a[(2, 2)] * w0, max_relative = 1e-14);
    }

    #[test]
    fn adaptation_opposes_drive() {
        // γ_w must be negative: a positive w pulls v down.
        let adlif = AdLif::new(AdLifParams::default()).unwrap();
        assert!(adlif.a_local()[(0, 2)] < 0.0);
    }
}
