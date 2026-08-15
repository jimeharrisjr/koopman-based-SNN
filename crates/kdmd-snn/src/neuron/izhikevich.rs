//! Izhikevich neuron (Izhikevich 2003, *IEEE Trans. Neural Networks*).
//!
//! ```text
//! dv/dt = 0.04·v² + 5·v + 140 − u + I
//! du/dt = a·(b·v − u)
//! if v ≥ 30 mV:  v ← c,  u ← u + d
//! ```
//!
//! The quadratic term makes the sub-threshold flow **genuinely nonlinear** —
//! no closed-form discrete propagator exists. This is the primary ground
//! truth for Phase 4's EDMD lifted surrogates (value case V2): the dictionary
//! must earn its keep against this simulator's trajectories.
//!
//! Integration follows the paper's practice: forward Euler with sub-steps of
//! the reported `dt` for numerical stability (the paper uses two half-steps
//! per ms). The 30 mV peak is part of the model definition, not a tunable
//! threshold, and the reset is inherently hard (`v ← c`).

use faer::MatRef;

use crate::error::SnnError;
use crate::neuron::{assert_step_dims, NeuronModel};
use crate::spikes::SpikeBatch;
use crate::state::LayerState;
use crate::util;

/// Spike cutoff (mV), fixed by the model definition.
const V_PEAK: f64 = 30.0;

/// Izhikevich parameters. `dt` is in ms; each step integrates with
/// `substeps` forward-Euler sub-steps.
#[derive(Debug, Clone)]
pub struct IzhikevichParams {
    pub a: f64,
    pub b: f64,
    /// Post-spike reset value for `v` (mV).
    pub c: f64,
    /// Post-spike increment for `u`.
    pub d: f64,
    /// Simulation time step (ms).
    pub dt: f64,
    /// Euler sub-steps per `dt` (the paper uses 2 per ms).
    pub substeps: usize,
}

impl IzhikevichParams {
    /// Regular spiking (RS) cortical neuron: a=0.02, b=0.2, c=−65, d=8.
    pub fn regular_spiking(dt: f64) -> Self {
        Self {
            a: 0.02,
            b: 0.2,
            c: -65.0,
            d: 8.0,
            dt,
            substeps: 4,
        }
    }

    /// Intrinsically bursting (IB): c=−55, d=4.
    pub fn intrinsically_bursting(dt: f64) -> Self {
        Self {
            c: -55.0,
            d: 4.0,
            ..Self::regular_spiking(dt)
        }
    }

    /// Chattering (CH): c=−50, d=2.
    pub fn chattering(dt: f64) -> Self {
        Self {
            c: -50.0,
            d: 2.0,
            ..Self::regular_spiking(dt)
        }
    }

    /// Fast spiking (FS) interneuron: a=0.1, d=2.
    pub fn fast_spiking(dt: f64) -> Self {
        Self {
            a: 0.1,
            d: 2.0,
            ..Self::regular_spiking(dt)
        }
    }
}

/// Izhikevich reference simulator.
///
/// State layout: variable 0 = membrane potential `v` (mV), variable 1 =
/// recovery variable `u`.
#[derive(Debug, Clone)]
pub struct Izhikevich {
    p: IzhikevichParams,
}

impl Izhikevich {
    pub fn new(p: IzhikevichParams) -> Result<Self, SnnError> {
        if !util::is_positive(p.dt) {
            return Err(SnnError::InvalidParameter(format!(
                "dt must be positive (dt = {})",
                p.dt
            )));
        }
        if p.substeps == 0 {
            return Err(SnnError::InvalidParameter(
                "substeps must be at least 1".into(),
            ));
        }
        if p.c >= V_PEAK {
            return Err(SnnError::InvalidParameter(format!(
                "reset value c = {} must be below the {V_PEAK} mV peak",
                p.c
            )));
        }
        Ok(Self { p })
    }

    pub fn params(&self) -> &IzhikevichParams {
        &self.p
    }
}

impl NeuronModel for Izhikevich {
    fn n_state_vars(&self) -> usize {
        2
    }

    fn dt(&self) -> f64 {
        self.p.dt
    }

    /// Paper initial condition: v = −65 mV, u = b·v.
    fn init_state(&self, state: &mut LayerState) {
        assert_eq!(state.n_state_vars(), 2, "Izhikevich state has 2 variables");
        state.var_mut(0).fill(-65.0);
        state.var_mut(1).fill(self.p.b * -65.0);
    }

    fn step(&self, state: &mut LayerState, input: MatRef<'_, f64>, spikes: &mut SpikeBatch) {
        assert_step_dims(2, state, input, spikes);
        let n = state.n_neurons();
        let batch = state.batch();
        let p = &self.p;
        let h = p.dt / p.substeps as f64;

        for bcol in 0..batch {
            for j in 0..n {
                let current = input[(j, bcol)];
                let mut v = state.var(0)[(j, bcol)];
                let mut u = state.var(1)[(j, bcol)];
                let mut fired = false;

                for _ in 0..p.substeps {
                    v += h * (0.04 * v * v + 5.0 * v + 140.0 - u + current);
                    u += h * (p.a * (p.b * v - u));
                    if v >= V_PEAK {
                        fired = true;
                        v = p.c;
                        u += p.d;
                    }
                }

                state.var_mut(0)[(j, bcol)] = v;
                state.var_mut(1)[(j, bcol)] = u;
                spikes.as_mat_mut()[(j, bcol)] = if fired { 1.0 } else { 0.0 };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::Mat;

    /// Run one neuron under constant current, returning spike step indices.
    fn run(params: IzhikevichParams, current: f64, t_steps: usize) -> (Vec<usize>, LayerState) {
        let model = Izhikevich::new(params).unwrap();
        let mut state = LayerState::zeros(1, 2, 1).unwrap();
        model.init_state(&mut state);
        let mut input = Mat::<f64>::zeros(1, 1);
        input[(0, 0)] = current;
        let mut spikes = SpikeBatch::zeros(1, 1).unwrap();

        let mut spike_steps = Vec::new();
        for t in 0..t_steps {
            model.step(&mut state, input.as_ref(), &mut spikes);
            assert!(
                state.var(0)[(0, 0)].is_finite(),
                "membrane potential diverged at step {t}"
            );
            if spikes.as_mat()[(0, 0)] == 1.0 {
                spike_steps.push(t);
            }
        }
        (spike_steps, state)
    }

    fn isis(spike_steps: &[usize]) -> Vec<f64> {
        spike_steps
            .windows(2)
            .map(|w| (w[1] - w[0]) as f64)
            .collect()
    }

    #[test]
    fn regular_spiking_is_tonic_under_constant_current() {
        // RS at I = 10 over 1000 ms: sustained, regular firing.
        let (spike_steps, _) = run(IzhikevichParams::regular_spiking(1.0), 10.0, 1000);
        assert!(
            spike_steps.len() >= 4 && spike_steps.len() <= 60,
            "RS spike count out of tonic range: {}",
            spike_steps.len()
        );
        // Tonic regularity: after the initial transient, ISIs are nearly
        // constant (coefficient of variation well under 10 %).
        let isis: Vec<f64> = isis(&spike_steps)[1..].to_vec();
        let mean = isis.iter().sum::<f64>() / isis.len() as f64;
        let var = isis.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / isis.len() as f64;
        let cv = var.sqrt() / mean;
        assert!(cv < 0.1, "RS firing not tonic: ISI CV = {cv:.3}");
    }

    #[test]
    fn chattering_bursts_under_constant_current() {
        // CH at I = 10: groups of closely spaced spikes separated by long
        // gaps — the ISI distribution must contain both short intra-burst and
        // long inter-burst intervals.
        let (spike_steps, _) = run(IzhikevichParams::chattering(1.0), 10.0, 1000);
        assert!(
            spike_steps.len() >= 6,
            "CH spike count too low: {}",
            spike_steps.len()
        );
        let isis = isis(&spike_steps);
        let min_isi = isis.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_isi = isis.iter().cloned().fold(0.0, f64::max);
        assert!(
            min_isi <= 15.0,
            "no intra-burst intervals found (min ISI = {min_isi} ms)"
        );
        assert!(
            max_isi >= 25.0,
            "no inter-burst gaps found (max ISI = {max_isi} ms)"
        );
        assert!(
            max_isi / min_isi >= 3.0,
            "ISI distribution not bimodal (min {min_isi}, max {max_isi})"
        );
    }

    #[test]
    fn fast_spiking_fires_faster_than_regular_spiking() {
        let (rs, _) = run(IzhikevichParams::regular_spiking(1.0), 10.0, 1000);
        let (fs, _) = run(IzhikevichParams::fast_spiking(1.0), 10.0, 1000);
        assert!(
            fs.len() > rs.len(),
            "FS ({}) should out-fire RS ({})",
            fs.len(),
            rs.len()
        );
    }

    #[test]
    fn spike_resets_v_to_c() {
        // With substeps = 1 the reset is the last thing applied in a firing
        // step, so v must sit exactly at c afterwards.
        let params = IzhikevichParams {
            substeps: 1,
            ..IzhikevichParams::regular_spiking(0.5)
        };
        let model = Izhikevich::new(params).unwrap();
        let mut state = LayerState::zeros(1, 2, 1).unwrap();
        model.init_state(&mut state);
        let mut input = Mat::<f64>::zeros(1, 1);
        input[(0, 0)] = 10.0;
        let mut spikes = SpikeBatch::zeros(1, 1).unwrap();

        let mut saw_spike = false;
        for _ in 0..4000 {
            model.step(&mut state, input.as_ref(), &mut spikes);
            if spikes.as_mat()[(0, 0)] == 1.0 {
                saw_spike = true;
                assert_eq!(state.var(0)[(0, 0)], -65.0);
            }
        }
        assert!(saw_spike, "neuron never fired");
    }

    #[test]
    fn quiescent_without_input() {
        let (spike_steps, state) = run(IzhikevichParams::regular_spiking(1.0), 0.0, 1000);
        assert!(spike_steps.is_empty(), "RS fired with no input");
        // Settles near the resting fixed point (≈ −70 mV for RS at I = 0).
        let v = state.var(0)[(0, 0)];
        assert!((-75.0..=-55.0).contains(&v), "unexpected rest v = {v}");
    }

    #[test]
    fn invalid_params_are_rejected() {
        assert!(Izhikevich::new(IzhikevichParams {
            dt: 0.0,
            ..IzhikevichParams::regular_spiking(1.0)
        })
        .is_err());
        assert!(Izhikevich::new(IzhikevichParams {
            substeps: 0,
            ..IzhikevichParams::regular_spiking(1.0)
        })
        .is_err());
        assert!(Izhikevich::new(IzhikevichParams {
            c: 40.0,
            ..IzhikevichParams::regular_spiking(1.0)
        })
        .is_err());
    }
}
