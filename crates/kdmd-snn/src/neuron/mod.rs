//! Ground-truth neuron models and their reference simulators.
//!
//! These simulators are the source of truth for the library: they generate
//! the trajectories that DMD/DMDc identification fits on, and they are the
//! baseline every fast path is validated (and benchmarked) against. They are
//! written for correctness and clarity, not speed — the hot paths live in the
//! operator/layer machinery.

pub mod adlif;
pub mod izhikevich;
pub mod lif;

use faer::MatRef;

use crate::spikes::SpikeBatch;
use crate::state::LayerState;

pub use adlif::{AdLif, AdLifParams};
pub use izhikevich::{Izhikevich, IzhikevichParams};
pub use lif::{lif_entry_grads, Lif, LifParams};

/// What happens to the membrane potential when a neuron fires.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ResetMode {
    /// Subtract the threshold-to-rest distance: `v ← v − (θ − v_rest)`.
    ///
    /// A neuron exactly at threshold lands at `v_rest`; overshoot is
    /// preserved. This is the canonical mode for the fast Koopman path: the
    /// reset is linear in the spike indicator, so the full step is exactly
    /// `x_{t+1} = A·x_t + B·u_t`.
    Subtractive,
    /// Assign a fixed value: `v ← v_reset`. Overshoot is discarded.
    ///
    /// Closer to the stereotyped repolarization of a biological action
    /// potential, but bilinear in (state, spike) — supported in the reference
    /// simulators only, with masked identification.
    HardTo(f64),
}

/// A ground-truth neuron model: sub-threshold dynamics advanced one `dt` at a
/// time (by an exact propagator or a numerical integrator), plus the model's
/// threshold/reset semantics.
pub trait NeuronModel {
    /// Number of state variables per neuron (`k`).
    fn n_state_vars(&self) -> usize;

    /// Simulation time step.
    fn dt(&self) -> f64;

    /// Set `state` to the model's resting point.
    ///
    /// # Panics
    /// Panics if `state.n_state_vars() != self.n_state_vars()`.
    fn init_state(&self, state: &mut LayerState);

    /// Advance one time step.
    ///
    /// `input` is the external drive, one value per neuron per batch sample
    /// (`n_neurons × batch`), held constant over the step (zero-order hold).
    /// Where it enters the dynamics is model-specific: through the synaptic
    /// current for LIF/adLIF, directly as the current `I` for Izhikevich.
    ///
    /// Every entry of `spikes` is overwritten with `1.0` (fired this step) or
    /// `0.0`.
    ///
    /// # Panics
    /// Panics if the dimensions of `state`, `input`, and `spikes` disagree.
    fn step(&self, state: &mut LayerState, input: MatRef<'_, f64>, spikes: &mut SpikeBatch);
}

/// Shared dimension check for `NeuronModel::step` implementations.
pub(crate) fn assert_step_dims(
    model_state_vars: usize,
    state: &LayerState,
    input: MatRef<'_, f64>,
    spikes: &SpikeBatch,
) {
    assert_eq!(
        state.n_state_vars(),
        model_state_vars,
        "state has wrong number of state variables for this model"
    );
    assert_eq!(input.nrows(), state.n_neurons(), "input rows != n_neurons");
    assert_eq!(input.ncols(), state.batch(), "input cols != batch");
    assert_eq!(
        spikes.n_neurons(),
        state.n_neurons(),
        "spike rows != n_neurons"
    );
    assert_eq!(spikes.batch(), state.batch(), "spike cols != batch");
}
