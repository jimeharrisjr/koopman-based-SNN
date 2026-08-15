//! Snapshot collection: turning simulated trajectories into the pair
//! matrices `(X₁, X₂, U)` that identification consumes.
//!
//! `koopman_dmd::dmd()` takes one **contiguous** trajectory and forms shift
//! pairs internally — it cannot consume concatenated trajectories or masked
//! pairs. The [`SnapshotSet`] builder exists for everything `dmd()` cannot
//! do: it accumulates explicit `(x_t, x_{t+1}, u_t)` triples across many
//! trajectories, drops masked pairs (e.g. hard-reset transitions), and hands
//! the assembled matrices to DMDc (Phase 3).

use faer::{Mat, MatRef};

use crate::error::SnnError;
use crate::neuron::NeuronModel;
use crate::spikes::SpikeBatch;
use crate::state::LayerState;

/// Assembled snapshot pairs: `x1`/`x2` are `n_state × m` (states at `t` and
/// `t+1`), `u` is `n_controls × m` (controls at `t`; zero rows when the
/// system is autonomous).
#[derive(Debug, Clone)]
pub struct SnapshotSet {
    x1: Mat<f64>,
    x2: Mat<f64>,
    u: Mat<f64>,
}

impl SnapshotSet {
    /// Assemble a snapshot set directly from pair matrices (`x1`/`x2`:
    /// `n_state × m`; `u`: `n_controls × m`, possibly zero rows).
    pub fn from_parts(x1: Mat<f64>, x2: Mat<f64>, u: Mat<f64>) -> Result<Self, SnnError> {
        if x1.nrows() == 0 || x1.ncols() == 0 {
            return Err(SnnError::InvalidParameter(
                "snapshot pairs must be nonempty".into(),
            ));
        }
        if x2.nrows() != x1.nrows() || x2.ncols() != x1.ncols() {
            return Err(SnnError::DimensionMismatch(format!(
                "x2 is {}×{}, expected {}×{}",
                x2.nrows(),
                x2.ncols(),
                x1.nrows(),
                x1.ncols()
            )));
        }
        if u.nrows() > 0 && u.ncols() != x1.ncols() {
            return Err(SnnError::DimensionMismatch(format!(
                "u has {} columns, expected {}",
                u.ncols(),
                x1.ncols()
            )));
        }
        Ok(Self { x1, x2, u })
    }

    pub fn builder(n_state: usize, n_controls: usize) -> SnapshotSetBuilder {
        SnapshotSetBuilder {
            n_state,
            n_controls,
            x1: Vec::new(),
            x2: Vec::new(),
            u: Vec::new(),
            n_pairs: 0,
        }
    }

    pub fn n_state(&self) -> usize {
        self.x1.nrows()
    }

    pub fn n_controls(&self) -> usize {
        self.u.nrows()
    }

    pub fn n_pairs(&self) -> usize {
        self.x1.ncols()
    }

    /// States at time `t`.
    pub fn x1(&self) -> MatRef<'_, f64> {
        self.x1.as_ref()
    }

    /// States at time `t + 1`.
    pub fn x2(&self) -> MatRef<'_, f64> {
        self.x2.as_ref()
    }

    /// Controls at time `t`.
    pub fn u(&self) -> MatRef<'_, f64> {
        self.u.as_ref()
    }

    /// Lift the state pairs through a **memoryless** dictionary (polynomial,
    /// trigonometric, RBF), producing a lifted snapshot set for EDMD(-c).
    /// Controls are passed through unchanged.
    ///
    /// Delay embedding is rejected: it couples adjacent columns, and snapshot
    /// pairs are not contiguous in time.
    pub fn lifted(
        &self,
        cfg: &koopman_dmd::LiftingConfig,
    ) -> Result<(SnapshotSet, koopman_dmd::LiftingInfo), SnnError> {
        if matches!(cfg, koopman_dmd::LiftingConfig::Delay { .. }) {
            return Err(SnnError::InvalidParameter(
                "delay embedding cannot lift snapshot pairs: pairs are not \
                 contiguous in time — apply delay lifting to whole trajectories \
                 before pair extraction instead"
                    .into(),
            ));
        }
        let (x1, info) = koopman_dmd::lift_data(&self.x1, cfg)?;
        let (x2, _) = koopman_dmd::lift_data(&self.x2, cfg)?;
        Ok((
            SnapshotSet {
                x1,
                x2,
                u: self.u.clone(),
            },
            info,
        ))
    }
}

/// Accumulates snapshot pairs trajectory by trajectory.
#[derive(Debug, Clone)]
pub struct SnapshotSetBuilder {
    n_state: usize,
    n_controls: usize,
    // Column-major flat storage, one column appended per kept pair.
    x1: Vec<f64>,
    x2: Vec<f64>,
    u: Vec<f64>,
    n_pairs: usize,
}

impl SnapshotSetBuilder {
    /// Append every shift pair of one trajectory.
    ///
    /// `states` is `n_state × T` (`T ≥ 2`, time-ordered columns). For a
    /// controlled system, `controls` must be `n_controls × (T − 1)`: column
    /// `t` is the input driving the transition `t → t+1`. Pass `None` iff the
    /// builder was created with `n_controls == 0`.
    pub fn push_trajectory(
        &mut self,
        states: MatRef<'_, f64>,
        controls: Option<MatRef<'_, f64>>,
    ) -> Result<&mut Self, SnnError> {
        let t = states.ncols();
        self.push_trajectory_masked(states, controls, &vec![true; t.saturating_sub(1)])
    }

    /// Like [`push_trajectory`](Self::push_trajectory), but keeps only pairs
    /// with `keep[t] == true` (`keep.len() == T − 1`). Used to excise
    /// transitions that straddle reset events when fitting sub-threshold
    /// dynamics.
    pub fn push_trajectory_masked(
        &mut self,
        states: MatRef<'_, f64>,
        controls: Option<MatRef<'_, f64>>,
        keep: &[bool],
    ) -> Result<&mut Self, SnnError> {
        if states.nrows() != self.n_state {
            return Err(SnnError::DimensionMismatch(format!(
                "trajectory has {} state rows, builder expects {}",
                states.nrows(),
                self.n_state
            )));
        }
        let t = states.ncols();
        if t < 2 {
            return Err(SnnError::InvalidParameter(format!(
                "trajectory needs at least 2 snapshots, got {t}"
            )));
        }
        if keep.len() != t - 1 {
            return Err(SnnError::DimensionMismatch(format!(
                "mask has {} entries, expected T − 1 = {}",
                keep.len(),
                t - 1
            )));
        }
        match (&controls, self.n_controls) {
            (None, 0) => {}
            (None, q) => {
                return Err(SnnError::InvalidParameter(format!(
                    "builder expects {q} control rows but no controls were given"
                )));
            }
            (Some(u), q) => {
                if u.nrows() != q {
                    return Err(SnnError::DimensionMismatch(format!(
                        "controls have {} rows, builder expects {q}",
                        u.nrows()
                    )));
                }
                if u.ncols() != t - 1 {
                    return Err(SnnError::DimensionMismatch(format!(
                        "controls have {} columns, expected T − 1 = {}",
                        u.ncols(),
                        t - 1
                    )));
                }
            }
        }

        for (step, &keep_pair) in keep.iter().enumerate() {
            if !keep_pair {
                continue;
            }
            for i in 0..self.n_state {
                self.x1.push(states[(i, step)]);
            }
            for i in 0..self.n_state {
                self.x2.push(states[(i, step + 1)]);
            }
            if let Some(u) = &controls {
                for i in 0..self.n_controls {
                    self.u.push(u[(i, step)]);
                }
            }
            self.n_pairs += 1;
        }
        Ok(self)
    }

    pub fn n_pairs(&self) -> usize {
        self.n_pairs
    }

    pub fn build(self) -> Result<SnapshotSet, SnnError> {
        if self.n_pairs == 0 {
            return Err(SnnError::InvalidParameter(
                "snapshot set has no pairs (all masked out, or nothing pushed)".into(),
            ));
        }
        let from_flat = |flat: &[f64], rows: usize, cols: usize| {
            let mut m = Mat::<f64>::zeros(rows, cols);
            for c in 0..cols {
                for r in 0..rows {
                    m[(r, c)] = flat[c * rows + r];
                }
            }
            m
        };
        Ok(SnapshotSet {
            x1: from_flat(&self.x1, self.n_state, self.n_pairs),
            x2: from_flat(&self.x2, self.n_state, self.n_pairs),
            u: from_flat(&self.u, self.n_controls, self.n_pairs),
        })
    }
}

/// A recorded simulation: the state trajectory (including the initial state)
/// plus per-step spikes and a per-transition spike mask.
#[derive(Debug, Clone)]
pub struct Recording {
    /// `state_dim × (t_steps + 1)`; column 0 is the initial state.
    pub states: Mat<f64>,
    /// Spikes emitted at each step (`t_steps` entries).
    pub spikes: Vec<SpikeBatch>,
    /// `any_spike[t]` — whether any neuron fired during transition `t → t+1`.
    /// `!any_spike[t]` is the keep-mask for sub-threshold-only fitting.
    pub any_spike: Vec<bool>,
}

impl Recording {
    /// Keep-mask selecting only spike-free transitions.
    pub fn subthreshold_mask(&self) -> Vec<bool> {
        self.any_spike.iter().map(|&s| !s).collect()
    }
}

/// Simulate `model` under a per-step input sequence (`inputs[t]` drives the
/// transition `t → t+1`), recording the trajectory. `state` must have
/// batch = 1 and is advanced in place.
pub fn record_input_sequence(
    model: &dyn NeuronModel,
    state: &mut LayerState,
    inputs: &[Mat<f64>],
) -> Result<Recording, SnnError> {
    if state.batch() != 1 {
        return Err(SnnError::InvalidParameter(format!(
            "trajectory recording needs batch = 1, got {}",
            state.batch()
        )));
    }
    if inputs.is_empty() {
        return Err(SnnError::InvalidParameter(
            "input sequence must be non-empty".into(),
        ));
    }
    let dim = state.state_dim();
    let n = state.n_neurons();
    let t_steps = inputs.len();
    let mut states = Mat::<f64>::zeros(dim, t_steps + 1);
    let mut spikes_out = Vec::with_capacity(t_steps);
    let mut any_spike = Vec::with_capacity(t_steps);

    for i in 0..dim {
        states[(i, 0)] = state.as_mat()[(i, 0)];
    }
    let mut spikes = SpikeBatch::zeros(n, 1)?;
    for (t, input) in inputs.iter().enumerate() {
        model.step(state, input.as_ref(), &mut spikes);
        for i in 0..dim {
            states[(i, t + 1)] = state.as_mat()[(i, 0)];
        }
        let fired = (0..n).any(|j| spikes.as_mat()[(j, 0)] != 0.0);
        any_spike.push(fired);
        spikes_out.push(spikes.clone());
    }
    Ok(Recording {
        states,
        spikes: spikes_out,
        any_spike,
    })
}

/// Simulate `model` for `t_steps` under a constant input, recording the
/// trajectory. `state` must have batch = 1 (a trajectory is a column
/// sequence) and is advanced in place.
pub fn record_constant_input(
    model: &dyn NeuronModel,
    state: &mut LayerState,
    input: MatRef<'_, f64>,
    t_steps: usize,
) -> Result<Recording, SnnError> {
    if state.batch() != 1 {
        return Err(SnnError::InvalidParameter(format!(
            "trajectory recording needs batch = 1, got {}",
            state.batch()
        )));
    }
    if t_steps == 0 {
        return Err(SnnError::InvalidParameter(
            "t_steps must be positive".into(),
        ));
    }
    let dim = state.state_dim();
    let n = state.n_neurons();
    let mut states = Mat::<f64>::zeros(dim, t_steps + 1);
    let mut spikes_out = Vec::with_capacity(t_steps);
    let mut any_spike = Vec::with_capacity(t_steps);

    for i in 0..dim {
        states[(i, 0)] = state.as_mat()[(i, 0)];
    }
    let mut spikes = SpikeBatch::zeros(n, 1)?;
    for t in 0..t_steps {
        model.step(state, input, &mut spikes);
        for i in 0..dim {
            states[(i, t + 1)] = state.as_mat()[(i, 0)];
        }
        let fired = (0..n).any(|j| spikes.as_mat()[(j, 0)] != 0.0);
        any_spike.push(fired);
        spikes_out.push(spikes.clone());
    }
    Ok(Recording {
        states,
        spikes: spikes_out,
        any_spike,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neuron::{Lif, LifParams, NeuronModel};

    fn traj(n_state: usize, t: usize, offset: f64) -> Mat<f64> {
        let mut m = Mat::<f64>::zeros(n_state, t);
        for c in 0..t {
            for r in 0..n_state {
                m[(r, c)] = offset + (c * n_state + r) as f64;
            }
        }
        m
    }

    #[test]
    fn pairs_are_shifted_columns() {
        let mut b = SnapshotSet::builder(2, 0);
        b.push_trajectory(traj(2, 4, 0.0).as_ref(), None).unwrap();
        let s = b.build().unwrap();
        assert_eq!(s.n_pairs(), 3);
        // Pair 0: x1 = col 0, x2 = col 1.
        assert_eq!(s.x1()[(0, 0)], 0.0);
        assert_eq!(s.x2()[(0, 0)], 2.0);
        // Pair 2: x1 = col 2, x2 = col 3.
        assert_eq!(s.x1()[(1, 2)], 5.0);
        assert_eq!(s.x2()[(1, 2)], 7.0);
    }

    #[test]
    fn trajectories_concatenate_without_cross_pairs() {
        let mut b = SnapshotSet::builder(1, 0);
        b.push_trajectory(traj(1, 3, 0.0).as_ref(), None).unwrap();
        b.push_trajectory(traj(1, 3, 100.0).as_ref(), None).unwrap();
        let s = b.build().unwrap();
        // 2 + 2 pairs; no pair joins column 2 of the first to column 0 of the
        // second.
        assert_eq!(s.n_pairs(), 4);
        assert_eq!(s.x1()[(0, 2)], 100.0);
        assert_eq!(s.x2()[(0, 2)], 101.0);
    }

    #[test]
    fn mask_drops_pairs() {
        let mut b = SnapshotSet::builder(1, 0);
        b.push_trajectory_masked(traj(1, 4, 0.0).as_ref(), None, &[true, false, true])
            .unwrap();
        let s = b.build().unwrap();
        assert_eq!(s.n_pairs(), 2);
        assert_eq!(s.x1()[(0, 1)], 2.0); // the middle pair is gone
    }

    #[test]
    fn controls_are_aligned_with_pairs() {
        let mut b = SnapshotSet::builder(1, 2);
        let mut u = Mat::<f64>::zeros(2, 2);
        u[(0, 0)] = 10.0;
        u[(1, 1)] = 20.0;
        b.push_trajectory(traj(1, 3, 0.0).as_ref(), Some(u.as_ref()))
            .unwrap();
        let s = b.build().unwrap();
        assert_eq!(s.n_controls(), 2);
        assert_eq!(s.u()[(0, 0)], 10.0);
        assert_eq!(s.u()[(1, 1)], 20.0);
    }

    #[test]
    fn dimension_errors_are_caught() {
        let mut b = SnapshotSet::builder(2, 1);
        // Missing controls.
        assert!(b.push_trajectory(traj(2, 3, 0.0).as_ref(), None).is_err());
        // Wrong control length (needs T − 1 = 2 columns).
        let u = Mat::<f64>::zeros(1, 3);
        assert!(b
            .push_trajectory(traj(2, 3, 0.0).as_ref(), Some(u.as_ref()))
            .is_err());
        // Wrong state rows.
        let u = Mat::<f64>::zeros(1, 2);
        assert!(b
            .push_trajectory(traj(3, 3, 0.0).as_ref(), Some(u.as_ref()))
            .is_err());
        // Too-short trajectory.
        assert!(b
            .push_trajectory(
                traj(2, 1, 0.0).as_ref(),
                Some(Mat::<f64>::zeros(1, 0).as_ref())
            )
            .is_err());
        // Empty build.
        assert!(SnapshotSet::builder(1, 0).build().is_err());
    }

    #[test]
    fn recording_captures_states_and_spike_mask() {
        let lif = Lif::new(LifParams {
            dt: 0.5,
            ..LifParams::default()
        })
        .unwrap();
        let mut state = LayerState::zeros(2, 2, 1).unwrap();
        lif.init_state(&mut state);
        // Strong drive on neuron 0 only: it will fire, neuron 1 stays silent.
        let mut input = Mat::<f64>::zeros(2, 1);
        input[(0, 0)] = 5.0;

        let rec = record_constant_input(&lif, &mut state, input.as_ref(), 200).unwrap();
        assert_eq!(rec.states.nrows(), 4);
        assert_eq!(rec.states.ncols(), 201);
        assert_eq!(rec.spikes.len(), 200);
        assert!(rec.any_spike.iter().any(|&s| s), "neuron 0 should fire");
        assert!(
            rec.subthreshold_mask().iter().any(|&k| k),
            "some transitions should be spike-free"
        );
        // Column 0 is the pre-step state (all rest = 0).
        assert_eq!(rec.states[(0, 0)], 0.0);
        // Mask agrees with recorded spikes.
        for (t, s) in rec.spikes.iter().enumerate() {
            let fired = s.as_mat()[(0, 0)] != 0.0 || s.as_mat()[(1, 0)] != 0.0;
            assert_eq!(fired, rec.any_spike[t]);
        }
    }

    #[test]
    fn recording_rejects_batched_state() {
        let lif = Lif::new(LifParams::default()).unwrap();
        let mut state = LayerState::zeros(2, 2, 3).unwrap();
        lif.init_state(&mut state);
        let input = Mat::<f64>::zeros(2, 3);
        assert!(record_constant_input(&lif, &mut state, input.as_ref(), 10).is_err());
    }
}
