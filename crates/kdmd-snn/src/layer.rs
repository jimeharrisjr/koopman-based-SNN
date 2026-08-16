//! The fast inference layer: linear advance + threshold/reset.
//!
//! One [`KoopmanLayer`] step is the canonical two-regime split
//! (IMPLEMENTATION_PLAN.md, core design decisions):
//!
//! ```text
//! 1. drive:      d = W · s_in            (sparse column-accum or dense matmul)
//! 2. advance:    y = A·x;  y_p += b_local[p] · d   (per state variable p)
//! 3. threshold:  s = Θ(v(y) − θ)         (v = variable 0)
//! 4. reset:      y_v −= θ ⊙ s            (subtractive — the only mode here)
//! ```
//!
//! Steps 2 + 4 together are exactly `x_{t+1} = A·x_t + B·u_t` with
//! `u = [s_in; s]` and `B = [b_local ⊗ W-columns, −θ·E_v]`; the elementwise
//! `Θ` is the network's only nonlinearity. Hard reset is deliberately absent
//! from the fast path (owner decision Q5) — it lives in the reference
//! simulators.
//!
//! For a homogeneous LIF layer, `A = A_local ⊗ I_N` ([`Operator::PerVariable`])
//! and `b_local = Lif::b_local()`, making this layer **exactly** the reference
//! simulator's update — the equivalence test demands identical spike trains
//! over 1000 steps. No allocation happens per step: all intermediates live in
//! the layer's scratch, sized at construction.

use faer::Mat;

use crate::error::SnnError;
use crate::neuron::Lif;
use crate::operator::Operator;
use crate::spikes::{SpikeBatch, SpikeVec};
use crate::state::LayerState;
use crate::util;

/// A fast-path layer: identified/closed-form `A`, synaptic weights `W`,
/// per-variable input coupling, subtractive threshold.
#[derive(Debug, Clone)]
pub struct KoopmanLayer {
    a: Operator,
    /// Synaptic weights, `n_neurons × n_inputs`, column-major (faer default)
    /// so sparse drive accumulation walks contiguous columns.
    w_in: Mat<f64>,
    /// Input coupling per state variable: drive `d_j` enters variable `p` of
    /// neuron `j` as `b_local[p] · d_j` (for exact LIF: `[δ, 1 − β]`).
    b_local: Vec<f64>,
    /// Firing threshold (subtractive reset subtracts exactly this).
    theta: f64,
    /// Spike-triggered jumps per state variable: on a spike, variable `p` of
    /// the firing neuron gets `jumps[p]` added. `[−θ, 0]` is the plain LIF
    /// subtractive reset; `[−θ, 0, b_jump]` adds adaptive-LIF adaptation.
    /// Linear in the spike indicator, so the step stays exactly
    /// `x_{t+1} = A·x_t + B·u_t`.
    jumps: Vec<f64>,
    /// Per-neuron input coupling override (`k × N`): entry `(p, j)` scales
    /// the drive into variable `p` of neuron `j`. `None` broadcasts
    /// `b_local` uniformly (the homogeneous case). Heterogeneous-τ layers
    /// need this because δ and 1−β differ per neuron.
    coupling_override: Option<Mat<f64>>,
    n_neurons: usize,
    n_state_vars: usize,
    /// Optional recurrent weights (`n_neurons × n_neurons`): the layer's own
    /// spikes from the **previous** step feed back into the drive.
    w_rec: Option<Mat<f64>>,
    /// The layer's own spikes at the previous step (`n × batch`); episodic
    /// state — cleared by [`reset_recurrent`](Self::reset_recurrent), which
    /// [`Network::reset_state`](crate::Network::reset_state) calls.
    prev_spikes: Mat<f64>,
    // Scratch, allocated once: next-state buffer and drive buffer.
    y: Mat<f64>,
    drive: Mat<f64>,
}

impl KoopmanLayer {
    /// Build a layer from an operator, weights, input coupling, and
    /// threshold. `batch` fixes the scratch capacity (and the batch size all
    /// step calls must use).
    pub fn new(
        a: Operator,
        w_in: Mat<f64>,
        b_local: Vec<f64>,
        theta: f64,
        batch: usize,
    ) -> Result<Self, SnnError> {
        let dim = a.state_dim();
        if dim == 0 {
            return Err(SnnError::InvalidParameter(
                "operator acts on a zero-dimensional state".into(),
            ));
        }
        if b_local.is_empty() || dim % b_local.len() != 0 {
            return Err(SnnError::DimensionMismatch(format!(
                "operator dim {dim} is not a multiple of the {} input-coupling entries",
                b_local.len()
            )));
        }
        let n_state_vars = b_local.len();
        let n_neurons = dim / n_state_vars;
        // A structured operator carries its own (k, N) factorization; a
        // b_local that implies a different one would silently misalign the
        // threshold block (adversarial-test finding, docs/11).
        match &a {
            Operator::PerVariable {
                a_local,
                n_neurons: n_op,
            } => {
                if a_local.nrows() != n_state_vars || *n_op != n_neurons {
                    return Err(SnnError::DimensionMismatch(format!(
                        "PerVariable operator is ({}, {n_op}) but b_local implies \
                         ({n_state_vars}, {n_neurons})",
                        a_local.nrows()
                    )));
                }
            }
            Operator::PerNeuron {
                blocks,
                n_state_vars: k_op,
            } => {
                if *k_op != n_state_vars || blocks.ncols() != n_neurons {
                    return Err(SnnError::DimensionMismatch(format!(
                        "PerNeuron operator is ({k_op}, {}) but b_local implies \
                         ({n_state_vars}, {n_neurons})",
                        blocks.ncols()
                    )));
                }
            }
            Operator::Dense(_) | Operator::LowRank { .. } => {}
        }
        if w_in.nrows() != n_neurons {
            return Err(SnnError::DimensionMismatch(format!(
                "W has {} rows, expected n_neurons = {n_neurons}",
                w_in.nrows()
            )));
        }
        if !util::is_positive(theta) {
            return Err(SnnError::InvalidParameter(format!(
                "threshold must be positive (got {theta})"
            )));
        }
        if batch == 0 {
            return Err(SnnError::InvalidParameter("batch must be nonzero".into()));
        }
        // Default jumps: plain subtractive reset on the potential.
        let mut jumps = vec![0.0; n_state_vars];
        jumps[0] = -theta;
        Ok(Self {
            a,
            b_local,
            theta,
            jumps,
            coupling_override: None,
            n_neurons,
            n_state_vars,
            w_rec: None,
            prev_spikes: Mat::zeros(n_neurons, batch),
            y: Mat::zeros(dim, batch),
            drive: Mat::zeros(n_neurons, batch),
            w_in,
        })
    }

    /// Override the spike-triggered jumps (length k; `jumps[0]` must stay
    /// `−θ` — the subtractive reset is not optional on the fast path).
    pub fn with_jumps(mut self, jumps: Vec<f64>) -> Result<Self, SnnError> {
        if jumps.len() != self.n_state_vars {
            return Err(SnnError::DimensionMismatch(format!(
                "{} jump entries for {} state variables",
                jumps.len(),
                self.n_state_vars
            )));
        }
        if jumps[0] != -self.theta {
            return Err(SnnError::InvalidParameter(format!(
                "jumps[0] must equal −θ = {} (got {})",
                -self.theta, jumps[0]
            )));
        }
        self.jumps = jumps;
        Ok(self)
    }

    /// Per-neuron input-coupling override (`k × N`), for heterogeneous
    /// layers where δ and 1−β differ per neuron.
    pub fn with_coupling(mut self, coupling: Mat<f64>) -> Result<Self, SnnError> {
        if coupling.nrows() != self.n_state_vars || coupling.ncols() != self.n_neurons {
            return Err(SnnError::DimensionMismatch(format!(
                "coupling is {}×{}, expected {}×{}",
                coupling.nrows(),
                coupling.ncols(),
                self.n_state_vars,
                self.n_neurons
            )));
        }
        self.coupling_override = Some(coupling);
        Ok(self)
    }

    /// Add recurrent connections: the layer's own previous-step spikes feed
    /// back through `w_rec` (`n_neurons × n_neurons`) into the same drive
    /// path as the feedforward input. Zero-initializing `w_rec` reproduces
    /// the feedforward layer exactly, letting training grow recurrence from
    /// nothing.
    pub fn with_recurrent(mut self, w_rec: Mat<f64>) -> Result<Self, SnnError> {
        if w_rec.nrows() != self.n_neurons || w_rec.ncols() != self.n_neurons {
            return Err(SnnError::DimensionMismatch(format!(
                "recurrent weights are {}×{}, expected {}×{}",
                w_rec.nrows(),
                w_rec.ncols(),
                self.n_neurons,
                self.n_neurons
            )));
        }
        self.w_rec = Some(w_rec);
        Ok(self)
    }

    /// Recurrent weights, if the layer has them.
    pub fn recurrent_weights(&self) -> Option<&Mat<f64>> {
        self.w_rec.as_ref()
    }

    /// Mutable recurrent weights (training updates them).
    pub fn recurrent_weights_mut(&mut self) -> Option<&mut Mat<f64>> {
        self.w_rec.as_mut()
    }

    /// Clear the episodic recurrent state (previous-step spikes). Called at
    /// the start of a trial via `Network::reset_state`.
    pub fn reset_recurrent(&mut self) {
        for b in 0..self.prev_spikes.ncols() {
            for i in 0..self.prev_spikes.nrows() {
                self.prev_spikes[(i, b)] = 0.0;
            }
        }
    }

    /// The exact-linear LIF layer: `A = A_local ⊗ I_N` from the closed-form
    /// propagator, coupling from [`Lif::b_local`], threshold from the params.
    /// This is the engine the Phase 5 benchmark gate measures.
    ///
    /// Requires `v_rest = 0` (the purely linear step cannot represent the
    /// affine rest term — shift coordinates to `v − v_rest` first) and the
    /// subtractive reset (the fast path's only mode, owner decision Q5).
    /// Both are rejected loudly rather than silently mis-simulated.
    pub fn lif(
        lif: &Lif,
        n_neurons: usize,
        w_in: Mat<f64>,
        batch: usize,
    ) -> Result<Self, SnnError> {
        if lif.params().v_rest != 0.0 {
            return Err(SnnError::InvalidParameter(format!(
                "KoopmanLayer::lif requires v_rest = 0 (got {}): the linear \
                 step has no affine term — shift coordinates to v − v_rest",
                lif.params().v_rest
            )));
        }
        if !matches!(lif.params().reset, crate::neuron::ResetMode::Subtractive) {
            return Err(SnnError::InvalidParameter(
                "the fast path supports subtractive reset only (Q5); hard \
                 reset lives in the reference simulators"
                    .into(),
            ));
        }
        let a = Operator::PerVariable {
            a_local: lif.a_local(),
            n_neurons,
        };
        Self::new(a, w_in, lif.b_local().to_vec(), lif.params().theta, batch)
    }

    /// The exact-linear **adaptive** LIF layer (k = 3: potential, synaptic
    /// current, adaptation). The spike-triggered adaptation increment is one
    /// more linear jump (`[−θ, 0, b_jump]`), so identification and training
    /// exactness carry over unchanged. Same v_rest/reset requirements as
    /// [`lif`](Self::lif).
    pub fn adlif(
        adlif: &crate::neuron::AdLif,
        n_neurons: usize,
        w_in: Mat<f64>,
        batch: usize,
    ) -> Result<Self, SnnError> {
        let p = adlif.params();
        if p.v_rest != 0.0 {
            return Err(SnnError::InvalidParameter(format!(
                "KoopmanLayer::adlif requires v_rest = 0 (got {})",
                p.v_rest
            )));
        }
        if !matches!(p.reset, crate::neuron::ResetMode::Subtractive) {
            return Err(SnnError::InvalidParameter(
                "the fast path supports subtractive reset only (Q5)".into(),
            ));
        }
        let theta = p.theta;
        let b_jump = p.b_jump;
        let a = Operator::PerVariable {
            a_local: adlif.a_local(),
            n_neurons,
        };
        Self::new(a, w_in, adlif.b_local().to_vec(), theta, batch)?
            .with_jumps(vec![-theta, 0.0, b_jump])
    }

    /// Heterogeneous adaptive-LIF layer: per-neuron time constants (one
    /// [`AdLif`](crate::neuron::AdLif) per neuron) with shared threshold and
    /// adaptation increment. Uses [`Operator::PerNeuron`] blocks plus a
    /// per-neuron coupling override.
    pub fn adlif_hetero(
        neurons: &[crate::neuron::AdLif],
        w_in: Mat<f64>,
        batch: usize,
    ) -> Result<Self, SnnError> {
        let n = neurons.len();
        if n == 0 {
            return Err(SnnError::InvalidParameter("no neurons given".into()));
        }
        let theta = neurons[0].params().theta;
        let b_jump = neurons[0].params().b_jump;
        let mut blocks = Mat::<f64>::zeros(9, n);
        let mut coupling = Mat::<f64>::zeros(3, n);
        for (j, cell) in neurons.iter().enumerate() {
            let p = cell.params();
            if p.v_rest != 0.0
                || !matches!(p.reset, crate::neuron::ResetMode::Subtractive)
                || p.theta != theta
                || p.b_jump != b_jump
            {
                return Err(SnnError::InvalidParameter(format!(
                    "neuron {j}: heterogeneous layers vary time constants only \
                     (v_rest = 0, subtractive reset, shared θ and b_jump)"
                )));
            }
            let a_local = cell.a_local();
            for p_row in 0..3 {
                for q in 0..3 {
                    blocks[(p_row * 3 + q, j)] = a_local[(p_row, q)];
                }
            }
            let b = cell.b_local();
            for (p_row, &c) in b.iter().enumerate() {
                coupling[(p_row, j)] = c;
            }
        }
        let a = Operator::PerNeuron {
            blocks,
            n_state_vars: 3,
        };
        Self::new(a, w_in, vec![0.0; 3], theta, batch)?
            .with_coupling(coupling)?
            .with_jumps(vec![-theta, 0.0, b_jump])
    }

    pub fn n_neurons(&self) -> usize {
        self.n_neurons
    }

    pub fn n_inputs(&self) -> usize {
        self.w_in.ncols()
    }

    pub fn n_state_vars(&self) -> usize {
        self.n_state_vars
    }

    pub fn operator(&self) -> &Operator {
        &self.a
    }

    pub fn weights(&self) -> &Mat<f64> {
        &self.w_in
    }

    /// Mutable weights (training updates them; dimensions must not change).
    pub fn weights_mut(&mut self) -> &mut Mat<f64> {
        &mut self.w_in
    }

    /// Input-coupling coefficients per state variable (uniform case; a
    /// heterogeneous layer's effective coupling is [`coupling_override`]).
    pub fn b_local(&self) -> &[f64] {
        &self.b_local
    }

    /// Per-neuron coupling override, if set.
    pub fn coupling_override(&self) -> Option<&Mat<f64>> {
        self.coupling_override.as_ref()
    }

    /// Effective drive coupling into variable `p` of neuron `j`.
    #[inline]
    pub fn coupling(&self, p: usize, j: usize) -> f64 {
        match &self.coupling_override {
            Some(c) => c[(p, j)],
            None => self.b_local[p],
        }
    }

    /// Spike-triggered jumps per state variable (`jumps[0] = −θ`).
    pub fn jumps(&self) -> &[f64] {
        &self.jumps
    }

    /// Firing threshold.
    pub fn theta(&self) -> f64 {
        self.theta
    }

    fn check_state(&self, state: &LayerState, batch: usize) -> Result<(), SnnError> {
        if state.n_neurons() != self.n_neurons || state.n_state_vars() != self.n_state_vars {
            return Err(SnnError::DimensionMismatch(format!(
                "state is {}×{} vars, layer is {}×{}",
                state.n_neurons(),
                state.n_state_vars(),
                self.n_neurons,
                self.n_state_vars
            )));
        }
        if state.batch() != batch || self.y.ncols() != batch {
            return Err(SnnError::DimensionMismatch(format!(
                "batch mismatch: state {}, expected {batch} (scratch {})",
                state.batch(),
                self.y.ncols()
            )));
        }
        Ok(())
    }

    /// Inference step, batch = 1, sparse input spikes. Appends this layer's
    /// spikes into `out` (cleared first; allocation-free once its capacity
    /// has grown).
    pub fn step(
        &mut self,
        state: &mut LayerState,
        s_in: &SpikeVec,
        out: &mut SpikeVec,
    ) -> Result<(), SnnError> {
        self.check_state(state, 1)?;
        if s_in.n_neurons() != self.n_inputs() {
            return Err(SnnError::DimensionMismatch(format!(
                "input spikes cover {} neurons, layer expects {}",
                s_in.n_neurons(),
                self.n_inputs()
            )));
        }
        let n = self.n_neurons;

        // 1. Sparse drive: d = Σ_{j active} W[:, j] — contiguous column adds.
        for i in 0..n {
            self.drive[(i, 0)] = 0.0;
        }
        for &j in s_in.active() {
            let j = j as usize;
            for i in 0..n {
                self.drive[(i, 0)] += self.w_in[(i, j)];
            }
        }
        // 1b. Recurrent drive from the layer's own previous-step spikes.
        if let Some(w_rec) = &self.w_rec {
            for j in 0..n {
                if self.prev_spikes[(j, 0)] == 0.0 {
                    continue;
                }
                for i in 0..n {
                    self.drive[(i, 0)] += w_rec[(i, j)];
                }
            }
        }

        // 2. Linear advance plus input coupling.
        self.a.apply(state.as_mat(), self.y.as_mut(), false);
        match &self.coupling_override {
            None => {
                for (p, &coef) in self.b_local.iter().enumerate() {
                    if coef == 0.0 {
                        continue;
                    }
                    for j in 0..n {
                        self.y[(p * n + j, 0)] += coef * self.drive[(j, 0)];
                    }
                }
            }
            Some(c) => {
                for p in 0..self.n_state_vars {
                    for j in 0..n {
                        self.y[(p * n + j, 0)] += c[(p, j)] * self.drive[(j, 0)];
                    }
                }
            }
        }

        // 3 + 4. Threshold on the potential block; spike-triggered jumps
        // (subtractive reset on v, adaptation increments, …).
        out.clear();
        for j in 0..n {
            if self.y[(j, 0)] >= self.theta {
                for (p, &jump) in self.jumps.iter().enumerate() {
                    if jump != 0.0 {
                        self.y[(p * n + j, 0)] += jump;
                    }
                }
                out.push(j as u32);
            }
        }

        // Commit.
        let mut sm = state.as_mat_mut();
        for i in 0..n * self.n_state_vars {
            sm[(i, 0)] = self.y[(i, 0)];
        }
        if self.w_rec.is_some() {
            for i in 0..n {
                self.prev_spikes[(i, 0)] = 0.0;
            }
            for &j in out.active() {
                self.prev_spikes[(j as usize, 0)] = 1.0;
            }
        }
        Ok(())
    }

    /// Batched step with dense spikes (the training-path representation).
    /// Writes this layer's spikes into `out` (overwritten entirely).
    pub fn step_batch(
        &mut self,
        state: &mut LayerState,
        s_in: &SpikeBatch,
        out: &mut SpikeBatch,
    ) -> Result<(), SnnError> {
        self.step_batch_impl(state, s_in, out, None)
    }

    /// [`step_batch`](Self::step_batch), additionally saving the **pre-reset**
    /// membrane potentials into `v_pre` (`n_neurons × batch`) — the tensor the
    /// BPTT tape needs for the surrogate derivative.
    pub fn step_batch_taped(
        &mut self,
        state: &mut LayerState,
        s_in: &SpikeBatch,
        out: &mut SpikeBatch,
        v_pre: &mut Mat<f64>,
    ) -> Result<(), SnnError> {
        if v_pre.nrows() != self.n_neurons || v_pre.ncols() != state.batch() {
            return Err(SnnError::DimensionMismatch(format!(
                "v_pre buffer is {}×{}, expected {}×{}",
                v_pre.nrows(),
                v_pre.ncols(),
                self.n_neurons,
                state.batch()
            )));
        }
        self.step_batch_impl(state, s_in, out, Some(v_pre))
    }

    fn step_batch_impl(
        &mut self,
        state: &mut LayerState,
        s_in: &SpikeBatch,
        out: &mut SpikeBatch,
        mut v_pre: Option<&mut Mat<f64>>,
    ) -> Result<(), SnnError> {
        let batch = state.batch();
        self.check_state(state, batch)?;
        if s_in.n_neurons() != self.n_inputs() || s_in.batch() != batch {
            return Err(SnnError::DimensionMismatch(format!(
                "input spikes are {}×{}, layer expects {}×{batch}",
                s_in.n_neurons(),
                s_in.batch(),
                self.n_inputs()
            )));
        }
        if out.n_neurons() != self.n_neurons || out.batch() != batch {
            return Err(SnnError::DimensionMismatch(format!(
                "output spikes are {}×{}, expected {}×{batch}",
                out.n_neurons(),
                out.batch(),
                self.n_neurons
            )));
        }
        let n = self.n_neurons;

        // 1. Dense drive: D = W · S (BLAS-3 shape; loops for now, faer matmul
        //    lands with the bench-driven optimization pass).
        for b in 0..batch {
            for i in 0..n {
                self.drive[(i, b)] = 0.0;
            }
        }
        for b in 0..batch {
            for j in 0..self.w_in.ncols() {
                let s = s_in.as_mat()[(j, b)];
                if s == 0.0 {
                    continue;
                }
                for i in 0..n {
                    self.drive[(i, b)] += self.w_in[(i, j)] * s;
                }
            }
        }
        // 1b. Recurrent drive from the layer's own previous-step spikes.
        if let Some(w_rec) = &self.w_rec {
            for b in 0..batch {
                for j in 0..n {
                    if self.prev_spikes[(j, b)] == 0.0 {
                        continue;
                    }
                    for i in 0..n {
                        self.drive[(i, b)] += w_rec[(i, j)];
                    }
                }
            }
        }

        // 2. Advance + coupling.
        self.a.apply(state.as_mat(), self.y.as_mut(), false);
        match &self.coupling_override {
            None => {
                for (p, &coef) in self.b_local.iter().enumerate() {
                    if coef == 0.0 {
                        continue;
                    }
                    for b in 0..batch {
                        for j in 0..n {
                            self.y[(p * n + j, b)] += coef * self.drive[(j, b)];
                        }
                    }
                }
            }
            Some(c) => {
                for p in 0..self.n_state_vars {
                    for b in 0..batch {
                        for j in 0..n {
                            self.y[(p * n + j, b)] += c[(p, j)] * self.drive[(j, b)];
                        }
                    }
                }
            }
        }

        // (Tape) pre-reset potentials, before the threshold mutates them.
        if let Some(v_pre) = v_pre.as_mut() {
            for b in 0..batch {
                for j in 0..n {
                    v_pre[(j, b)] = self.y[(j, b)];
                }
            }
        }

        // 3 + 4. Threshold + spike-triggered jumps (reset, adaptation, …).
        for b in 0..batch {
            for j in 0..n {
                let fired = self.y[(j, b)] >= self.theta;
                if fired {
                    for (p, &jump) in self.jumps.iter().enumerate() {
                        if jump != 0.0 {
                            self.y[(p * n + j, b)] += jump;
                        }
                    }
                }
                out.as_mat_mut()[(j, b)] = if fired { 1.0 } else { 0.0 };
            }
        }

        // Commit.
        let mut sm = state.as_mat_mut();
        for b in 0..batch {
            for i in 0..n * self.n_state_vars {
                sm[(i, b)] = self.y[(i, b)];
            }
        }
        if self.w_rec.is_some() {
            for b in 0..batch {
                for i in 0..n {
                    self.prev_spikes[(i, b)] = out.as_mat()[(i, b)];
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neuron::{LifParams, NeuronModel};

    fn identity_w(n: usize) -> Mat<f64> {
        let mut w = Mat::zeros(n, n);
        for i in 0..n {
            w[(i, i)] = 2.0; // strong one-to-one drive
        }
        w
    }

    #[test]
    fn constructor_validates_dimensions() {
        let lif = Lif::new(LifParams::default()).unwrap();
        assert!(KoopmanLayer::lif(&lif, 4, identity_w(4), 1).is_ok());
        // Wrong W rows.
        assert!(KoopmanLayer::lif(&lif, 4, identity_w(3), 1).is_err());
        // Zero batch.
        assert!(KoopmanLayer::lif(&lif, 4, identity_w(4), 0).is_err());
    }

    #[test]
    fn sparse_and_batch_paths_agree() {
        let lif = Lif::new(LifParams {
            dt: 0.5,
            ..LifParams::default()
        })
        .unwrap();
        let n = 6;
        let mut sparse_layer = KoopmanLayer::lif(&lif, n, identity_w(n), 1).unwrap();
        let mut batch_layer = KoopmanLayer::lif(&lif, n, identity_w(n), 1).unwrap();

        let mut s_state = LayerState::zeros(n, 2, 1).unwrap();
        let mut b_state = LayerState::zeros(n, 2, 1).unwrap();
        lif.init_state(&mut s_state);
        lif.init_state(&mut b_state);

        let mut out_sparse = SpikeVec::new(n);
        let mut out_batch = SpikeBatch::zeros(n, 1).unwrap();
        for t in 0..400 {
            // Alternate input pattern on neurons 0 and 3.
            let active: &[u32] = if t % 2 == 0 { &[0] } else { &[0, 3] };
            let s_in = SpikeVec::from_indices(active.to_vec(), n).unwrap();
            let mut dense_in = SpikeBatch::zeros(n, 1).unwrap();
            for &j in active {
                dense_in.as_mat_mut()[(j as usize, 0)] = 1.0;
            }
            sparse_layer
                .step(&mut s_state, &s_in, &mut out_sparse)
                .unwrap();
            batch_layer
                .step_batch(&mut b_state, &dense_in, &mut out_batch)
                .unwrap();

            let batch_active = out_batch.column_to_sparse(0);
            assert_eq!(
                out_sparse.active(),
                batch_active.active(),
                "sparse and batch paths diverged at step {t}"
            );
            for i in 0..2 * n {
                assert_eq!(
                    s_state.as_mat()[(i, 0)],
                    b_state.as_mat()[(i, 0)],
                    "state diverged at step {t}, row {i}"
                );
            }
        }
    }

    #[test]
    fn adlif_layer_matches_reference_simulator_spike_for_spike() {
        use crate::neuron::{AdLif, AdLifParams};
        let adlif = AdLif::new(AdLifParams {
            dt: 0.5,
            b_jump: 0.3,
            ..AdLifParams::default()
        })
        .unwrap();
        let n = 6;
        let mut layer = KoopmanLayer::adlif(&adlif, n, identity_w(n), 1).unwrap();
        let mut fast_state = LayerState::zeros(n, 3, 1).unwrap();
        adlif.init_state(&mut fast_state);
        let mut ref_state = LayerState::zeros(n, 3, 1).unwrap();
        adlif.init_state(&mut ref_state);
        let mut ref_spikes = SpikeBatch::zeros(n, 1).unwrap();
        let mut out = SpikeVec::new(n);

        for t in 0..800 {
            // Drive neurons 0 and 3 (through identity W, gain 2).
            let active: &[u32] = if t % 3 == 0 { &[0, 3] } else { &[0] };
            let s_in = SpikeVec::from_indices(active.to_vec(), n).unwrap();
            let mut drive = Mat::<f64>::zeros(n, 1);
            for &j in active {
                drive[(j as usize, 0)] += 2.0;
            }
            adlif.step(&mut ref_state, drive.as_ref(), &mut ref_spikes);
            layer.step(&mut fast_state, &s_in, &mut out).unwrap();

            let ref_active: Vec<u32> = (0..n as u32)
                .filter(|&j| ref_spikes.as_mat()[(j as usize, 0)] == 1.0)
                .collect();
            assert_eq!(
                out.active(),
                ref_active.as_slice(),
                "spikes diverged at {t}"
            );
            for i in 0..3 * n {
                let a = fast_state.as_mat()[(i, 0)];
                let b = ref_state.as_mat()[(i, 0)];
                assert!(
                    (a - b).abs() <= 1e-12 * b.abs().max(1.0),
                    "state diverged at step {t}, row {i}: {a} vs {b}"
                );
            }
        }
        // Adaptation must actually have engaged.
        assert!(fast_state.var(2)[(0, 0)] > 0.0, "adaptation never engaged");
    }

    #[test]
    fn heterogeneous_adlif_layer_matches_per_neuron_references() {
        use crate::neuron::{AdLif, AdLifParams};
        // Three neurons with different time constants; the hetero layer must
        // match each neuron's own reference simulator exactly.
        let cells: Vec<AdLif> = [(10.0, 5.0, 80.0), (20.0, 8.0, 150.0), (30.0, 12.0, 250.0)]
            .iter()
            .map(|&(tau_m, tau_s, tau_w)| {
                AdLif::new(AdLifParams {
                    tau_m,
                    tau_s,
                    tau_w,
                    dt: 0.5,
                    b_jump: 0.2,
                    ..AdLifParams::default()
                })
                .unwrap()
            })
            .collect();
        let n = cells.len();
        let mut layer = KoopmanLayer::adlif_hetero(&cells, identity_w(n), 1).unwrap();
        let mut fast_state = LayerState::zeros(n, 3, 1).unwrap();
        // Per-neuron single-cell references.
        let mut ref_states: Vec<LayerState> = (0..n)
            .map(|_| LayerState::zeros(1, 3, 1).unwrap())
            .collect();
        let mut ref_spikes = SpikeBatch::zeros(1, 1).unwrap();
        let mut out = SpikeVec::new(n);

        for t in 0..600 {
            let s_in = SpikeVec::from_indices(vec![0, 1, 2], n).unwrap();
            layer.step(&mut fast_state, &s_in, &mut out).unwrap();
            for (j, cell) in cells.iter().enumerate() {
                let mut drive = Mat::<f64>::zeros(1, 1);
                drive[(0, 0)] = 2.0;
                cell.step(&mut ref_states[j], drive.as_ref(), &mut ref_spikes);
                for p in 0..3 {
                    let a = fast_state.as_mat()[(p * n + j, 0)];
                    let b = ref_states[j].as_mat()[(p, 0)];
                    assert!(
                        (a - b).abs() <= 1e-12 * b.abs().max(1.0),
                        "neuron {j} var {p} diverged at step {t}: {a} vs {b}"
                    );
                }
            }
        }
    }

    #[test]
    fn hetero_constructor_rejects_mixed_thresholds() {
        use crate::neuron::{AdLif, AdLifParams};
        let a = AdLif::new(AdLifParams::default()).unwrap();
        let b = AdLif::new(AdLifParams {
            theta: 2.0,
            ..AdLifParams::default()
        })
        .unwrap();
        assert!(KoopmanLayer::adlif_hetero(&[a, b], identity_w(2), 1).is_err());
    }

    #[test]
    fn recurrent_self_excitation_sustains_and_reset_silences() {
        // Strong self-excitation: after a single kick, the layer keeps
        // itself firing with no further input; reset_recurrent (plus a state
        // reset) returns it to silence.
        let lif = Lif::new(LifParams {
            dt: 1.0,
            ..LifParams::default()
        })
        .unwrap();
        let n = 4;
        let w_rec = Mat::from_fn(n, n, |_, _| 4.0);
        let mut layer = KoopmanLayer::lif(&lif, n, identity_w(n), 1)
            .unwrap()
            .with_recurrent(w_rec)
            .unwrap();
        let mut state = LayerState::zeros(n, 2, 1).unwrap();
        lif.init_state(&mut state);
        let mut out = SpikeVec::new(n);

        // Kick every neuron for a few steps, then go silent.
        let kick = SpikeVec::from_indices((0..n as u32).collect(), n).unwrap();
        let quiet = SpikeVec::new(n);
        for _ in 0..30 {
            layer.step(&mut state, &kick, &mut out).unwrap();
        }
        let mut fired_after_kick = 0usize;
        let mut fired_late = 0usize;
        for t in 0..100 {
            layer.step(&mut state, &quiet, &mut out).unwrap();
            fired_after_kick += out.count();
            if t >= 70 {
                fired_late += out.count();
            }
        }
        assert!(
            fired_after_kick > 10,
            "self-excitation died out ({fired_after_kick} spikes in 100 quiet steps)"
        );
        assert!(
            fired_late > 0,
            "self-excitation decayed away instead of persisting (no spikes in the last 30 steps)"
        );

        // Full reset: silent thereafter.
        state.fill(0.0);
        layer.reset_recurrent();
        for _ in 0..50 {
            layer.step(&mut state, &quiet, &mut out).unwrap();
            assert!(out.is_empty(), "layer fired after a full reset");
        }
    }

    #[test]
    fn zero_recurrence_matches_feedforward_exactly() {
        let lif = Lif::new(LifParams {
            dt: 0.5,
            ..LifParams::default()
        })
        .unwrap();
        let n = 5;
        let mut ff = KoopmanLayer::lif(&lif, n, identity_w(n), 1).unwrap();
        let mut rec = KoopmanLayer::lif(&lif, n, identity_w(n), 1)
            .unwrap()
            .with_recurrent(Mat::zeros(n, n))
            .unwrap();
        let mut s1 = LayerState::zeros(n, 2, 1).unwrap();
        let mut s2 = LayerState::zeros(n, 2, 1).unwrap();
        let mut o1 = SpikeVec::new(n);
        let mut o2 = SpikeVec::new(n);
        let drive = SpikeVec::from_indices(vec![0, 2], n).unwrap();
        for t in 0..300 {
            ff.step(&mut s1, &drive, &mut o1).unwrap();
            rec.step(&mut s2, &drive, &mut o2).unwrap();
            assert_eq!(o1.active(), o2.active(), "diverged at step {t}");
        }
    }

    #[test]
    fn recurrent_weight_dimensions_are_validated() {
        let lif = Lif::new(LifParams::default()).unwrap();
        let layer = KoopmanLayer::lif(&lif, 4, identity_w(4), 1).unwrap();
        assert!(layer.with_recurrent(Mat::zeros(3, 4)).is_err());
    }

    #[test]
    fn batched_columns_evolve_independently() {
        let lif = Lif::new(LifParams::default()).unwrap();
        let n = 3;
        let batch = 2;
        let mut layer = KoopmanLayer::lif(&lif, n, identity_w(n), batch).unwrap();
        let mut state = LayerState::zeros(n, 2, batch).unwrap();
        lif.init_state(&mut state);

        // Drive only batch column 0.
        let mut s_in = SpikeBatch::zeros(n, batch).unwrap();
        s_in.as_mat_mut()[(0, 0)] = 1.0;
        let mut out = SpikeBatch::zeros(n, batch).unwrap();
        for _ in 0..200 {
            layer.step_batch(&mut state, &s_in, &mut out).unwrap();
        }
        // Column 0's neuron 0 accumulated drive; column 1 stayed at rest.
        assert!(state.var(1)[(0, 0)] > 0.1);
        assert_eq!(state.var(0)[(0, 1)], 0.0);
        assert_eq!(state.var(1)[(0, 1)], 0.0);
    }
}
