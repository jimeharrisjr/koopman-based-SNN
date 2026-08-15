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
    n_neurons: usize,
    n_state_vars: usize,
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
        Ok(Self {
            a,
            b_local,
            theta,
            n_neurons,
            n_state_vars,
            y: Mat::zeros(dim, batch),
            drive: Mat::zeros(n_neurons, batch),
            w_in,
        })
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

    /// Input-coupling coefficients per state variable.
    pub fn b_local(&self) -> &[f64] {
        &self.b_local
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

        // 2. Linear advance plus input coupling.
        self.a.apply(state.as_mat(), self.y.as_mut(), false);
        for (p, &coef) in self.b_local.iter().enumerate() {
            if coef == 0.0 {
                continue;
            }
            for j in 0..n {
                self.y[(p * n + j, 0)] += coef * self.drive[(j, 0)];
            }
        }

        // 3 + 4. Threshold on the potential block, subtractive reset.
        out.clear();
        for j in 0..n {
            if self.y[(j, 0)] >= self.theta {
                self.y[(j, 0)] -= self.theta;
                out.push(j as u32);
            }
        }

        // Commit.
        let mut sm = state.as_mat_mut();
        for i in 0..n * self.n_state_vars {
            sm[(i, 0)] = self.y[(i, 0)];
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

        // 2. Advance + coupling.
        self.a.apply(state.as_mat(), self.y.as_mut(), false);
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

        // (Tape) pre-reset potentials, before the threshold mutates them.
        if let Some(v_pre) = v_pre.as_mut() {
            for b in 0..batch {
                for j in 0..n {
                    v_pre[(j, b)] = self.y[(j, b)];
                }
            }
        }

        // 3 + 4. Threshold + subtractive reset.
        for b in 0..batch {
            for j in 0..n {
                let fired = self.y[(j, b)] >= self.theta;
                if fired {
                    self.y[(j, b)] -= self.theta;
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
