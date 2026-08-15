//! Layer state storage.

use faer::{Mat, MatMut, MatRef};

use crate::error::SnnError;

/// State of one layer: a `(k·N) × batch` matrix in **variable-major** order.
///
/// For a layer of `N` neurons with a `k`-variable neuron model (LIF: `k = 2`,
/// membrane potential and synaptic current; adaptive LIF: `k = 3`, adding an
/// adaptation variable), rows `[0, N)` hold state variable 0 for every neuron,
/// rows `[N, 2N)` variable 1, and so on. **Variable 0 is the membrane
/// potential by convention.** Batch samples are columns; batch = 1 inference
/// uses the same layout with a single column.
///
/// Variable-major ordering keeps each variable's values for all neurons in one
/// contiguous row block, so per-variable operators (`A = A_local ⊗ I_N`) apply
/// as contiguous passes and "the potential block" is a zero-copy row view.
///
/// The state carries no history; the training tape owns per-step saved
/// tensors.
#[derive(Debug, Clone)]
pub struct LayerState {
    x: Mat<f64>,
    n_neurons: usize,
    n_state_vars: usize,
}

impl LayerState {
    /// Create a zero-initialized state for `n_neurons` neurons with
    /// `n_state_vars` variables each and `batch` samples.
    pub fn zeros(n_neurons: usize, n_state_vars: usize, batch: usize) -> Result<Self, SnnError> {
        if n_neurons == 0 || n_state_vars == 0 || batch == 0 {
            return Err(SnnError::InvalidParameter(format!(
                "LayerState dimensions must be nonzero \
                 (n_neurons = {n_neurons}, n_state_vars = {n_state_vars}, batch = {batch})"
            )));
        }
        Ok(Self {
            x: Mat::zeros(n_neurons * n_state_vars, batch),
            n_neurons,
            n_state_vars,
        })
    }

    /// Number of neurons in the layer.
    pub fn n_neurons(&self) -> usize {
        self.n_neurons
    }

    /// Number of state variables per neuron (`k`).
    pub fn n_state_vars(&self) -> usize {
        self.n_state_vars
    }

    /// Number of batch samples (columns).
    pub fn batch(&self) -> usize {
        self.x.ncols()
    }

    /// Total state dimension `k·N` (rows).
    pub fn state_dim(&self) -> usize {
        self.x.nrows()
    }

    /// Full state block, for operator application.
    pub fn as_mat(&self) -> MatRef<'_, f64> {
        self.x.as_ref()
    }

    /// Mutable full state block.
    pub fn as_mat_mut(&mut self) -> MatMut<'_, f64> {
        self.x.as_mut()
    }

    /// View of state variable `var`: rows `[var·N, (var+1)·N)`.
    ///
    /// # Panics
    /// Panics if `var >= n_state_vars`.
    pub fn var(&self, var: usize) -> MatRef<'_, f64> {
        assert!(
            var < self.n_state_vars,
            "state variable index {var} out of range (n_state_vars = {})",
            self.n_state_vars
        );
        self.x
            .as_ref()
            .subrows(var * self.n_neurons, self.n_neurons)
    }

    /// Mutable view of state variable `var`.
    ///
    /// # Panics
    /// Panics if `var >= n_state_vars`.
    pub fn var_mut(&mut self, var: usize) -> MatMut<'_, f64> {
        assert!(
            var < self.n_state_vars,
            "state variable index {var} out of range (n_state_vars = {})",
            self.n_state_vars
        );
        self.x
            .as_mut()
            .subrows_mut(var * self.n_neurons, self.n_neurons)
    }

    /// Membrane potentials (state variable 0).
    pub fn potentials(&self) -> MatRef<'_, f64> {
        self.var(0)
    }

    /// Mutable membrane potentials (state variable 0).
    pub fn potentials_mut(&mut self) -> MatMut<'_, f64> {
        self.var_mut(0)
    }

    /// Set every state entry to `value`.
    pub fn fill(&mut self, value: f64) {
        for j in 0..self.x.ncols() {
            for i in 0..self.x.nrows() {
                self.x[(i, j)] = value;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zeros_has_requested_shape_and_is_zero() {
        let s = LayerState::zeros(4, 2, 3).unwrap();
        assert_eq!(s.n_neurons(), 4);
        assert_eq!(s.n_state_vars(), 2);
        assert_eq!(s.batch(), 3);
        assert_eq!(s.state_dim(), 8);
        for j in 0..3 {
            for i in 0..8 {
                assert_eq!(s.as_mat()[(i, j)], 0.0);
            }
        }
    }

    #[test]
    fn zero_dimensions_are_rejected() {
        assert!(LayerState::zeros(0, 2, 1).is_err());
        assert!(LayerState::zeros(4, 0, 1).is_err());
        assert!(LayerState::zeros(4, 2, 0).is_err());
    }

    #[test]
    fn var_views_map_to_variable_major_rows() {
        let mut s = LayerState::zeros(3, 2, 2).unwrap();
        // Write potentials (var 0) and currents (var 1) through the views.
        for n in 0..3 {
            for b in 0..2 {
                s.var_mut(0)[(n, b)] = 10.0 + n as f64;
                s.var_mut(1)[(n, b)] = 20.0 + n as f64;
            }
        }
        // Variable-major layout: var 0 in rows [0, 3), var 1 in rows [3, 6).
        let full = s.as_mat();
        for b in 0..2 {
            for n in 0..3 {
                assert_eq!(full[(n, b)], 10.0 + n as f64);
                assert_eq!(full[(3 + n, b)], 20.0 + n as f64);
            }
        }
    }

    #[test]
    fn potentials_is_var_zero() {
        let mut s = LayerState::zeros(2, 3, 1).unwrap();
        s.var_mut(0)[(1, 0)] = 7.5;
        assert_eq!(s.potentials()[(1, 0)], 7.5);
    }

    #[test]
    fn fill_sets_every_entry() {
        let mut s = LayerState::zeros(2, 2, 2).unwrap();
        s.fill(-1.25);
        for j in 0..2 {
            for i in 0..4 {
                assert_eq!(s.as_mat()[(i, j)], -1.25);
            }
        }
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn var_out_of_range_panics() {
        let s = LayerState::zeros(2, 2, 1).unwrap();
        let _ = s.var(2);
    }
}
