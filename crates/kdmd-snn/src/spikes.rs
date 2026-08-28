//! Spike representations.
//!
//! Two canonical forms, chosen per code path:
//!
//! - [`SpikeVec`] — sorted active-neuron indices, the **inference**
//!   representation. At typical SNN activity (1–10 %), computing the synaptic
//!   drive `W·s` by accumulating the active columns of a column-major `W` does
//!   far less work than a dense matvec (crossover ≈ 25 % activity).
//! - [`SpikeBatch`] — a dense `N × batch` matrix of `{0.0, 1.0}`, the
//!   **training** representation: the BPTT tape needs dense reals, and batched
//!   `W·S` becomes a BLAS-3 matmul.

use faer::{Col, Mat, MatMut, MatRef};

use crate::error::SnnError;

/// Spikes from one layer at one time step (batch = 1): sorted, deduplicated
/// indices of the neurons that fired.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpikeVec {
    active: Vec<u32>,
    n_neurons: usize,
}

impl SpikeVec {
    /// Empty spike vector (no neuron fired) for a layer of `n_neurons`.
    pub fn new(n_neurons: usize) -> Self {
        Self {
            active: Vec::new(),
            n_neurons,
        }
    }

    /// Build from explicit indices. Indices must be strictly increasing and
    /// less than `n_neurons`.
    pub fn from_indices(active: Vec<u32>, n_neurons: usize) -> Result<Self, SnnError> {
        for (pos, &idx) in active.iter().enumerate() {
            if idx as usize >= n_neurons {
                return Err(SnnError::InvalidParameter(format!(
                    "spike index {idx} out of range for {n_neurons} neurons"
                )));
            }
            if pos > 0 && active[pos - 1] >= idx {
                return Err(SnnError::InvalidParameter(
                    "spike indices must be strictly increasing".into(),
                ));
            }
        }
        Ok(Self { active, n_neurons })
    }

    /// Sorted indices of the neurons that fired.
    pub fn active(&self) -> &[u32] {
        &self.active
    }

    /// Number of neurons in the layer.
    pub fn n_neurons(&self) -> usize {
        self.n_neurons
    }

    /// Number of neurons that fired.
    pub fn count(&self) -> usize {
        self.active.len()
    }

    /// Whether no neuron fired.
    pub fn is_empty(&self) -> bool {
        self.active.is_empty()
    }

    /// Fraction of the layer that fired, in `[0, 1]` (0 for an empty layer).
    pub fn activity(&self) -> f64 {
        if self.n_neurons == 0 {
            0.0
        } else {
            self.active.len() as f64 / self.n_neurons as f64
        }
    }

    /// Append a firing neuron. Used by the threshold scan, which visits
    /// neurons in ascending index order; ordering is checked only in debug
    /// builds to keep the hot path branch-free.
    pub fn push(&mut self, idx: u32) {
        debug_assert!((idx as usize) < self.n_neurons, "spike index out of range");
        debug_assert!(
            self.active.last().is_none_or(|&last| last < idx),
            "spike indices must be pushed in strictly increasing order"
        );
        self.active.push(idx);
    }

    /// Remove all spikes, keeping the allocation for reuse across steps.
    pub fn clear(&mut self) {
        self.active.clear();
    }

    /// Densify into an `N`-vector of `{0.0, 1.0}`.
    pub fn to_dense(&self) -> Col<f64> {
        let mut out = Col::zeros(self.n_neurons);
        for &idx in &self.active {
            out[idx as usize] = 1.0;
        }
        out
    }
}

/// Batched spikes: a dense `n_neurons × batch` matrix with entries in
/// `{0.0, 1.0}`.
#[derive(Debug, Clone)]
pub struct SpikeBatch {
    s: Mat<f64>,
}

impl SpikeBatch {
    /// All-zero (no spikes) batch.
    pub fn zeros(n_neurons: usize, batch: usize) -> Result<Self, SnnError> {
        if n_neurons == 0 || batch == 0 {
            return Err(SnnError::InvalidParameter(format!(
                "SpikeBatch dimensions must be nonzero (n_neurons = {n_neurons}, batch = {batch})"
            )));
        }
        Ok(Self {
            s: Mat::zeros(n_neurons, batch),
        })
    }

    /// Wrap an existing matrix, validating that every entry is exactly 0.0 or
    /// 1.0. Cold-path constructor: the validation is O(N·batch).
    pub fn from_mat(s: Mat<f64>) -> Result<Self, SnnError> {
        if s.nrows() == 0 || s.ncols() == 0 {
            return Err(SnnError::InvalidParameter(
                "SpikeBatch dimensions must be nonzero".into(),
            ));
        }
        for j in 0..s.ncols() {
            for i in 0..s.nrows() {
                let v = s[(i, j)];
                if v != 0.0 && v != 1.0 {
                    return Err(SnnError::InvalidParameter(format!(
                        "SpikeBatch entry ({i}, {j}) = {v} is not in {{0.0, 1.0}}"
                    )));
                }
            }
        }
        Ok(Self { s })
    }

    /// Number of neurons (rows).
    pub fn n_neurons(&self) -> usize {
        self.s.nrows()
    }

    /// Number of batch samples (columns).
    pub fn batch(&self) -> usize {
        self.s.ncols()
    }

    /// Dense view, for matmuls.
    pub fn as_mat(&self) -> MatRef<'_, f64> {
        self.s.as_ref()
    }

    /// Mutable dense view. Callers must write only `{0.0, 1.0}`.
    pub fn as_mat_mut(&mut self) -> MatMut<'_, f64> {
        self.s.as_mut()
    }

    /// Sparsify one batch column. Entries are treated as spikes iff nonzero.
    ///
    /// # Panics
    /// Panics if `col >= batch`.
    pub fn column_to_sparse(&self, col: usize) -> SpikeVec {
        assert!(col < self.batch(), "batch column {col} out of range");
        let mut out = SpikeVec::new(self.n_neurons());
        for i in 0..self.s.nrows() {
            if self.s[(i, col)] != 0.0 {
                out.push(i as u32);
            }
        }
        out
    }

    /// Sparsify every batch column.
    pub fn to_sparse(&self) -> Vec<SpikeVec> {
        (0..self.batch())
            .map(|j| self.column_to_sparse(j))
            .collect()
    }

    /// Copy a contiguous column range into a new batch (the training path's
    /// data-parallel chunking uses this to split a minibatch across threads).
    pub fn column_range(&self, start: usize, len: usize) -> Result<SpikeBatch, SnnError> {
        if len == 0 || start + len > self.batch() {
            return Err(SnnError::DimensionMismatch(format!(
                "column range {start}..{} out of batch {}",
                start + len,
                self.batch()
            )));
        }
        let mut out = SpikeBatch::zeros(self.n_neurons(), len)?;
        {
            let mut m = out.as_mat_mut();
            for c in 0..len {
                for i in 0..self.s.nrows() {
                    m[(i, c)] = self.s[(i, start + c)];
                }
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_indices_validates_range_and_order() {
        assert!(SpikeVec::from_indices(vec![0, 2, 5], 6).is_ok());
        assert!(SpikeVec::from_indices(vec![0, 2, 6], 6).is_err()); // out of range
        assert!(SpikeVec::from_indices(vec![2, 2], 6).is_err()); // duplicate
        assert!(SpikeVec::from_indices(vec![3, 1], 6).is_err()); // unsorted
    }

    #[test]
    fn activity_and_counts() {
        let s = SpikeVec::from_indices(vec![1, 3], 8).unwrap();
        assert_eq!(s.count(), 2);
        assert!(!s.is_empty());
        assert_eq!(s.activity(), 0.25);
        assert_eq!(s.n_neurons(), 8);
    }

    #[test]
    fn to_dense_round_trips() {
        let s = SpikeVec::from_indices(vec![0, 4], 5).unwrap();
        let d = s.to_dense();
        assert_eq!(d.nrows(), 5);
        let expected = [1.0, 0.0, 0.0, 0.0, 1.0];
        for (i, &e) in expected.iter().enumerate() {
            assert_eq!(d[i], e);
        }
    }

    #[test]
    fn push_and_clear_reuse() {
        let mut s = SpikeVec::new(4);
        s.push(1);
        s.push(3);
        assert_eq!(s.active(), &[1, 3]);
        s.clear();
        assert!(s.is_empty());
        assert_eq!(s.n_neurons(), 4);
    }

    #[test]
    fn batch_round_trips_through_sparse() {
        let mut b = SpikeBatch::zeros(4, 2).unwrap();
        b.as_mat_mut()[(1, 0)] = 1.0;
        b.as_mat_mut()[(3, 0)] = 1.0;
        b.as_mat_mut()[(0, 1)] = 1.0;
        let sparse = b.to_sparse();
        assert_eq!(sparse[0].active(), &[1, 3]);
        assert_eq!(sparse[1].active(), &[0]);
    }

    #[test]
    fn from_mat_rejects_non_binary_entries() {
        let mut m = Mat::<f64>::zeros(2, 2);
        m[(0, 0)] = 0.5;
        assert!(SpikeBatch::from_mat(m).is_err());

        let mut ok = Mat::<f64>::zeros(2, 2);
        ok[(1, 1)] = 1.0;
        assert!(SpikeBatch::from_mat(ok).is_ok());
    }

    #[test]
    fn zero_dimensions_are_rejected() {
        assert!(SpikeBatch::zeros(0, 1).is_err());
        assert!(SpikeBatch::zeros(1, 0).is_err());
    }
}
