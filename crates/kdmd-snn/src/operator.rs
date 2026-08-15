//! Discrete-time state-advance operators: `x_{t+1} = A·x_t (+ inputs)`.
//!
//! The [`Operator`] enum stores `A` in the cheapest structure the dynamics
//! permit. For a homogeneous linear neuron model (LIF, adLIF) the layer
//! operator is exactly `A_local ⊗ I_N` — a k×k block replicated per neuron —
//! so applying it is O(k²·N·batch), not O((kN)²·batch). Dense is the general
//! fallback; LowRank carries the reduced-order path.
//!
//! Extraction from `koopman-dmd` results happens **once, at identification
//! time**: `DmdResult.a_matrix` is complex (`Vec<Vec<C64>>`) and never touches
//! the stepping path. A correctly identified operator from real data is real
//! up to conjugate-pair roundoff, which [`dense_from_dmd`] enforces.

use faer::{Mat, MatMut, MatRef};
use koopman_dmd::DmdResult;

use crate::error::SnnError;

/// Discrete-time state-advance operator for one layer.
#[derive(Debug, Clone)]
pub enum Operator {
    /// Full dense `A`: `n × n`.
    Dense(Mat<f64>),

    /// `A = A_local ⊗ I_n` in variable-major ordering: every neuron shares one
    /// `k × k` local matrix. Exact for homogeneous linear neuron models.
    PerVariable {
        /// k × k local propagator.
        a_local: Mat<f64>,
        n_neurons: usize,
    },

    /// Heterogeneous per-neuron blocks: neuron `j` has its own `k × k` matrix.
    /// `blocks` is `(k·k) × n` with entry `(p·k + q, j) = A_j[(p, q)]`.
    PerNeuron {
        blocks: Mat<f64>,
        n_state_vars: usize,
    },

    /// Rank-r factorization `A ≈ P · A_r · Pᵀ` with `P` orthonormal
    /// (`n × r`), `A_r` the reduced operator (`r × r`).
    LowRank { p: Mat<f64>, a_r: Mat<f64> },
}

impl Operator {
    /// Dimension of the state this operator acts on.
    pub fn state_dim(&self) -> usize {
        match self {
            Operator::Dense(a) => a.nrows(),
            Operator::PerVariable { a_local, n_neurons } => a_local.nrows() * n_neurons,
            Operator::PerNeuron {
                blocks,
                n_state_vars,
            } => n_state_vars * blocks.ncols(),
            Operator::LowRank { p, .. } => p.nrows(),
        }
    }

    /// `y = A·x` (or `y += A·x` with `accumulate`), into a caller-owned
    /// buffer. `x` and `y` are `state_dim × batch`.
    ///
    /// # Panics
    /// Panics if the shapes disagree with `state_dim`.
    pub fn apply(&self, x: MatRef<'_, f64>, y: MatMut<'_, f64>, accumulate: bool) {
        self.apply_impl(x, y, accumulate, false);
    }

    /// `y = Aᵀ·x` (or `y += Aᵀ·x`): the exact Jacobian-transpose product used
    /// by the BPTT backward pass.
    ///
    /// # Panics
    /// Panics if the shapes disagree with `state_dim`.
    pub fn apply_transpose(&self, x: MatRef<'_, f64>, y: MatMut<'_, f64>, accumulate: bool) {
        self.apply_impl(x, y, accumulate, true);
    }

    fn apply_impl(
        &self,
        x: MatRef<'_, f64>,
        mut y: MatMut<'_, f64>,
        accumulate: bool,
        transpose: bool,
    ) {
        let n = self.state_dim();
        assert_eq!(x.nrows(), n, "x rows != state_dim");
        assert_eq!(y.nrows(), n, "y rows != state_dim");
        assert_eq!(x.ncols(), y.ncols(), "x and y batch sizes differ");
        let batch = x.ncols();

        if !accumulate {
            for b in 0..batch {
                for i in 0..n {
                    y[(i, b)] = 0.0;
                }
            }
        }

        match self {
            Operator::Dense(a) => {
                for b in 0..batch {
                    for i in 0..n {
                        let mut acc = 0.0;
                        for j in 0..n {
                            let a_ij = if transpose { a[(j, i)] } else { a[(i, j)] };
                            acc += a_ij * x[(j, b)];
                        }
                        y[(i, b)] += acc;
                    }
                }
            }
            Operator::PerVariable { a_local, n_neurons } => {
                let k = a_local.nrows();
                let nn = *n_neurons;
                for b in 0..batch {
                    for p in 0..k {
                        for q in 0..k {
                            let coef = if transpose {
                                a_local[(q, p)]
                            } else {
                                a_local[(p, q)]
                            };
                            if coef == 0.0 {
                                continue;
                            }
                            for j in 0..nn {
                                y[(p * nn + j, b)] += coef * x[(q * nn + j, b)];
                            }
                        }
                    }
                }
            }
            Operator::PerNeuron {
                blocks,
                n_state_vars,
            } => {
                let k = *n_state_vars;
                let nn = blocks.ncols();
                for b in 0..batch {
                    for j in 0..nn {
                        for p in 0..k {
                            let mut acc = 0.0;
                            for q in 0..k {
                                let idx = if transpose { q * k + p } else { p * k + q };
                                acc += blocks[(idx, j)] * x[(q * nn + j, b)];
                            }
                            y[(p * nn + j, b)] += acc;
                        }
                    }
                }
            }
            Operator::LowRank { p, a_r } => {
                let r = a_r.nrows();
                // z = Pᵀ x; z2 = A_r z (or A_rᵀ z); y += P z2.
                // Allocates two r×batch temporaries — reduced-path scratch
                // buffers arrive with the Phase 5 hot-path work.
                let mut z = Mat::<f64>::zeros(r, batch);
                for b in 0..batch {
                    for i in 0..r {
                        let mut acc = 0.0;
                        for j in 0..n {
                            acc += p[(j, i)] * x[(j, b)];
                        }
                        z[(i, b)] = acc;
                    }
                }
                let mut z2 = Mat::<f64>::zeros(r, batch);
                for b in 0..batch {
                    for i in 0..r {
                        let mut acc = 0.0;
                        for j in 0..r {
                            let a_ij = if transpose { a_r[(j, i)] } else { a_r[(i, j)] };
                            acc += a_ij * z[(j, b)];
                        }
                        z2[(i, b)] = acc;
                    }
                }
                for b in 0..batch {
                    for i in 0..n {
                        let mut acc = 0.0;
                        for j in 0..r {
                            acc += p[(i, j)] * z2[(j, b)];
                        }
                        y[(i, b)] += acc;
                    }
                }
            }
        }
    }
}

/// Copy `DmdResult.a_matrix` into a real dense [`Operator`].
///
/// Errors if any imaginary part exceeds `max_imag · max(1, max|Re|)`: an
/// operator identified from real data is real up to conjugate-pair roundoff,
/// so a large imaginary residue means the fit is unusable (unpaired complex
/// modes, rank cut through a conjugate pair, or corrupted data).
pub fn dense_from_dmd(res: &DmdResult, max_imag: f64) -> Result<Operator, SnnError> {
    let n = res.data_dim.0;
    let mut max_abs_re: f64 = 0.0;
    let mut max_abs_im: f64 = 0.0;
    for row in &res.a_matrix {
        for c in row {
            max_abs_re = max_abs_re.max(c.re.abs());
            max_abs_im = max_abs_im.max(c.im.abs());
        }
    }
    let scale = max_abs_re.max(1.0);
    if max_abs_im > max_imag * scale {
        return Err(SnnError::NumericalError(format!(
            "identified A is not real: max |Im| = {max_abs_im:.3e} \
             exceeds {max_imag:.1e} · {scale:.3e} — check rank (conjugate pairs \
             must not be split) and data quality"
        )));
    }
    let mut a = Mat::<f64>::zeros(n, n);
    for (i, row) in res.a_matrix.iter().enumerate() {
        for (j, c) in row.iter().enumerate() {
            a[(i, j)] = c.re;
        }
    }
    Ok(Operator::Dense(a))
}

/// Build the reduced-order [`Operator::LowRank`] from a DMD result:
/// `P = svd.u` (orthonormal, `n × r`), `A_r = Ã` (real by construction in
/// `koopman-dmd`'s `dmd()`).
pub fn low_rank_from_dmd(res: &DmdResult) -> Result<Operator, SnnError> {
    let r = res.rank;
    let p = res.svd.u.clone();
    if p.ncols() != r {
        return Err(SnnError::DimensionMismatch(format!(
            "svd.u has {} columns, expected rank {r}",
            p.ncols()
        )));
    }
    let mut a_r = Mat::<f64>::zeros(r, r);
    for (i, row) in res.a_tilde.iter().enumerate() {
        for (j, c) in row.iter().enumerate() {
            a_r[(i, j)] = c.re;
        }
    }
    Ok(Operator::LowRank { p, a_r })
}

/// Detect `A_local ⊗ I_N` structure in a dense operator and compress it to
/// [`Operator::PerVariable`].
///
/// `a_local[(p, q)]` is estimated as the mean of the `(p·N + j, q·N + j)`
/// diagonal-block entries; the fit is accepted only if every such entry
/// deviates from the shared value by at most `tol`, and every off-pattern
/// entry is at most `tol` in magnitude. Returns an error (not a silent
/// fallback) when the structure is absent.
pub fn per_variable_from_dense(
    a: MatRef<'_, f64>,
    n_neurons: usize,
    tol: f64,
) -> Result<Operator, SnnError> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(SnnError::DimensionMismatch(format!(
            "operator must be square, got {n} × {}",
            a.ncols()
        )));
    }
    if n_neurons == 0 || n % n_neurons != 0 {
        return Err(SnnError::InvalidParameter(format!(
            "state dim {n} is not a multiple of n_neurons {n_neurons}"
        )));
    }
    let k = n / n_neurons;
    let mut a_local = Mat::<f64>::zeros(k, k);
    for p in 0..k {
        for q in 0..k {
            let mut mean = 0.0;
            for j in 0..n_neurons {
                mean += a[(p * n_neurons + j, q * n_neurons + j)];
            }
            mean /= n_neurons as f64;
            a_local[(p, q)] = mean;
        }
    }
    // Verify the whole matrix against the compressed structure.
    for i in 0..n {
        for j in 0..n {
            let (p, jn) = (i / n_neurons, i % n_neurons);
            let (q, jn2) = (j / n_neurons, j % n_neurons);
            let expected = if jn == jn2 { a_local[(p, q)] } else { 0.0 };
            let dev = (a[(i, j)] - expected).abs();
            if dev > tol {
                return Err(SnnError::NumericalError(format!(
                    "dense operator deviates from A_local ⊗ I structure at \
                     ({i}, {j}): |{} − {expected}| = {dev:.3e} > {tol:.1e}",
                    a[(i, j)]
                )));
            }
        }
    }
    Ok(Operator::PerVariable { a_local, n_neurons })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn mat_from(rows: &[&[f64]]) -> Mat<f64> {
        let mut m = Mat::<f64>::zeros(rows.len(), rows[0].len());
        for (i, row) in rows.iter().enumerate() {
            for (j, &v) in row.iter().enumerate() {
                m[(i, j)] = v;
            }
        }
        m
    }

    /// Dense reference for `A_local ⊗ I_N` in variable-major ordering.
    fn kron_dense(a_local: MatRef<'_, f64>, n: usize) -> Mat<f64> {
        let k = a_local.nrows();
        let mut a = Mat::<f64>::zeros(k * n, k * n);
        for p in 0..k {
            for q in 0..k {
                for j in 0..n {
                    a[(p * n + j, q * n + j)] = a_local[(p, q)];
                }
            }
        }
        a
    }

    #[test]
    fn dense_apply_matches_hand_matmul() {
        let a = mat_from(&[&[1.0, 2.0], &[3.0, 4.0]]);
        let x = mat_from(&[&[5.0], &[6.0]]);
        let mut y = Mat::<f64>::zeros(2, 1);
        Operator::Dense(a.clone()).apply(x.as_ref(), y.as_mut(), false);
        assert_eq!(y[(0, 0)], 17.0);
        assert_eq!(y[(1, 0)], 39.0);

        // Accumulate adds on top.
        Operator::Dense(a.clone()).apply(x.as_ref(), y.as_mut(), true);
        assert_eq!(y[(0, 0)], 34.0);

        let mut yt = Mat::<f64>::zeros(2, 1);
        Operator::Dense(a).apply_transpose(x.as_ref(), yt.as_mut(), false);
        assert_eq!(yt[(0, 0)], 1.0 * 5.0 + 3.0 * 6.0);
        assert_eq!(yt[(1, 0)], 2.0 * 5.0 + 4.0 * 6.0);
    }

    #[test]
    fn per_variable_apply_matches_kron_dense() {
        let a_local = mat_from(&[&[0.9, 0.05], &[0.0, 0.8]]);
        let n = 4;
        let dense = Operator::Dense(kron_dense(a_local.as_ref(), n));
        let pv = Operator::PerVariable {
            a_local,
            n_neurons: n,
        };

        let mut x = Mat::<f64>::zeros(2 * n, 3);
        for b in 0..3 {
            for i in 0..2 * n {
                x[(i, b)] = (i as f64 + 1.0) * (b as f64 + 1.0) * 0.1;
            }
        }
        let mut y1 = Mat::<f64>::zeros(2 * n, 3);
        let mut y2 = Mat::<f64>::zeros(2 * n, 3);
        dense.apply(x.as_ref(), y1.as_mut(), false);
        pv.apply(x.as_ref(), y2.as_mut(), false);
        for b in 0..3 {
            for i in 0..2 * n {
                assert_relative_eq!(y1[(i, b)], y2[(i, b)], max_relative = 1e-14);
            }
        }

        let mut t1 = Mat::<f64>::zeros(2 * n, 3);
        let mut t2 = Mat::<f64>::zeros(2 * n, 3);
        dense.apply_transpose(x.as_ref(), t1.as_mut(), false);
        pv.apply_transpose(x.as_ref(), t2.as_mut(), false);
        for b in 0..3 {
            for i in 0..2 * n {
                assert_relative_eq!(t1[(i, b)], t2[(i, b)], max_relative = 1e-14);
            }
        }
    }

    #[test]
    fn per_neuron_apply_matches_blockwise_dense() {
        // Two neurons with different 2×2 blocks.
        let b0 = mat_from(&[&[0.9, 0.1], &[0.0, 0.8]]);
        let b1 = mat_from(&[&[0.7, 0.2], &[0.1, 0.6]]);
        let n = 2;
        let k = 2;
        let mut blocks = Mat::<f64>::zeros(k * k, n);
        for p in 0..k {
            for q in 0..k {
                blocks[(p * k + q, 0)] = b0[(p, q)];
                blocks[(p * k + q, 1)] = b1[(p, q)];
            }
        }
        let mut dense = Mat::<f64>::zeros(k * n, k * n);
        for p in 0..k {
            for q in 0..k {
                dense[(p * n, q * n)] = b0[(p, q)];
                dense[(p * n + 1, q * n + 1)] = b1[(p, q)];
            }
        }

        let pn = Operator::PerNeuron {
            blocks,
            n_state_vars: k,
        };
        let dn = Operator::Dense(dense);
        let x = mat_from(&[&[1.0], &[2.0], &[3.0], &[4.0]]);
        let mut y1 = Mat::<f64>::zeros(4, 1);
        let mut y2 = Mat::<f64>::zeros(4, 1);
        pn.apply(x.as_ref(), y1.as_mut(), false);
        dn.apply(x.as_ref(), y2.as_mut(), false);
        for i in 0..4 {
            assert_relative_eq!(y1[(i, 0)], y2[(i, 0)], max_relative = 1e-14);
        }

        let mut t1 = Mat::<f64>::zeros(4, 1);
        let mut t2 = Mat::<f64>::zeros(4, 1);
        pn.apply_transpose(x.as_ref(), t1.as_mut(), false);
        dn.apply_transpose(x.as_ref(), t2.as_mut(), false);
        for i in 0..4 {
            assert_relative_eq!(t1[(i, 0)], t2[(i, 0)], max_relative = 1e-14);
        }
    }

    #[test]
    fn low_rank_apply_matches_factored_product() {
        // P = first two columns of I₄ (orthonormal), so P·A_r·Pᵀ acts on the
        // first two coordinates and zeroes the rest.
        let mut p = Mat::<f64>::zeros(4, 2);
        p[(0, 0)] = 1.0;
        p[(1, 1)] = 1.0;
        let a_r = mat_from(&[&[0.5, 0.25], &[0.0, 0.5]]);
        let lr = Operator::LowRank { p, a_r };

        let x = mat_from(&[&[2.0], &[4.0], &[8.0], &[16.0]]);
        let mut y = Mat::<f64>::zeros(4, 1);
        lr.apply(x.as_ref(), y.as_mut(), false);
        assert_eq!(y[(0, 0)], 0.5 * 2.0 + 0.25 * 4.0);
        assert_eq!(y[(1, 0)], 0.5 * 4.0);
        assert_eq!(y[(2, 0)], 0.0);
        assert_eq!(y[(3, 0)], 0.0);
    }

    #[test]
    fn per_variable_detection_round_trips() {
        let a_local = mat_from(&[&[0.99, 0.048], &[0.0, 0.98]]);
        let dense = kron_dense(a_local.as_ref(), 5);
        let op = per_variable_from_dense(dense.as_ref(), 5, 1e-12).unwrap();
        match op {
            Operator::PerVariable {
                a_local: got,
                n_neurons,
            } => {
                assert_eq!(n_neurons, 5);
                for p in 0..2 {
                    for q in 0..2 {
                        assert_relative_eq!(got[(p, q)], a_local[(p, q)], max_relative = 1e-14);
                    }
                }
            }
            other => panic!("expected PerVariable, got {other:?}"),
        }
    }

    #[test]
    fn per_variable_detection_rejects_unstructured_matrices() {
        let mut dense = kron_dense(mat_from(&[&[0.9, 0.0], &[0.0, 0.8]]).as_ref(), 3);
        dense[(0, 1)] = 0.5; // cross-neuron coupling breaks the pattern
        assert!(per_variable_from_dense(dense.as_ref(), 3, 1e-9).is_err());
        // Wrong divisor.
        assert!(per_variable_from_dense(dense.as_ref(), 4, 1e-9).is_err());
    }
}
