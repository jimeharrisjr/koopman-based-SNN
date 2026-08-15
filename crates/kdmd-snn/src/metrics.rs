//! Evaluation metrics: firing statistics, trajectory error, spike-train
//! agreement.
//!
//! These back the phase exit gates: reference-vs-fast-path equivalence uses
//! [`trajectory_rmse`] and [`spike_coincidence`], and rate sanity checks use
//! [`firing_rates`].

use faer::{Mat, MatRef};

use crate::error::SnnError;
use crate::spikes::SpikeBatch;
use crate::util;

/// Mean firing rate per neuron per batch sample, in spikes per time unit:
/// `count / (t_steps · dt)`.
pub fn firing_rates(spikes: &[SpikeBatch], dt: f64) -> Result<Mat<f64>, SnnError> {
    let first = spikes.first().ok_or_else(|| {
        SnnError::InvalidParameter("firing_rates needs at least one time step".into())
    })?;
    if !util::is_positive(dt) {
        return Err(SnnError::InvalidParameter(format!(
            "dt must be positive (dt = {dt})"
        )));
    }
    let (n, batch) = (first.n_neurons(), first.batch());
    let mut counts = Mat::<f64>::zeros(n, batch);
    for (t, s) in spikes.iter().enumerate() {
        if s.n_neurons() != n || s.batch() != batch {
            return Err(SnnError::DimensionMismatch(format!(
                "spike batch at step {t} has shape ({}, {}), expected ({n}, {batch})",
                s.n_neurons(),
                s.batch()
            )));
        }
        for j in 0..batch {
            for i in 0..n {
                counts[(i, j)] += s.as_mat()[(i, j)];
            }
        }
    }
    let duration = spikes.len() as f64 * dt;
    for j in 0..batch {
        for i in 0..n {
            counts[(i, j)] /= duration;
        }
    }
    Ok(counts)
}

/// Root-mean-square difference between two equally shaped matrices (e.g. two
/// state trajectories laid out as `state_dim × t_steps`).
pub fn trajectory_rmse(a: MatRef<'_, f64>, b: MatRef<'_, f64>) -> Result<f64, SnnError> {
    if a.nrows() != b.nrows() || a.ncols() != b.ncols() {
        return Err(SnnError::DimensionMismatch(format!(
            "shapes ({}, {}) vs ({}, {})",
            a.nrows(),
            a.ncols(),
            b.nrows(),
            b.ncols()
        )));
    }
    if a.nrows() == 0 || a.ncols() == 0 {
        return Err(SnnError::InvalidParameter(
            "trajectory_rmse needs nonempty matrices".into(),
        ));
    }
    let mut sum_sq = 0.0;
    for j in 0..a.ncols() {
        for i in 0..a.nrows() {
            let d = a[(i, j)] - b[(i, j)];
            sum_sq += d * d;
        }
    }
    Ok((sum_sq / (a.nrows() * a.ncols()) as f64).sqrt())
}

/// Symmetric spike-train agreement in `[0, 1]`.
///
/// A spike in one train is *matched* if the other train has a spike on the
/// same neuron and batch sample within `±tol_steps` time steps. The score is
/// the mean of the two directional match fractions (precision/recall style);
/// identical trains score exactly 1.0, and two trains with no coincident
/// spikes score 0.0. Two empty trains score 1.0 by convention.
pub fn spike_coincidence(
    a: &[SpikeBatch],
    b: &[SpikeBatch],
    tol_steps: usize,
) -> Result<f64, SnnError> {
    if a.len() != b.len() {
        return Err(SnnError::DimensionMismatch(format!(
            "trains have different lengths: {} vs {}",
            a.len(),
            b.len()
        )));
    }
    let first = a.first().ok_or_else(|| {
        SnnError::InvalidParameter("spike_coincidence needs at least one time step".into())
    })?;
    let (n, batch) = (first.n_neurons(), first.batch());

    // Per (neuron, batch) sorted spike-step lists for both trains.
    let collect = |train: &[SpikeBatch]| -> Result<Vec<Vec<usize>>, SnnError> {
        let mut lists = vec![Vec::new(); n * batch];
        for (t, s) in train.iter().enumerate() {
            if s.n_neurons() != n || s.batch() != batch {
                return Err(SnnError::DimensionMismatch(format!(
                    "spike batch at step {t} has shape ({}, {}), expected ({n}, {batch})",
                    s.n_neurons(),
                    s.batch()
                )));
            }
            for j in 0..batch {
                for i in 0..n {
                    if s.as_mat()[(i, j)] != 0.0 {
                        lists[j * n + i].push(t);
                    }
                }
            }
        }
        Ok(lists)
    };
    let la = collect(a)?;
    let lb = collect(b)?;

    // Fraction of spikes in `from` that have a partner in `to` within tol.
    let matched_fraction = |from: &[Vec<usize>], to: &[Vec<usize>]| -> (usize, usize) {
        let (mut matched, mut total) = (0usize, 0usize);
        for (list_f, list_t) in from.iter().zip(to) {
            total += list_f.len();
            for &t in list_f {
                let hit = match list_t.binary_search(&t) {
                    Ok(_) => true,
                    Err(ins) => {
                        let before_ok = ins > 0 && t - list_t[ins - 1] <= tol_steps;
                        let after_ok = ins < list_t.len() && list_t[ins] - t <= tol_steps;
                        before_ok || after_ok
                    }
                };
                if hit {
                    matched += 1;
                }
            }
        }
        (matched, total)
    };

    let (ma, ta) = matched_fraction(&la, &lb);
    let (mb, tb) = matched_fraction(&lb, &la);
    if ta == 0 && tb == 0 {
        return Ok(1.0);
    }
    let frac_a = if ta == 0 { 1.0 } else { ma as f64 / ta as f64 };
    let frac_b = if tb == 0 { 1.0 } else { mb as f64 / tb as f64 };
    Ok(0.5 * (frac_a + frac_b))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn train_from(steps: &[&[u32]], n: usize) -> Vec<SpikeBatch> {
        steps
            .iter()
            .map(|active| {
                let mut b = SpikeBatch::zeros(n, 1).unwrap();
                for &i in *active {
                    b.as_mat_mut()[(i as usize, 0)] = 1.0;
                }
                b
            })
            .collect()
    }

    #[test]
    fn firing_rates_count_and_normalize() {
        // Neuron 0 fires every step, neuron 1 never, over 4 steps of dt = 0.5.
        let train = train_from(&[&[0], &[0], &[0], &[0]], 2);
        let rates = firing_rates(&train, 0.5).unwrap();
        assert_eq!(rates[(0, 0)], 2.0); // 4 spikes / 2.0 time units
        assert_eq!(rates[(1, 0)], 0.0);
    }

    #[test]
    fn rmse_of_identical_trajectories_is_zero() {
        let mut a = Mat::<f64>::zeros(2, 3);
        a[(0, 0)] = 1.0;
        a[(1, 2)] = -2.0;
        assert_eq!(trajectory_rmse(a.as_ref(), a.as_ref()).unwrap(), 0.0);
    }

    #[test]
    fn rmse_matches_hand_computation() {
        let a = Mat::<f64>::zeros(1, 2);
        let mut b = Mat::<f64>::zeros(1, 2);
        b[(0, 0)] = 3.0;
        b[(0, 1)] = 4.0;
        // sqrt((9 + 16) / 2)
        let expected = (25.0f64 / 2.0).sqrt();
        assert!((trajectory_rmse(a.as_ref(), b.as_ref()).unwrap() - expected).abs() < 1e-15);
    }

    #[test]
    fn coincidence_identical_trains_is_one() {
        let t = train_from(&[&[0], &[], &[1], &[0, 1]], 2);
        assert_eq!(spike_coincidence(&t, &t, 0).unwrap(), 1.0);
    }

    #[test]
    fn coincidence_disjoint_trains_is_zero() {
        let a = train_from(&[&[0], &[], &[], &[]], 1);
        let b = train_from(&[&[], &[], &[], &[0]], 1);
        assert_eq!(spike_coincidence(&a, &b, 0).unwrap(), 0.0);
    }

    #[test]
    fn coincidence_tolerance_window_matches_shifted_spikes() {
        let a = train_from(&[&[0], &[], &[], &[]], 1);
        let b = train_from(&[&[], &[0], &[], &[]], 1);
        assert_eq!(spike_coincidence(&a, &b, 0).unwrap(), 0.0);
        assert_eq!(spike_coincidence(&a, &b, 1).unwrap(), 1.0);
    }

    #[test]
    fn coincidence_of_empty_trains_is_one() {
        let a = train_from(&[&[], &[]], 2);
        let b = train_from(&[&[], &[]], 2);
        assert_eq!(spike_coincidence(&a, &b, 0).unwrap(), 1.0);
    }

    #[test]
    fn coincidence_requires_same_neuron() {
        let a = train_from(&[&[0], &[]], 2);
        let b = train_from(&[&[1], &[]], 2);
        assert_eq!(spike_coincidence(&a, &b, 1).unwrap(), 0.0);
    }
}
