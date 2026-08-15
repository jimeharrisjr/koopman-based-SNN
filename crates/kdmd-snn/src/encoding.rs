//! Input encoders: turning analog values into spike trains or currents.

use faer::{Mat, MatRef};
use rand::Rng;

use crate::error::SnnError;
use crate::spikes::SpikeBatch;
use crate::util;

/// Rate (Poisson) encoder: each input value is a firing rate, realized as an
/// independent Bernoulli draw per time bin with `p = min(rate·dt, 1)`.
///
/// The Bernoulli-per-bin scheme is the standard discrete-time approximation
/// of a Poisson process; it undercounts slightly once `rate·dt` approaches 1,
/// so keep `rate·dt ≪ 1` for faithful rates.
#[derive(Debug, Clone)]
pub struct PoissonEncoder {
    /// Time bin width, in the same time unit as the rates' denominator.
    pub dt: f64,
}

impl PoissonEncoder {
    pub fn new(dt: f64) -> Result<Self, SnnError> {
        if !util::is_positive(dt) {
            return Err(SnnError::InvalidParameter(format!(
                "dt must be positive (dt = {dt})"
            )));
        }
        Ok(Self { dt })
    }

    /// Encode `rates` (`n_inputs × batch`, non-negative) into `t_steps` spike
    /// batches.
    pub fn encode<R: Rng>(
        &self,
        rates: MatRef<'_, f64>,
        t_steps: usize,
        rng: &mut R,
    ) -> Result<Vec<SpikeBatch>, SnnError> {
        for j in 0..rates.ncols() {
            for i in 0..rates.nrows() {
                if rates[(i, j)] < 0.0 {
                    return Err(SnnError::InvalidParameter(format!(
                        "rate ({i}, {j}) = {} is negative",
                        rates[(i, j)]
                    )));
                }
            }
        }
        let mut out = Vec::with_capacity(t_steps);
        for _ in 0..t_steps {
            let mut batch = SpikeBatch::zeros(rates.nrows(), rates.ncols())?;
            for j in 0..rates.ncols() {
                for i in 0..rates.nrows() {
                    let p = (rates[(i, j)] * self.dt).min(1.0);
                    if rng.random::<f64>() < p {
                        batch.as_mat_mut()[(i, j)] = 1.0;
                    }
                }
            }
            out.push(batch);
        }
        Ok(out)
    }
}

/// Constant current injection: an `n × batch` drive matrix holding `value`
/// everywhere, for direct-current protocols (f–I curves, step responses).
pub fn constant_current(n: usize, batch: usize, value: f64) -> Result<Mat<f64>, SnnError> {
    if n == 0 || batch == 0 {
        return Err(SnnError::InvalidParameter(
            "current dimensions must be nonzero".into(),
        ));
    }
    let mut m = Mat::zeros(n, batch);
    for j in 0..batch {
        for i in 0..n {
            m[(i, j)] = value;
        }
    }
    Ok(m)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn empirical_rate_tracks_requested_rate() {
        let enc = PoissonEncoder::new(0.001).unwrap(); // 1 ms bins, rates in Hz
        let mut rates = Mat::<f64>::zeros(2, 1);
        rates[(0, 0)] = 100.0; // 100 Hz
        rates[(1, 0)] = 10.0;
        let t_steps = 50_000; // 50 s
        let mut rng = StdRng::seed_from_u64(7);
        let spikes = enc.encode(rates.as_ref(), t_steps, &mut rng).unwrap();

        let mut counts = [0u32; 2];
        for s in &spikes {
            for (i, count) in counts.iter_mut().enumerate() {
                if s.as_mat()[(i, 0)] == 1.0 {
                    *count += 1;
                }
            }
        }
        let duration = t_steps as f64 * 0.001;
        // 3σ bands for a binomial count.
        for (i, expected_rate) in [100.0, 10.0].iter().enumerate() {
            let expected = expected_rate * duration;
            let sigma = (expected * (1.0 - expected_rate * 0.001)).sqrt();
            let got = counts[i] as f64;
            assert!(
                (got - expected).abs() < 3.0 * sigma,
                "neuron {i}: got {got} spikes, expected {expected} ± {sigma}"
            );
        }
    }

    #[test]
    fn seeded_encoding_is_reproducible() {
        let enc = PoissonEncoder::new(0.001).unwrap();
        let mut rates = Mat::<f64>::zeros(3, 2);
        for i in 0..3 {
            for j in 0..2 {
                rates[(i, j)] = 50.0;
            }
        }
        let a = enc
            .encode(rates.as_ref(), 100, &mut StdRng::seed_from_u64(42))
            .unwrap();
        let b = enc
            .encode(rates.as_ref(), 100, &mut StdRng::seed_from_u64(42))
            .unwrap();
        for (sa, sb) in a.iter().zip(&b) {
            for j in 0..2 {
                for i in 0..3 {
                    assert_eq!(sa.as_mat()[(i, j)], sb.as_mat()[(i, j)]);
                }
            }
        }
    }

    #[test]
    fn negative_rates_are_rejected() {
        let enc = PoissonEncoder::new(0.001).unwrap();
        let mut rates = Mat::<f64>::zeros(1, 1);
        rates[(0, 0)] = -1.0;
        let mut rng = StdRng::seed_from_u64(0);
        assert!(enc.encode(rates.as_ref(), 10, &mut rng).is_err());
    }

    #[test]
    fn constant_current_fills_value() {
        let m = constant_current(2, 3, 1.5).unwrap();
        for j in 0..3 {
            for i in 0..2 {
                assert_eq!(m[(i, j)], 1.5);
            }
        }
        assert!(constant_current(0, 1, 0.0).is_err());
    }
}
