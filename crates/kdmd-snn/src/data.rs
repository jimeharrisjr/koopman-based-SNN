//! Synthetic tasks, plus (behind the `datasets` feature) the Spiking
//! Heidelberg Digits loader.

#[cfg(feature = "datasets")]
pub mod shd;

use rand::Rng;

use faer::Mat;

use crate::encoding::PoissonEncoder;
use crate::error::SnnError;
use crate::spikes::SpikeBatch;

/// Poisson pattern classification: each class is a fixed random subset of
/// input channels firing at `high_rate`, the rest at `low_rate`; a sample is
/// a Poisson realization of its class pattern.
///
/// The canonical zero-dependency training task (Phase 6): linearly separable
/// in rate space, but only through temporal integration of noisy spikes.
#[derive(Debug, Clone)]
pub struct PoissonPatternTask {
    /// Per class: rate for every input channel.
    class_rates: Vec<Vec<f64>>,
    n_inputs: usize,
}

impl PoissonPatternTask {
    /// Draw `n_classes` random patterns over `n_inputs` channels: each channel
    /// is "active" (rate `high_rate`) with probability `active_fraction`,
    /// otherwise `low_rate`.
    pub fn new<R: Rng>(
        n_inputs: usize,
        n_classes: usize,
        high_rate: f64,
        low_rate: f64,
        active_fraction: f64,
        rng: &mut R,
    ) -> Result<Self, SnnError> {
        if n_inputs == 0 || n_classes == 0 {
            return Err(SnnError::InvalidParameter(
                "n_inputs and n_classes must be nonzero".into(),
            ));
        }
        if !(0.0..=1.0).contains(&active_fraction) {
            return Err(SnnError::InvalidParameter(format!(
                "active_fraction must be in [0, 1] (got {active_fraction})"
            )));
        }
        if high_rate < 0.0 || low_rate < 0.0 || high_rate <= low_rate {
            return Err(SnnError::InvalidParameter(format!(
                "rates must satisfy 0 <= low_rate < high_rate \
                 (low = {low_rate}, high = {high_rate})"
            )));
        }
        let class_rates = (0..n_classes)
            .map(|_| {
                (0..n_inputs)
                    .map(|_| {
                        if rng.random::<f64>() < active_fraction {
                            high_rate
                        } else {
                            low_rate
                        }
                    })
                    .collect()
            })
            .collect();
        Ok(Self {
            class_rates,
            n_inputs,
        })
    }

    pub fn n_classes(&self) -> usize {
        self.class_rates.len()
    }

    pub fn n_inputs(&self) -> usize {
        self.n_inputs
    }

    /// The rate vector defining `class`.
    ///
    /// # Panics
    /// Panics if `class >= n_classes`.
    pub fn rates(&self, class: usize) -> &[f64] {
        &self.class_rates[class]
    }

    /// Draw one spike-train sample of `class` (`t_steps` batches of batch
    /// size 1), with time bins of `dt`.
    pub fn sample<R: Rng>(
        &self,
        class: usize,
        t_steps: usize,
        dt: f64,
        rng: &mut R,
    ) -> Result<Vec<SpikeBatch>, SnnError> {
        if class >= self.n_classes() {
            return Err(SnnError::InvalidParameter(format!(
                "class {class} out of range ({} classes)",
                self.n_classes()
            )));
        }
        let mut rates = Mat::<f64>::zeros(self.n_inputs, 1);
        for (i, &r) in self.class_rates[class].iter().enumerate() {
            rates[(i, 0)] = r;
        }
        PoissonEncoder::new(dt)?.encode(rates.as_ref(), t_steps, rng)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn patterns_are_distinct_across_classes() {
        let mut rng = StdRng::seed_from_u64(3);
        let task = PoissonPatternTask::new(64, 4, 100.0, 5.0, 0.3, &mut rng).unwrap();
        for a in 0..4 {
            for b in (a + 1)..4 {
                assert_ne!(
                    task.rates(a),
                    task.rates(b),
                    "classes {a} and {b} drew identical patterns"
                );
            }
        }
    }

    #[test]
    fn sample_realizes_the_class_rates() {
        let mut rng = StdRng::seed_from_u64(11);
        let task = PoissonPatternTask::new(32, 2, 200.0, 2.0, 0.5, &mut rng).unwrap();
        let dt = 0.001;
        let t_steps = 20_000;
        let spikes = task.sample(0, t_steps, dt, &mut rng).unwrap();

        // Active channels should out-fire inactive ones by a wide margin.
        let mut rates_seen = vec![0.0f64; 32];
        for s in &spikes {
            for (i, rate) in rates_seen.iter_mut().enumerate() {
                *rate += s.as_mat()[(i, 0)];
            }
        }
        for r in &mut rates_seen {
            *r /= t_steps as f64 * dt;
        }
        for (i, &target) in task.rates(0).iter().enumerate() {
            let got = rates_seen[i];
            assert!(
                (got - target).abs() < 0.25 * target.max(4.0),
                "channel {i}: empirical {got:.1} Hz vs target {target:.1} Hz"
            );
        }
    }

    #[test]
    fn invalid_configs_are_rejected() {
        let mut rng = StdRng::seed_from_u64(0);
        assert!(PoissonPatternTask::new(0, 2, 100.0, 5.0, 0.3, &mut rng).is_err());
        assert!(PoissonPatternTask::new(8, 2, 5.0, 100.0, 0.3, &mut rng).is_err());
        assert!(PoissonPatternTask::new(8, 2, 100.0, 5.0, 1.5, &mut rng).is_err());
        let task = PoissonPatternTask::new(8, 2, 100.0, 5.0, 0.3, &mut rng).unwrap();
        assert!(task.sample(2, 10, 0.001, &mut rng).is_err());
    }
}
