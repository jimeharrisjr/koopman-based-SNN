//! Plain optimizers — no dependencies, per the Q6 decision.

use faer::Mat;

/// Optimizer configuration.
#[derive(Debug, Clone, Copy)]
pub enum OptimConfig {
    Sgd {
        lr: f64,
        momentum: f64,
    },
    Adam {
        lr: f64,
        beta1: f64,
        beta2: f64,
        eps: f64,
    },
}

/// Per-parameter optimizer state (first/second moments; the SGD path uses
/// only the first as its velocity).
#[derive(Debug, Clone)]
pub struct Adam {
    m: Mat<f64>,
    v: Mat<f64>,
    t: u64,
}

impl Adam {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            m: Mat::zeros(rows, cols),
            v: Mat::zeros(rows, cols),
            t: 0,
        }
    }

    /// In-place parameter update.
    pub fn update(&mut self, param: &mut Mat<f64>, grad: &Mat<f64>, cfg: &OptimConfig) {
        debug_assert_eq!(param.nrows(), grad.nrows());
        debug_assert_eq!(param.ncols(), grad.ncols());
        match *cfg {
            OptimConfig::Sgd { lr, momentum } => {
                for c in 0..param.ncols() {
                    for r in 0..param.nrows() {
                        let vel = momentum * self.m[(r, c)] - lr * grad[(r, c)];
                        self.m[(r, c)] = vel;
                        param[(r, c)] += vel;
                    }
                }
            }
            OptimConfig::Adam {
                lr,
                beta1,
                beta2,
                eps,
            } => {
                self.t += 1;
                let bc1 = 1.0 - beta1.powi(self.t as i32);
                let bc2 = 1.0 - beta2.powi(self.t as i32);
                for c in 0..param.ncols() {
                    for r in 0..param.nrows() {
                        let g = grad[(r, c)];
                        self.m[(r, c)] = beta1 * self.m[(r, c)] + (1.0 - beta1) * g;
                        self.v[(r, c)] = beta2 * self.v[(r, c)] + (1.0 - beta2) * g * g;
                        let m_hat = self.m[(r, c)] / bc1;
                        let v_hat = self.v[(r, c)] / bc2;
                        param[(r, c)] -= lr * m_hat / (v_hat.sqrt() + eps);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adam_descends_a_quadratic() {
        // Minimize f(x) = (x − 3)² from x = 0.
        let mut x = Mat::<f64>::zeros(1, 1);
        let mut opt = Adam::new(1, 1);
        let cfg = OptimConfig::Adam {
            lr: 0.1,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
        };
        for _ in 0..500 {
            let g = Mat::from_fn(1, 1, |_, _| 2.0 * (x[(0, 0)] - 3.0));
            opt.update(&mut x, &g, &cfg);
        }
        assert!((x[(0, 0)] - 3.0).abs() < 0.05, "x = {}", x[(0, 0)]);
    }

    #[test]
    fn sgd_with_momentum_descends() {
        let mut x = Mat::<f64>::zeros(1, 1);
        let mut opt = Adam::new(1, 1);
        let cfg = OptimConfig::Sgd {
            lr: 0.05,
            momentum: 0.9,
        };
        for _ in 0..200 {
            let g = Mat::from_fn(1, 1, |_, _| 2.0 * (x[(0, 0)] - 3.0));
            opt.update(&mut x, &g, &cfg);
        }
        assert!((x[(0, 0)] - 3.0).abs() < 0.1, "x = {}", x[(0, 0)]);
    }
}
