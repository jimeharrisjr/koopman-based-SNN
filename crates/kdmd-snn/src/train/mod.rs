//! Surrogate-gradient BPTT for [`Network`]s — hand-rolled, per the owner's
//! Q6 decision (no autograd dependency).
//!
//! The per-step graph is fixed (`linear advance → threshold → linear
//! continuation`), so the backward pass is a short, exact sequence of
//! transposed operator applications and elementwise products; the **only**
//! approximation is the surrogate derivative `σ'` standing in for `Θ'` at the
//! threshold. With the subtractive reset expressed as `−θ·s` inside the
//! linear step, the reset's gradient path is retained exactly (no detach
//! trick).
//!
//! Backward recursion per layer ℓ and step t (λ = ∂L/∂x_{ℓ,t+1}):
//!
//! ```text
//! g_s   = W_{ℓ+1}ᵀ·(∂L/∂d_{ℓ+1,t})  [+ Rᵀ·∂L/∂logits at the top layer]
//!         − θ·λ_v                    (reset path)
//! g_v   = σ'(v_pre − θ) ⊙ g_s       (the surrogate — the only approximation)
//! ∂L/∂y = [λ_v + g_v ; λ_i]
//! λ'    = Aᵀ·(∂L/∂y)                (exact Jacobian of the linear part)
//! ∂L/∂d = Σ_p b_local[p]·(∂L/∂y_p)
//! ∂L/∂W += (∂L/∂d)·s_inᵀ
//! ```
//!
//! The readout is a linear map `logits = R·(Σ_t s_out)/T` on output spike
//! counts, trained jointly. Loss is softmax cross-entropy. Note the gradient
//! caveat documented on [`Trainer::check_readout_gradient`]: the forward pass
//! is piecewise-constant in the spikes, so finite differences can only
//! validate the smooth (readout) part — the surrogate path is validated
//! end-to-end by learning curves instead.

pub mod optim;
pub mod surrogate;

use faer::Mat;

use crate::error::SnnError;
use crate::network::Network;
use crate::spikes::SpikeBatch;

pub use optim::{Adam, OptimConfig};
pub use surrogate::SurrogateKind;

/// Training configuration.
#[derive(Debug, Clone)]
pub struct TrainConfig {
    pub surrogate: SurrogateKind,
    pub optim: OptimConfig,
    /// Elementwise gradient clip (absolute value), applied before the update.
    pub grad_clip: Option<f64>,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            surrogate: SurrogateKind::FastSigmoid { beta: 5.0 },
            optim: OptimConfig::Adam {
                lr: 5e-3,
                beta1: 0.9,
                beta2: 0.999,
                eps: 1e-8,
            },
            grad_clip: Some(1.0),
        }
    }
}

/// Per-minibatch statistics.
#[derive(Debug, Clone, Copy)]
pub struct StepStats {
    pub loss: f64,
    pub accuracy: f64,
}

/// Trainer: owns the readout, gradients, and optimizer state for one network
/// shape.
#[derive(Debug)]
pub struct Trainer {
    cfg: TrainConfig,
    /// Readout `n_classes × n_out`.
    r: Mat<f64>,
    opt_w: Vec<Adam>,
    opt_r: Adam,
}

impl Trainer {
    /// Build a trainer for `net` with `n_classes` outputs. The readout is
    /// deterministically initialized (small alternating values — no RNG
    /// dependency; symmetry is broken by the spike statistics).
    pub fn new(net: &Network, n_classes: usize, cfg: TrainConfig) -> Result<Self, SnnError> {
        if n_classes < 2 {
            return Err(SnnError::InvalidParameter(
                "need at least two classes".into(),
            ));
        }
        let n_out = net.layer(net.n_layers() - 1).n_neurons();
        let r = Mat::from_fn(n_classes, n_out, |i, j| {
            0.01 * ((i + 2 * j) % 5) as f64 - 0.02
        });
        let mut opt_w = Vec::with_capacity(net.n_layers());
        for l in 0..net.n_layers() {
            let w = net.layer(l).weights();
            opt_w.push(Adam::new(w.nrows(), w.ncols()));
        }
        let opt_r = Adam::new(n_classes, n_out);
        Ok(Self {
            cfg,
            r,
            opt_w,
            opt_r,
        })
    }

    pub fn readout(&self) -> &Mat<f64> {
        &self.r
    }

    /// Forward pass with tape; returns (per-step v_pre, per-step s_out,
    /// output spike counts `n_out × batch`).
    #[allow(clippy::type_complexity)]
    fn forward(
        &self,
        net: &mut Network,
        inputs: &[SpikeBatch],
    ) -> Result<(Vec<Vec<Mat<f64>>>, Vec<Vec<SpikeBatch>>, Mat<f64>), SnnError> {
        // The trainer is bound to one network shape at construction; a
        // mismatched network must error, not panic mid-matmul.
        let readout_inputs = net.layer(net.n_layers() - 1).n_neurons();
        if self.r.ncols() != readout_inputs {
            return Err(SnnError::DimensionMismatch(format!(
                "trainer readout covers {} output neurons, network has {readout_inputs}",
                self.r.ncols()
            )));
        }
        let batch = net.batch();
        let n_layers = net.n_layers();
        net.reset_state();
        let mut v_pre_tape = Vec::with_capacity(inputs.len());
        let mut s_out_tape = Vec::with_capacity(inputs.len());
        let n_out = net.layer(n_layers - 1).n_neurons();
        let mut counts = Mat::<f64>::zeros(n_out, batch);

        for input in inputs {
            let mut v_pre: Vec<Mat<f64>> = (0..n_layers)
                .map(|l| Mat::zeros(net.layer(l).n_neurons(), batch))
                .collect();
            let mut s_out: Vec<SpikeBatch> = (0..n_layers)
                .map(|l| SpikeBatch::zeros(net.layer(l).n_neurons(), batch))
                .collect::<Result<_, _>>()?;
            net.step_batch_taped(input, &mut v_pre, &mut s_out)?;
            let top = s_out[n_layers - 1].as_mat();
            for b in 0..batch {
                for i in 0..n_out {
                    counts[(i, b)] += top[(i, b)];
                }
            }
            v_pre_tape.push(v_pre);
            s_out_tape.push(s_out);
        }
        Ok((v_pre_tape, s_out_tape, counts))
    }

    /// Softmax cross-entropy over logits; returns (mean loss, ∂L/∂logits,
    /// accuracy).
    fn loss_and_grad(
        logits: &Mat<f64>,
        targets: &[usize],
    ) -> Result<(f64, Mat<f64>, f64), SnnError> {
        let (classes, batch) = (logits.nrows(), logits.ncols());
        if targets.len() != batch {
            return Err(SnnError::DimensionMismatch(format!(
                "{} targets for batch {batch}",
                targets.len()
            )));
        }
        let mut dlogits = Mat::<f64>::zeros(classes, batch);
        let mut loss = 0.0;
        let mut correct = 0usize;
        for b in 0..batch {
            let target = targets[b];
            if target >= classes {
                return Err(SnnError::InvalidParameter(format!(
                    "target {target} out of range ({classes} classes)"
                )));
            }
            let max = (0..classes)
                .map(|i| logits[(i, b)])
                .fold(f64::MIN, f64::max);
            let mut z = 0.0;
            for i in 0..classes {
                z += (logits[(i, b)] - max).exp();
            }
            let log_z = z.ln() + max;
            loss += log_z - logits[(target, b)];
            let mut best = 0usize;
            for i in 0..classes {
                let p = (logits[(i, b)] - log_z).exp();
                dlogits[(i, b)] = (p - if i == target { 1.0 } else { 0.0 }) / batch as f64;
                if logits[(i, b)] > logits[(best, b)] {
                    best = i;
                }
            }
            if best == target {
                correct += 1;
            }
        }
        Ok((loss / batch as f64, dlogits, correct as f64 / batch as f64))
    }

    /// One minibatch: taped forward, hand-rolled backward, optimizer step on
    /// every layer's `W` and the readout.
    pub fn train_step(
        &mut self,
        net: &mut Network,
        inputs: &[SpikeBatch],
        targets: &[usize],
    ) -> Result<StepStats, SnnError> {
        if inputs.is_empty() {
            return Err(SnnError::InvalidParameter("empty input sequence".into()));
        }
        let t_steps = inputs.len();
        let batch = net.batch();
        let n_layers = net.n_layers();
        let (v_pre_tape, s_out_tape, counts) = self.forward(net, inputs)?;

        // logits = R · counts / T.
        let n_out = counts.nrows();
        let classes = self.r.nrows();
        let mut logits = Mat::<f64>::zeros(classes, batch);
        for b in 0..batch {
            for i in 0..classes {
                let mut acc = 0.0;
                for j in 0..n_out {
                    acc += self.r[(i, j)] * counts[(j, b)];
                }
                logits[(i, b)] = acc / t_steps as f64;
            }
        }
        let (loss, dlogits, accuracy) = Self::loss_and_grad(&logits, targets)?;

        // ∂L/∂R = dlogits · countsᵀ / T.
        let mut grad_r = Mat::<f64>::zeros(classes, n_out);
        for i in 0..classes {
            for j in 0..n_out {
                let mut acc = 0.0;
                for b in 0..batch {
                    acc += dlogits[(i, b)] * counts[(j, b)];
                }
                grad_r[(i, j)] = acc / t_steps as f64;
            }
        }
        // ∂L/∂s_top (per step, same every step): Rᵀ · dlogits / T.
        let mut ds_top = Mat::<f64>::zeros(n_out, batch);
        for j in 0..n_out {
            for b in 0..batch {
                let mut acc = 0.0;
                for i in 0..classes {
                    acc += self.r[(i, j)] * dlogits[(i, b)];
                }
                ds_top[(j, b)] = acc / t_steps as f64;
            }
        }

        // Backward through time.
        let mut grad_w: Vec<Mat<f64>> = (0..n_layers)
            .map(|l| {
                Mat::zeros(
                    net.layer(l).weights().nrows(),
                    net.layer(l).weights().ncols(),
                )
            })
            .collect();
        // λ_l = ∂L/∂x_{l, t+1}, zero at the horizon.
        let mut lambda: Vec<Mat<f64>> = (0..n_layers)
            .map(|l| {
                Mat::zeros(
                    net.layer(l).n_neurons() * net.layer(l).n_state_vars(),
                    batch,
                )
            })
            .collect();
        let mut dldy: Vec<Mat<f64>> = lambda.clone();
        // ∂L/∂d of layer l at the current step (consumed by layer l−1).
        let mut dldd: Vec<Mat<f64>> = (0..n_layers)
            .map(|l| Mat::zeros(net.layer(l).n_neurons(), batch))
            .collect();

        for t in (0..t_steps).rev() {
            for l in (0..n_layers).rev() {
                let n = net.layer(l).n_neurons();
                let k = net.layer(l).n_state_vars();
                let theta = net.layer(l).theta();
                // g_s: downstream uses of this layer's spikes at step t.
                let mut g_s = Mat::<f64>::zeros(n, batch);
                if l == n_layers - 1 {
                    for b in 0..batch {
                        for j in 0..n {
                            g_s[(j, b)] += ds_top[(j, b)];
                        }
                    }
                }
                if l + 1 < n_layers {
                    // W_{l+1}ᵀ · dldd_{l+1} (that layer was processed first).
                    let w_next = net.layer(l + 1).weights();
                    let d_next = &dldd[l + 1];
                    for b in 0..batch {
                        for j in 0..n {
                            let mut acc = 0.0;
                            for i in 0..w_next.nrows() {
                                acc += w_next[(i, j)] * d_next[(i, b)];
                            }
                            g_s[(j, b)] += acc;
                        }
                    }
                }
                // Reset path: x_{t+1,v} = y_v − θ·s.
                for b in 0..batch {
                    for j in 0..n {
                        g_s[(j, b)] -= theta * lambda[l][(j, b)];
                    }
                }
                // Surrogate through the threshold; ∂L/∂y.
                let v_pre = &v_pre_tape[t][l];
                for b in 0..batch {
                    for j in 0..n {
                        let g_v =
                            self.cfg.surrogate.derivative(v_pre[(j, b)] - theta) * g_s[(j, b)];
                        dldy[l][(j, b)] = lambda[l][(j, b)] + g_v;
                    }
                    for p in 1..k {
                        for j in 0..n {
                            dldy[l][(p * n + j, b)] = lambda[l][(p * n + j, b)];
                        }
                    }
                }
                // ∂L/∂d = Σ_p b_local[p] · ∂L/∂y_p ; accumulate ∂L/∂W.
                let b_local: Vec<f64> = net.layer(l).b_local().to_vec();
                for b in 0..batch {
                    for j in 0..n {
                        let mut acc = 0.0;
                        for (p, &coef) in b_local.iter().enumerate() {
                            acc += coef * dldy[l][(p * n + j, b)];
                        }
                        dldd[l][(j, b)] = acc;
                    }
                }
                let s_in_mat = if l == 0 {
                    inputs[t].as_mat()
                } else {
                    s_out_tape[t][l - 1].as_mat()
                };
                for b in 0..batch {
                    for j in 0..s_in_mat.nrows() {
                        let s = s_in_mat[(j, b)];
                        if s == 0.0 {
                            continue;
                        }
                        for i in 0..n {
                            grad_w[l][(i, j)] += dldd[l][(i, b)] * s;
                        }
                    }
                }
                // λ ← Aᵀ·∂L/∂y (the exact linear-part Jacobian).
                net.layer(l).operator().apply_transpose(
                    dldy[l].as_ref(),
                    lambda[l].as_mut(),
                    false,
                );
            }
        }

        // Clip + update.
        if let Some(clip) = self.cfg.grad_clip {
            for g in grad_w.iter_mut().chain(std::iter::once(&mut grad_r)) {
                for b in 0..g.ncols() {
                    for i in 0..g.nrows() {
                        g[(i, b)] = g[(i, b)].clamp(-clip, clip);
                    }
                }
            }
        }
        for (l, (opt, grad)) in self.opt_w.iter_mut().zip(&grad_w).enumerate() {
            opt.update(net.layer_mut(l).weights_mut(), grad, &self.cfg.optim);
        }
        let mut r = std::mem::replace(&mut self.r, Mat::zeros(0, 0));
        self.opt_r.update(&mut r, &grad_r, &self.cfg.optim);
        self.r = r;

        Ok(StepStats { loss, accuracy })
    }

    /// Classify a batch: argmax of the readout on output spike counts.
    pub fn predict(
        &self,
        net: &mut Network,
        inputs: &[SpikeBatch],
    ) -> Result<Vec<usize>, SnnError> {
        let (_, _, counts) = self.forward(net, inputs)?;
        let batch = counts.ncols();
        let classes = self.r.nrows();
        let mut out = Vec::with_capacity(batch);
        for b in 0..batch {
            let mut best = 0usize;
            let mut best_v = f64::MIN;
            for i in 0..classes {
                let mut acc = 0.0;
                for j in 0..counts.nrows() {
                    acc += self.r[(i, j)] * counts[(j, b)];
                }
                if acc > best_v {
                    best_v = acc;
                    best = i;
                }
            }
            out.push(best);
        }
        Ok(out)
    }

    /// Finite-difference check of the READOUT gradient (the smooth part of
    /// the loss — exact to first order). The W-gradient cannot be
    /// finite-difference-checked: the forward pass is piecewise constant in
    /// the spikes, so the true derivative is zero almost everywhere and the
    /// surrogate deliberately reports something else. W's path is validated
    /// end-to-end by learning curves (see the training test).
    pub fn check_readout_gradient(
        &self,
        net: &mut Network,
        inputs: &[SpikeBatch],
        targets: &[usize],
        eps: f64,
    ) -> Result<f64, SnnError> {
        let (_, _, counts) = self.forward(net, inputs)?;
        let t_steps = inputs.len();
        let (classes, n_out) = (self.r.nrows(), self.r.ncols());
        let logits_for = |r: &Mat<f64>| {
            let mut logits = Mat::<f64>::zeros(classes, counts.ncols());
            for b in 0..counts.ncols() {
                for i in 0..classes {
                    let mut acc = 0.0;
                    for j in 0..n_out {
                        acc += r[(i, j)] * counts[(j, b)];
                    }
                    logits[(i, b)] = acc / t_steps as f64;
                }
            }
            logits
        };
        let (_, dlogits, _) = Self::loss_and_grad(&logits_for(&self.r), targets)?;
        // Analytic ∂L/∂R.
        let mut grad_r = Mat::<f64>::zeros(classes, n_out);
        for i in 0..classes {
            for j in 0..n_out {
                let mut acc = 0.0;
                for b in 0..counts.ncols() {
                    acc += dlogits[(i, b)] * counts[(j, b)];
                }
                grad_r[(i, j)] = acc / t_steps as f64;
            }
        }
        // Central finite differences, worst relative error over entries with
        // non-negligible gradient.
        let mut worst: f64 = 0.0;
        for i in 0..classes {
            for j in 0..n_out {
                let mut r_hi = self.r.clone();
                r_hi[(i, j)] += eps;
                let mut r_lo = self.r.clone();
                r_lo[(i, j)] -= eps;
                let (l_hi, _, _) = Self::loss_and_grad(&logits_for(&r_hi), targets)?;
                let (l_lo, _, _) = Self::loss_and_grad(&logits_for(&r_lo), targets)?;
                let fd = (l_hi - l_lo) / (2.0 * eps);
                let denom = fd.abs().max(grad_r[(i, j)].abs());
                if denom > 1e-8 {
                    worst = worst.max((fd - grad_r[(i, j)]).abs() / denom);
                }
            }
        }
        Ok(worst)
    }
}
