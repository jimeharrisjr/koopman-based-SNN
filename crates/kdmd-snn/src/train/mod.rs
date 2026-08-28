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
    /// Decoupled (AdamW-style) weight decay applied to the layer weights
    /// (`W` and `W_rec`, not the readout) after each optimizer step:
    /// `w ← w · (1 − lr·λ)`. Zero disables it.
    pub weight_decay: f64,
    /// Temporal readout: `Some(κ)` (κ < 1) replaces the plain spike count
    /// with a leaky trace `trace_t = κ·trace_{t−1} + s_t`, weighting recent
    /// spikes more — a readout memory of ≈ 1/(1−κ) steps. `None` keeps the
    /// uniform count readout.
    pub readout_decay: Option<f64>,
    /// Data-parallel threads for the batch dimension of `train_step` and
    /// `logits`. `1` (the default) is the exact single-threaded path that
    /// reproduces all recorded results bit-for-bit. With `t > 1` the batch
    /// splits into `t` column chunks, each processed on its own thread with
    /// its own network copy; chunk gradients are summed in fixed chunk order,
    /// so results are deterministic for a given thread count, but floating-
    /// point summation order differs from the serial path (differences
    /// ~1e-15 per step, which can drift across a long run the way any seed
    /// perturbation does).
    pub threads: usize,
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
            weight_decay: 0.0,
            readout_decay: None,
            threads: 1,
        }
    }
}

/// Gradients and loss statistics from one forward+backward pass over a batch
/// (or a column chunk of one). Chunks sum entrywise: every gradient inside is
/// normalized by the *full* batch's `loss_denom`, so summation needs no
/// rescaling.
struct BatchGrads {
    w: Vec<Mat<f64>>,
    rec: Vec<Option<Mat<f64>>>,
    r: Mat<f64>,
    /// Unnormalized Σ_b per-sample loss.
    loss_sum: f64,
    correct: usize,
}

impl BatchGrads {
    fn add(&mut self, other: &BatchGrads) {
        for (a, b) in self.w.iter_mut().zip(&other.w) {
            for c in 0..a.ncols() {
                for i in 0..a.nrows() {
                    a[(i, c)] += b[(i, c)];
                }
            }
        }
        for (a, b) in self.rec.iter_mut().zip(&other.rec) {
            if let (Some(a), Some(b)) = (a.as_mut(), b.as_ref()) {
                for c in 0..a.ncols() {
                    for i in 0..a.nrows() {
                        a[(i, c)] += b[(i, c)];
                    }
                }
            }
        }
        for c in 0..self.r.ncols() {
            for i in 0..self.r.nrows() {
                self.r[(i, c)] += other.r[(i, c)];
            }
        }
        self.loss_sum += other.loss_sum;
        self.correct += other.correct;
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
    /// Optimizer state for recurrent weights, per layer (None = feedforward).
    opt_rec: Vec<Option<Adam>>,
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
        let mut opt_rec = Vec::with_capacity(net.n_layers());
        for l in 0..net.n_layers() {
            let w = net.layer(l).weights();
            opt_w.push(Adam::new(w.nrows(), w.ncols()));
            opt_rec.push(
                net.layer(l)
                    .recurrent_weights()
                    .map(|wr| Adam::new(wr.nrows(), wr.ncols())),
            );
        }
        let opt_r = Adam::new(n_classes, n_out);
        Ok(Self {
            cfg,
            r,
            opt_w,
            opt_rec,
            opt_r,
        })
    }

    pub fn readout(&self) -> &Mat<f64> {
        &self.r
    }

    /// Change the learning rate in place (for decay schedules). Optimizer
    /// moments are preserved.
    pub fn set_learning_rate(&mut self, new_lr: f64) {
        match &mut self.cfg.optim {
            OptimConfig::Sgd { lr, .. } | OptimConfig::Adam { lr, .. } => *lr = new_lr,
        }
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
            match self.cfg.readout_decay {
                None => {
                    for b in 0..batch {
                        for i in 0..n_out {
                            counts[(i, b)] += top[(i, b)];
                        }
                    }
                }
                // Leaky trace: trace ← κ·trace + s_t.
                Some(kappa) => {
                    for b in 0..batch {
                        for i in 0..n_out {
                            counts[(i, b)] = kappa * counts[(i, b)] + top[(i, b)];
                        }
                    }
                }
            }
            v_pre_tape.push(v_pre);
            s_out_tape.push(s_out);
        }
        Ok((v_pre_tape, s_out_tape, counts))
    }

    /// Readout normalization: the effective mass of the (possibly leaky)
    /// count over `t_steps` steps, so logits stay O(rate) regardless of κ.
    fn readout_norm(&self, t_steps: usize) -> f64 {
        match self.cfg.readout_decay {
            None => t_steps as f64,
            Some(kappa) => (1.0 - kappa.powi(t_steps as i32)) / (1.0 - kappa),
        }
    }

    /// Softmax cross-entropy over logits; returns (Σ_b per-sample loss,
    /// ∂L/∂logits normalized by `denom`, correct-prediction count). `denom`
    /// is the full-minibatch size — a column chunk passes the full size so
    /// chunk gradients sum to exactly the full-batch gradient.
    fn loss_and_grad(
        logits: &Mat<f64>,
        targets: &[usize],
        denom: usize,
    ) -> Result<(f64, Mat<f64>, usize), SnnError> {
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
                dlogits[(i, b)] = (p - if i == target { 1.0 } else { 0.0 }) / denom as f64;
                if logits[(i, b)] > logits[(best, b)] {
                    best = i;
                }
            }
            if best == target {
                correct += 1;
            }
        }
        Ok((loss, dlogits, correct))
    }

    /// Taped forward + hand-rolled backward over one batch (or one column
    /// chunk of a larger minibatch), producing gradients normalized by
    /// `loss_denom` (the full minibatch size). Does not touch the optimizer;
    /// [`train_step`](Self::train_step) sums chunks and applies the update.
    fn forward_backward(
        &self,
        net: &mut Network,
        inputs: &[SpikeBatch],
        targets: &[usize],
        loss_denom: usize,
    ) -> Result<BatchGrads, SnnError> {
        if inputs.is_empty() {
            return Err(SnnError::InvalidParameter("empty input sequence".into()));
        }
        let t_steps = inputs.len();
        let batch = net.batch();
        let n_layers = net.n_layers();
        let (v_pre_tape, s_out_tape, counts) = self.forward(net, inputs)?;

        // logits = R · trace / norm (norm = T for the count readout, the
        // leaky-trace mass otherwise).
        let norm = self.readout_norm(t_steps);
        let n_out = counts.nrows();
        let classes = self.r.nrows();
        let mut logits = Mat::<f64>::zeros(classes, batch);
        for b in 0..batch {
            for i in 0..classes {
                let mut acc = 0.0;
                for j in 0..n_out {
                    acc += self.r[(i, j)] * counts[(j, b)];
                }
                logits[(i, b)] = acc / norm;
            }
        }
        let (loss_sum, dlogits, correct) = Self::loss_and_grad(&logits, targets, loss_denom)?;

        // ∂L/∂R = dlogits · traceᵀ / norm.
        let mut grad_r = Mat::<f64>::zeros(classes, n_out);
        for i in 0..classes {
            for j in 0..n_out {
                let mut acc = 0.0;
                for b in 0..batch {
                    acc += dlogits[(i, b)] * counts[(j, b)];
                }
                grad_r[(i, j)] = acc / norm;
            }
        }
        // ∂L/∂s_top base: Rᵀ · dlogits / norm. Under the leaky readout the
        // per-step contribution is scaled by κ^(T−1−t) in the backward loop
        // (∂trace_T/∂s_t); with the count readout the scale is 1 every step.
        let mut ds_top = Mat::<f64>::zeros(n_out, batch);
        for j in 0..n_out {
            for b in 0..batch {
                let mut acc = 0.0;
                for i in 0..classes {
                    acc += self.r[(i, j)] * dlogits[(i, b)];
                }
                ds_top[(j, b)] = acc / norm;
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
        // ∂L/∂d at step t+1 (consumed by the recurrent path: the layer's
        // spikes at t drove its own input at t+1). Double-buffered with dldd.
        let mut dldd_next: Vec<Mat<f64>> = dldd.clone();
        // Recurrent-weight gradients, where present.
        let mut grad_rec: Vec<Option<Mat<f64>>> = (0..n_layers)
            .map(|l| {
                net.layer(l)
                    .recurrent_weights()
                    .map(|wr| Mat::zeros(wr.nrows(), wr.ncols()))
            })
            .collect();

        // κ^(T−1−t) factor for the leaky readout (1.0 throughout for counts).
        let mut readout_scale = 1.0f64;
        for t in (0..t_steps).rev() {
            // dldd currently holds step t+1's values (zero at t = T−1);
            // stash them for the recurrent path and overwrite dldd below.
            std::mem::swap(&mut dldd, &mut dldd_next);
            for l in (0..n_layers).rev() {
                let n = net.layer(l).n_neurons();
                let k = net.layer(l).n_state_vars();
                let theta = net.layer(l).theta();
                // g_s: downstream uses of this layer's spikes at step t.
                let mut g_s = Mat::<f64>::zeros(n, batch);
                if l == n_layers - 1 {
                    for b in 0..batch {
                        for j in 0..n {
                            g_s[(j, b)] += readout_scale * ds_top[(j, b)];
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
                // Recurrent path: this layer's spikes at t drove its own
                // input at t+1 through W_rec.
                if let Some(w_rec) = net.layer(l).recurrent_weights() {
                    let d_next = &dldd_next[l];
                    for b in 0..batch {
                        for j in 0..n {
                            let mut acc = 0.0;
                            for i in 0..n {
                                acc += w_rec[(i, j)] * d_next[(i, b)];
                            }
                            g_s[(j, b)] += acc;
                        }
                    }
                }
                // Spike-jump paths: x_{t+1,p} += jumps[p]·s (subtractive
                // reset on v, adaptation increment on w, …).
                let jumps: Vec<f64> = net.layer(l).jumps().to_vec();
                for (p, &jump) in jumps.iter().enumerate() {
                    if jump == 0.0 {
                        continue;
                    }
                    for b in 0..batch {
                        for j in 0..n {
                            g_s[(j, b)] += jump * lambda[l][(p * n + j, b)];
                        }
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
                // ∂L/∂d = Σ_p coupling(p, j) · ∂L/∂y_p ; accumulate ∂L/∂W.
                let layer_l = net.layer(l);
                for b in 0..batch {
                    for j in 0..n {
                        let mut acc = 0.0;
                        for p in 0..k {
                            acc += layer_l.coupling(p, j) * dldy[l][(p * n + j, b)];
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
                // ∂L/∂W_rec += ∂L/∂d_t · s_own(t−1)ᵀ (t = 0 saw zero
                // recurrent input — the post-reset convention).
                if t > 0 {
                    if let Some(grad) = grad_rec[l].as_mut() {
                        let s_prev = s_out_tape[t - 1][l].as_mat();
                        for b in 0..batch {
                            for j in 0..n {
                                if s_prev[(j, b)] == 0.0 {
                                    continue;
                                }
                                for i in 0..n {
                                    grad[(i, j)] += dldd[l][(i, b)];
                                }
                            }
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
            if let Some(kappa) = self.cfg.readout_decay {
                readout_scale *= kappa;
            }
        }

        Ok(BatchGrads {
            w: grad_w,
            rec: grad_rec,
            r: grad_r,
            loss_sum,
            correct,
        })
    }

    /// One minibatch: taped forward, hand-rolled backward, optimizer step on
    /// every layer's `W` and the readout. With `cfg.threads > 1` the batch is
    /// split into column chunks processed in parallel (each on its own copy
    /// of the network) and the chunk gradients are summed in fixed order
    /// before the single optimizer update.
    pub fn train_step(
        &mut self,
        net: &mut Network,
        inputs: &[SpikeBatch],
        targets: &[usize],
    ) -> Result<StepStats, SnnError> {
        let batch = net.batch();
        if targets.len() != batch {
            return Err(SnnError::DimensionMismatch(format!(
                "{} targets for batch {batch}",
                targets.len()
            )));
        }
        let threads = self.cfg.threads.max(1).min(batch);
        let grads = if threads == 1 {
            self.forward_backward(net, inputs, targets, batch)?
        } else {
            self.chunked_grads(net, inputs, targets, threads)?
        };
        self.apply_update(net, &grads);
        Ok(StepStats {
            loss: grads.loss_sum / batch as f64,
            accuracy: grads.correct as f64 / batch as f64,
        })
    }

    /// Data-parallel gradients: split the batch columns into `threads`
    /// chunks, run [`forward_backward`](Self::forward_backward) on each in
    /// its own thread with a chunk-sized network copy, and sum in chunk
    /// order (deterministic for a fixed thread count).
    fn chunked_grads(
        &self,
        net: &Network,
        inputs: &[SpikeBatch],
        targets: &[usize],
        threads: usize,
    ) -> Result<BatchGrads, SnnError> {
        if inputs.is_empty() {
            return Err(SnnError::InvalidParameter("empty input sequence".into()));
        }
        let batch = net.batch();
        let base = batch / threads;
        let rem = batch % threads;
        // Chunk boundaries: the first `rem` chunks carry one extra column.
        let mut spans = Vec::with_capacity(threads);
        let mut start = 0usize;
        for c in 0..threads {
            let len = base + usize::from(c < rem);
            spans.push((start, len));
            start += len;
        }
        // Per-chunk nets and column-sliced inputs, prepared up front.
        let mut chunk_nets: Vec<Network> = spans
            .iter()
            .map(|&(_, len)| net.clone_with_batch(len))
            .collect::<Result<_, _>>()?;
        let chunk_inputs: Vec<Vec<SpikeBatch>> = spans
            .iter()
            .map(|&(s, len)| {
                inputs
                    .iter()
                    .map(|step| step.column_range(s, len))
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<_, _>>()?;

        let results: Vec<Result<BatchGrads, SnnError>> = std::thread::scope(|scope| {
            let handles: Vec<_> = chunk_nets
                .iter_mut()
                .zip(&chunk_inputs)
                .zip(&spans)
                .map(|((cnet, cin), &(s, len))| {
                    let ctargets = &targets[s..s + len];
                    scope.spawn(move || self.forward_backward(cnet, cin, ctargets, batch))
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().expect("training worker panicked"))
                .collect()
        });
        let mut iter = results.into_iter();
        let mut total = iter.next().expect("at least one chunk")?;
        for r in iter {
            total.add(&r?);
        }
        Ok(total)
    }

    /// Clip the summed gradients and apply the optimizer + weight decay to
    /// the live network and readout.
    fn apply_update(&mut self, net: &mut Network, grads: &BatchGrads) {
        let n_layers = net.n_layers();
        let mut grad_w = grads.w.clone();
        let mut grad_rec = grads.rec.clone();
        let mut grad_r = grads.r.clone();
        if let Some(clip) = self.cfg.grad_clip {
            for g in grad_w
                .iter_mut()
                .chain(grad_rec.iter_mut().flatten())
                .chain(std::iter::once(&mut grad_r))
            {
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
        for (l, (opt_slot, grad_slot)) in self.opt_rec.iter_mut().zip(&grad_rec).enumerate() {
            if let (Some(opt), Some(grad)) = (opt_slot.as_mut(), grad_slot.as_ref()) {
                if let Some(w_rec) = net.layer_mut(l).recurrent_weights_mut() {
                    opt.update(w_rec, grad, &self.cfg.optim);
                }
            }
        }
        // Decoupled weight decay (AdamW-style), after the optimizer step.
        if self.cfg.weight_decay > 0.0 {
            let lr = match self.cfg.optim {
                OptimConfig::Sgd { lr, .. } | OptimConfig::Adam { lr, .. } => lr,
            };
            let shrink = 1.0 - lr * self.cfg.weight_decay;
            for l in 0..n_layers {
                let w = net.layer_mut(l).weights_mut();
                for c in 0..w.ncols() {
                    for i in 0..w.nrows() {
                        w[(i, c)] *= shrink;
                    }
                }
                if let Some(wr) = net.layer_mut(l).recurrent_weights_mut() {
                    for c in 0..wr.ncols() {
                        for i in 0..wr.nrows() {
                            wr[(i, c)] *= shrink;
                        }
                    }
                }
            }
        }
        let mut r = std::mem::replace(&mut self.r, Mat::zeros(0, 0));
        self.opt_r.update(&mut r, &grad_r, &self.cfg.optim);
        self.r = r;
    }

    /// Classify a batch: argmax of the readout on output spike counts.
    pub fn predict(
        &self,
        net: &mut Network,
        inputs: &[SpikeBatch],
    ) -> Result<Vec<usize>, SnnError> {
        let logits = self.logits(net, inputs)?;
        let (classes, batch) = (logits.nrows(), logits.ncols());
        let mut out = Vec::with_capacity(batch);
        for b in 0..batch {
            let mut best = 0usize;
            for i in 1..classes {
                if logits[(i, b)] > logits[(best, b)] {
                    best = i;
                }
            }
            out.push(best);
        }
        Ok(out)
    }

    /// Readout logits (`n_classes × batch`) for a batch — forward pass only.
    /// Lets callers combine models (ensembling) or inspect confidence. With
    /// `cfg.threads > 1` the batch columns are evaluated in parallel chunks
    /// (forward only, so the result is exactly the serial one).
    pub fn logits(&self, net: &mut Network, inputs: &[SpikeBatch]) -> Result<Mat<f64>, SnnError> {
        let batch = net.batch();
        let threads = self.cfg.threads.max(1).min(batch);
        let counts = if threads == 1 || inputs.is_empty() {
            self.forward(net, inputs)?.2
        } else {
            let base = batch / threads;
            let rem = batch % threads;
            let mut spans = Vec::with_capacity(threads);
            let mut start = 0usize;
            for c in 0..threads {
                let len = base + usize::from(c < rem);
                spans.push((start, len));
                start += len;
            }
            let mut chunk_nets: Vec<Network> = spans
                .iter()
                .map(|&(_, len)| net.clone_with_batch(len))
                .collect::<Result<_, _>>()?;
            let chunk_inputs: Vec<Vec<SpikeBatch>> = spans
                .iter()
                .map(|&(s, len)| {
                    inputs
                        .iter()
                        .map(|step| step.column_range(s, len))
                        .collect::<Result<Vec<_>, _>>()
                })
                .collect::<Result<_, _>>()?;
            let results: Vec<Result<Mat<f64>, SnnError>> = std::thread::scope(|scope| {
                let handles: Vec<_> = chunk_nets
                    .iter_mut()
                    .zip(&chunk_inputs)
                    .map(|(cnet, cin)| scope.spawn(move || Ok(self.forward(cnet, cin)?.2)))
                    .collect();
                handles
                    .into_iter()
                    .map(|h| h.join().expect("eval worker panicked"))
                    .collect()
            });
            let n_out = self.r.ncols();
            let mut counts = Mat::<f64>::zeros(n_out, batch);
            for (res, &(s, len)) in results.into_iter().zip(&spans) {
                let c = res?;
                for col in 0..len {
                    for i in 0..n_out {
                        counts[(i, s + col)] = c[(i, col)];
                    }
                }
            }
            counts
        };
        let norm = self.readout_norm(inputs.len());
        let classes = self.r.nrows();
        let mut logits = Mat::<f64>::zeros(classes, batch);
        for b in 0..batch {
            for i in 0..classes {
                let mut acc = 0.0;
                for j in 0..counts.nrows() {
                    acc += self.r[(i, j)] * counts[(j, b)];
                }
                logits[(i, b)] = acc / norm;
            }
        }
        Ok(logits)
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
                    logits[(i, b)] = acc / self.readout_norm(t_steps);
                }
            }
            logits
        };
        let batch = counts.ncols();
        let (_, dlogits, _) = Self::loss_and_grad(&logits_for(&self.r), targets, batch)?;
        // Analytic ∂L/∂R.
        let mut grad_r = Mat::<f64>::zeros(classes, n_out);
        for i in 0..classes {
            for j in 0..n_out {
                let mut acc = 0.0;
                for b in 0..counts.ncols() {
                    acc += dlogits[(i, b)] * counts[(j, b)];
                }
                grad_r[(i, j)] = acc / self.readout_norm(t_steps);
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
                let (l_hi, _, _) = Self::loss_and_grad(&logits_for(&r_hi), targets, batch)?;
                let (l_lo, _, _) = Self::loss_and_grad(&logits_for(&r_lo), targets, batch)?;
                // loss_and_grad returns the per-sample SUM; the analytic
                // gradient is normalized by batch, so divide the FD to match.
                let fd = (l_hi - l_lo) / (2.0 * eps * batch as f64);
                let denom = fd.abs().max(grad_r[(i, j)].abs());
                if denom > 1e-8 {
                    worst = worst.max((fd - grad_r[(i, j)]).abs() / denom);
                }
            }
        }
        Ok(worst)
    }
}
