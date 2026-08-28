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
    /// Learn per-neuron LIF time constants (improvements.md P1.1): layers
    /// built with [`KoopmanLayer::lif_hetero`](crate::KoopmanLayer::lif_hetero)
    /// get their τ_m/τ_s trained by backprop through the closed-form
    /// propagator entries (α, β, γ, δ, 1−β are analytic functions of τ).
    /// Parameters are learned in log-space (scale-free) under the same
    /// optimizer, then clamped to τ_m ∈ [5, 100] ms, τ_s ∈ [2, 50] ms with
    /// τ_m ≥ 1.2·τ_s (keeps the γ formulas away from the degenerate limit).
    pub learn_tau: bool,
    /// Temporal aggregation of the readout (see [`ReadoutMode`]).
    /// `readout_decay` is only valid with `ReadoutMode::Count`.
    pub readout_mode: ReadoutMode,
}

/// Learnable-τ clamp bounds (ms) — see [`TrainConfig::learn_tau`].
const TAU_M_RANGE: (f64, f64) = (5.0, 100.0);
const TAU_S_RANGE: (f64, f64) = (2.0, 50.0);
const TAU_SEPARATION: f64 = 1.2;

/// How the readout aggregates output spikes over time (docs/16). Every mode
/// reduces **exactly** to the uniform count readout at its initialization, so
/// a trained-readout run starts bit-for-bit (Count/StaticProfile) or
/// FP-roundoff-close (SpikeAttention) at the baseline it is compared against
/// — the round-2/round-6 "grow from identity" discipline.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ReadoutMode {
    /// `c = Σ_t s_t / T` — the campaign default (with `readout_decay` this
    /// becomes the leaky trace of round 5's AC).
    Count,
    /// Learned static per-bin profile: `c = Σ_t w_t·s_t / T`, `w` initialized
    /// to all-ones (= Count exactly) and trained jointly.
    StaticProfile { t_steps: usize },
    /// Spike-driven temporal attention: scores `z_t = u·s_t`, weights
    /// `a = softmax_t(z)` per sample, `c = Σ_t a_t·s_t`; the query `u` is
    /// zero-initialized (uniform attention = Count) and trained jointly.
    SpikeAttention,
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
            learn_tau: false,
            readout_mode: ReadoutMode::Count,
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
    skip: Vec<Option<Mat<f64>>>,
    r: Mat<f64>,
    /// Per-layer propagator-entry gradients for learnable τ (`5 × n`, rows
    /// ordered `[α, β, γ, δ, b₂]` to match `lif_entry_grads`); `None` for
    /// layers without τ metadata or when `learn_tau` is off. The chain rule
    /// to τ itself is applied once, at update time.
    tau: Vec<Option<Mat<f64>>>,
    /// Static-profile readout gradient (`1 × t_steps`), when that mode is on.
    prof: Option<Mat<f64>>,
    /// Attention-query gradient (`n_out × 1`), when that mode is on.
    attn_u: Option<Mat<f64>>,
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
        for (a, b) in self
            .rec
            .iter_mut()
            .zip(&other.rec)
            .chain(self.skip.iter_mut().zip(&other.skip))
        {
            if let (Some(a), Some(b)) = (a.as_mut(), b.as_ref()) {
                for c in 0..a.ncols() {
                    for i in 0..a.nrows() {
                        a[(i, c)] += b[(i, c)];
                    }
                }
            }
        }
        for (a, b) in self.tau.iter_mut().zip(&other.tau) {
            if let (Some(a), Some(b)) = (a.as_mut(), b.as_ref()) {
                for c in 0..a.ncols() {
                    for i in 0..a.nrows() {
                        a[(i, c)] += b[(i, c)];
                    }
                }
            }
        }
        for (a, b) in [
            (self.prof.as_mut(), other.prof.as_ref()),
            (self.attn_u.as_mut(), other.attn_u.as_ref()),
        ] {
            if let (Some(a), Some(b)) = (a, b) {
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
    /// Optimizer state for skip weights, per layer (None = no skip).
    opt_skip: Vec<Option<Adam>>,
    opt_r: Adam,
    /// Learnable-τ parameters in log-space, per layer (`2 × n`: row 0 =
    /// ln τ_m, row 1 = ln τ_s); `None` for layers without τ metadata.
    tau_rho: Vec<Option<Mat<f64>>>,
    opt_tau: Vec<Option<Adam>>,
    /// Static temporal profile `1 × t_steps` (StaticProfile mode; init 1).
    w_prof: Option<Mat<f64>>,
    opt_prof: Option<Adam>,
    /// Attention query `n_out × 1` (SpikeAttention mode; init 0).
    attn_u: Option<Mat<f64>>,
    opt_attn: Option<Adam>,
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
        let mut opt_skip = Vec::with_capacity(net.n_layers());
        let mut tau_rho = Vec::with_capacity(net.n_layers());
        let mut opt_tau = Vec::with_capacity(net.n_layers());
        for l in 0..net.n_layers() {
            let w = net.layer(l).weights();
            opt_w.push(Adam::new(w.nrows(), w.ncols()));
            opt_rec.push(
                net.layer(l)
                    .recurrent_weights()
                    .map(|wr| Adam::new(wr.nrows(), wr.ncols())),
            );
            opt_skip.push(
                net.layer(l)
                    .skip_weights()
                    .map(|ws| Adam::new(ws.nrows(), ws.ncols())),
            );
            let meta = if cfg.learn_tau {
                net.layer(l).lif_taus()
            } else {
                None
            };
            match meta {
                Some(meta) => {
                    let n = meta.taus_m.len();
                    let mut rho = Mat::<f64>::zeros(2, n);
                    for j in 0..n {
                        rho[(0, j)] = meta.taus_m[j].ln();
                        rho[(1, j)] = meta.taus_s[j].ln();
                    }
                    tau_rho.push(Some(rho));
                    opt_tau.push(Some(Adam::new(2, n)));
                }
                None => {
                    tau_rho.push(None);
                    opt_tau.push(None);
                }
            }
        }
        if cfg.learn_tau && tau_rho.iter().all(Option::is_none) {
            return Err(SnnError::InvalidParameter(
                "learn_tau is set but no layer carries LIF-τ metadata — build \
                 layers with KoopmanLayer::lif_hetero"
                    .into(),
            ));
        }
        if cfg.readout_decay.is_some() && cfg.readout_mode != ReadoutMode::Count {
            return Err(SnnError::InvalidParameter(
                "readout_decay is only valid with ReadoutMode::Count".into(),
            ));
        }
        let (w_prof, opt_prof) = match cfg.readout_mode {
            ReadoutMode::StaticProfile { t_steps } => {
                if t_steps == 0 {
                    return Err(SnnError::InvalidParameter(
                        "StaticProfile needs t_steps ≥ 1".into(),
                    ));
                }
                // All-ones profile: exactly the count readout at step 0.
                (
                    Some(Mat::from_fn(1, t_steps, |_, _| 1.0)),
                    Some(Adam::new(1, t_steps)),
                )
            }
            _ => (None, None),
        };
        let (attn_u, opt_attn) = match cfg.readout_mode {
            // Zero query: uniform attention = the count readout at step 0.
            ReadoutMode::SpikeAttention => {
                (Some(Mat::zeros(n_out, 1)), Some(Adam::new(n_out, 1)))
            }
            _ => (None, None),
        };
        let opt_r = Adam::new(n_classes, n_out);
        Ok(Self {
            cfg,
            r,
            opt_w,
            opt_rec,
            opt_skip,
            opt_r,
            tau_rho,
            opt_tau,
            w_prof,
            opt_prof,
            attn_u,
            opt_attn,
        })
    }

    /// The learned static temporal profile (StaticProfile mode).
    pub fn temporal_profile(&self) -> Option<&Mat<f64>> {
        self.w_prof.as_ref()
    }

    /// The learned attention query (SpikeAttention mode).
    pub fn attention_query(&self) -> Option<&Mat<f64>> {
        self.attn_u.as_ref()
    }

    /// Attention concentration on a batch: the mean over samples of
    /// `max_t a_t`. Uniform attention gives exactly `1/t_steps`; larger
    /// values mean the readout has learned to weight some bins over others.
    /// `Ok(None)` when the readout is not SpikeAttention.
    pub fn attention_concentration(
        &self,
        net: &mut Network,
        inputs: &[SpikeBatch],
    ) -> Result<Option<f64>, SnnError> {
        let (_, _, _, attn, _) = self.forward(net, inputs, false)?;
        Ok(attn.map(|a| {
            let (t_total, batch) = (a.nrows(), a.ncols());
            let mut acc = 0.0;
            for b in 0..batch {
                let mut mx = 0.0f64;
                for t in 0..t_total {
                    mx = mx.max(a[(t, b)]);
                }
                acc += mx;
            }
            acc / batch as f64
        }))
    }

    /// Current per-neuron time constants of learnable-τ layers, per layer
    /// (`None` for layers without τ metadata). For inspection and logging.
    pub fn taus(&self, net: &Network) -> Vec<Option<(Vec<f64>, Vec<f64>)>> {
        (0..net.n_layers())
            .map(|l| {
                net.layer(l)
                    .lif_taus()
                    .map(|m| (m.taus_m.clone(), m.taus_s.clone()))
            })
            .collect()
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
    /// optional learnable-τ tape of per-step (x_pre, drive), optional
    /// attention weights `t_steps × batch`, readout feature `c`
    /// (`n_out × batch`)). The τ tape is collected only when
    /// `cfg.learn_tau` is set and `tau_tape` is requested.
    #[allow(clippy::type_complexity)]
    fn forward(
        &self,
        net: &mut Network,
        inputs: &[SpikeBatch],
        tau_tape: bool,
    ) -> Result<
        (
            Vec<Vec<Mat<f64>>>,
            Vec<Vec<SpikeBatch>>,
            Option<Vec<Vec<(Mat<f64>, Mat<f64>)>>>,
            Option<Mat<f64>>,
            Mat<f64>,
        ),
        SnnError,
    > {
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
        let want_tau = tau_tape && self.cfg.learn_tau;
        net.reset_state();
        let mut v_pre_tape = Vec::with_capacity(inputs.len());
        let mut s_out_tape = Vec::with_capacity(inputs.len());
        let mut tau_tapes: Option<Vec<Vec<(Mat<f64>, Mat<f64>)>>> =
            want_tau.then(|| Vec::with_capacity(inputs.len()));
        let n_out = net.layer(n_layers - 1).n_neurons();
        let mut counts = Mat::<f64>::zeros(n_out, batch);
        if let ReadoutMode::StaticProfile { t_steps } = self.cfg.readout_mode {
            if inputs.len() != t_steps {
                return Err(SnnError::DimensionMismatch(format!(
                    "StaticProfile readout is sized for {t_steps} steps, got {}",
                    inputs.len()
                )));
            }
        }
        // Attention scores z_t = u·s_t, filled online; softmaxed after the
        // rollout (the normalization needs every step).
        let mut attn_z: Option<Mat<f64>> =
            matches!(self.cfg.readout_mode, ReadoutMode::SpikeAttention)
                .then(|| Mat::zeros(inputs.len(), batch));

        for (t, input) in inputs.iter().enumerate() {
            let mut v_pre: Vec<Mat<f64>> = (0..n_layers)
                .map(|l| Mat::zeros(net.layer(l).n_neurons(), batch))
                .collect();
            let mut s_out: Vec<SpikeBatch> = (0..n_layers)
                .map(|l| SpikeBatch::zeros(net.layer(l).n_neurons(), batch))
                .collect::<Result<_, _>>()?;
            if let Some(tapes) = tau_tapes.as_mut() {
                let mut x_pre: Vec<Mat<f64>> = (0..n_layers)
                    .map(|l| {
                        Mat::zeros(net.layer(l).n_neurons() * net.layer(l).n_state_vars(), batch)
                    })
                    .collect();
                let mut drive: Vec<Mat<f64>> = (0..n_layers)
                    .map(|l| Mat::zeros(net.layer(l).n_neurons(), batch))
                    .collect();
                net.step_batch_taped_tau(input, &mut v_pre, &mut s_out, &mut x_pre, &mut drive)?;
                tapes.push(x_pre.into_iter().zip(drive).collect());
            } else {
                net.step_batch_taped(input, &mut v_pre, &mut s_out)?;
            }
            let top = s_out[n_layers - 1].as_mat();
            match self.cfg.readout_mode {
                ReadoutMode::Count => match self.cfg.readout_decay {
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
                },
                ReadoutMode::StaticProfile { .. } => {
                    let w = self.w_prof.as_ref().expect("profile allocated")[(0, t)];
                    for b in 0..batch {
                        for i in 0..n_out {
                            counts[(i, b)] += w * top[(i, b)];
                        }
                    }
                }
                ReadoutMode::SpikeAttention => {
                    let u = self.attn_u.as_ref().expect("query allocated");
                    let z = attn_z.as_mut().expect("scores allocated");
                    for b in 0..batch {
                        let mut acc = 0.0;
                        for i in 0..n_out {
                            acc += u[(i, 0)] * top[(i, b)];
                        }
                        z[(t, b)] = acc;
                    }
                }
            }
            v_pre_tape.push(v_pre);
            s_out_tape.push(s_out);
        }
        // Attention: per-sample softmax over time, then c = Σ_t a_t·s_t.
        let attn = match attn_z {
            None => None,
            Some(z) => {
                let t_total = inputs.len();
                let mut a = Mat::<f64>::zeros(t_total, batch);
                for b in 0..batch {
                    let mut mx = f64::NEG_INFINITY;
                    for t in 0..t_total {
                        mx = mx.max(z[(t, b)]);
                    }
                    let mut sum = 0.0;
                    for t in 0..t_total {
                        let e = (z[(t, b)] - mx).exp();
                        a[(t, b)] = e;
                        sum += e;
                    }
                    for t in 0..t_total {
                        a[(t, b)] /= sum;
                    }
                }
                for (t, s_out) in s_out_tape.iter().enumerate() {
                    let top = s_out[n_layers - 1].as_mat();
                    for b in 0..batch {
                        let w = a[(t, b)];
                        for i in 0..n_out {
                            counts[(i, b)] += w * top[(i, b)];
                        }
                    }
                }
                Some(a)
            }
        };
        Ok((v_pre_tape, s_out_tape, tau_tapes, attn, counts))
    }

    /// Readout normalization: the effective mass of the aggregated feature,
    /// so logits stay O(rate) across modes. Attention weights already sum to
    /// one, so that mode needs no normalization — which is exactly what makes
    /// zero-query attention identical to the count readout.
    fn readout_norm(&self, t_steps: usize) -> f64 {
        match self.cfg.readout_mode {
            ReadoutMode::Count => match self.cfg.readout_decay {
                None => t_steps as f64,
                Some(kappa) => (1.0 - kappa.powi(t_steps as i32)) / (1.0 - kappa),
            },
            ReadoutMode::StaticProfile { .. } => t_steps as f64,
            ReadoutMode::SpikeAttention => 1.0,
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
        let (v_pre_tape, s_out_tape, tau_tapes, attn, counts) = self.forward(net, inputs, true)?;

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
        // Attention backward, softmax part: with ∂L/∂c = ds_top and
        // dLda_t = ds_top·s_t, the score gradient is the softmax Jacobian
        // dz_t = a_t·(dLda_t − Σ_k a_k·dLda_k), computed per sample before
        // the time loop (it couples all steps).
        let attn_dz: Option<Mat<f64>> = attn.as_ref().map(|a| {
            let mut dlda = Mat::<f64>::zeros(t_steps, batch);
            for (t, s_out) in s_out_tape.iter().enumerate() {
                let s_top = s_out[n_layers - 1].as_mat();
                for b in 0..batch {
                    let mut acc = 0.0;
                    for j in 0..n_out {
                        acc += ds_top[(j, b)] * s_top[(j, b)];
                    }
                    dlda[(t, b)] = acc;
                }
            }
            let mut dz = Mat::<f64>::zeros(t_steps, batch);
            for b in 0..batch {
                let mut dot = 0.0;
                for t in 0..t_steps {
                    dot += a[(t, b)] * dlda[(t, b)];
                }
                for t in 0..t_steps {
                    dz[(t, b)] = a[(t, b)] * (dlda[(t, b)] - dot);
                }
            }
            dz
        });

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
        // Skip-weight gradients, where present.
        let mut grad_skip: Vec<Option<Mat<f64>>> = (0..n_layers)
            .map(|l| {
                net.layer(l)
                    .skip_weights()
                    .map(|ws| Mat::zeros(ws.nrows(), ws.ncols()))
            })
            .collect();
        // Propagator-entry gradients for learnable τ (rows [α, β, γ, δ, b₂]).
        let mut grad_tau: Vec<Option<Mat<f64>>> = (0..n_layers)
            .map(|l| {
                self.tau_rho[l]
                    .as_ref()
                    .map(|_| Mat::zeros(5, net.layer(l).n_neurons()))
            })
            .collect();
        // Trained-readout gradients, where those modes are active.
        let mut grad_prof: Option<Mat<f64>> =
            self.w_prof.as_ref().map(|w| Mat::zeros(1, w.ncols()));
        let mut grad_attn: Option<Mat<f64>> =
            self.attn_u.as_ref().map(|u| Mat::zeros(u.nrows(), 1));

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
                    match self.cfg.readout_mode {
                        ReadoutMode::Count => {
                            for b in 0..batch {
                                for j in 0..n {
                                    g_s[(j, b)] += readout_scale * ds_top[(j, b)];
                                }
                            }
                        }
                        ReadoutMode::StaticProfile { .. } => {
                            // ∂c/∂s_t = w_t/T (the /T is inside ds_top via
                            // norm); ∂L/∂w_t = ds_top·s_t summed over (j, b).
                            let w = self.w_prof.as_ref().expect("profile")[(0, t)];
                            let s_top = s_out_tape[t][l].as_mat();
                            let gp = grad_prof.as_mut().expect("profile grads");
                            let mut acc = 0.0;
                            for b in 0..batch {
                                for j in 0..n {
                                    g_s[(j, b)] += w * ds_top[(j, b)];
                                    acc += ds_top[(j, b)] * s_top[(j, b)];
                                }
                            }
                            gp[(0, t)] += acc;
                        }
                        ReadoutMode::SpikeAttention => {
                            // Two spike paths: through the weighted sum
                            // (a_t·ds_top) and through the score z_t = u·s_t
                            // (dz_t·u); ∂L/∂u = Σ_t dz_t·s_t.
                            let a = attn.as_ref().expect("attention weights");
                            let dz = attn_dz.as_ref().expect("score grads");
                            let u = self.attn_u.as_ref().expect("query");
                            let s_top = s_out_tape[t][l].as_mat();
                            let gu = grad_attn.as_mut().expect("query grads");
                            for b in 0..batch {
                                let (a_tb, dz_tb) = (a[(t, b)], dz[(t, b)]);
                                for j in 0..n {
                                    g_s[(j, b)] += a_tb * ds_top[(j, b)] + dz_tb * u[(j, 0)];
                                    gu[(j, 0)] += dz_tb * s_top[(j, b)];
                                }
                            }
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
                // Skip path: layer l+2 also read this layer's spikes at t.
                if l + 2 < n_layers {
                    if let Some(w_skip) = net.layer(l + 2).skip_weights() {
                        let d_next = &dldd[l + 2];
                        for b in 0..batch {
                            for j in 0..n {
                                let mut acc = 0.0;
                                for i in 0..w_skip.nrows() {
                                    acc += w_skip[(i, j)] * d_next[(i, b)];
                                }
                                g_s[(j, b)] += acc;
                            }
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
                // Learnable τ: dldy is exactly ∂L/∂(A·x_t + b⊙d), so the
                // entry gradients are outer products with the taped pre-step
                // state and drive:
                //   ∂L/∂α_j = Σ_b dldy_v · v_t     ∂L/∂β_j  = Σ_b dldy_i · i_t
                //   ∂L/∂γ_j = Σ_b dldy_v · i_t     ∂L/∂δ_j  = Σ_b dldy_v · d_j
                //   ∂L/∂b₂_j = Σ_b dldy_i · d_j
                if let (Some(gt), Some(tapes)) = (grad_tau[l].as_mut(), tau_tapes.as_ref()) {
                    let (x_pre, drv) = &tapes[t][l];
                    for b in 0..batch {
                        for j in 0..n {
                            let gv = dldy[l][(j, b)];
                            let gi = dldy[l][(n + j, b)];
                            let v_t = x_pre[(j, b)];
                            let i_t = x_pre[(n + j, b)];
                            let d_j = drv[(j, b)];
                            gt[(0, j)] += gv * v_t;
                            gt[(1, j)] += gi * i_t;
                            gt[(2, j)] += gv * i_t;
                            gt[(3, j)] += gv * d_j;
                            gt[(4, j)] += gi * d_j;
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
                // ∂L/∂W_skip += ∂L/∂d_t · s_{l−2}(t)ᵀ (same-step input;
                // Network::new guarantees l ≥ 2 when skip weights exist).
                if let Some(grad) = grad_skip[l].as_mut() {
                    let s_skip_mat = s_out_tape[t][l - 2].as_mat();
                    for b in 0..batch {
                        for j in 0..s_skip_mat.nrows() {
                            let s = s_skip_mat[(j, b)];
                            if s == 0.0 {
                                continue;
                            }
                            for i in 0..n {
                                grad[(i, j)] += dldd[l][(i, b)] * s;
                            }
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
            skip: grad_skip,
            r: grad_r,
            tau: grad_tau,
            prof: grad_prof,
            attn_u: grad_attn,
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
        let mut grad_skip = grads.skip.clone();
        let mut grad_r = grads.r.clone();
        if let Some(clip) = self.cfg.grad_clip {
            for g in grad_w
                .iter_mut()
                .chain(grad_rec.iter_mut().flatten())
                .chain(grad_skip.iter_mut().flatten())
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
        for (l, (opt_slot, grad_slot)) in self.opt_skip.iter_mut().zip(&grad_skip).enumerate() {
            if let (Some(opt), Some(grad)) = (opt_slot.as_mut(), grad_slot.as_ref()) {
                if let Some(w_skip) = net.layer_mut(l).skip_weights_mut() {
                    opt.update(w_skip, grad, &self.cfg.optim);
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

        // Trained-readout parameters, clipped consistently with the other
        // parameter groups.
        let clip_mat = |g: &Mat<f64>, clip: Option<f64>| -> Mat<f64> {
            let mut g = g.clone();
            if let Some(c) = clip {
                for col in 0..g.ncols() {
                    for i in 0..g.nrows() {
                        g[(i, col)] = g[(i, col)].clamp(-c, c);
                    }
                }
            }
            g
        };
        if let (Some(w), Some(opt), Some(g)) = (
            self.w_prof.as_mut(),
            self.opt_prof.as_mut(),
            grads.prof.as_ref(),
        ) {
            let g = clip_mat(g, self.cfg.grad_clip);
            opt.update(w, &g, &self.cfg.optim);
        }
        if let (Some(u), Some(opt), Some(g)) = (
            self.attn_u.as_mut(),
            self.opt_attn.as_mut(),
            grads.attn_u.as_ref(),
        ) {
            let g = clip_mat(g, self.cfg.grad_clip);
            opt.update(u, &g, &self.cfg.optim);
        }

        // Learnable τ: chain the summed entry gradients through the analytic
        // ∂entry/∂τ, then through τ = exp(ρ) (log-space parameters), update ρ
        // with the shared optimizer, clamp, and write the new propagator
        // entries back into the layer.
        for l in 0..n_layers {
            let (Some(rho), Some(opt), Some(gt)) = (
                self.tau_rho[l].as_mut(),
                self.opt_tau[l].as_mut(),
                grads.tau[l].as_ref(),
            ) else {
                continue;
            };
            let Some(meta) = net.layer(l).lif_taus() else {
                continue;
            };
            let n = meta.taus_m.len();
            let (taus_m, taus_s, r_gain, dt) =
                (meta.taus_m.clone(), meta.taus_s.clone(), meta.r, meta.dt);
            let mut g_rho = Mat::<f64>::zeros(2, n);
            for j in 0..n {
                // Clamps keep the pair separated, so the formula cannot fail;
                // treat failure as a zero gradient rather than aborting a run.
                let Ok((_, dm, ds)) =
                    crate::neuron::lif_entry_grads(taus_m[j], taus_s[j], r_gain, dt)
                else {
                    continue;
                };
                let mut g_tm = 0.0;
                let mut g_ts = 0.0;
                for k in 0..5 {
                    g_tm += gt[(k, j)] * dm[k];
                    g_ts += gt[(k, j)] * ds[k];
                }
                // dτ/dρ = τ.
                g_rho[(0, j)] = g_tm * taus_m[j];
                g_rho[(1, j)] = g_ts * taus_s[j];
            }
            if let Some(clip) = self.cfg.grad_clip {
                for j in 0..n {
                    g_rho[(0, j)] = g_rho[(0, j)].clamp(-clip, clip);
                    g_rho[(1, j)] = g_rho[(1, j)].clamp(-clip, clip);
                }
            }
            opt.update(rho, &g_rho, &self.cfg.optim);
            // Map back to τ, clamp to the trusted region, re-sync ρ.
            let mut new_tm = vec![0.0; n];
            let mut new_ts = vec![0.0; n];
            for j in 0..n {
                let mut tm = rho[(0, j)].exp().clamp(TAU_M_RANGE.0, TAU_M_RANGE.1);
                let mut ts = rho[(1, j)].exp().clamp(TAU_S_RANGE.0, TAU_S_RANGE.1);
                if tm < TAU_SEPARATION * ts {
                    ts = tm / TAU_SEPARATION;
                    if ts < TAU_S_RANGE.0 {
                        ts = TAU_S_RANGE.0;
                        tm = TAU_SEPARATION * ts;
                    }
                }
                rho[(0, j)] = tm.ln();
                rho[(1, j)] = ts.ln();
                new_tm[j] = tm;
                new_ts[j] = ts;
            }
            net.layer_mut(l)
                .set_lif_taus(&new_tm, &new_ts)
                .expect("clamped taus are always valid");
        }
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
            self.forward(net, inputs, false)?.4
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
                    .map(|(cnet, cin)| scope.spawn(move || Ok(self.forward(cnet, cin, false)?.4)))
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
        let (_, _, _, _, counts) = self.forward(net, inputs, false)?;
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
