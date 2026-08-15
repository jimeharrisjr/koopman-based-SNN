//! ISI / Poincaré return-map surrogate for the Izhikevich neuron (V2b).
//!
//! The V2 step-by-step surrogate failed on cumulative phase drift: a ~5–10 %
//! per-interval timing bias compounds across a 1000 ms rollout (docs/05,
//! docs/06). This module attacks the timing problem directly by modeling the
//! **return map on the post-reset section** instead of the per-step flow:
//!
//! - Section coordinate: `(u₊, I)` — after a spike, `v` is pinned at `c`, so
//!   the only free state is the recovery variable `u₊ = u + d` (plus the
//!   constant current `I`).
//! - **ISI map** `T ≈ w_T · Ψ(u₊, I)`: the time to the next spike.
//! - **Next-section map** `u₊' ≈ w_u · Ψ(u₊, I)`.
//! - **First-spike map** `T₀, u₊¹ ≈ w₀ · Ψ₀(v, u, I)`, trained (per the
//!   registered refinement, docs/07 §1) on states sampled every 0.5 ms along
//!   each training trajectory before its first crossing, labeled with the
//!   remaining time to that crossing.
//!
//! Rollout jumps spike-to-spike: timing error per cycle is the regression
//! residual of the ISI map — the quantity the fit minimizes — rather than an
//! accumulation of per-step operator bias. This is the Koopman view of the
//! *Poincaré map* (cf. isostables, Mauroy–Mezić–Moehlis 2013) rather than of
//! the flow. The surrogate is a **spike-timing model only** (docs/07 scope
//! limitation): it produces no membrane trace.
//!
//! **Experiment outcome: the pre-registered V2b gates FAILED** (repository
//! docs/08): the ISI map achieved perfect ±2 ms spike-train prediction at the
//! interior test current (I = 10) for both gated neuron types, but the
//! first-spike/approach map, sub-rheobase quiescence extrapolation, and the
//! near-rheobase edge all missed their gates. This module is retained as the
//! experimental record and harness support (`examples/v2b_return_map.rs`);
//! it is **not** a supported simulation path.
//!
//! Everything protocol-relevant is frozen by `docs/07-v2b-preregistration.md`:
//! the two dictionary families (B adds the saddle-node observable
//! `g(I) = min(10, (I−4)^(−1/2))`), the z-scored SVD least squares with
//! cutoff 1e-10, the quiescence rule `T_max = min(horizon, 2 × max training
//! interval)`, and the invalid-rollout convention (a non-positive or
//! non-finite predicted interval is an ERROR, never quiescence — an insane
//! fit must not pass the sub-rheobase gate by accident).

use faer::Mat;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::error::SnnError;
use crate::neuron::IzhikevichParams;
use crate::util;

/// Spike cutoff of the Izhikevich model (mV).
const V_PEAK: f64 = 30.0;
/// Registered saddle-node location for the Family-B observable (docs/07 §1).
const RHEOBASE: f64 = 4.0;
/// Registered clamp of the Family-B observable.
const G_CLAMP: f64 = 10.0;
/// Registered SVD cutoff for the least-squares solves.
const SVD_CUTOFF: f64 = 1e-10;
/// Registered runaway guard: more emitted spikes aborts as invalid.
const MAX_SPIKES: usize = 5000;
/// Registered quiescence factor: T_max = min(horizon, factor × max interval).
const T_MAX_FACTOR: f64 = 2.0;

/// The registered saddle-node observable `g(I)`.
fn g_of_i(i_inj: f64) -> f64 {
    if i_inj <= RHEOBASE {
        G_CLAMP
    } else {
        (1.0 / (i_inj - RHEOBASE).sqrt()).min(G_CLAMP)
    }
}

/// The two registered dictionary families (docs/07 §1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DictionaryFamily {
    /// Plain polynomial cross-terms.
    A,
    /// A plus the physics-informed features: `{g, u₊·g}` in Ψ and
    /// `{g, v·g, u·g}` in Ψ₀.
    B,
}

/// Configuration for fitting a return-map surrogate. Defaults follow the
/// registered protocol.
#[derive(Debug, Clone)]
pub struct ReturnMapConfig {
    /// Dictionary degree, registered grid {2, 3, 4}.
    pub degree: usize,
    pub family: DictionaryFamily,
    /// Trajectories per training current (randomized ICs,
    /// `v₀ ~ U[−80, −50]`, `u₀ = b·v₀ + U[−2, 2]`).
    pub n_trajectories: usize,
    /// Held out per current, split by whole trajectory.
    pub n_holdout: usize,
    /// Training trajectory length (ms).
    pub horizon_ms: f64,
    /// Sampling cadence for first-spike training pairs (registered 0.5 ms).
    pub first_sample_interval_ms: f64,
    pub seed: u64,
}

impl Default for ReturnMapConfig {
    fn default() -> Self {
        Self {
            degree: 3,
            family: DictionaryFamily::B,
            n_trajectories: 10,
            n_holdout: 2,
            horizon_ms: 1000.0,
            first_sample_interval_ms: 0.5,
            seed: 42,
        }
    }
}

/// Fit diagnostics; the V2b preconditions (docs/07 §4) gate on these.
#[derive(Debug, Clone)]
pub struct FitDiagnostics {
    pub n_section_samples: usize,
    pub n_first_samples: usize,
    /// Training ISI pairs per spiking current (data floor: ≥ 50 each).
    pub section_counts: Vec<(f64, usize)>,
    /// Training u₊ range per spiking current (for the P-B fixed-point check).
    pub u_ranges: Vec<(f64, f64, f64)>,
    /// Condition numbers of the two least-squares problems.
    pub section_cond: f64,
    pub first_cond: f64,
    /// Numerical full-rank at the registered SVD cutoff.
    pub section_full_rank: bool,
    pub first_full_rank: bool,
    /// P-C screens: relative RMS on held-out trajectories.
    pub heldout_isi_rel_rms: Option<f64>,
    pub heldout_first_rel_rms: Option<f64>,
    /// Reported, not gated.
    pub heldout_next_u_err: Option<f64>,
    /// The registered quiescence cap.
    pub t_max: f64,
}

/// A fitted return-map surrogate.
#[derive(Debug, Clone)]
pub struct IzhikevichReturnMap {
    params: IzhikevichParams,
    degree: usize,
    family: DictionaryFamily,
    w_isi: Vec<f64>,
    w_next: Vec<f64>,
    w_first_t: Vec<f64>,
    w_first_u: Vec<f64>,
    t_max: f64,
}

#[derive(Debug, Clone, Copy)]
struct SpikeRecord {
    t: f64,
    u_plus: f64,
}

/// One fine-Euler training trajectory: crossing-interpolated spikes plus the
/// pre-first-spike state samples at the registered cadence.
struct TrajData {
    spikes: Vec<SpikeRecord>,
    /// `(t, v, u)` sampled every `interval` from t = 0 until the first
    /// crossing (whole horizon if none).
    pre_first: Vec<(f64, f64, f64)>,
}

fn simulate(
    p: &IzhikevichParams,
    h: f64,
    i_inj: f64,
    v0: f64,
    u0: f64,
    horizon_ms: f64,
    sample_interval: f64,
) -> TrajData {
    let steps = (horizon_ms / h).round() as usize;
    let mut v = v0;
    let mut u = u0;
    let mut spikes = Vec::new();
    let mut pre_first = vec![(0.0, v0, u0)];
    let mut next_sample = sample_interval;
    for t in 0..steps {
        let v_prev = v;
        v += h * (0.04 * v * v + 5.0 * v + 140.0 - u + i_inj);
        u += h * (p.a * (p.b * v - u));
        let now = (t + 1) as f64 * h;
        if v >= V_PEAK {
            let frac = if v > v_prev {
                (V_PEAK - v_prev) / (v - v_prev)
            } else {
                1.0
            };
            v = p.c;
            u += p.d;
            spikes.push(SpikeRecord {
                t: (t as f64 + frac) * h,
                u_plus: u,
            });
            next_sample = f64::INFINITY; // stop pre-first sampling forever
        }
        if now >= next_sample {
            pre_first.push((now, v, u));
            next_sample += sample_interval;
        }
    }
    TrajData { spikes, pre_first }
}

/// Monomial features with constant term plus the family-B extras.
/// `vars` are the raw coordinates (section: `[u₊, I]`; first: `[v, u, I]`).
fn build_features(vars: &[f64], i_inj: f64, degree: usize, family: DictionaryFamily) -> Vec<f64> {
    let mut out = vec![1.0];
    fn emit(vars: &[f64], idx: usize, remaining: usize, acc: f64, out: &mut Vec<f64>) {
        if idx == vars.len() {
            out.push(acc);
            return;
        }
        let mut power = 1.0;
        for e in 0..=remaining {
            emit(vars, idx + 1, remaining - e, acc * power, out);
            power *= vars[idx];
        }
    }
    let mut monos = Vec::new();
    emit(vars, 0, degree, 1.0, &mut monos);
    out.extend(monos.into_iter().skip(1)); // skip the duplicate constant
    if family == DictionaryFamily::B {
        let g = g_of_i(i_inj);
        out.push(g);
        // u₊·g on the section; v·g and u·g on the first-spike state.
        for &v in &vars[..vars.len() - 1] {
            out.push(v * g);
        }
    }
    out
}

/// z-scored SVD least squares `Y ≈ W·Φ` with the registered cutoff.
/// Returns (weight rows in RAW feature coordinates, condition number,
/// full-rank flag). The constant feature (row 0) is exempt from z-scoring
/// and absorbs the centering offsets.
fn solve_ls(
    phi_cols: &[Vec<f64>],
    targets: &[Vec<f64>],
) -> Result<(Vec<Vec<f64>>, f64, bool), SnnError> {
    let m = phi_cols.len();
    if m == 0 {
        return Err(SnnError::InvalidParameter(
            "no samples to fit the return map on".into(),
        ));
    }
    let d = phi_cols[0].len();
    if m < d {
        return Err(SnnError::InvalidParameter(format!(
            "underdetermined fit: {m} samples for {d} features"
        )));
    }
    // Training-stat z-scores (row 0 = constant: exempt).
    let mut mu = vec![0.0f64; d];
    let mut sigma = vec![1.0f64; d];
    for i in 1..d {
        let mean = phi_cols.iter().map(|c| c[i]).sum::<f64>() / m as f64;
        let var = phi_cols.iter().map(|c| (c[i] - mean).powi(2)).sum::<f64>() / m as f64;
        mu[i] = mean;
        sigma[i] = var.sqrt().max(1e-12);
    }
    let phi = Mat::from_fn(d, m, |i, c| {
        if i == 0 {
            1.0
        } else {
            (phi_cols[c][i] - mu[i]) / sigma[i]
        }
    });
    let svd = phi
        .svd()
        .map_err(|e| SnnError::NumericalError(format!("SVD failed: {e:?}")))?;
    let s_col = svd.S().column_vector();
    let s: Vec<f64> = (0..s_col.nrows()).map(|i| s_col[i]).collect();
    let s_max = s.first().copied().unwrap_or(0.0);
    if s_max <= 0.0 {
        return Err(SnnError::NumericalError("zero feature matrix".into()));
    }
    let rank = s.iter().filter(|&&x| x >= SVD_CUTOFF * s_max).count();
    let full_rank = rank == d;
    let cond = s_max / s[rank.saturating_sub(1)].max(f64::MIN_POSITIVE);
    let u_mat = svd.U(); // d × d
    let v_mat = svd.V(); // m × d

    let mut weights = Vec::with_capacity(targets.len());
    for y in targets {
        debug_assert_eq!(y.len(), m);
        // w_z = y · V · Σ⁺ · Uᵀ, cutoff-regularized.
        let mut yv = vec![0.0f64; d];
        for (j, yv_j) in yv.iter_mut().enumerate() {
            if s[j] < SVD_CUTOFF * s_max {
                continue;
            }
            let mut acc = 0.0;
            for c in 0..m {
                acc += y[c] * v_mat[(c, j)];
            }
            *yv_j = acc / s[j];
        }
        let mut w_z = vec![0.0f64; d];
        for (i, w_zi) in w_z.iter_mut().enumerate() {
            let mut acc = 0.0;
            for (j, &yv_j) in yv.iter().enumerate() {
                acc += yv_j * u_mat[(i, j)];
            }
            *w_zi = acc;
        }
        // Fold the z-scoring back into raw-feature weights.
        let mut w = vec![0.0f64; d];
        let mut const_shift = 0.0;
        for i in 1..d {
            w[i] = w_z[i] / sigma[i];
            const_shift += w_z[i] * mu[i] / sigma[i];
        }
        w[0] = w_z[0] - const_shift;
        weights.push(w);
    }
    Ok((weights, cond, full_rank))
}

fn dot(w: &[f64], phi: &[f64]) -> f64 {
    w.iter().zip(phi).map(|(a, b)| a * b).sum()
}

impl IzhikevichReturnMap {
    /// Fit the surrogate from fine-Euler reference data at step `h_ref`
    /// (the per-type self-converged h), following the registered protocol.
    pub fn fit(
        params: &IzhikevichParams,
        h_ref: f64,
        train_currents: &[f64],
        cfg: &ReturnMapConfig,
    ) -> Result<(Self, FitDiagnostics), SnnError> {
        if train_currents.is_empty() {
            return Err(SnnError::InvalidParameter(
                "at least one training current is required".into(),
            ));
        }
        if !(1..=6).contains(&cfg.degree) {
            return Err(SnnError::InvalidParameter(format!(
                "degree {} outside the sane range 1..=6",
                cfg.degree
            )));
        }
        if !(util::is_positive(h_ref)
            && util::is_positive(cfg.horizon_ms)
            && util::is_positive(cfg.first_sample_interval_ms))
        {
            return Err(SnnError::InvalidParameter(
                "h_ref, horizon_ms, first_sample_interval_ms must be positive".into(),
            ));
        }
        if cfg.n_trajectories == 0 || cfg.n_holdout >= cfg.n_trajectories {
            return Err(SnnError::InvalidParameter(format!(
                "need n_holdout < n_trajectories (got {} / {})",
                cfg.n_holdout, cfg.n_trajectories
            )));
        }

        let mut rng = StdRng::seed_from_u64(cfg.seed);
        // Section samples [u₊, I, T, u₊'] and first samples [v, u, I, T_rem, u₊¹].
        let mut sec_train: Vec<[f64; 4]> = Vec::new();
        let mut sec_hold: Vec<[f64; 4]> = Vec::new();
        let mut first_train: Vec<[f64; 5]> = Vec::new();
        // Holdout first-spike check uses only the t = 0 state per trajectory
        // (time-to-first-spike from the IC), avoiding the small-denominator
        // blowup of late-segment samples.
        let mut first_hold_ic: Vec<[f64; 4]> = Vec::new(); // v0, u0, I, t1
        let mut max_interval = 0.0f64;
        let mut section_counts: Vec<(f64, usize)> = Vec::new();
        let mut u_ranges: Vec<(f64, f64, f64)> = Vec::new();

        for &i_inj in train_currents {
            let mut count_here = 0usize;
            let mut u_min = f64::INFINITY;
            let mut u_max = f64::NEG_INFINITY;
            for r in 0..cfg.n_trajectories {
                let v0 = rng.random_range(-80.0..-50.0);
                let u0 = params.b * v0 + rng.random_range(-2.0..2.0);
                let traj = simulate(
                    params,
                    h_ref,
                    i_inj,
                    v0,
                    u0,
                    cfg.horizon_ms,
                    cfg.first_sample_interval_ms,
                );
                let is_train = r < cfg.n_trajectories - cfg.n_holdout;
                if let Some(first) = traj.spikes.first() {
                    if is_train {
                        max_interval = max_interval.max(first.t);
                        for &(t, v, u) in &traj.pre_first {
                            if t < first.t {
                                first_train.push([v, u, i_inj, first.t - t, first.u_plus]);
                            }
                        }
                    } else {
                        first_hold_ic.push([v0, u0, i_inj, first.t]);
                    }
                }
                for w in traj.spikes.windows(2) {
                    let isi = w[1].t - w[0].t;
                    let sample = [w[0].u_plus, i_inj, isi, w[1].u_plus];
                    if is_train {
                        max_interval = max_interval.max(isi);
                        u_min = u_min.min(w[0].u_plus);
                        u_max = u_max.max(w[0].u_plus);
                        count_here += 1;
                        sec_train.push(sample);
                    } else {
                        sec_hold.push(sample);
                    }
                }
            }
            if count_here > 0 {
                section_counts.push((i_inj, count_here));
                u_ranges.push((i_inj, u_min, u_max));
            }
        }
        if sec_train.len() < 10 || first_train.is_empty() {
            return Err(SnnError::InvalidParameter(format!(
                "insufficient spiking training data ({} section samples, {} \
                 first-spike samples) — are the training currents supra-threshold?",
                sec_train.len(),
                first_train.len()
            )));
        }

        let sec_phi: Vec<Vec<f64>> = sec_train
            .iter()
            .map(|s| build_features(&s[0..2], s[1], cfg.degree, cfg.family))
            .collect();
        let sec_targets = vec![
            sec_train.iter().map(|s| s[2]).collect::<Vec<f64>>(),
            sec_train.iter().map(|s| s[3]).collect::<Vec<f64>>(),
        ];
        let (sec_w, section_cond, section_full_rank) = solve_ls(&sec_phi, &sec_targets)?;

        let first_phi: Vec<Vec<f64>> = first_train
            .iter()
            .map(|s| build_features(&s[0..3], s[2], cfg.degree, cfg.family))
            .collect();
        let first_targets = vec![
            first_train.iter().map(|s| s[3]).collect::<Vec<f64>>(),
            first_train.iter().map(|s| s[4]).collect::<Vec<f64>>(),
        ];
        let (first_w, first_cond, first_full_rank) = solve_ls(&first_phi, &first_targets)?;

        let surrogate = Self {
            params: params.clone(),
            degree: cfg.degree,
            family: cfg.family,
            w_isi: sec_w[0].clone(),
            w_next: sec_w[1].clone(),
            w_first_t: first_w[0].clone(),
            w_first_u: first_w[1].clone(),
            t_max: (T_MAX_FACTOR * max_interval).min(cfg.horizon_ms),
        };

        let (mut isi_sq, mut next_sq) = (0.0f64, 0.0f64);
        for s in &sec_hold {
            let rel = (surrogate.predicted_isi(s[0], s[1]) - s[2]) / s[2];
            isi_sq += rel * rel;
            let du = surrogate.next_section(s[0], s[1]) - s[3];
            next_sq += du * du;
        }
        let mut first_sq = 0.0f64;
        for s in &first_hold_ic {
            let rel = (surrogate.predicted_first(s[0], s[1], s[2]) - s[3]) / s[3];
            first_sq += rel * rel;
        }
        let diagnostics = FitDiagnostics {
            n_section_samples: sec_train.len(),
            n_first_samples: first_train.len(),
            section_counts,
            u_ranges,
            section_cond,
            first_cond,
            section_full_rank,
            first_full_rank,
            heldout_isi_rel_rms: (!sec_hold.is_empty())
                .then(|| (isi_sq / sec_hold.len() as f64).sqrt()),
            heldout_first_rel_rms: (!first_hold_ic.is_empty())
                .then(|| (first_sq / first_hold_ic.len() as f64).sqrt()),
            heldout_next_u_err: (!sec_hold.is_empty())
                .then(|| (next_sq / sec_hold.len() as f64).sqrt()),
            t_max: surrogate.t_max,
        };
        Ok((surrogate, diagnostics))
    }

    /// The registered quiescence cap.
    pub fn t_max(&self) -> f64 {
        self.t_max
    }

    /// One application of the ISI map.
    pub fn predicted_isi(&self, u_plus: f64, i_inj: f64) -> f64 {
        dot(
            &self.w_isi,
            &build_features(&[u_plus, i_inj], i_inj, self.degree, self.family),
        )
    }

    /// One application of the next-section map.
    pub fn next_section(&self, u_plus: f64, i_inj: f64) -> f64 {
        dot(
            &self.w_next,
            &build_features(&[u_plus, i_inj], i_inj, self.degree, self.family),
        )
    }

    /// Predicted time from an arbitrary state to the first crossing.
    pub fn predicted_first(&self, v: f64, u: f64, i_inj: f64) -> f64 {
        dot(
            &self.w_first_t,
            &build_features(&[v, u, i_inj], i_inj, self.degree, self.family),
        )
    }

    /// Numerical contraction |∂u₊'/∂u₊| at `(u_plus, i_inj)` — the P-B
    /// stability check (a stable limit cycle's return map contracts at its
    /// fixed point).
    pub fn contraction_at(&self, u_plus: f64, i_inj: f64) -> f64 {
        let eps = 1e-4 * u_plus.abs().max(1.0);
        let hi = self.next_section(u_plus + eps, i_inj);
        let lo = self.next_section(u_plus - eps, i_inj);
        ((hi - lo) / (2.0 * eps)).abs()
    }

    /// Predicted spike times over `horizon_ms` from the model's standard
    /// initial condition (`v = −65`, `u = b·v`) at constant current `i_inj`.
    ///
    /// Registered semantics: predicted interval > `T_max` → quiescence (train
    /// ends); non-positive or non-finite prediction → **error** (invalid
    /// configuration, never quiescence); > 5000 spikes → error.
    pub fn rollout(&self, i_inj: f64, horizon_ms: f64) -> Result<Vec<f64>, SnnError> {
        if !i_inj.is_finite() || !util::is_positive(horizon_ms) {
            return Err(SnnError::InvalidParameter(format!(
                "bad rollout inputs (I = {i_inj}, horizon = {horizon_ms})"
            )));
        }
        let v0 = -65.0;
        let u0 = self.params.b * v0;
        let phi0 = build_features(&[v0, u0, i_inj], i_inj, self.degree, self.family);
        let t0 = dot(&self.w_first_t, &phi0);
        if !t0.is_finite() || t0 <= 0.0 {
            return Err(SnnError::NumericalError(format!(
                "invalid first-spike prediction {t0} at I = {i_inj}"
            )));
        }
        if t0 > self.t_max || t0 > horizon_ms {
            return Ok(Vec::new()); // quiescent
        }
        let mut spikes = vec![t0];
        let mut u_plus = dot(&self.w_first_u, &phi0);
        let mut t = t0;
        loop {
            if !u_plus.is_finite() {
                return Err(SnnError::NumericalError(format!(
                    "section state diverged after {} spikes",
                    spikes.len()
                )));
            }
            let isi = self.predicted_isi(u_plus, i_inj);
            if !isi.is_finite() || isi <= 0.0 {
                return Err(SnnError::NumericalError(format!(
                    "invalid ISI prediction {isi} after {} spikes",
                    spikes.len()
                )));
            }
            if isi > self.t_max {
                break; // quiescent from here on
            }
            t += isi;
            if t > horizon_ms {
                break;
            }
            spikes.push(t);
            if spikes.len() > MAX_SPIKES {
                return Err(SnnError::NumericalError(
                    "rollout exceeded the registered spike-count guard".into(),
                ));
            }
            u_plus = self.next_section(u_plus, i_inj);
        }
        Ok(spikes)
    }

    /// Feature dimension of the section maps.
    pub fn section_dim(&self) -> usize {
        self.w_isi.len()
    }

    /// Registered cost convention (docs/07 §5): 2·d² multiply-adds per
    /// emitted spike.
    pub fn flops_per_spike(&self) -> usize {
        2 * self.section_dim() * self.section_dim()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rs() -> IzhikevichParams {
        IzhikevichParams::regular_spiking(1.0)
    }

    /// Coarser-than-experiment h keeps unit tests fast; the science runs at
    /// the self-converged h in examples/v2b_return_map.rs.
    const H_TEST: f64 = 0.005;

    #[test]
    fn dictionary_dimensions_match_the_registration() {
        // docs/07 §1: section d (A) = 6/10/15, (B) = 8/12/17;
        // first-spike d₀ (A) = 10/20/35, (B) = 13/23/38.
        for (deg, d_a, d_b, d0_a, d0_b) in
            [(2, 6, 8, 10, 13), (3, 10, 12, 20, 23), (4, 15, 17, 35, 38)]
        {
            assert_eq!(
                build_features(&[1.0, 2.0], 2.0, deg, DictionaryFamily::A).len(),
                d_a
            );
            assert_eq!(
                build_features(&[1.0, 2.0], 2.0, deg, DictionaryFamily::B).len(),
                d_b
            );
            assert_eq!(
                build_features(&[1.0, 2.0, 3.0], 3.0, deg, DictionaryFamily::A).len(),
                d0_a
            );
            assert_eq!(
                build_features(&[1.0, 2.0, 3.0], 3.0, deg, DictionaryFamily::B).len(),
                d0_b
            );
        }
    }

    #[test]
    fn g_observable_matches_the_registered_clamp() {
        assert_eq!(g_of_i(2.0), 10.0); // at/below rheobase: clamp value
        assert_eq!(g_of_i(4.0), 10.0);
        assert!((g_of_i(8.0) - 0.5).abs() < 1e-12); // 1/sqrt(4)
        assert_eq!(g_of_i(4.005), 10.0); // clamped just above rheobase
    }

    #[test]
    fn fit_reports_heldout_diagnostics_and_reasonable_isi_error() {
        // Degree 2 / Family B: the highest-capacity config whose pure-I
        // feature set ({1, I, I², g} — 4 functions) can still be full-rank on
        // the 4 spiking training currents. Higher degrees are exactly
        // collinear on 4 nodes and legitimately fail the P-A precondition —
        // a protocol consequence recorded in the V2b results.
        let (_, diag) = IzhikevichReturnMap::fit(
            &rs(),
            H_TEST,
            &[5.0, 7.0, 9.0, 11.0],
            &ReturnMapConfig {
                degree: 2,
                ..ReturnMapConfig::default()
            },
        )
        .unwrap();
        assert!(diag.n_section_samples > 50);
        // The registered ≥ 500 floor is enforced by the experiment harness
        // (with its extend-trajectories contingency); here just confirm the
        // segment-sampling refinement multiplies the per-trajectory yield.
        assert!(diag.n_first_samples > 100);
        assert!(diag.section_full_rank && diag.first_full_rank);
        let rel = diag.heldout_isi_rel_rms.unwrap();
        assert!(
            rel < 0.05,
            "held-out ISI relative error {rel:.4} above the P-C screen"
        );
    }

    #[test]
    fn rollout_tracks_reference_spike_count_at_training_current() {
        let (map, _) = IzhikevichReturnMap::fit(
            &rs(),
            H_TEST,
            &[5.0, 7.0, 9.0, 11.0],
            &ReturnMapConfig::default(),
        )
        .unwrap();
        let reference = simulate(&rs(), H_TEST, 9.0, -65.0, 0.2 * -65.0, 1000.0, 0.5);
        let predicted = map.rollout(9.0, 1000.0).unwrap();
        assert!(!predicted.is_empty());
        let diff = (predicted.len() as i64 - reference.spikes.len() as i64).abs();
        assert!(
            diff <= 2,
            "spike count {} vs reference {} at a training current",
            predicted.len(),
            reference.spikes.len()
        );
    }

    #[test]
    fn contraction_is_stable_at_the_tonic_fixed_point() {
        let (map, _) = IzhikevichReturnMap::fit(
            &rs(),
            H_TEST,
            &[5.0, 7.0, 9.0, 11.0],
            &ReturnMapConfig::default(),
        )
        .unwrap();
        let mut u = 0.0;
        for _ in 0..100 {
            let next = map.next_section(u, 9.0);
            if (next - u).abs() < 1e-12 {
                break;
            }
            u = next;
        }
        let contraction = map.contraction_at(u, 9.0);
        assert!(
            contraction < 1.0,
            "next-section map not contracting at its fixed point: {contraction}"
        );
    }

    #[test]
    fn config_validation_rejects_bad_inputs() {
        let cfg = ReturnMapConfig::default();
        assert!(IzhikevichReturnMap::fit(&rs(), H_TEST, &[], &cfg).is_err());
        assert!(IzhikevichReturnMap::fit(
            &rs(),
            H_TEST,
            &[9.0],
            &ReturnMapConfig {
                degree: 0,
                ..cfg.clone()
            }
        )
        .is_err());
        // All-sub-rheobase currents → no spiking data.
        assert!(IzhikevichReturnMap::fit(&rs(), H_TEST, &[1.0, 2.0], &cfg).is_err());
        let (map, _) = IzhikevichReturnMap::fit(&rs(), H_TEST, &[5.0, 9.0], &cfg).unwrap();
        assert!(map.rollout(f64::NAN, 1000.0).is_err());
        assert!(map.rollout(9.0, 0.0).is_err());
    }
}
