//! V2 go/no-go experiment (Phase 4), implementing the frozen protocol of
//! `docs/04-v2-preregistration.md`. This binary measures, applies the
//! registered gates, and prints the decision; the results document records
//! the outcome.
//!
//! Protocol deviations applied via the pre-registered contingencies:
//! - The registered ground-truth h = 0.025 ms FAILED its own self-convergence
//!   check; per §1 the harness halves h per neuron type until every spike at
//!   I = 13 moves < 0.1 ms under a further halving (RS → 0.0015625 ms,
//!   FS → ~0.000195 ms). Thresholds are unchanged.
//!
//! Run: `cargo run --release --example v2_rollout`

use std::time::Instant;

use kdmd_snn::neuron::IzhikevichParams;
use kdmd_snn::surrogate::{IzhikevichSurrogate, SurrogateConfig};
use kdmd_snn::Izhikevich;

const HORIZON_MS: f64 = 1000.0;
const TRAIN_CURRENTS: &[f64] = &[3.0, 5.0, 7.0, 9.0, 11.0];
const EULER_FRONTIER_H: &[f64] = &[0.5, 0.25, 0.1, 0.05];
const DT_GRID: &[f64] = &[0.5, 1.0, 2.0];
const DEGREES: &[usize] = &[2, 3];
const COINC_WINDOW_MS: f64 = 2.0;
const SEGMENT_GUARD_MS: f64 = 5.0;
/// Registered starting point for the ground-truth step.
const H_START: f64 = 0.025;
/// Contingency floor: 0.025 / 2⁸.
const H_FLOOR: f64 = 0.025 / 256.0;

#[derive(Clone, Copy, PartialEq)]
enum Regime {
    Interp,
    Extrap,
    SubRheo,
    Reported,
}

const TEST_CURRENTS: &[(f64, Regime)] = &[
    (6.0, Regime::Interp),
    (10.0, Regime::Interp),
    (13.0, Regime::Extrap),
    (2.0, Regime::SubRheo),
    (4.0, Regime::Reported), // saddle-node ghost: reported, never gated
];

/// A simulated membrane trace with interpolated spike times.
struct Trace {
    /// Sample interval (ms).
    h: f64,
    /// v at k·h, k = 0..=steps (post-reset values).
    vs: Vec<f64>,
    /// Linearly interpolated v = 30 crossings (ms).
    spike_times: Vec<f64>,
    wall_secs: f64,
}

impl Trace {
    fn v_at(&self, t_ms: f64) -> f64 {
        let x = (t_ms / self.h).max(0.0);
        let k = x.floor() as usize;
        let f = x - k as f64;
        if k + 1 >= self.vs.len() {
            *self.vs.last().unwrap()
        } else {
            self.vs[k] * (1.0 - f) + self.vs[k + 1] * f
        }
    }
}

/// Forward-Euler Izhikevich at step `h`, with crossing-interpolated spike
/// times (the reference and the cost-frontier baselines).
fn euler_trace(p: &IzhikevichParams, h: f64, i_inj: f64) -> Trace {
    let steps = (HORIZON_MS / h).round() as usize;
    let mut v = -65.0;
    let mut u = p.b * v;
    let mut vs = Vec::with_capacity(steps + 1);
    vs.push(v);
    let mut spike_times = Vec::new();
    let start = Instant::now();
    for t in 0..steps {
        let v_prev = v;
        v += h * (0.04 * v * v + 5.0 * v + 140.0 - u + i_inj);
        u += h * (p.a * (p.b * v - u));
        if v >= 30.0 {
            let frac = if v > v_prev {
                (30.0 - v_prev) / (v - v_prev)
            } else {
                1.0
            };
            spike_times.push((t as f64 + frac) * h);
            v = p.c;
            u += p.d;
        }
        vs.push(v);
    }
    Trace {
        h,
        vs,
        spike_times,
        wall_secs: start.elapsed().as_secs_f64(),
    }
}

fn surrogate_trace(surrogate: &IzhikevichSurrogate, dt: f64, i_inj: f64) -> Result<Trace, String> {
    let steps = (HORIZON_MS / dt).round() as usize;
    let start = Instant::now();
    let rollout = surrogate.rollout(i_inj, steps).map_err(|e| e.to_string())?;
    let wall_secs = start.elapsed().as_secs_f64();
    let vs: Vec<f64> = (0..=steps).map(|t| rollout.states[(0, t)]).collect();
    let mut spike_times = Vec::new();
    for (t, &fired) in rollout.spikes.iter().enumerate() {
        if fired {
            let v_prev = rollout.states[(0, t)];
            let v_peak = rollout.pre_reset_v[t];
            let frac = if v_peak > v_prev {
                ((30.0 - v_prev) / (v_peak - v_prev)).clamp(0.0, 1.0)
            } else {
                1.0
            };
            spike_times.push((t as f64 + frac) * dt);
        }
    }
    Ok(Trace {
        h: dt,
        vs,
        spike_times,
        wall_secs,
    })
}

/// Greedy one-to-one spike matching within ±window.
fn matched_pairs(reference: &[f64], candidate: &[f64], window: f64) -> Vec<(f64, f64)> {
    let mut used = vec![false; candidate.len()];
    let mut out = Vec::new();
    for &rt in reference {
        let mut best: Option<(usize, f64)> = None;
        for (j, &st) in candidate.iter().enumerate() {
            if used[j] {
                continue;
            }
            let d = (st - rt).abs();
            if d <= window && best.is_none_or(|(_, bd)| d < bd) {
                best = Some((j, d));
            }
        }
        if let Some((j, _)) = best {
            used[j] = true;
            out.push((rt, candidate[j]));
        }
    }
    out
}

struct Accuracy {
    n_ref: usize,
    n_model: usize,
    count_rel_err: f64,
    coincidence: f64,
    first_spike_err: f64,
    v_rmse: f64,
}

fn evaluate(reference: &Trace, model: &Trace) -> Accuracy {
    let n_ref = reference.spike_times.len();
    let n_model = model.spike_times.len();
    let count_rel_err = if n_ref > 0 {
        (n_model as f64 - n_ref as f64).abs() / n_ref as f64
    } else if n_model > 0 {
        f64::INFINITY
    } else {
        0.0
    };
    let pairs = matched_pairs(&reference.spike_times, &model.spike_times, COINC_WINDOW_MS);
    let coincidence = if n_ref > 0 {
        pairs.len() as f64 / n_ref as f64
    } else {
        1.0
    };
    let first_spike_err = match (reference.spike_times.first(), model.spike_times.first()) {
        (Some(&r), Some(&s)) => (r - s).abs(),
        (None, None) => 0.0,
        _ => f64::INFINITY,
    };

    // A4: per matched inter-spike segment, re-indexed from its own spike
    // time, excluding ±5 ms around spikes. Spikeless traces compare the full
    // horizon directly.
    let mut sum_sq = 0.0;
    let mut n_samples = 0usize;
    if n_ref == 0 && n_model == 0 {
        let mut t = 0.0;
        while t <= HORIZON_MS {
            let d = reference.v_at(t) - model.v_at(t);
            sum_sq += d * d;
            n_samples += 1;
            t += model.h;
        }
    } else {
        for w in pairs.windows(2) {
            let (r0, s0) = w[0];
            let (r1, _) = w[1];
            let seg_len = r1 - r0;
            let mut tau = SEGMENT_GUARD_MS;
            while tau <= seg_len - SEGMENT_GUARD_MS {
                let d = reference.v_at(r0 + tau) - model.v_at(s0 + tau);
                sum_sq += d * d;
                n_samples += 1;
                tau += model.h;
            }
        }
    }
    let v_rmse = if n_samples > 0 {
        (sum_sq / n_samples as f64).sqrt()
    } else {
        f64::NAN
    };

    Accuracy {
        n_ref,
        n_model,
        count_rel_err,
        coincidence,
        first_spike_err,
        v_rmse,
    }
}

/// Apply the §3 gates for one current. `Reported` never gates.
fn passes_gates(acc: &Accuracy, regime: Regime) -> bool {
    match regime {
        Regime::Reported => true,
        Regime::SubRheo => acc.n_model == 0 && acc.v_rmse <= 0.5,
        Regime::Interp | Regime::Extrap => {
            let (count_thr, coinc_thr, first_thr, rmse_thr) = match regime {
                Regime::Interp => (0.10, 0.80, 2.0, 1.0),
                _ => (0.15, 0.70, 4.0, 2.0),
            };
            // Absolute slack when N_ref < 10.
            let count_ok = if acc.n_ref < 10 {
                (acc.n_model as i64 - acc.n_ref as i64).abs() <= 1
            } else {
                acc.count_rel_err <= count_thr
            };
            count_ok
                && acc.coincidence >= coinc_thr
                && acc.first_spike_err <= first_thr
                && acc.v_rmse <= rmse_thr
        }
    }
}

fn regime_name(r: Regime) -> &'static str {
    match r {
        Regime::Interp => "interp",
        Regime::Extrap => "extrap",
        Regime::SubRheo => "sub-rheo",
        Regime::Reported => "reported",
    }
}

/// §1 contingency: halve h until every I = 13 spike moves < 0.1 ms under a
/// further halving.
fn find_reference_h(name: &str, p: &IzhikevichParams) -> f64 {
    let mut h = H_START;
    loop {
        let a = euler_trace(p, h, 13.0);
        let b = euler_trace(p, h / 2.0, 13.0);
        let converged = a.spike_times.len() == b.spike_times.len()
            && a.spike_times
                .iter()
                .zip(&b.spike_times)
                .all(|(x, y)| (x - y).abs() < 0.1);
        if converged {
            println!("- {name}: ground-truth h = {h:.7} ms (self-converged)");
            return h;
        }
        if h <= H_FLOOR {
            panic!("{name}: no self-converged h found above the floor {H_FLOOR}");
        }
        h /= 2.0;
    }
}

struct ConfigResult {
    dt: f64,
    preconditions_ok: bool,
    /// All interpolation + sub-rheobase gates passed.
    interp_subrheo_ok: bool,
    /// The extrapolation gate passed.
    extrap_ok: bool,
    flops_per_ms: f64,
    c1_ok: bool,
    wall_ratio: f64,
}

impl ConfigResult {
    fn full_accuracy(&self) -> bool {
        self.interp_subrheo_ok && self.extrap_ok
    }
}

fn main() {
    println!("# V2 experiment — pre-registered protocol (docs/04-v2-preregistration.md)");
    println!("horizon {HORIZON_MS} ms · train I {TRAIN_CURRENTS:?}\n");

    let types: Vec<(&str, IzhikevichParams, bool)> = vec![
        ("RS", IzhikevichParams::regular_spiking(1.0), true),
        ("FS", IzhikevichParams::fast_spiking(1.0), true),
        ("CH", IzhikevichParams::chattering(1.0), false), // reported, not gated
    ];

    println!("## Ground-truth self-convergence (§1 contingency: h halved until < 0.1 ms shift at I = 13)");
    let h_refs: Vec<f64> = types
        .iter()
        .map(|(name, p, _)| find_reference_h(name, p))
        .collect();
    println!();

    // Cache reference traces per (type, current).
    let reference_traces: Vec<Vec<(f64, Trace)>> = types
        .iter()
        .zip(&h_refs)
        .map(|((_, p, _), &h)| {
            TEST_CURRENTS
                .iter()
                .map(|&(i_inj, _)| (i_inj, euler_trace(p, h, i_inj)))
                .collect()
        })
        .collect();
    let reference_for = |type_idx: usize, i_inj: f64| -> &Trace {
        &reference_traces[type_idx]
            .iter()
            .find(|(i, _)| *i == i_inj)
            .unwrap()
            .1
    };

    // §4 frontier: cheapest Euler h that passes every accuracy gate.
    let mut frontier_flops: Vec<(String, Option<f64>, bool)> = Vec::new();
    for (type_idx, (name, p, gated)) in types.iter().enumerate() {
        if !gated {
            continue;
        }
        println!("## Euler cost–accuracy frontier — {name}");
        println!("| h (ms) | flops/ms | verdict |");
        println!("|---|---|---|");
        let mut cheapest: Option<f64> = None;
        let mut coarse_passes = false;
        for &h in EULER_FRONTIER_H {
            let mut all_ok = true;
            for &(i_inj, regime) in TEST_CURRENTS {
                if regime == Regime::Reported {
                    continue;
                }
                let baseline = euler_trace(p, h, i_inj);
                let acc = evaluate(reference_for(type_idx, i_inj), &baseline);
                if !passes_gates(&acc, regime) {
                    all_ok = false;
                }
            }
            let flops = 15.0 / h;
            println!(
                "| {h} | {flops:.0} | {} |",
                if all_ok {
                    "passes accuracy"
                } else {
                    "fails accuracy"
                }
            );
            if all_ok {
                cheapest = Some(match cheapest {
                    Some(c) if c < flops => c,
                    _ => flops,
                });
                if h == 0.5 {
                    coarse_passes = true;
                }
            }
        }
        match cheapest {
            Some(f) => println!("cheapest accuracy-passing Euler: {f:.0} flops/ms\n"),
            None => {
                println!("no Euler on the registered grid passes all accuracy gates\n")
            }
        }
        frontier_flops.push((name.to_string(), cheapest, coarse_passes));
    }

    // Surrogate grid.
    let mut per_type_results: Vec<(String, Vec<ConfigResult>)> = Vec::new();
    for (type_idx, (name, p, gated)) in types.iter().enumerate() {
        println!(
            "## Surrogate grid — {name}{}",
            if *gated { "" } else { " (reported only)" }
        );
        let h_ref = h_refs[type_idx];
        let mut results = Vec::new();
        for &degree in DEGREES {
            for &dt in DT_GRID {
                let substeps = (dt / h_ref).round() as usize;
                let fit_model = Izhikevich::new(IzhikevichParams {
                    dt,
                    substeps,
                    ..p.clone()
                })
                .unwrap();
                let cfg = SurrogateConfig {
                    degree,
                    t_steps: (HORIZON_MS / dt).round() as usize,
                    ..SurrogateConfig::default()
                };
                let fitted = IzhikevichSurrogate::fit(&fit_model, TRAIN_CURRENTS, &cfg);
                let (surrogate, report) = match fitted {
                    Ok(x) => x,
                    Err(e) => {
                        println!("degree {degree}, Δt {dt}: FIT FAILED — {e}");
                        continue;
                    }
                };
                let heldout = report.heldout_rmse.unwrap_or(f64::INFINITY);
                let rho = report.stability.spectral_radius;
                let i_row = surrogate.i_row_deviation();
                let preconditions_ok = heldout <= 1e-3 && rho < 1.0 + 1e-6 && i_row < 1e-6;
                println!(
                    "### degree {degree}, Δt* = {dt} ms  (d = {}, holdout {heldout:.2e}, ρ {rho:.6}, I-row {i_row:.2e}{})",
                    surrogate.lifted_dim(),
                    if preconditions_ok {
                        ""
                    } else {
                        " — PRECONDITION FAILED"
                    },
                );
                println!(
                    "| I | regime | ref N | sur N | count err | coinc ±2ms | first-spike err | v-RMSE | gates |"
                );
                println!("|---|---|---|---|---|---|---|---|---|");
                let mut interp_subrheo_ok = true;
                let mut extrap_ok = true;
                let mut wall_ref_total = 0.0;
                let mut wall_sur_total = 0.0;
                for &(i_inj, regime) in TEST_CURRENTS {
                    let reference = reference_for(type_idx, i_inj);
                    let sur = match surrogate_trace(&surrogate, dt, i_inj) {
                        Ok(t) => t,
                        Err(e) => {
                            println!(
                                "| {i_inj} | {} | — | — | — | — | — | — | DIVERGED: {e} |",
                                regime_name(regime)
                            );
                            match regime {
                                Regime::Reported => {}
                                Regime::Extrap => extrap_ok = false,
                                _ => interp_subrheo_ok = false,
                            }
                            continue;
                        }
                    };
                    let acc = evaluate(reference, &sur);
                    let ok = passes_gates(&acc, regime);
                    if regime != Regime::Reported {
                        wall_ref_total += reference.wall_secs;
                        wall_sur_total += sur.wall_secs;
                        if !ok {
                            match regime {
                                Regime::Extrap => extrap_ok = false,
                                _ => interp_subrheo_ok = false,
                            }
                        }
                    }
                    println!(
                        "| {i_inj} | {} | {} | {} | {:.3} | {:.3} | {:.2} | {:.3} | {} |",
                        regime_name(regime),
                        acc.n_ref,
                        acc.n_model,
                        acc.count_rel_err,
                        acc.coincidence,
                        acc.first_spike_err,
                        acc.v_rmse,
                        if regime == Regime::Reported {
                            "n/a"
                        } else if ok {
                            "PASS"
                        } else {
                            "fail"
                        },
                    );
                }
                let flops_per_ms = surrogate.step_flops() as f64 / dt;
                let c1_ok = flops_per_ms <= 300.0;
                let wall_ratio = wall_sur_total / wall_ref_total.max(1e-12);
                println!(
                    "cost: {flops_per_ms:.0} flops/ms (C1 {}), wall-clock ratio {wall_ratio:.4} (C3 target ≤ 0.7)\n",
                    if c1_ok { "PASS" } else { "fail" },
                );
                results.push(ConfigResult {
                    dt,
                    preconditions_ok,
                    interp_subrheo_ok,
                    extrap_ok,
                    flops_per_ms,
                    c1_ok,
                    wall_ratio,
                });
            }
        }
        if *gated {
            per_type_results.push((name.to_string(), results));
        }
    }

    // §5 decision.
    println!("## Decision (§5)");
    let mut type_pass = Vec::new();
    let mut type_conditional = Vec::new();
    for (name, results) in &per_type_results {
        let frontier = frontier_flops
            .iter()
            .find(|(n, _, _)| n == name)
            .expect("frontier computed for every gated type");
        if frontier.2 {
            println!(
                "- {name}: C2 NO-GO — paper-standard Euler h = 0.5 passes all accuracy gates."
            );
            type_pass.push(false);
            type_conditional.push(false);
            continue;
        }
        let c2_limit = frontier.1.unwrap_or(f64::INFINITY);
        let cost_ok = |r: &ConfigResult| {
            r.preconditions_ok && r.c1_ok && r.flops_per_ms < c2_limit && r.wall_ratio <= 0.7
        };
        let full_pass = results
            .iter()
            .any(|r| cost_ok(r) && r.full_accuracy() && r.dt >= 1.0);
        // Conditional routes (§5 row 2): (a) everything but extrapolation
        // passes at Δt ≥ 1; (b) full accuracy but only at Δt = 0.5.
        let conditional = results.iter().any(|r| {
            cost_ok(r)
                && ((r.interp_subrheo_ok && !r.extrap_ok && r.dt >= 1.0)
                    || (r.full_accuracy() && r.dt < 1.0))
        });
        println!(
            "- {name}: full PASS config exists: {full_pass}; conditional route: {conditional} \
             (C2 limit {c2_limit:.0} flops/ms)"
        );
        type_pass.push(full_pass);
        type_conditional.push(conditional);
    }
    let rs_pass = type_pass.first().copied().unwrap_or(false);
    let verdict = if type_pass.iter().all(|&p| p) {
        "PASS"
    } else if type_pass
        .iter()
        .zip(&type_conditional)
        .all(|(&p, &c)| p || c)
        || rs_pass
    {
        "CONDITIONAL PASS (claim to be narrowed in writing — see §5 row 2)"
    } else {
        "FAIL — pause for owner decision with skeptic re-review (§5 row 3)"
    };
    println!("\nVERDICT: {verdict}");
    println!("Record the outcome in docs/05-v2-results.md.");
}
