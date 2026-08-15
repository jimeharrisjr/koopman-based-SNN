//! V2b rescue experiment: the ISI/Poincaré return-map surrogate, under the
//! frozen protocol of `docs/07-v2b-preregistration.md`.
//!
//! Gates: A1/A2/A3 carried at V2 thresholds; A2b (chance-corrected Γ) and A5
//! (teacher-forced per-ISI bias) as registered; preconditions P-A (full
//! rank), P-B (fixed-point contraction), P-C (5 % held-out screens); cost C1
//! (≤ 0.5× converged reference) with the C2 frontier carried from V2 (no
//! Euler passed accuracy — not re-run). Decision: PASS / FAIL only.
//!
//! Run: `cargo run --release --example v2b_return_map`

use kdmd_snn::neuron::IzhikevichParams;
use kdmd_snn::return_map::{DictionaryFamily, IzhikevichReturnMap, ReturnMapConfig};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const HORIZON_MS: f64 = 1000.0;
const TRAIN_CURRENTS: &[f64] = &[3.0, 5.0, 7.0, 9.0, 11.0];
const SPIKING_TRAIN_CURRENTS: &[f64] = &[5.0, 7.0, 9.0, 11.0];
const DEGREES: &[usize] = &[2, 3, 4];
const FAMILIES: &[DictionaryFamily] = &[DictionaryFamily::A, DictionaryFamily::B];
const COINC_WINDOW_MS: f64 = 2.0;
const H_START: f64 = 0.025;
const H_FLOOR: f64 = 0.025 / 256.0;
/// Teacher-forced A5 evaluation trajectories per test current.
const A5_TRAJECTORIES: usize = 10;
const A5_SEED: u64 = 20260814;

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
    (4.0, Regime::Reported),
];

/// Fine-Euler run: crossing-interpolated spike times plus post-reset u₊.
fn euler_spikes(p: &IzhikevichParams, h: f64, i_inj: f64, v0: f64, u0: f64) -> Vec<(f64, f64)> {
    let steps = (HORIZON_MS / h).round() as usize;
    let mut v = v0;
    let mut u = u0;
    let mut out = Vec::new();
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
            v = p.c;
            u += p.d;
            out.push(((t as f64 + frac) * h, u));
        }
    }
    out
}

fn standard_ic(p: &IzhikevichParams) -> (f64, f64) {
    (-65.0, p.b * -65.0)
}

fn find_reference_h(name: &str, p: &IzhikevichParams) -> f64 {
    let (v0, u0) = standard_ic(p);
    let mut h = H_START;
    loop {
        let a = euler_spikes(p, h, 13.0, v0, u0);
        let b = euler_spikes(p, h / 2.0, 13.0, v0, u0);
        let converged =
            a.len() == b.len() && a.iter().zip(&b).all(|(x, y)| (x.0 - y.0).abs() < 0.1);
        if converged {
            println!("- {name}: ground-truth h = {h:.7} ms (self-converged, re-verified)");
            return h;
        }
        assert!(h > H_FLOOR, "{name}: no converged h above floor");
        h /= 2.0;
    }
}

fn matched_count(reference: &[f64], candidate: &[f64], window: f64) -> usize {
    let mut used = vec![false; candidate.len()];
    let mut matched = 0;
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
            matched += 1;
        }
    }
    matched
}

struct Accuracy {
    n_ref: usize,
    n_model: usize,
    coincidence: f64,
    gamma: f64,
    first_err: f64,
}

fn evaluate(reference: &[f64], model: &[f64]) -> Accuracy {
    let n_ref = reference.len();
    let n_model = model.len();
    let coincidence = if n_ref > 0 {
        matched_count(reference, model, COINC_WINDOW_MS) as f64 / n_ref as f64
    } else {
        1.0
    };
    // A2b: Γ = (c − f)/(1 − f), f = min(1, 2Δ·N_sur/T); Γ := 0 if f ≥ 1.
    let f = (2.0 * COINC_WINDOW_MS * n_model as f64 / HORIZON_MS).min(1.0);
    let gamma = if f >= 1.0 {
        0.0
    } else {
        (coincidence - f) / (1.0 - f)
    };
    let first_err = match (reference.first(), model.first()) {
        (Some(&r), Some(&s)) => (r - s).abs(),
        (None, None) => 0.0,
        _ => f64::INFINITY,
    };
    Accuracy {
        n_ref,
        n_model,
        coincidence,
        gamma,
        first_err,
    }
}

fn passes_gates(acc: &Accuracy, a5_bias: Option<f64>, regime: Regime) -> bool {
    match regime {
        Regime::Reported => true,
        Regime::SubRheo => acc.n_model == 0,
        Regime::Interp | Regime::Extrap => {
            let (count_thr, coinc_thr, gamma_thr, first_thr, bias_thr) = match regime {
                Regime::Interp => (0.10, 0.80, 0.60, 2.0, 0.0025),
                _ => (0.15, 0.70, 0.50, 4.0, 0.0030),
            };
            let count_ok = if acc.n_ref < 10 {
                (acc.n_model as i64 - acc.n_ref as i64).abs() <= 1
            } else {
                (acc.n_model as f64 - acc.n_ref as f64).abs() / acc.n_ref as f64 <= count_thr
            };
            count_ok
                && acc.coincidence >= coinc_thr
                && acc.gamma >= gamma_thr
                && acc.first_err <= first_thr
                && a5_bias.is_some_and(|b| b.abs() <= bias_thr)
        }
    }
}

/// A5: teacher-forced per-ISI bias at a test current — evaluate the ISI map
/// on the reference's own section states across `A5_TRAJECTORIES`
/// randomized-IC reference trajectories; bias relative to the mean ISI.
fn teacher_forced_bias(
    map: &IzhikevichReturnMap,
    p: &IzhikevichParams,
    h: f64,
    i_inj: f64,
) -> Option<(f64, f64)> {
    let mut rng = StdRng::seed_from_u64(A5_SEED);
    let mut errs = Vec::new();
    let mut isis = Vec::new();
    for _ in 0..A5_TRAJECTORIES {
        let v0 = rng.random_range(-80.0..-50.0);
        let u0 = p.b * v0 + rng.random_range(-2.0..2.0);
        let spikes = euler_spikes(p, h, i_inj, v0, u0);
        for w in spikes.windows(2) {
            let (t0, u_plus) = w[0];
            let t_true = w[1].0 - t0;
            errs.push(map.predicted_isi(u_plus, i_inj) - t_true);
            isis.push(t_true);
        }
    }
    if errs.is_empty() {
        return None;
    }
    let mean_isi = isis.iter().sum::<f64>() / isis.len() as f64;
    let bias = errs.iter().sum::<f64>() / errs.len() as f64 / mean_isi;
    let rms = (errs.iter().map(|e| e * e).sum::<f64>() / errs.len() as f64).sqrt() / mean_isi;
    Some((bias, rms))
}

fn regime_name(r: Regime) -> &'static str {
    match r {
        Regime::Interp => "interp",
        Regime::Extrap => "extrap",
        Regime::SubRheo => "sub-rheo",
        Regime::Reported => "reported",
    }
}

fn family_name(f: DictionaryFamily) -> &'static str {
    match f {
        DictionaryFamily::A => "A",
        DictionaryFamily::B => "B",
    }
}

fn main() {
    println!("# V2b return-map experiment — protocol docs/07-v2b-preregistration.md");
    println!("horizon {HORIZON_MS} ms · train I {TRAIN_CURRENTS:?} · grid deg {DEGREES:?} × family {{A, B}}\n");

    let types: Vec<(&str, IzhikevichParams, bool, f64)> = vec![
        // (name, params, gated, converged-reference flops/ms)
        ("RS", IzhikevichParams::regular_spiking(1.0), true, 9600.0),
        ("FS", IzhikevichParams::fast_spiking(1.0), true, 76900.0),
        ("CH", IzhikevichParams::chattering(1.0), false, 38400.0),
    ];

    println!("## Ground-truth self-convergence (re-verified)");
    let h_refs: Vec<f64> = types
        .iter()
        .map(|(name, p, _, _)| find_reference_h(name, p))
        .collect();
    println!();

    let mut type_pass = Vec::new();
    for (type_idx, (name, p, gated, ref_flops)) in types.iter().enumerate() {
        let h_ref = h_refs[type_idx];
        let (v0, u0) = standard_ic(p);
        println!(
            "## Return-map grid — {name}{}",
            if *gated { "" } else { " (reported only)" }
        );
        let references: Vec<(f64, Regime, Vec<f64>)> = TEST_CURRENTS
            .iter()
            .map(|&(i, r)| {
                let times = euler_spikes(p, h_ref, i, v0, u0)
                    .into_iter()
                    .map(|(t, _)| t)
                    .collect();
                (i, r, times)
            })
            .collect();

        let mut any_config_passes = false;
        for &family in FAMILIES {
            for &degree in DEGREES {
                // Registered contingency (docs/07 §2): if the data floors
                // (≥ 50 ISI pairs per spiking current, ≥ 500 first-spike
                // pairs) are unmet, extend the trajectory budget and refit.
                let mut fitted = None;
                let mut extended = false;
                for n_traj in [10usize, 20, 40] {
                    let cfg = ReturnMapConfig {
                        degree,
                        family,
                        horizon_ms: HORIZON_MS,
                        n_trajectories: n_traj,
                        n_holdout: n_traj / 5,
                        ..ReturnMapConfig::default()
                    };
                    match IzhikevichReturnMap::fit(p, h_ref, TRAIN_CURRENTS, &cfg) {
                        Ok((map, diag)) => {
                            let floors_ok = diag.section_counts.iter().all(|&(_, n)| n >= 50)
                                && diag.n_first_samples >= 500;
                            let last_chance = n_traj == 40;
                            if floors_ok || last_chance {
                                extended = n_traj > 10;
                                fitted = Some((map, diag));
                                break;
                            }
                        }
                        Err(e) => {
                            println!(
                                "### deg {degree} fam {}: FIT FAILED — {e}",
                                family_name(family)
                            );
                            break;
                        }
                    }
                }
                let Some((map, diag)) = fitted else {
                    continue;
                };
                if extended {
                    println!(
                        "(contingency applied: trajectory budget extended to meet data floors)"
                    );
                }

                // Preconditions (docs/07 §4).
                let data_floor_ok = diag.section_counts.iter().all(|&(_, n)| n >= 50)
                    && diag.n_first_samples >= 500;
                let p_a = diag.section_full_rank && diag.first_full_rank;
                let mut p_b = true;
                let mut contractions = Vec::new();
                for &i_inj in SPIKING_TRAIN_CURRENTS {
                    let range = diag.u_ranges.iter().find(|r| r.0 == i_inj);
                    let (u_lo, u_hi) = match range {
                        Some(&(_, lo, hi)) => (lo, hi),
                        None => {
                            p_b = false;
                            continue;
                        }
                    };
                    let mut u = 0.5 * (u_lo + u_hi);
                    let mut converged = false;
                    for _ in 0..200 {
                        let next = map.next_section(u, i_inj);
                        if !next.is_finite() {
                            break;
                        }
                        if (next - u).abs() < 1e-10 {
                            u = next;
                            converged = true;
                            break;
                        }
                        u = next;
                    }
                    let c = map.contraction_at(u, i_inj);
                    contractions.push((i_inj, c));
                    if !(converged && u >= u_lo && u <= u_hi && c < 1.0) {
                        p_b = false;
                    }
                }
                let isi_screen = diag.heldout_isi_rel_rms.unwrap_or(f64::INFINITY);
                let first_screen = diag.heldout_first_rel_rms.unwrap_or(f64::INFINITY);
                let p_c = isi_screen <= 0.05 && first_screen <= 0.05;
                let preconditions_ok = data_floor_ok && p_a && p_b && p_c;

                println!(
                    "### deg {degree} fam {} (d = {}, samples {}/{}, cond {:.1e}/{:.1e}, \
                     holdout ISI {:.2e}, first {:.2e}, contraction {:?}{})",
                    family_name(family),
                    map.section_dim(),
                    diag.n_section_samples,
                    diag.n_first_samples,
                    diag.section_cond,
                    diag.first_cond,
                    isi_screen,
                    first_screen,
                    contractions
                        .iter()
                        .map(|&(i, c)| format!("I{i}:{c:.2}"))
                        .collect::<Vec<_>>(),
                    if preconditions_ok {
                        ""
                    } else {
                        " — PRECONDITION FAILED"
                    },
                );

                println!(
                    "| I | regime | ref N | sur N | coinc | Γ | first err | A5 bias | A5 rms | gates |"
                );
                println!("|---|---|---|---|---|---|---|---|---|---|");
                let mut all_ok = true;
                let mut max_rate = 0.0f64;
                for (i_inj, regime, reference) in &references {
                    let predicted = match map.rollout(*i_inj, HORIZON_MS) {
                        Ok(s) => s,
                        Err(e) => {
                            println!(
                                "| {i_inj} | {} | — | — | — | — | — | — | — | INVALID: {e} |",
                                regime_name(*regime)
                            );
                            if *regime != Regime::Reported {
                                all_ok = false;
                            }
                            continue;
                        }
                    };
                    let acc = evaluate(reference, &predicted);
                    let a5 = teacher_forced_bias(&map, p, h_ref, *i_inj);
                    let a5_bias = a5.map(|(b, _)| b);
                    let ok = passes_gates(&acc, a5_bias, *regime);
                    if *regime != Regime::Reported {
                        if !ok {
                            all_ok = false;
                        }
                        max_rate = max_rate.max(acc.n_model as f64 / HORIZON_MS);
                    }
                    println!(
                        "| {i_inj} | {} | {} | {} | {:.3} | {:.3} | {:.2} | {} | {} | {} |",
                        regime_name(*regime),
                        acc.n_ref,
                        acc.n_model,
                        acc.coincidence,
                        acc.gamma,
                        acc.first_err,
                        a5.map_or("n/a".into(), |(b, _)| format!("{:+.4}%", b * 100.0)),
                        a5.map_or("n/a".into(), |(_, r)| format!("{:.4}%", r * 100.0)),
                        if *regime == Regime::Reported {
                            "n/a"
                        } else if ok {
                            "PASS"
                        } else {
                            "fail"
                        },
                    );
                }
                // In-sample quiescence report at I = 3 (docs/07 §2).
                let mut rng = StdRng::seed_from_u64(A5_SEED);
                let (mut quiet, mut total) = (0usize, 0usize);
                for _ in 0..A5_TRAJECTORIES {
                    let v0s = rng.random_range(-80.0..-50.0);
                    let u0s = p.b * v0s + rng.random_range(-2.0..2.0);
                    let pred = map.predicted_first(v0s, u0s, 3.0);
                    total += 1;
                    if pred > map.t_max() {
                        quiet += 1;
                    }
                }
                let flops_per_ms = max_rate * map.flops_per_spike() as f64;
                let c1_ok = flops_per_ms <= 0.5 * ref_flops;
                println!(
                    "I=3 in-sample quiescence: {quiet}/{total} · cost {flops_per_ms:.1} flops/ms \
                     (C1 {} vs 0.5×{ref_flops:.0}; C2 frontier carried: reference itself)\n",
                    if c1_ok { "PASS" } else { "fail" },
                );
                if *gated && preconditions_ok && all_ok && c1_ok {
                    any_config_passes = true;
                }
            }
        }
        if *gated {
            println!("{name}: passing configuration exists: {any_config_passes}\n");
            type_pass.push(any_config_passes);
        } else {
            println!();
        }
    }

    let verdict = if type_pass.iter().all(|&x| x) {
        "PASS — RS and FS meet every V2b gate; Phase 5 includes the event-level \
         spike-timing surrogate (scope: constant I ∈ [5, 13], registered IC box, \
         spike-timing workloads only)"
    } else {
        "FAIL — per the owner's standing decision: immediate pivot to V1/V3/V4; \
         the nonlinear-surrogate track closes"
    };
    println!("VERDICT: {verdict}");
    println!("Record the outcome in docs/08-v2b-results.md.");
}
