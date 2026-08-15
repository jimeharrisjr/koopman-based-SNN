//! Phase 5 benches. Three questions, one per group:
//!
//! - `step_kernels`: does the structured PerVariable operator actually beat
//!   Dense at scale (the C2 architecture claim)?
//! - `spike_drive`: where is the sparse-accumulation vs dense-matvec
//!   crossover (the §2.3 claim: ~25 % activity)?
//! - `inference_e2e`: **the benchmark gate** — the KoopmanLayer fast path
//!   must be within ~2× of the plain reference simulator per step at equal
//!   accuracy (owner-accepted kill criterion, plan Q8), and the dense
//!   fitted-operator variant is measured against the same bar.
//!
//! Run: `cargo bench` (results under target/criterion/).

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use faer::Mat;
use kdmd_snn::neuron::{Lif, LifParams, NeuronModel};
use kdmd_snn::{KoopmanLayer, LayerState, Operator, SpikeBatch, SpikeVec};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::hint::black_box;

fn dense_from_per_variable(a_local: &Mat<f64>, n: usize) -> Mat<f64> {
    let k = a_local.nrows();
    let mut a = Mat::zeros(k * n, k * n);
    for p in 0..k {
        for q in 0..k {
            for j in 0..n {
                a[(p * n + j, q * n + j)] = a_local[(p, q)];
            }
        }
    }
    a
}

fn bench_step_kernels(c: &mut Criterion) {
    let lif = Lif::new(LifParams::default()).unwrap();
    let mut group = c.benchmark_group("step_kernels");
    group.sample_size(20);
    for &n in &[256usize, 1024, 4096] {
        let pv = Operator::PerVariable {
            a_local: lif.a_local(),
            n_neurons: n,
        };
        let x = Mat::from_fn(2 * n, 1, |i, _| (i as f64 * 0.37).sin());
        let mut y = Mat::zeros(2 * n, 1);
        group.bench_with_input(BenchmarkId::new("per_variable", n), &n, |b, _| {
            b.iter(|| pv.apply(black_box(x.as_ref()), y.as_mut(), false));
        });
        // Dense only up to 1024: 4096 would be an 8192² matrix (0.5 GB).
        if n <= 1024 {
            let dense = Operator::Dense(dense_from_per_variable(&lif.a_local(), n));
            group.bench_with_input(BenchmarkId::new("dense", n), &n, |b, _| {
                b.iter(|| dense.apply(black_box(x.as_ref()), y.as_mut(), false));
            });
        }
    }
    group.finish();
}

fn bench_spike_drive(c: &mut Criterion) {
    let n = 1024usize;
    let mut rng = StdRng::seed_from_u64(3);
    let w = Mat::from_fn(n, n, |_, _| rng.random_range(-0.5..0.5));
    let mut group = c.benchmark_group("spike_drive");
    group.sample_size(20);
    for &activity in &[0.01f64, 0.05, 0.20, 0.50] {
        let active: Vec<u32> = (0..n as u32)
            .filter(|_| rng.random::<f64>() < activity)
            .collect();
        let sparse = SpikeVec::from_indices(active.clone(), n as u32 as usize).unwrap();
        let mut dense_s = Mat::<f64>::zeros(n, 1);
        for &j in sparse.active() {
            dense_s[(j as usize, 0)] = 1.0;
        }
        let mut out = Mat::<f64>::zeros(n, 1);

        group.bench_with_input(
            BenchmarkId::new("sparse_accum", format!("{:.0}%", activity * 100.0)),
            &activity,
            |b, _| {
                b.iter(|| {
                    for i in 0..n {
                        out[(i, 0)] = 0.0;
                    }
                    for &j in black_box(sparse.active()) {
                        let j = j as usize;
                        for i in 0..n {
                            out[(i, 0)] += w[(i, j)];
                        }
                    }
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("dense_matvec", format!("{:.0}%", activity * 100.0)),
            &activity,
            |b, _| {
                b.iter(|| {
                    for i in 0..n {
                        let mut acc = 0.0;
                        for j in 0..n {
                            acc += w[(i, j)] * dense_s[(j, 0)];
                        }
                        out[(i, 0)] = acc;
                    }
                });
            },
        );
    }
    group.finish();
}

/// The gate: reference simulator vs fast paths, N = 1024, T = 100, ~8 %
/// input activity.
fn bench_inference_e2e(c: &mut Criterion) {
    const N: usize = 1024;
    const T: usize = 100;
    let lif = Lif::new(LifParams {
        dt: 0.5,
        ..LifParams::default()
    })
    .unwrap();
    let mut rng = StdRng::seed_from_u64(9);
    let w = Mat::from_fn(N, N, |_, _| rng.random_range(-0.1..0.4));
    let inputs: Vec<SpikeVec> = (0..T)
        .map(|_| {
            SpikeVec::from_indices(
                (0..N as u32)
                    .filter(|_| rng.random::<f64>() < 0.08)
                    .collect(),
                N,
            )
            .unwrap()
        })
        .collect();

    let mut group = c.benchmark_group("inference_e2e");
    group.sample_size(10);

    // (a) Reference simulator with explicit dense drive.
    group.bench_function("reference_lif", |b| {
        b.iter(|| {
            let mut state = LayerState::zeros(N, 2, 1).unwrap();
            lif.init_state(&mut state);
            let mut spikes = SpikeBatch::zeros(N, 1).unwrap();
            let mut drive = Mat::<f64>::zeros(N, 1);
            for s_in in &inputs {
                for i in 0..N {
                    drive[(i, 0)] = 0.0;
                }
                for &j in s_in.active() {
                    for i in 0..N {
                        drive[(i, 0)] += w[(i, j as usize)];
                    }
                }
                lif.step(&mut state, drive.as_ref(), &mut spikes);
            }
            black_box(state);
        });
    });

    // (b) Fast path, structured operator (the product configuration).
    group.bench_function("koopman_per_variable", |b| {
        b.iter(|| {
            let mut layer = KoopmanLayer::lif(&lif, N, w.clone(), 1).unwrap();
            let mut state = LayerState::zeros(N, 2, 1).unwrap();
            lif.init_state(&mut state);
            let mut out = SpikeVec::new(N);
            for s_in in &inputs {
                layer.step(&mut state, s_in, &mut out).unwrap();
            }
            black_box(state);
        });
    });

    // (c) Fast path, dense fitted-operator variant (the kill-criterion
    // measurement: how far outside 2× does an unstructured identified A land?).
    group.bench_function("koopman_dense", |b| {
        let dense = dense_from_per_variable(&lif.a_local(), N);
        b.iter(|| {
            let mut layer = KoopmanLayer::new(
                Operator::Dense(dense.clone()),
                w.clone(),
                lif.b_local().to_vec(),
                lif.params().theta,
                1,
            )
            .unwrap();
            let mut state = LayerState::zeros(N, 2, 1).unwrap();
            lif.init_state(&mut state);
            let mut out = SpikeVec::new(N);
            for s_in in &inputs {
                layer.step(&mut state, s_in, &mut out).unwrap();
            }
            black_box(state);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_step_kernels,
    bench_spike_drive,
    bench_inference_e2e
);
criterion_main!(benches);
