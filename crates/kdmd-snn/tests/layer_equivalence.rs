//! Phase 5 exit gate: the fast KoopmanLayer must reproduce the reference LIF
//! simulator **spike-for-spike** over 1000 steps.
//!
//! The layer's PerVariable operator is the closed-form propagator, so this is
//! an identity check, not an approximation bound: identical spike trains, and
//! state agreement at machine precision.

use faer::Mat;
use kdmd_snn::neuron::{Lif, LifParams, NeuronModel};
use kdmd_snn::{KoopmanLayer, LayerState, SpikeBatch, SpikeVec};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const N: usize = 64;
const N_IN: usize = 32;
const T: usize = 1000;

/// Random-ish weights and a Poisson-ish input schedule, both seeded.
fn setup(seed: u64) -> (Mat<f64>, Vec<Vec<u32>>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let w = Mat::from_fn(N, N_IN, |_, _| rng.random_range(-0.2..0.8));
    let inputs: Vec<Vec<u32>> = (0..T)
        .map(|_| {
            (0..N_IN as u32)
                .filter(|_| rng.random::<f64>() < 0.08)
                .collect()
        })
        .collect();
    (w, inputs)
}

#[test]
fn fast_layer_reproduces_reference_lif_spike_for_spike() {
    let lif = Lif::new(LifParams {
        dt: 0.5,
        ..LifParams::default()
    })
    .unwrap();
    let (w, inputs) = setup(7);

    // Reference path: explicit drive d = W·s fed to the reference simulator.
    let mut ref_state = LayerState::zeros(N, 2, 1).unwrap();
    lif.init_state(&mut ref_state);
    let mut ref_spikes = SpikeBatch::zeros(N, 1).unwrap();

    // Fast path.
    let mut layer = KoopmanLayer::lif(&lif, N, w.clone(), 1).unwrap();
    let mut fast_state = LayerState::zeros(N, 2, 1).unwrap();
    lif.init_state(&mut fast_state);
    let mut fast_out = SpikeVec::new(N);

    let mut total_spikes = 0usize;
    for (t, active) in inputs.iter().enumerate() {
        // Reference drive: dense W·s in the same accumulation order the
        // sparse path uses (sum over active columns), so agreement is exact.
        let mut drive = Mat::<f64>::zeros(N, 1);
        for &j in active {
            for i in 0..N {
                drive[(i, 0)] += w[(i, j as usize)];
            }
        }
        lif.step(&mut ref_state, drive.as_ref(), &mut ref_spikes);

        let s_in = SpikeVec::from_indices(active.clone(), N_IN).unwrap();
        layer.step(&mut fast_state, &s_in, &mut fast_out).unwrap();

        // Spike-for-spike agreement.
        let ref_active: Vec<u32> = (0..N as u32)
            .filter(|&j| ref_spikes.as_mat()[(j as usize, 0)] == 1.0)
            .collect();
        assert_eq!(
            fast_out.active(),
            ref_active.as_slice(),
            "spike trains diverged at step {t}"
        );
        total_spikes += ref_active.len();

        // State agreement at machine precision.
        for i in 0..2 * N {
            let a = fast_state.as_mat()[(i, 0)];
            let b = ref_state.as_mat()[(i, 0)];
            assert!(
                (a - b).abs() <= 1e-12 * b.abs().max(1.0),
                "state diverged at step {t}, row {i}: {a} vs {b}"
            );
        }
    }
    assert!(
        total_spikes > 100,
        "test regime produced too few spikes ({total_spikes}) to be meaningful"
    );
}

#[test]
fn dense_identified_operator_matches_the_structured_path() {
    // A Dense operator holding the same values as PerVariable must produce
    // identical spikes — the fast path's correctness must not depend on which
    // operator variant identification happened to emit.
    let lif = Lif::new(LifParams {
        dt: 0.5,
        ..LifParams::default()
    })
    .unwrap();
    let (w, inputs) = setup(11);

    let a_local = lif.a_local();
    let mut dense = Mat::<f64>::zeros(2 * N, 2 * N);
    for p in 0..2 {
        for q in 0..2 {
            for j in 0..N {
                dense[(p * N + j, q * N + j)] = a_local[(p, q)];
            }
        }
    }
    let mut structured = KoopmanLayer::lif(&lif, N, w.clone(), 1).unwrap();
    let mut dense_layer = KoopmanLayer::new(
        kdmd_snn::Operator::Dense(dense),
        w,
        lif.b_local().to_vec(),
        lif.params().theta,
        1,
    )
    .unwrap();

    let mut s1 = LayerState::zeros(N, 2, 1).unwrap();
    let mut s2 = LayerState::zeros(N, 2, 1).unwrap();
    let mut o1 = SpikeVec::new(N);
    let mut o2 = SpikeVec::new(N);
    for (t, active) in inputs.iter().enumerate().take(300) {
        let s_in = SpikeVec::from_indices(active.clone(), N_IN).unwrap();
        structured.step(&mut s1, &s_in, &mut o1).unwrap();
        dense_layer.step(&mut s2, &s_in, &mut o2).unwrap();
        assert_eq!(o1.active(), o2.active(), "variants diverged at step {t}");
    }
}
