"""S-C: spiking-regime reduced-order models — a feasibility map (docs/25).

Recurrent LIF layer (N = 64, exact propagator, tau = 20/10 ms, h = 1 ms),
random W_rec ~ N(0, 0.4/sqrt(N)), Poisson input at three drive levels.
Ground truth: full 2N spiking rollout, T = 1000. ROM: POD basis P_r from
ground-truth snapshots; reduced step z <- (P^T A P) z + P^T B u with spikes
computed from lifted voltages v = (P z)_v and fed back (reset + recurrence)
exactly as in the full model. Metrics: spike coincidence (+/-2 bins) and
mean absolute per-neuron rate error, per (rank, drive).
"""

import numpy as np

TAU_M, TAU_S, H = 20.0, 10.0, 1.0
THETA = 1.0
N = 64
T = 1000
RANKS = [8, 16, 32, 64, 128]


def propagator():
    """The library's convention: tau_m v' = -v + R i (R = 1), so
    gamma = R tau_s (alpha - beta)/(tau_m - tau_s) — matching Lif::a_local.
    (The first run of this study used a 20x hotter coupling by mistake;
    recorded as a deviation in docs/26.)"""
    a = np.exp(-H / TAU_M)
    b = np.exp(-H / TAU_S)
    g = TAU_S * (a - b) / (TAU_M - TAU_S)
    d = (1.0 - a) - g
    return np.array([[a, g], [0.0, b]]), np.array([d, 1.0 - b])


def build(rng):
    a_local, b_local = propagator()
    a_true = np.kron(np.eye(N), a_local)
    # Harness-style init: positive input weights U(0, 35/N); recurrent
    # weights small mean-zero.
    w_in = rng.uniform(0.0, 35.0 / N, size=(N, N))
    w_rec = rng.normal(0.0, 0.4 / np.sqrt(N), size=(N, N))
    b_full = np.zeros((2 * N, 3 * N))  # [input; own-reset; recurrent]
    for j in range(N):
        b_full[2 * j, :N] = b_local[0] * w_in[j, :]
        b_full[2 * j + 1, :N] = b_local[1] * w_in[j, :]
        b_full[2 * j, N + j] = -THETA
        b_full[2 * j, 2 * N :] = b_local[0] * w_rec[j, :]
        b_full[2 * j + 1, 2 * N :] = b_local[1] * w_rec[j, :]
    return a_true, b_full


def rollout_full(a, b, drive, rng_in):
    x = np.zeros(2 * N)
    prev = np.zeros(N)
    spikes = np.zeros((T, N))
    snaps = np.zeros((2 * N, T))
    inputs = (rng_in.random((T, N)) < drive).astype(float)
    for t in range(T):
        v = x[0::2]
        s = (v >= THETA).astype(float)
        spikes[t] = s
        u = np.concatenate([inputs[t], s, prev])
        snaps[:, t] = x
        x = a @ x + b @ u
        prev = s
    return spikes, snaps, inputs


def rollout_rom(a, b, p, inputs):
    a_r = p.T @ a @ p
    b_r = p.T @ b
    v_rows = p[0::2, :]  # lift z -> voltages
    z = np.zeros(p.shape[1])
    prev = np.zeros(N)
    spikes = np.zeros((T, N))
    for t in range(T):
        v = v_rows @ z
        s = (v >= THETA).astype(float)
        spikes[t] = s
        u = np.concatenate([inputs[t], s, prev])
        z = a_r @ z + b_r @ u
        prev = s
    return spikes


def coincidence(ref, hyp, tol=2):
    """Fraction of reference spikes matched by a hypothesis spike within
    +/-tol bins on the same neuron (and symmetrically), averaged."""
    hits_ref = 0
    n_ref = int(ref.sum())
    hits_hyp = 0
    n_hyp = int(hyp.sum())
    # Dilate along time.
    def dilate(m):
        d = np.zeros_like(m)
        for dt in range(-tol, tol + 1):
            d = np.maximum(d, np.roll(m, dt, axis=0))
        return d
    d_hyp = dilate(hyp)
    d_ref = dilate(ref)
    hits_ref = int((ref * d_hyp).sum())
    hits_hyp = int((hyp * d_ref).sum())
    if n_ref == 0 and n_hyp == 0:
        return 1.0
    if n_ref == 0 or n_hyp == 0:
        return 0.0
    return 0.5 * (hits_ref / n_ref + hits_hyp / n_hyp)


def main():
    rng = np.random.default_rng(17)
    a, b = build(rng)
    print(f"N = {N}, T = {T}, ranks {RANKS}")
    print("drive | fire/step | rank | coincidence | mean |rate err| (rel)")
    for drive in (0.08, 0.13, 0.28):  # measured rates ~0.036/0.080/0.203
        rng_in = np.random.default_rng(23)
        ref_spikes, snaps, inputs = rollout_full(a, b, drive, rng_in)
        rate = ref_spikes.mean()
        uu, ss, _ = np.linalg.svd(snaps, full_matrices=False)
        for r in RANKS:
            p = uu[:, :r]
            hyp = rollout_rom(a, b, p, inputs)
            c = coincidence(ref_spikes, hyp)
            ref_rates = ref_spikes.mean(axis=0)
            hyp_rates = hyp.mean(axis=0)
            denom = max(ref_rates.mean(), 1e-9)
            rate_err = np.abs(hyp_rates - ref_rates).mean() / denom
            print(
                f"{drive:5.2f} | {rate:9.4f} | {r:4d} | {c:11.4f} | {rate_err:10.4f}"
            )


if __name__ == "__main__":
    main()
