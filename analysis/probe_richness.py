"""S-A: probe richness for identification (docs/25, frozen setup).

Known-B recovery of a dense 2N x 2N LIF-layer propagator as the number of
independently driven input channels m falls below the layer width N.
Exact propagator (tau = 20/10 ms, h = 1 ms), W ~ N(0, 0.35) restricted to m
active channels, Poisson probes at 0.15/channel, T = 4000, state noise
sigma = 1e-4, known-B least squares. Metrics: full-A relative Frobenius
error, cond(X), excited-subspace error (top-2m left singular vectors of X),
and the v-row / i-row error split (prediction PA4).
"""

import numpy as np

TAU_M, TAU_S, H = 20.0, 10.0, 1.0
THETA = 1.0
SIGMA = 1e-4
T = 4000
RATE = 0.15


def propagator():
    a = np.exp(-H / TAU_M)
    b = np.exp(-H / TAU_S)
    g = (a - b) / (-1.0 / TAU_M + 1.0 / TAU_S)
    return np.array([[a, g], [0.0, b]])


def run(n, m, rng):
    a_true = np.kron(np.eye(n), propagator())
    # W: n x n, but only the first m columns are ever driven.
    w = rng.normal(0.0, 0.35, size=(n, n))
    b_ff = np.zeros((2 * n, n))
    b_reset = np.zeros((2 * n, n))
    for j in range(n):
        b_ff[2 * j + 1, :] = w[j, :]
        b_reset[2 * j, j] = -THETA
    b_true = np.hstack([b_ff, b_reset])

    x = np.zeros(2 * n)
    xs, us, ys = [], [], []
    for _ in range(T):
        s_in = np.zeros(n)
        s_in[:m] = (rng.random(m) < RATE).astype(float)
        v = x[0::2]
        s_own = (v >= THETA).astype(float)
        u = np.concatenate([s_in, s_own])
        xn = a_true @ x + b_true @ u
        xs.append(x)
        us.append(u)
        ys.append(xn)
        x = xn
    xs = np.array(xs).T
    us = np.array(us).T
    ys = np.array(ys).T

    xn_ = xs + SIGMA * rng.normal(size=xs.shape)
    yn_ = ys + SIGMA * rng.normal(size=ys.shape)
    a_hat = np.linalg.lstsq(xn_.T, (yn_ - b_true @ us).T, rcond=None)[0].T

    err = np.linalg.norm(a_hat - a_true, "fro") / np.linalg.norm(a_true, "fro")
    cond_x = np.linalg.cond(xs)
    # Excited-subspace error: project both operators onto the top-2m left
    # singular directions of X.
    uu, _, _ = np.linalg.svd(xs, full_matrices=False)
    p = uu[:, : min(2 * m, 2 * n)]
    err_sub = np.linalg.norm(p.T @ (a_hat - a_true) @ p, "fro") / np.linalg.norm(
        p.T @ a_true @ p, "fro"
    )
    # Row split: rows 2j are voltage, 2j+1 current.
    dv = a_hat - a_true
    v_rows = np.linalg.norm(dv[0::2, :], "fro") / np.linalg.norm(a_true[0::2, :], "fro")
    i_rows = np.linalg.norm(dv[1::2, :], "fro") / np.linalg.norm(a_true[1::2, :], "fro")
    rate = us[n:, :].mean()
    return err, err_sub, cond_x, v_rows, i_rows, rate


def main():
    rng = np.random.default_rng(11)
    for n in (8, 32):
        ms = [1]
        while ms[-1] < n:
            ms.append(min(ms[-1] * 2, n))
        print(f"\nN = {n} (state dim {2 * n}), T = {T}, sigma = {SIGMA}")
        print("  m | full-A err | excited-sub err | cond(X) | v-rows | i-rows | fire/step")
        for m in ms:
            err, err_sub, cx, vr, ir, rate = run(n, m, rng)
            print(
                f"{m:3d} | {err:10.3e} | {err_sub:15.3e} | {cx:8.1e} "
                f"| {vr:6.3f} | {ir:6.3f} | {rate:.3f}"
            )


if __name__ == "__main__":
    main()
