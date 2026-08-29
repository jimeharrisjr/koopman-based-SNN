"""Generate all figures for the kdmd-SNN draft paper.

Figures 1, 3, 4, 5 come from fresh numerical experiments that reproduce the
library's mathematics in numpy; figures 6-8 are plotted from the repository's
recorded sweep logs (demo/sweep-*.txt); figure 9 replots the recorded gate
numbers from docs/05 and docs/08; figure 2 is a schematic.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

import figstyle as st

st.apply()

HERE = Path(__file__).parent
OUT = Path("/Users/jimharris/Documents/kdmd-SNN/paper/figures")
OUT.mkdir(parents=True, exist_ok=True)

DATA = json.loads((HERE / "sweep_data.json").read_text())
CURVES, RESULTS = DATA["curves"], DATA["results"]

# ---------------------------------------------------------------- LIF machinery
TAU_M, TAU_S = 20.0, 10.0  # ms
THETA = 1.0

def propagator(h):
    a = np.exp(-h / TAU_M)
    b = np.exp(-h / TAU_S)
    g = (a - b) / (-1.0 / TAU_M + 1.0 / TAU_S)
    return np.array([[a, g], [0.0, b]])

A_C = np.array([[-1.0 / TAU_M, 1.0], [0.0, -1.0 / TAU_S]])
B_C = np.array([0.0, 1.0])  # external drive enters the current equation


# ------------------------------------------------------------------- Figure 1
def fig1_lif_trace():
    rng = np.random.default_rng(3)
    h = 0.1  # ms
    A = propagator(h)
    T = 400.0
    n = int(T / h)
    x = np.zeros(2)
    t_axis = np.arange(n) * h
    v_tr = np.zeros(n)
    i_tr = np.zeros(n)
    spikes = []
    # Poisson input spikes through one effective synapse
    w, rate = 0.55, 0.013  # per-ms rate, silent window carved out below
    for k in range(n):
        t = k * h
        drive = rate * h if not (150 <= t < 230) else 0.0
        s_in = 1.0 if rng.random() < drive else 0.0
        v_tr[k], i_tr[k] = x
        if x[0] >= THETA:
            spikes.append(t)
            x[0] -= THETA          # subtractive reset (a -theta control input)
        x = A @ x
        x[1] += w * s_in           # input spike lands in the current
    fig, ax = plt.subplots(figsize=(6.6, 2.9), layout="constrained")
    ax.axhline(THETA, color=st.CRITICAL, lw=1.1, ls=(0, (4, 3)), zorder=1)
    ax.text(T - 2, THETA + 0.05, r"threshold $\theta$", color=st.CRITICAL,
            ha="right", fontsize=8.5)
    ax.plot(t_axis, v_tr, color=st.BLUE, lw=1.6, label="membrane voltage $v$")
    ax.plot(t_axis, i_tr, color=st.ORANGE, lw=1.2, alpha=0.85,
            label="synaptic current $i$")
    for s in spikes:
        ax.plot([s, s], [1.28, 1.42], color=st.INK, lw=1.4, solid_capstyle="butt")
    ax.text(T - 2, 1.35, "spikes", ha="right", va="center",
            fontsize=8.5, color=st.INK2)
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("state")
    ax.set_xlim(0, T)
    ax.set_ylim(-0.08, 1.5)
    ax.legend(loc="center", bbox_to_anchor=(0.68, 0.62), ncols=1)
    fig.savefig(OUT / "fig01-lif-trace.png")
    plt.close(fig)
    print(f"fig1: {len(spikes)} spikes")


# ------------------------------------------------------------------- Figure 2
def _box(ax, xy, w_, h_, text, fc, ec, tc=st.INK, fs=9.5):
    b = FancyBboxPatch(xy, w_, h_, boxstyle="round,pad=0.012",
                       fc=fc, ec=ec, lw=1.2, mutation_aspect=1.0)
    ax.add_patch(b)
    ax.text(xy[0] + w_ / 2, xy[1] + h_ / 2, text, ha="center", va="center",
            fontsize=fs, color=tc)

def _arrow(ax, p0, p1, color=st.INK2, rad=0.0, lw=1.4):
    a = FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=13,
                        color=color, lw=lw,
                        connectionstyle=f"arc3,rad={rad}", zorder=3)
    ax.add_patch(a)

def fig2_formulation():
    fig, ax = plt.subplots(figsize=(6.8, 3.1), layout="constrained")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.6)
    ax.axis("off")
    ax.grid(False)
    # input assembly
    _box(ax, (0.25, 2.6), 2.1, 1.25,
         "assemble input $u_t$\n$[\\,W s^{(l-1)}_t;\\ s^{(l)}_{t-1}\\,]$",
         "#ffffff", st.BASELINE)
    # linear advance
    _box(ax, (3.35, 2.6), 3.0, 1.25,
         "exact linear advance\n$x_{t+1} = A\\,x_t + B\\,u_t$",
         st.SEQ100, st.BLUE)
    # threshold
    _box(ax, (7.35, 2.6), 2.35, 1.25,
         "threshold\n$s_t = \\Theta(v_t - \\theta)$",
         "#ffffff", st.BASELINE)
    # spikes
    _box(ax, (3.9, 0.35), 1.9, 0.95, "spikes $s_t$", "#fdeeee", st.CRITICAL)
    _arrow(ax, (2.35, 3.22), (3.35, 3.22))
    _arrow(ax, (6.35, 3.22), (7.35, 3.22))
    _arrow(ax, (8.5, 2.6), (5.8, 1.05), color=st.CRITICAL, rad=-0.25)
    _arrow(ax, (3.9, 0.9), (1.15, 2.6), color=st.CRITICAL, rad=-0.25)
    ax.text(0.15, 0.30, "own spikes re-enter $u$ as a\n"
                        "$-\\theta$ kick to the voltage\n(subtractive reset)",
            fontsize=8.5, color=st.CRITICAL, ha="left", va="bottom")
    ax.text(7.6, 1.15, "the only nonlinearity:\na comparison, not\nan approximation",
            fontsize=8.5, color=st.INK2, ha="left")
    ax.text(4.85, 4.35, "input spikes from the previous layer, scaled by learned $W$",
            fontsize=8.5, color=st.INK2, ha="center")
    _arrow(ax, (1.3, 4.25), (1.3, 3.85), color=st.INK2)
    fig.savefig(OUT / "fig02-formulation.png")
    plt.close(fig)
    print("fig2 done")


# ------------------------------------------------------------------- Figure 3
def fig3_exactness():
    h = 1.0
    A_d = propagator(h)

    def exact_flow(x, i_ext):
        x_ss = -np.linalg.solve(A_C, B_C * i_ext)
        return x_ss + A_d @ (x - x_ss)

    def euler_flow(x, i_ext, nsub):
        hs = h / nsub
        for _ in range(nsub):
            x = x + hs * (A_C @ x + B_C * i_ext)
        return x

    # (a) sub-threshold voltage error vs analytic solution
    i_ext = 0.004  # v_ss = 0.8 < theta
    T = 300
    x_ss = -np.linalg.solve(A_C, B_C * i_ext)
    eigval, eigvec = np.linalg.eig(A_C)
    x0 = np.zeros(2)
    c0 = np.linalg.solve(eigvec, x0 - x_ss)

    def analytic(t):
        return (x_ss + (eigvec * np.exp(eigval * t)) @ c0).real

    t_axis = np.arange(1, T + 1) * h
    paths = {"euler_1": 1, "euler_01": 10, "euler_001": 100}
    errs = {k: [] for k in paths}
    errs["prop"] = []
    xs = {k: x0.copy() for k in paths}
    xp = x0.copy()
    for t in t_axis:
        ref = analytic(t)
        xp = exact_flow(xp, i_ext)
        errs["prop"].append(abs(xp[0] - ref[0]))
        for k, nsub in paths.items():
            xs[k] = euler_flow(xs[k], i_ext, nsub)
            errs[k].append(abs(xs[k][0] - ref[0]))

    # (b) spike-time drift under a supra-threshold drive
    i_ext2 = 0.008  # v_ss = 1.6 > theta -> periodic firing

    def run(flow, nsteps=1000):
        x = np.zeros(2)
        times = []
        for k in range(nsteps):
            if x[0] >= THETA:
                times.append(k * h)
                x[0] -= THETA
            x = flow(x)
        return np.array(times)

    t_ref = run(lambda x: exact_flow(x, i_ext2))
    drift = {}
    for k, nsub in [("euler_1", 1), ("euler_05", 2), ("euler_01", 10)]:
        t_e = run(lambda x, n=nsub: euler_flow(x, i_ext2, n))
        m = min(len(t_e), len(t_ref))
        drift[k] = t_e[:m] - t_ref[:m]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), layout="constrained")
    ax = axes[0]
    labels = {"euler_1": "Euler, $h$ = 1 ms", "euler_01": "Euler, $h$ = 0.1 ms",
              "euler_001": "Euler, $h$ = 0.01 ms"}
    colors = {"euler_1": st.ORANGE, "euler_01": st.AQUA, "euler_001": "#eda100"}
    for k in paths:
        ax.semilogy(t_axis, errs[k], color=colors[k], label=labels[k], lw=1.6)
    ax.semilogy(t_axis, np.maximum(errs["prop"], 1e-18), color=st.BLUE, lw=1.8,
                label="closed-form propagator, $h$ = 1 ms")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel(r"$|v_{\mathrm{num}} - v_{\mathrm{analytic}}|$")
    ax.set_title("(a) sub-threshold voltage error", fontsize=9.5, loc="left")
    ax.set_ylim(1e-18, 1e-1)
    ax.legend(loc="center right", fontsize=7.8)
    ax = axes[1]
    labels_b = {"euler_1": "Euler, $h$ = 1 ms", "euler_05": "Euler, $h$ = 0.5 ms",
                "euler_01": "Euler, $h$ = 0.1 ms"}
    colors_b = {"euler_1": st.ORANGE, "euler_05": st.AQUA, "euler_01": "#eda100"}
    for k in drift:
        m = len(drift[k])
        ax.plot(np.arange(1, m + 1), drift[k], color=colors_b[k],
                label=labels_b[k], lw=1.6)
    ax.axhline(0.0, color=st.BLUE, lw=1.8, zorder=1)
    ax.text(3, 1.0, "propagator = reference (zero drift)", color=st.BLUE,
            fontsize=8)
    ax.set_xlabel("spike index")
    ax.set_ylabel("spike-time offset (ms)")
    ax.set_title("(b) spike-time drift, periodic firing", fontsize=9.5, loc="left")
    ax.legend(loc="lower left", fontsize=7.8)
    fig.savefig(OUT / "fig03-exactness.png")
    plt.close(fig)
    print(f"fig3: ref spikes {len(t_ref)}, final drifts",
          {k: round(v[-1], 2) for k, v in drift.items()})


# ------------------------------------------------------------------- Figure 4
def fig4_identification():
    N, M = 8, 8
    A_true = np.kron(np.eye(N), propagator(1.0))
    rel = lambda Ah: np.linalg.norm(Ah - A_true, "fro") / np.linalg.norm(A_true, "fro")

    def build_B(rng):
        W = rng.normal(0.0, 0.25, size=(N, M))
        B = np.zeros((2 * N, M + N))
        for n_ in range(N):
            B[2 * n_ + 1, :M] = W[n_, :]
            B[2 * n_, M + n_] = -THETA
        return B

    def simulate(T, rng, B, drive):
        x = np.zeros(2 * N)
        X, U, Y = [], [], []
        for t in range(T):
            v = x[0::2]
            s_own = (v >= THETA).astype(float)
            u = np.concatenate([drive(t, rng), s_own])
            xn = A_true @ x + B @ u
            X.append(x); U.append(u); Y.append(xn)
            x = xn
        return np.array(X).T, np.array(U).T, np.array(Y).T

    def fit_both(X, U, Y, B, sigma, rng):
        Xn = X + sigma * rng.normal(size=X.shape)
        Yn = Y + sigma * rng.normal(size=Y.shape)
        A_kb = np.linalg.lstsq(Xn.T, (Yn - B @ U).T, rcond=1e-12)[0].T
        G = np.linalg.lstsq(np.vstack([Xn, U]).T, Yn.T, rcond=1e-12)[0].T
        return rel(A_kb), rel(G[:, : 2 * N])

    poisson = lambda t, rng: (rng.random(M) < 0.15).astype(float)
    bursty = lambda t, rng: (rng.random(M) <
                             (0.35 if (t // 200) % 2 == 0 else 0.05)).astype(float)
    const = lambda t, rng: np.ones(M)

    # (a) Poisson-probed recording: error vs snapshot count, sigma = 1e-4,
    #     mean over 5 noise realizations
    rng = np.random.default_rng(7)
    B = build_B(rng)
    Xp, Up, Yp = simulate(4000, rng, B, poisson)
    Ts = [50, 100, 200, 500, 1000, 2000, 4000]
    kb_T, j_T = [], []
    for T in Ts:
        kb_r, j_r = [], []
        for _ in range(5):
            kb, j = fit_both(Xp[:, :T], Up[:, :T], Yp[:, :T], B, 1e-4, rng)
            kb_r.append(kb); j_r.append(j)
        kb_T.append(np.mean(kb_r)); j_T.append(np.mean(j_r))

    # (b) excitation regimes: 3 seeds x {poisson, bursty, constant}
    regimes = [("Poisson\nprobe", poisson), ("bursty\ndrive", bursty),
               ("constant\ndrive", const)]
    err_kb = np.zeros((3, 3))
    err_j = np.zeros((3, 3))
    for s_i, seed in enumerate([300, 301, 302]):
        rng_s = np.random.default_rng(seed)
        B_s = build_B(rng_s)
        for r_i, (_, drv) in enumerate(regimes):
            X, U, Y = simulate(4000, rng_s, B_s, drv)
            kb, j = fit_both(X, U, Y, B_s, 1e-4, rng_s)
            err_kb[s_i, r_i] = kb
            err_j[s_i, r_i] = j

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), layout="constrained")
    ax = axes[0]
    ax.loglog(Ts, j_T, "o-", color=st.ORANGE, ms=4.5, label="joint $[A,B]$ fit")
    ax.loglog(Ts, kb_T, "o-", color=st.BLUE, ms=4.5, label="known-$B$ fit")
    ax.set_xlabel("snapshots $m$")
    ax.set_ylabel(r"$\|\hat A - A\|_F \,/\, \|A\|_F$")
    ax.set_title("(a) error vs. recording length\n(Poisson probe, $\\sigma = 10^{-4}$)",
                 fontsize=9.5, loc="left")
    ax.legend()
    ax = axes[1]
    xs = np.arange(3)
    for r_i in range(3):
        ax.semilogy([xs[r_i] - 0.13] * 3, err_kb[:, r_i], "o", color=st.BLUE,
                    ms=5.5, alpha=0.85, label="known-$B$ fit" if r_i == 0 else None)
        ax.semilogy([xs[r_i] + 0.13] * 3, err_j[:, r_i], "o", color=st.ORANGE,
                    ms=5.5, alpha=0.85, label="joint $[A,B]$ fit" if r_i == 0 else None)
    ax.set_xticks(xs, [r[0] for r in regimes], fontsize=8.3)
    ax.set_ylabel(r"$\|\hat A - A\|_F \,/\, \|A\|_F$")
    ax.set_ylim(1e-6, 3)
    ax.set_title("(b) excitation decides identifiability\n(3 seeds, $\\sigma = 10^{-4}$)",
                 fontsize=9.5, loc="left")
    ax.text(2.0, 2.5e-1, "unidentifiable:\nstates never leave\na low-dim. set",
            fontsize=7.8, color=st.INK2, ha="center", va="top")
    ax.legend(loc="lower left", fontsize=7.8)
    ax.grid(axis="x", visible=False)
    fig.savefig(OUT / "fig04-identification.png")
    plt.close(fig)
    print("fig4 small-sample:", f"kb={kb_T[0]:.2e} j={j_T[0]:.2e}",
          "| const-drive:", f"kb={err_kb[:,2].mean():.2e} j={err_j[:,2].mean():.2e}",
          "| poisson:", f"kb={err_kb[:,0].mean():.2e}")


# ------------------------------------------------------------------- Figure 5
def fig5_surrogate():
    v = np.linspace(-1.0, 1.0, 800)
    step = (v >= 0).astype(float)
    beta = 10.0
    sg = 1.0 / (1.0 + beta * np.abs(v)) ** 2
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5), layout="constrained")
    ax = axes[0]
    ax.plot(v[v < 0], step[v < 0], color=st.BLUE, lw=1.8)
    ax.plot(v[v >= 0], step[v >= 0], color=st.BLUE, lw=1.8)
    ax.plot([0], [1.0], "o", color=st.BLUE, ms=4)
    ax.set_title("(a) forward pass: hard step", fontsize=9.5, loc="left")
    ax.set_xlabel(r"$v - \theta$")
    ax.set_ylabel(r"spike output $\Theta$")
    ax.set_yticks([0, 1])
    ax = axes[1]
    ax.plot(v, sg, color=st.ORANGE, lw=1.8)
    ax.set_title("(b) backward pass: surrogate slope", fontsize=9.5, loc="left")
    ax.set_xlabel(r"$v - \theta$")
    ax.set_ylabel(r"$(1 + \beta|v - \theta|)^{-2}$")
    fig.savefig(OUT / "fig08-surrogate.png")
    plt.close(fig)
    print("fig5 done")


# ------------------------------------------------------------------- Figure 6
def fig6_training_curves():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), layout="constrained")
    ax = axes[0]
    for tag, color, label in [("L", st.BLUE, "recurrent (L)"),
                              ("I", st.ORANGE, "feedforward (I)")]:
        c = np.array(CURVES[tag], dtype=float)
        ax.plot(c[:, 0], c[:, 1], color=color, lw=1.6, label=label)
    ax.text(2950, 0.60, "test 0.679", ha="right", fontsize=8.5, color=st.ORANGE)
    ax.text(2950, 0.30, "test 0.808", ha="right", fontsize=8.5, color=st.BLUE)
    ax.set_xlabel("minibatch")
    ax.set_ylabel("training loss (50-step mean)")
    ax.set_title("(a) recurrence: +12.8 points", fontsize=9.5, loc="left")
    ax.legend()
    ax = axes[1]
    for tag, color, label in [("R", st.BLUE, "augmented (R)"),
                              ("O", st.ORANGE, "unaugmented (O)")]:
        c = np.array(CURVES[tag], dtype=float)
        ax.plot(c[:, 0], c[:, 1], color=color, lw=1.6, label=label)
    ax.text(5950, 0.66, "test 0.873", ha="right", fontsize=8.5, color=st.BLUE)
    ax.text(5950, 0.47, "test 0.777 (train 0.038)", ha="right",
            fontsize=8.5, color=st.ORANGE)
    ax.set_xlabel("minibatch")
    ax.set_title("(b) augmentation at 6,000 minibatches", fontsize=9.5, loc="left")
    ax.legend()
    fig.savefig(OUT / "fig09-training-curves.png")
    plt.close(fig)
    print("fig6 done")


# ------------------------------------------------------------------- Figure 7
def fig7_campaign():
    # Honest values: multi-seed means where measured (rounds 3/4 revised by
    # the round-6 audits; rounds 6-7 were single-axis nulls, shown flat).
    stages = [
        ("first\ndemo", 0.502),
        ("round 1\nfiner input\n+ budget", 0.680),
        ("round 2\nrecurrence", 0.808),
        ("round 3\naug ×\nbudget", 0.850),
        ("round 4\n+ 2nd\nlayer", 0.856),
        ("rounds 5–7\naudits +\nsingle-axis\nnulls", 0.856),
        ("round 8\ncombination", 0.8888),
        ("round 9\nensemble\n×3", 0.9000),
        ("round 10\ndiverse\nensemble", 0.9179),
        ("round 11\ndiverse +\nstrong", 0.9366),
    ]
    xs = np.arange(len(stages))
    ys = [s[1] for s in stages]
    fig, ax = plt.subplots(figsize=(7.2, 3.8), layout="constrained")
    ax.axhspan(0.48, 0.71, color=st.BAND_FF, zorder=0)
    ax.axhspan(0.71, 0.83, color=st.BAND_REC, zorder=0)
    ax.axhspan(0.90, 0.94, color=st.SEQ100, zorder=0)
    ax.text(9.55, 0.60, "published\nfeedforward SNNs", fontsize=8, color=st.MUTED,
            ha="left", va="center")
    ax.text(9.55, 0.77, "published\nrecurrent SNNs", fontsize=8, color=st.MUTED,
            ha="left", va="center")
    ax.text(9.62, 0.985, "published state of the art", fontsize=8, color=st.INK2,
            ha="right", va="center")
    ax.errorbar([3, 4, 6], [0.850, 0.856, 0.8888],
                yerr=[[0.027, 0.036, 0.019], [0.027, 0.036, 0.019]],
                color=st.CRITICAL, lw=1.2, capsize=3, zorder=4, fmt="none")
    ax.text(3.0, 0.79, "seed spreads\n(3-seed audits)", fontsize=7.8,
            color=st.CRITICAL, va="center", ha="center")
    ax.plot(xs, ys, "-", color=st.BLUE, lw=1.9, zorder=3)
    ax.plot(xs, ys, "o", color=st.BLUE, ms=6, zorder=4)
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.3f}", xy=(x, y), xytext=(0, 8),
                    textcoords="offset points", ha="center", fontsize=8.7,
                    color=st.INK)
    ax.set_xticks(xs, [s[0] for s in stages], fontsize=6.8)
    ax.set_ylabel("SHD test accuracy")
    ax.set_xlim(-0.4, 11.7)
    ax.set_ylim(0.46, 0.96)
    ax.grid(axis="x", visible=False)
    fig.savefig(OUT / "fig10-campaign.png")
    plt.close(fig)
    print("fig7 done")


# ------------------------------------------------------------------- Figure 8
def fig8_round5():
    mean_r = np.mean([0.873, 0.819, 0.858])
    spread = 0.027
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), layout="constrained",
                             width_ratios=[1, 1.6])
    ax = axes[0]
    seeds = [42, 43, 44]
    accs = [0.8728, 0.8192, 0.8580]
    ax.axhspan(mean_r - spread, mean_r + spread, color=st.SEQ100, zorder=0)
    ax.axhline(mean_r, color=st.INK2, lw=0.9, ls=(0, (4, 3)))
    ax.plot(seeds, accs, "o", color=st.BLUE, ms=7)
    for s, a in zip(seeds, accs):
        ax.annotate(f"{a:.3f}", xy=(s, a), xytext=(10, -3),
                    textcoords="offset points", fontsize=8.5, color=st.INK)
    ax.text(41.7, mean_r + 0.003, f"mean {mean_r:.3f}", fontsize=8, color=st.INK2)
    ax.set_xticks(seeds)
    ax.set_xlim(41.6, 44.7)
    ax.set_ylim(0.78, 0.92)
    ax.set_xlabel("weight-init seed")
    ax.set_ylabel("test accuracy")
    ax.set_title("(a) identical recipe, three seeds", fontsize=9.5, loc="left")
    ax.grid(axis="x", visible=False)

    ax = axes[1]
    variants = [
        ("ensemble ×3 of two-layer (AF)", 0.8821),
        ("5 ms bins (AB)", 0.871),
        ("full 1.4 s duration (AA)", 0.850),
        ("512 wide, no decay (AD)", 0.850),
        ("three layers (AE)", 0.830),
        ("recency-weighted readout (AC)", 0.709),
    ]
    ys = np.arange(len(variants))[::-1]
    vals = [v[1] for v in variants]
    ax.axvspan(mean_r - spread, mean_r + spread, color=st.SEQ100, zorder=0)
    ax.axvline(mean_r, color=st.INK2, lw=0.9, ls=(0, (4, 3)))
    ax.hlines(ys, mean_r, vals, color=st.BLUE, lw=1.6)
    ax.plot(vals, ys, "o", color=st.BLUE, ms=6)
    for (label, v), y in zip(variants, ys):
        ax.annotate(f"{v:.3f}", xy=(v, y), xytext=(0, -13),
                    textcoords="offset points", fontsize=8,
                    ha="center", color=st.INK)
    ax.set_yticks(ys, [v[0] for v in variants], fontsize=8.3)
    ax.set_xlim(0.68, 0.93)
    ax.set_xlabel("test accuracy")
    ax.set_ylim(-0.75, 5.5)
    ax.set_title("(b) round-5 variations vs. baseline mean",
                 fontsize=9.5, loc="left")
    ax.grid(axis="y", visible=False)
    fig.savefig(OUT / "fig11-round5.png")
    plt.close(fig)
    print("fig8 done")


# ------------------------------------------------------------------- Figure 9
def fig9_negative():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), layout="constrained")
    ax = axes[0]
    labels = [r"$\Delta t^*$ 0.5 ms" + "\nI = 10", r"$\Delta t^*$ 1 ms" + "\nI = 10",
              r"$\Delta t^*$ 1 ms" + "\nI = 6", r"$\Delta t^*$ 2 ms" + "\nI = 6"]
    vals = [0.087, 0.130, 0.071, 0.43]
    xs = np.arange(len(vals))
    ax.bar(xs, vals, width=0.62, color=st.BLUE)
    ax.axhline(0.80, color=st.CRITICAL, lw=1.2, ls=(0, (4, 3)))
    ax.text(-0.4, 0.83, "pre-registered gate: ≥ 0.80", fontsize=8,
            color=st.CRITICAL)
    ax.annotate("+907%\nspike count", xy=(3, 0.44), xytext=(0, 4),
                textcoords="offset points", ha="center", fontsize=7.5,
                color=st.INK2)
    ax.set_xticks(xs, labels, fontsize=7.8)
    ax.set_ylim(0, 1.06)
    ax.set_ylabel("coincidence factor (±2 ms)")
    ax.set_title("(a) V2: per-step EDMD surrogate\n(Izhikevich RS, degree 2)",
                 fontsize=9.5, loc="left")
    ax.grid(axis="x", visible=False)
    ax = axes[1]
    labels2 = ["I = 2\nquiescent", "I = 6\nrheobase edge",
               "I = 10\ninterior", "I = 13\nextrapolated"]
    vals2 = [0.0, 0.714, 1.000, 0.13]
    xs2 = np.arange(len(vals2))
    ax.bar(xs2, vals2, width=0.62, color=st.BLUE)
    ax.axhline(0.80, color=st.CRITICAL, lw=1.2, ls=(0, (4, 3)))
    ax.annotate("spurious\nspikes", xy=(0, 0.02), xytext=(0, 6),
                textcoords="offset points", ha="center", fontsize=7.5,
                color=st.CRITICAL)
    ax.annotate("every spike\nwithin ±2 ms", xy=(2, 1.0), xytext=(0, 5),
                textcoords="offset points", ha="center", fontsize=7.5,
                color=st.INK2)
    ax.annotate("closed-loop\namplification", xy=(3, 0.14), xytext=(0, 6),
                textcoords="offset points", ha="center", fontsize=7.5,
                color=st.INK2)
    ax.text(-0.42, 0.83, "gate", fontsize=8, color=st.CRITICAL)
    ax.set_xticks(xs2, labels2, fontsize=7.8)
    ax.set_ylim(0, 1.20)
    ax.set_title("(b) V2b: spike-to-spike return map\n(Izhikevich RS, degree 3)",
                 fontsize=9.5, loc="left")
    ax.grid(axis="x", visible=False)
    fig.savefig(OUT / "fig07-negative-results.png")
    plt.close(fig)
    print("fig9 done")


# ------------------------------------------------------------------- Figure 5
def fig5_probe_richness():
    """S-A (docs/26): known-B recovery vs driven-channel count, N = 32.
    Values transcribed from demo/probe-richness-log.txt."""
    m = [1, 2, 4, 8, 16, 32]
    full_err = [8.960e-1, 8.669e-1, 7.693e-1, 7.073e-1, 5.783e-1, 2.316e-4]
    sub_err = [1.262e-6, 2.042e-6, 1.575e-6, 3.225e-6, 5.471e-6, 2.316e-4]
    fig, ax = plt.subplots(figsize=(5.6, 3.0), layout="constrained")
    ax.semilogy(m, full_err, "o-", color=st.BLUE, ms=5, label="full $\\hat A$ error")
    ax.semilogy(m, sub_err, "o-", color=st.AQUA, ms=5,
                label="error on the excited subspace")
    ax.set_xscale("log", base=2)
    ax.set_xticks(m, [str(v) for v in m])
    ax.set_xlabel("independently driven input channels $m$ (of $N$ = 32)")
    ax.set_ylabel(r"$\|\hat A - A\|_F / \|A\|_F$")
    ax.annotate("the $m = N$ cliff:\nfour orders in one doubling",
                xy=(32, 2.3e-4), xytext=(7, 3e-3), fontsize=8, color=st.INK2,
                arrowprops=dict(arrowstyle="->", color=st.MUTED, lw=1.0))
    ax.legend(loc="center left", fontsize=8)
    fig.savefig(OUT / "fig05-probe-richness.png")
    plt.close(fig)
    print("fig5 (probe richness) done")


# ------------------------------------------------------------------- Figure 6
def fig6_gradient_horizon():
    """S-B (docs/26): measured backward decay factor vs the spectral
    prediction, both regimes. Values from demo/grad-horizon-log.txt."""
    taus = [10, 20, 40, 80]
    silent = [0.946, 0.927, 0.955, 0.968]   # rho_hat / alpha
    active = [1.021, 0.947, 0.880, 0.749]
    fig, ax = plt.subplots(figsize=(5.6, 3.0), layout="constrained")
    ax.axhline(1.0, color=st.CRITICAL, lw=1.1, ls=(0, (4, 3)))
    ax.text(78, 1.015, "leak bound $\\rho = \\alpha$", color=st.CRITICAL,
            fontsize=8, ha="right")
    ax.plot(taus, silent, "o-", color=st.BLUE, ms=5.5,
            label="sub-threshold (silent) regime")
    ax.plot(taus, active, "o-", color=st.ORANGE, ms=5.5,
            label="spiking (trained) regime")
    ax.annotate("trained recurrence\nbeats the leak bound", xy=(10, 1.021),
                xytext=(14, 1.05), fontsize=8, color=st.ORANGE,
                arrowprops=dict(arrowstyle="->", color=st.ORANGE, lw=1.0))
    ax.set_xscale("log", base=2)
    ax.set_xticks(taus, [str(t) for t in taus])
    ax.set_xlabel(r"membrane time constant $\tau_m$ (ms)")
    ax.set_ylabel(r"measured $\hat\rho \, / \, \alpha$")
    ax.set_ylim(0.7, 1.12)
    ax.legend(loc="lower left", fontsize=8)
    fig.savefig(OUT / "fig06-gradient-horizon.png")
    plt.close(fig)
    print("fig6 (gradient horizon) done")


# ------------------------------------------------------------------ Figure 12
def fig12_rom_map():
    """S-C (docs/26): spiking-ROM coincidence vs POD rank at three firing
    rates. Values from demo/spiking-rom-log.txt (N = 64, state dim 128)."""
    ranks = [8, 16, 32, 64, 128]
    rates = [
        ("3.6% firing", [0.1166, 0.1286, 0.1653, 0.1777, 1.0], st.BLUE),
        ("8.0% firing", [0.3044, 0.3376, 0.3694, 0.4052, 1.0], st.ORANGE),
        ("20.3% firing", [0.7059, 0.7692, 0.8259, 0.9490, 1.0], st.AQUA),
    ]
    fig, ax = plt.subplots(figsize=(5.6, 3.0), layout="constrained")
    for label, ys, color in rates:
        ax.plot(ranks, ys, "o-", color=color, ms=5, label=label)
    ax.axvline(64, color=st.BASELINE, lw=0.9, ls=(0, (3, 3)))
    ax.text(64, 0.06, "half state\n($r = N$)", fontsize=7.5, color=st.MUTED,
            ha="center")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ranks, [str(r) for r in ranks])
    ax.set_xlabel("POD rank $r$ (full state = 128)")
    ax.set_ylabel("spike coincidence (±2 bins)")
    ax.set_ylim(0, 1.06)
    ax.legend(loc="upper left", fontsize=8)
    fig.savefig(OUT / "fig12-rom-map.png")
    plt.close(fig)
    print("fig12 (ROM map) done")


if __name__ == "__main__":
    fig1_lif_trace()
    fig2_formulation()
    fig3_exactness()
    fig4_identification()
    fig5_probe_richness()
    fig6_gradient_horizon()
    fig9_negative()
    fig5_surrogate()
    fig6_training_curves()
    fig7_campaign()
    fig8_round5()
    fig12_rom_map()
    print("all figures written to", OUT)
