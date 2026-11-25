"""Figures for blog post 4, "Read the equations back out of a recording".

Plots the CSV dump of the `spectral_probe` example, so every curve comes
from the same run whose printed output the post quotes:

    cargo run --release --example spectral_probe -- --dump /tmp/probe
    python analysis/probe_figures.py /tmp/probe blogs/images

Needs numpy and matplotlib; the style is paper/scripts/figstyle.py.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "paper" / "scripts"))
import figstyle as st  # noqa: E402

st.apply()

DT_DECAY = 0.1  # ms, experiments 1-2
DT_SPIKING = 0.5  # ms, experiments 3-4
TEXT_SEED = 7  # the noise seed whose table the post prints


def load(path):
    """CSV -> dict of numpy columns (string columns stay strings)."""
    with open(path) as f:
        header = f.readline().strip().split(",")
        rows = [line.strip().split(",") for line in f if line.strip()]
    cols = {}
    for k, name in enumerate(header):
        vals = [r[k] for r in rows]
        try:
            cols[name] = np.array([float(v) for v in vals])
        except ValueError:
            cols[name] = np.array(vals)
    return cols


# ------------------------------------------------------------ figure 8: decay
def fig_decay(data, out):
    d = load(data / "decay.csv")
    modes = load(data / "decay_modes.csv")
    # order the two modes slowest first, and label them by recovered tau
    order = np.argsort(-modes["tau_ms"])
    t = d["t_ms"]

    fig, ax = plt.subplots(figsize=(6.6, 2.9), layout="constrained")
    ax.axhline(0, color=st.BASELINE, lw=0.8, zorder=0)
    ax.plot(t, d["v"], color=st.BLUE, lw=2.0, label="voltage, as recorded")
    styles = ["--", ":"]
    for k, idx in enumerate(order):
        idx = int(modes["mode"][idx])
        tau = modes["tau_ms"][order[k]]
        ax.plot(t, d[f"v_mode{idx}"], styles[k], color=st.BLUE, lw=1.4,
                label=f"its {tau:.0f} ms mode (eigenvalue {modes['re'][order[k]]:.4f})")
    ax.plot(t, d["i"], color=st.ORANGE, lw=1.6, label="synaptic current, as recorded")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("state")
    ax.set_xlim(0, t[-1])
    ax.legend(loc="upper right", fontsize=8)
    ax.text(0.5, 0.05, "voltage = 10 ms mode + 5 ms mode, to round-off",
            transform=ax.transAxes, fontsize=8.3, color=st.INK2, ha="center", va="bottom")
    fig.savefig(out / "fig-08-decay-modes.png")
    plt.close(fig)


# ------------------------------------------------------ figure 9: convergence
def fig_convergence(data, out):
    c = load(data / "convergence.csv")
    seeds = np.unique(c["seed"]).astype(int)
    sizes = np.unique(c["steps"]).astype(int)

    fig, ax = plt.subplots(figsize=(6.6, 3.1), layout="constrained")
    for truth, name in [(10.0, "true membrane constant, 10 ms"),
                        (5.0, "true synapse constant, 5 ms")]:
        ax.plot([sizes[0] * 0.9, sizes[-1] * 1.04], [truth, truth],
                color=st.INK2, lw=0.9, ls="--", zorder=1)
        ax.text(sizes[-1] * 1.09, truth, name, fontsize=8, color=st.INK2,
                va="center", ha="left")
    for s in seeds:
        m = c["seed"] == s
        bold = s == TEXT_SEED
        for col, color in [("tau_m", st.BLUE), ("tau_s", st.ORANGE)]:
            ax.plot(c["steps"][m], c[col][m], "o-" if bold else "-",
                    color=color, lw=2.0 if bold else 0.9,
                    alpha=1.0 if bold else 0.3, ms=4 if bold else 0,
                    zorder=3 if bold else 2)
    ax.set_xscale("log")
    ticks = sizes[[0, 2, 4, 6, 8, 11]]
    ax.set_xticks(ticks, [str(v) for v in ticks])
    ax.minorticks_off()
    ax.set_xlim(sizes[0] * 0.9, sizes[-1] * 2.4)
    ax.set_ylim(0, 16)
    ax.set_xlabel(f"steps of the recording used (one step = {DT_DECAY} ms)")
    ax.set_ylabel("recovered time constant (ms)")
    ax.text(sizes[0] * 0.95, 0.5,
            "bold: the noise seed printed in the post; light: 19 other seeds\n"
            "(a few 10-step fits land above the axis)",
            fontsize=8, color=st.INK2, va="bottom")
    fig.savefig(out / "fig-09-convergence-noise.png")
    plt.close(fig)


# ------------------------------------------ figure 10: the reset in disguise
def fig_reset(data, out):
    sp = load(data / "spiking.csv")
    ev = load(data / "eigs.csv")
    t = sp["t_ms"]
    any_spike = (sp["s0"] + sp["s1"] + sp["s2"]) > 0

    window = 200.0  # ms shown in (b) and (c); the fit and the 85 % use all 400
    fig = plt.figure(figsize=(7.4, 3.8), layout="constrained")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.55], height_ratios=[1.0, 1.0])
    ax_e = fig.add_subplot(gs[:, 0])
    ax_v = fig.add_subplot(gs[0, 1])
    ax_r = fig.add_subplot(gs[1, 1], sharex=ax_v)

    # (a) eigenvalues against the unit circle, with a zoom on the crowd near 1
    kb = ev["fit"] == "knownb"
    nv = ev["fit"] == "naive"

    def draw_plane(ax, ms_scale=1.0):
        th = np.linspace(-0.4, 0.4, 400)
        ax.plot(np.cos(th), np.sin(th), color=st.INK2, lw=1.0, zorder=1)
        ax.axvspan(1.0, 1.1, color=st.CRITICAL, alpha=0.07, lw=0, zorder=0)
        ax.plot(ev["re"][kb], ev["im"][kb], "o", color=st.BLUE, ms=8 * ms_scale,
                mfc="none", mew=1.8, label="known-B fit (3 copies each)", zorder=3)
        ax.plot(ev["re"][nv], ev["im"][nv], "x", color=st.ORANGE, ms=7 * ms_scale,
                mew=1.8, label="plain DMD, inputs ignored", zorder=4)

    draw_plane(ax_e)
    ax_e.text(0.997, 0.058, "unit circle", fontsize=7.8, color=st.INK2, ha="right",
              va="center", rotation=90)
    for k, x in enumerate(sorted(set(np.round(ev["re"][kb], 6)))):
        tau = -DT_SPIKING / np.log(x)
        below = k == 0
        ax_e.annotate(f"{tau:.0f} ms\n×3", (x, 0), xytext=(0, -13 if below else 13),
                      textcoords="offset points", fontsize=8, color=st.BLUE,
                      ha="center", va="top" if below else "bottom")
    ax_e.set_xlim(0.70, 1.03)
    ax_e.set_ylim(-0.085, 0.085)
    ax_e.set_xlabel("real part of eigenvalue")
    ax_e.set_ylabel("imaginary part")
    ax_e.set_title("(a) the two spectra", fontsize=9.5, loc="left")
    ax_e.legend(loc="lower left", fontsize=7.4)

    axins = ax_e.inset_axes([0.06, 0.60, 0.46, 0.34])
    draw_plane(axins, ms_scale=0.9)
    axins.set_xlim(0.990, 1.003)
    axins.set_ylim(-0.0025, 0.0025)
    axins.set_xticks([0.99, 1.0], ["0.99", "1.00"])
    axins.set_yticks([])
    axins.tick_params(labelsize=7)
    axins.grid(False)
    grow = nv & (np.hypot(ev["re"], ev["im"]) > 1)
    axins.annotate("growing\n|λ| = 1.0008", (ev["re"][grow][0], 0), xytext=(-4, 9),
                   textcoords="offset points", fontsize=7.4, color=st.CRITICAL,
                   ha="right", va="bottom")
    axins.text(0.9903, -0.0021, "zoom", fontsize=7, color=st.INK2, ha="left", va="bottom")
    ax_e.indicate_inset_zoom(axins, edgecolor=st.MUTED, lw=0.8)

    # (b) voltages with spike ticks
    colors = [st.BLUE, st.ORANGE, st.AQUA]
    for j, color in enumerate(colors):
        ax_v.plot(t, sp[f"v{j}"], color=color, lw=0.8, alpha=0.9)
        ts = t[sp[f"s{j}"] > 0]
        ax_v.vlines(ts, 1.08 + 0.09 * j, 1.16 + 0.09 * j, color=color, lw=0.9)
    ax_v.axhline(1.0, color=st.BASELINE, lw=0.8, ls="--")
    ax_v.text(1.0, 1.0, " threshold", fontsize=7.4, color=st.INK2, ha="left", va="bottom")
    ax_v.set_ylim(-0.05, 1.4)
    ax_v.set_yticks([0, 0.5, 1.0])
    ax_v.set_ylabel("voltage")
    ax_v.tick_params(labelbottom=False)
    ax_v.set_title(f"(b) three driven neurons, first {window:.0f} ms; ticks mark spikes",
                   fontsize=9.5, loc="left")

    # (c) per-step residual of the plain fit, spikes highlighted
    ax_r.plot(t, sp["resid_naive"], color=st.INK, lw=0.7, label="plain DMD, per step")
    ax_r.plot(t[any_spike], sp["resid_naive"][any_spike], "o", color=st.ORANGE, ms=3.2,
              label="steps with a spike")
    ax_r.plot(t, sp["resid_knownb"], color=st.BLUE, lw=1.4,
              label=f"known-B fit (max {sp['resid_knownb'].max():.0e})")
    ax_r.set_xlim(0, window)
    ax_r.set_ylim(0, 2.0)
    ax_r.set_yticks([0, 0.5, 1.0, 1.5])
    ax_r.set_xlabel("time (ms)")
    ax_r.set_ylabel("residual")
    ax_r.set_title("(c) where the plain fit's error lives", fontsize=9.5, loc="left")
    ax_r.legend(loc="upper left", fontsize=7.2, ncol=3, columnspacing=0.8, handlelength=1.4)

    fig.savefig(out / "fig-10-reset-in-disguise.png")
    plt.close(fig)

    share = (sp["resid_naive"][any_spike] ** 2).sum() / (sp["resid_naive"] ** 2).sum()
    print(f"fig10: {any_spike.sum()} spiking steps carry {100 * share:.1f}% of the residual energy")


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    data, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    fig_decay(data, out)
    fig_convergence(data, out)
    fig_reset(data, out)
    print("wrote fig-08, fig-09, fig-10 to", out)


if __name__ == "__main__":
    main()
