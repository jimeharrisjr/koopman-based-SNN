"""Shared style for the kdmd-SNN paper figures (dataviz reference palette, light mode)."""
import matplotlib as mpl

# categorical slots (fixed order, never cycled)
BLUE = "#2a78d6"     # slot 1
ORANGE = "#eb6834"   # slot 2
AQUA = "#1baf7a"     # slot 3
CRITICAL = "#d03b3b" # status: critical (gates, thresholds, failure marks)

INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#ffffff"
SEQ100 = "#cde2fb"   # sequential blue, lightest band

BAND_FF = "#f2f1ee"   # neutral band fills for published ranges
BAND_REC = "#e6e5df"

def apply():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "savefig.dpi": 200,
        "font.family": "sans-serif",
        "font.size": 9.5,
        "axes.titlesize": 10,
        "axes.labelsize": 9.5,
        "axes.edgecolor": BASELINE,
        "axes.labelcolor": INK2,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "lines.linewidth": 1.8,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "text.color": INK,
        "axes.titlecolor": INK,
        "mathtext.default": "it",
    })
