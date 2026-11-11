"""Shared matplotlib style: validated colorblind-safe categorical order, thin marks,
recessive grid. Colors are assigned to controllers by name so a controller keeps its color
in every figure."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE = "#fcfcfb"
INK = "#1f1f1e"
MUTED = "#6b6a63"

CONTROLLER_COLOR = {
    "motor_only": SERIES[7],
    "motor_defl": SERIES[0],
    "motor_defl_velnotch": SERIES[4],
    "motor_defl_dob": SERIES[3],
    "dual_pi_notch": SERIES[1],
    "dual_pi_nonotch": SERIES[7],
    "dual_rls": SERIES[3],
    "adrc_dual": SERIES[3],
    "stsmc_dual": SERIES[6],
    "lqg": SERIES[2],
    "lqg_nodist": SERIES[4],
    "lqg_rls": SERIES[0],
}


def apply():
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.grid": True,
        "grid.color": "#e4e3dd",
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.6,
        "font.size": 10,
        "axes.titlesize": 11,
        "legend.frameon": False,
        "figure.dpi": 130,
    })
