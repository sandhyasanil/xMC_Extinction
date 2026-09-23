"""plot_curves.py -- main results figure: the three analytic curves with their data.

figures/average_curves.png : one panel per family, showing the published average
curve data, the analytic form at the nominal R(V), and the family of curves at
the extremes of the calibrated R(V) range.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paperlib import FAMILIES, FIGURES, model_class, average_curve


def run():
    xg = np.linspace(0.3, 8.7, 900)
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.9), sharey=True)
    for ax, fam, col in zip(axes, FAMILIES, ["tab:red", "tab:blue", "tab:green"]):
        c = model_class(fam)
        d = average_curve(fam)
        lo, hi = c.RV_RANGE
        ax.plot(xg, c.axav(xg, lo), color=col, lw=1, ls="--", alpha=0.7,
                label=f"R(V) = {lo:g}")
        ax.plot(xg, c.axav(xg, hi), color=col, lw=1, ls=":", alpha=0.7,
                label=f"R(V) = {hi:g}")
        ax.plot(xg, c.axav(xg, c.RV_DEFAULT), color="k", lw=1.8,
                label=f"this work, R(V) = {c.RV_DEFAULT:g}")
        ax.errorbar(d.x, d.axav, yerr=d.err, fmt="o", ms=3.5, lw=1, capsize=2,
                    color=col, alpha=0.85, label="average curve data")
        ax.set_xlabel(r"$x$ ($\mu$m$^{-1}$)")
        ax.set_title(fam)
        ax.set_xlim(0, 9); ax.set_ylim(0, 7)
        ax.tick_params(direction="in", which="both")
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    axes[0].set_ylabel(r"$A(x)/A(V)$")
    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / "average_curves.png", dpi=200)
    print(f"wrote {FIGURES / 'average_curves.png'}")


if __name__ == "__main__":
    run()
