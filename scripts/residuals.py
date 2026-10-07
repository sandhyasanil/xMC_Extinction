"""residuals.py -- residual of each sightline from the model, versus wavelength.

Replaces the six-panel delta-vs-delta scatter figures (Figs 11, 18, 19 of the
draft) with a single figure per family that shows directly where in wavelength
the one-parameter description succeeds and where it degrades.

Residual definition:   R(x) = A(x)/A(V)|observed  -  A(x)/A(V)|model(R(V)_published)
i.e. the model is evaluated at the independently published R(V), with no fitted
parameters, so the residuals are not artificially flattened by a fit.

Outputs
-------
results/residual_profiles.csv     median and 16th/84th percentile vs x
figures/residuals_vs_wavelength.png
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
from paperlib import FAMILIES, RESULTS, FIGURES, model_class, sightlines
from _common import OPT_BREAK

GRID = np.arange(3.35, 8.05, 0.15)     # stop at x = 8, the stated validity limit
HALF = 0.075
FLAG = {"SMC": ["azv214", "smc5-000398", "mr12-star09"]}   # model failures + data-quality case (Sec. 5.3)


def profiles(fam, exclude_flagged=True):
    cls = model_class(fam)
    curves = []
    for name, d, rv_pub, _ in sightlines(fam):
        if exclude_flagged and name in FLAG.get(fam, []):
            continue
        m = d.x.values > OPT_BREAK
        x, y = d.x.values[m], d.axav.values[m]
        res = y - cls.axav(x, rv_pub)
        prof = np.full(GRID.size, np.nan)
        for i, xc in enumerate(GRID):
            sel = np.abs(x - xc) < HALF
            if sel.sum():
                prof[i] = np.median(res[sel])
        curves.append(prof)
    C = np.array(curves)
    with np.errstate(invalid="ignore"):
        lo, med, hi = (np.nanpercentile(C, p, axis=0) for p in (16, 50, 84))
    return GRID, med, lo, hi, C


def run():
    font = {'family': 'sans', 'weight': 'normal', 'size': 20} #Set font type to sans-serif, normal font (not bold) and font size.
    plt.rc('font', **font) #Setting the plot parameters to the ones we've set above
    mpl.rcParams['axes.linewidth'] = 1.5 #Here we set the line width of the axes.
    figure_size = [18, 6] #Figure size. 
    fig, axes = plt.subplots(1, 3, figsize=(figure_size[0], figure_size[1]), sharey=True)
    fig.subplots_adjust(wspace=0.12)
    out = []
    for ax, fam, col in zip(axes, FAMILIES, ["tab:red", "tab:blue", "tab:green"]):
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(1.5)
        g, med, lo, hi, C = profiles(fam)
        for c in C:
            ax.plot(g, c, color="0.75", lw=0.6, alpha=0.8)
        ax.fill_between(g, lo, hi, color=col, alpha=0.25, lw=0)
        ax.plot(g, med, color=col, lw=2.5)
        ax.axhline(0, color="k", lw=0.8, ls=":")
        ax.set_xlabel(r"$x$ ($\mu$m$^{-1}$)")
        ax.set_title(f"{fam}  (n = {C.shape[0]})", fontsize = 20)
        ax.tick_params(direction='in', length=5, width=1.5)
        ax.tick_params(direction='in', which='minor', length=3.5, width=1.5)
        
        ax.yaxis.set_ticks_position('both')
        ax.xaxis.set_ticks_position('both')
        ax.set_ylim(-1.1, 1.1)
        for i, xc in enumerate(g):
            out.append(dict(family=fam, x=xc, median=med[i], p16=lo[i], p84=hi[i]))
    axes[0].set_ylabel(r"$A(x)/A(V)$  observed $-$ model")
    fig.tight_layout()
    RESULTS.mkdir(exist_ok=True); FIGURES.mkdir(exist_ok=True)
    pd.DataFrame(out).to_csv(RESULTS / "residual_profiles.csv", index=False)
    fig.savefig(FIGURES / "residuals_vs_wavelength.png", dpi=200)
    fig.savefig(FIGURES / "residuals_vs_wavelength.pdf")
    print(f"wrote {FIGURES / 'residuals_vs_wavelength.png'}")

    t = pd.DataFrame(out)
    print("\nscatter (84th-16th percentile) at selected wavenumbers:")
    for xq in (4.0, 4.6, 6.0, 7.0, 8.0):
        s = t[np.isclose(t.x, t.x[(t.x - xq).abs().idxmin()])]
        line = "  x = %.1f :  " % xq
        for fam in FAMILIES:
            r = s[s.family == fam]
            if len(r):
                line += f"{fam} {float(r.p84.iloc[0] - r.p16.iloc[0]):.3f}   "
        print(line)
    return t


if __name__ == "__main__":
    run()
