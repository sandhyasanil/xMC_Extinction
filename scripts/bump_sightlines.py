"""bump_sightlines.py -- external test on the four G24 'SMC bump' sightlines.

None of these sightlines was used to calibrate any of the three prescriptions.
Each binned (10 A) UV curve (x > 3.3 micron^-1) is fitted with the SMC, LMC and
LMC2 prescriptions and with the Milky Way CCM89 relation (R(V) free, 1-8), and
each curve is also evaluated at the photometric R(V) of Gordon et al. (2024).

Outputs
-------
results/bump_sightlines.csv, results/bump_sightlines.tex
figures/bump_sightlines.png / .pdf
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from paperlib import RESULTS, FIGURES, DATA
from SMC import SMCExtinction
from LMC import LMCExtinction
from LMC2 import LMC2Extinction
from _common import ccm89, OPT_BREAK

CURVES = [("SMC", SMCExtinction.axav, "tab:red", ":"),
          ("LMC", LMCExtinction.axav, "tab:blue", "--"),
          ("LMC2", LMC2Extinction.axav, "tab:green", "-"),
          ("Milky Way (CCM89)", ccm89, "k", "-.")]


def run():
    pub = pd.read_csv(DATA / "smc_bumps_published_rv.csv")
    rows = []
    fig, axes = plt.subplots(2, 2, figsize=(9, 7), sharex=True)
    for ax, (_, s) in zip(axes.flat, pub.iterrows()):
        d = pd.read_csv(DATA / "smc_bumps" / f"{s.sightline}_binned10A.csv")
        m = d.x.values > OPT_BREAK
        x, y, e = d.x.values[m], d.axav.values[m], d.err.values[m]
        ax.errorbar(x, y, yerr=e, fmt="o", ms=2, color="0.6", lw=0.5, alpha=0.7)
        for name, fn, col, ls in CURVES:
            p, cov = curve_fit(lambda xx, r: fn(xx, r), x, y, sigma=e, absolute_sigma=True,
                               p0=[3.0], bounds=(1.0, 8.0))
            rv = p[0]
            chi_fit = np.sum(((y - fn(x, rv)) / e) ** 2) / (len(x) - 1)
            chi_pub = np.sum(((y - fn(x, s.rv_pub)) / e) ** 2) / len(x)
            rows.append(dict(sightline=s["name"], rv_pub=s.rv_pub, rv_pub_err=s.rv_pub_err, curve=name,
                             rv_fit=rv, rv_fit_err=np.sqrt(cov[0, 0]), chi2_fit=chi_fit,
                             chi2_at_pub=chi_pub, n=len(x)))
            ax.plot(x, fn(x, rv), color=col, ls=ls, lw=1.5,
                    label=f"{name}: R(V)={rv:.2f}, $\\chi^2_\\nu$={chi_fit:.1f}")
        ax.axvspan(8.0, 8.8, color="0.92", zorder=0)
        ax.set_title(f"{s['name']}  (photometric R(V) = {s.rv_pub:.2f} $\\pm$ {s.rv_pub_err:.2f})", fontsize=9)
        ax.legend(frameon=False, fontsize=6.5, loc="upper left")
        ax.tick_params(direction="in", which="both")
    for ax in axes[1]:
        ax.set_xlabel(r"$x$ ($\mu$m$^{-1}$)")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$A(x)/A(V)$")
    fig.tight_layout()
    RESULTS.mkdir(exist_ok=True); FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / "bump_sightlines.png", dpi=200)
    fig.savefig(FIGURES / "bump_sightlines.pdf")
    t = pd.DataFrame(rows)
    t.to_csv(RESULTS / "bump_sightlines.csv", index=False, float_format="%.4f")
    lines = [r"\begin{tabular}{llrrrr}", r"\toprule",
             r"Sightline & Curve & $R(V)_{\rm fit}$ & $\chi^2_{\nu,\rm fit}$ & $\chi^2_{\nu}$ at $R(V)_{\rm phot}$ \\",
             r"\midrule"]
    for sl, g in t.groupby("sightline", sort=False):
        for i, (_, r) in enumerate(g.sort_values("chi2_fit").iterrows()):
            lab = f"{sl} ({r.rv_pub:.2f} $\\pm$ {r.rv_pub_err:.2f})" if i == 0 else ""
            lines.append(f"{lab} & {r.curve} & {r.rv_fit:.2f} & {r.chi2_fit:.2f} & {r.chi2_at_pub:.2f} \\\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    (RESULTS / "bump_sightlines.tex").write_text("\n".join(lines) + "\n")
    print(t.round(2).to_string(index=False))
    print(f"wrote {RESULTS / 'bump_sightlines.csv'} and {FIGURES / 'bump_sightlines.png'}")


if __name__ == "__main__":
    run()
