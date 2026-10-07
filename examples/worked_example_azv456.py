"""
worked_example_azv456.py  --  the worked example for Section 6 (Using the prescriptions).

What it does
  1. reads the binned AzV 456 extinction curve (Gordon et al. 2024 "bump" sightline,
     NOT used in any of our calibrations)
  2. keeps only the UV points (x > 3.3 micron^-1)
  3. fits R(V) with each of our three curves, and with the Milky Way CCM89 curve
  4. prints a small table (best-fit R(V), its formal error, reduced chi^2)
  5. makes the figure for the paper: data + four best-fit curves + residuals

How to run (from the xMC_Extinction repository root):
  python worked_example_azv456.py  "/path/to/binned_azv456.xlsx"
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from SMC import SMCExtinction
from LMC import LMCExtinction
from LMC2 import LMC2Extinction
from _common import ccm89

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/smc_bumps"
FIGURES = ROOT /"figures"
# ---------------------------------------------------------------- 1. read the data
path = DATA / "azv456_binned10A.csv"
d = pd.read_csv(path)                      # columns: wl (micron), A(x), Err
x = d["x"].to_numpy()                 # wavenumber in micron^-1
y = d["axav"].to_numpy()                     # A(x)/A(V)
e = d["err"].to_numpy()                      # uncertainty on A(x)/A(V)

# ---------------------------------------------------------------- 2. UV only
uv = x > 3.3
x, y, e = x[uv], y[uv], e[uv]

# ---------------------------------------------------------------- 3. fit R(V)
models = {
    "LMC2 (this work)": LMC2Extinction.axav,
    "LMC (this work)":  LMCExtinction.axav,
    "SMC (this work)":  SMCExtinction.axav,
    "Milky Way (CCM89)": ccm89,
}
RV_PHOT, RV_PHOT_ERR = 2.39, 0.10            # Gordon et al. (2024), for comparison only

results = {}
for name, fn in models.items():
    p, cov = curve_fit(lambda xx, rv: fn(xx, rv), x, y, sigma=e,
                       absolute_sigma=True, p0=[3.0], bounds=(1.0, 8.0))
    rv, rv_err = p[0], np.sqrt(cov[0, 0])
    chi2 = np.sum(((y - fn(x, rv)) / e) ** 2) / (len(x) - 1)
    results[name] = (rv, rv_err, chi2)

# ---------------------------------------------------------------- 4. print the table
print(f"AzV 456, binned, UV only: {len(x)} points.  Photometric R(V) = {RV_PHOT} +/- {RV_PHOT_ERR}")
print(f"{'curve':20s} {'R(V)':>6s} {'err':>6s} {'red. chi2':>10s}")
for name, (rv, err, chi2) in sorted(results.items(), key=lambda kv: kv[1][2]):
    print(f"{name:20s} {rv:6.2f} {err:6.2f} {chi2:10.2f}")

# ---------------------------------------------------------------- 5. the figure
xg = np.linspace(3.3, 8.7, 400)
fig, (ax, axr) = plt.subplots(2, 1, figsize=(6.5, 6.2), sharex=True,
                              gridspec_kw=dict(height_ratios=[3, 1.3], hspace=0.05))
ax.errorbar(x, y, yerr=e, fmt="o", ms=2.5, color="0.55", alpha=0.7, lw=0.6, label="AzV 456 (binned)")
styles = {"LMC2 (this work)": ("tab:green", "-"), "LMC (this work)": ("tab:blue", "--"),
          "SMC (this work)": ("tab:red", ":"), "Milky Way (CCM89)": ("k", "-.")}
for name, fn in models.items():
    rv, _, chi2 = results[name]
    c, ls = styles[name]
    ax.plot(xg, fn(xg, rv), color=c, ls=ls, lw=1.8,
            label=f"{name}: R(V) = {rv:.2f}, $\\chi^2_\\nu$ = {chi2:.1f}")
    axr.plot(x, y - fn(x, rv), ".", color=c, ms=2.5, alpha=0.7)
axr.axhline(0, color="k", lw=0.8)
axr.axvspan(8.0, 8.7, color="0.9", zorder=0)
ax.axvspan(8.0, 8.7, color="0.9", zorder=0)
ax.set_ylabel(r"$A(x)/A(V)$")
axr.set_ylabel("data $-$ model")
axr.set_xlabel(r"$x$ ($\mu$m$^{-1}$)")
ax.legend(frameon=False, fontsize=7.5, loc="upper left")
for a in (ax, axr):
    a.tick_params(direction="in", which="both")
fig.savefig(FIGURES / "worked_example_azv456.png", dpi=200, bbox_inches="tight")
print("wrote worked_example_azv456.png")
