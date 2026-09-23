"""Minimal example: evaluate and plot the three Magellanic Cloud extinction curves.

Run from the repository root:
    python examples/sample.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib.pyplot as plt

from SMC import SMCExtinction
from LMC import LMCExtinction
from LMC2 import LMC2Extinction

# Wavenumber grid in inverse microns.  A bare array is interpreted as
# micron^-1; an astropy Quantity with length or wavenumber units also works, e.g.
#     import astropy.units as u
#     x = np.linspace(1200, 20000, 500) * u.AA
x = np.linspace(0.3, 8.7, 500)

fig, ax = plt.subplots(figsize=(7, 5))
for cls, colour in [(SMCExtinction, "tab:red"),
                    (LMCExtinction, "tab:blue"),
                    (LMC2Extinction, "tab:green")]:
    curve = cls(x)                      # default R(V) reproduces the average curve
    ax.plot(x, curve.evaluate(), color=colour,
            label=f"{cls.NAME}, R(V) = {curve.rv:g}")
    for rv, ls in [(cls.RV_RANGE[0], "--"), (cls.RV_RANGE[1], ":")]:
        ax.plot(x, cls.axav(x, rv), color=colour, ls=ls, lw=1, alpha=0.6,
                label=f"{cls.NAME}, R(V) = {rv:g}")

ax.set_xlabel(r"$x$ ($\mu$m$^{-1}$)")
ax.set_ylabel(r"$A(\lambda)/A(V)$")
ax.legend(frameon=False, fontsize=9)
ax.set_title("xMC_Extinction curves")
fig.tight_layout()
plt.show()
