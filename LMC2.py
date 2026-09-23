"""LMC2 (LMC supershell / 30 Dor) average extinction curve in CCM form."""
from __future__ import annotations

from _common import BaseMCExtinction


class LMC2Extinction(BaseMCExtinction):
    """A(x)/A(V) for LMC2-type dust with R(V) as the single shape parameter.

    Parameters
    ----------
    x : array_like or astropy Quantity
        Wavenumber in micron^-1, or a Quantity with length/wavenumber units.
    rv : float, optional
        Default RV_DEFAULT (see below).

    Notes
    -----
    The *average* LMC2 curve is reproduced well at its nominal R(V) = 2.76.
    Individual LMC2 sightlines, however, return fitted R(V) values that are
    systematically lower than the Gordon et al. (2003) photometric values; the
    LMC2 R(V) scale should therefore be treated as approximate.  See the
    Limitations section of the paper.
    """

    NAME = "LMC2"
    A_PARS = (1.6177, -0.2814, -0.0507, 0.0421, 0.0044)
    B_PARS = (-2.8298, 1.8610, 0.4174, 0.0421, 0.0044)
    GAMMA = 0.1856
    RV_DEFAULT = 2.76          # reproduces the G03 LMC2 average (reduced chi2 = 1.1)
    X_RANGE = (0.3, 8.7)
    RV_RANGE = (1.7, 3.5)
