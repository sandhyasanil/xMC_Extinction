"""LMC average extinction curve (Gordon et al. 2003 sample) in CCM form.
"""
from __future__ import annotations

from _common import BaseMCExtinction


class LMCExtinction(BaseMCExtinction):
    """A(x)/A(V) for LMC-type dust with R(V) as the single shape parameter.

    Parameters
    ----------
    x : array_like or astropy Quantity
        Wavenumber in micron^-1 (bare arrays are assumed to be in micron^-1),
        or a Quantity with length or wavenumber units.
    rv : float, optional
        Total-to-selective extinction R(V) = A(V)/E(B-V).
        Default 3.41, the value at which this curve reproduces the Gordon et al.
        (2003) LMC average.

    """

    NAME = "LMC"
    #        a1       a2       a3       fa1      fa2
    A_PARS = (1.4358, -0.2311, -0.0226, 0.0787, -0.0125)
    #        b1       b2       b3       fb1      fb2
    B_PARS = (-3.4221, 1.9297, 1.1886, 0.0787, -0.0125)
    GAMMA = 0.3309
    RV_DEFAULT = 3.41
    X_RANGE = (0.3, 8.7)
    RV_RANGE = (3.15, 3.96)      # range actually spanned by the LMC sample
