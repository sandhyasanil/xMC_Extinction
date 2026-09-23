"""SMC average extinction curve (Gordon et al. 2024 steep sample) in CCM form."""
from __future__ import annotations

from _common import BaseMCExtinction


class SMCExtinction(BaseMCExtinction):
    """A(x)/A(V) for SMC-type dust with R(V) as the single shape parameter.

    Parameters
    ----------
    x : array_like or astropy Quantity
        Wavenumber in micron^-1, or a Quantity with length/wavenumber units.
    rv : float, optional
        Default 3.02, the R(V) at which this curve reproduces the Gordon et al.
        (2024) SMC Average curve.  (Note 3.02 is also the R(V) quoted by Gordon
        et al. 2024 for that average; 2.74 is the older G03 "SMCBar" value and
        should not be used as the default here.)
    """

    NAME = "SMC"
    A_PARS = (0.5503, 0.03266, 0.007102, 0.05234, -0.0001)
    B_PARS = (-3.6040, 2.1618, 0.007156, 0.04273, 0.0001)
    GAMMA = 0.0714
    RV_DEFAULT = 3.02
    X_RANGE = (0.3, 8.7)
    RV_RANGE = (2.0, 5.2)
