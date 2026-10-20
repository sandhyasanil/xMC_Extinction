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
    A_PARS = (0.8762, -0.06310, -0.01062, 0.08299, 0.00001)
    B_PARS = (-4.6370, 2.46595, 0.06051, -0.05923, -0.00005)
    GAMMA = 0.1215
    RV_DEFAULT = 3.02
    X_RANGE = (0.3, 8.7)
    RV_RANGE = (2.0, 5.6)
