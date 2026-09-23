"""LMC average extinction curve (Gordon et al. 2003 sample) in CCM form.

CORRECTION NOTE
---------------
The coefficient b1 was published as -3.534 in Table 1 of the manuscript and in
the first release of this code.  That value is wrong: it breaks the continuity
condition b(3.3) = 3.53402 imposed when the UV segment was fitted, and it is not
the value that was used to produce the fitted R(V) values in the paper.  The
correct value, which both restores continuity and reproduces the published
sightline R(V) values, is

    b1 = -3.4221                       (see scripts/check_model.py)

The number 3.534 is the *value of b at the breakpoint*, not the intercept b1;
the two were transcribed into each other at some point.
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

    Examples
    --------
    >>> import numpy as np
    >>> from LMC import LMCExtinction
    >>> x = np.linspace(0.3, 8.0, 5)
    >>> np.round(LMCExtinction(x, rv=3.41).evaluate(), 3)
    array([0.06 , 1.264, 2.433, 2.599, 3.437])
    """

    NAME = "LMC"
    #        a1       a2       a3       fa1      fa2
    A_PARS = (1.4358, -0.2311, -0.0226, 0.0787, -0.0125)
    #        b1       b2       b3       fb1      fb2
    B_PARS = (-3.4221, 1.9297, 1.1886, 0.0787, -0.0125)
    GAMMA = 0.3309
    RV_DEFAULT = 3.41
    X_RANGE = (0.3, 8.7)
    RV_RANGE = (2.8, 4.0)      # range actually spanned by the LMC sample
