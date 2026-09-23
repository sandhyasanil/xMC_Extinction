"""Shared pieces of the xMC extinction curves.

All three Magellanic Cloud curves in this package share, by construction, the
Cardelli, Clayton & Mathis (1989, CCM) near-infrared and optical segments
(x <= 3.3 micron^-1).  Only the ultraviolet segment (x > 3.3 micron^-1) differs
between the SMC, LMC and LMC2 families.  Keeping the shared part in one place
guarantees that the three classes cannot drift apart.
"""
from __future__ import annotations

import numpy as np

__all__ = ["OPT_BREAK", "A_BREAK", "B_BREAK", "ccm_optical_nir", "BaseMCExtinction", "ccm89"]

#: optical/UV breakpoint, micron^-1
OPT_BREAK = 3.3

# CCM (1989) optical polynomial coefficients, in y = x - 1.82
_CCM_A = (1.0, 0.17699, -0.50447, -0.02427, 0.72085, 0.01979, -0.77530, 0.32999)
_CCM_B = (0.0, 1.41338, 2.28305, 1.07233, -5.38434, -0.62251, 5.30260, -2.09002)


def _poly(y, coeffs):
    out = np.zeros_like(y)
    for i, c in enumerate(coeffs):
        out = out + c * y ** i
    return out


def ccm_optical_nir(x):
    """CCM (1989) a(x), b(x) for x <= 3.3 micron^-1 (NIR power law + optical polynomial)."""
    x = np.asarray(x, dtype=float)
    a = np.zeros_like(x)
    b = np.zeros_like(x)
    m_ir = x <= 1.1
    a[m_ir] = 0.574 * x[m_ir] ** 1.61
    b[m_ir] = -0.527 * x[m_ir] ** 1.61
    m_opt = (x > 1.1) & (x <= OPT_BREAK)
    y = x[m_opt] - 1.82
    a[m_opt] = _poly(y, _CCM_A)
    b[m_opt] = _poly(y, _CCM_B)
    return a, b


#: value of the CCM optical polynomials at the breakpoint; the UV segment of
#: every curve in this package is constrained to match these two numbers.
A_BREAK = float(ccm_optical_nir([OPT_BREAK])[0][0])   # 0.66208...
B_BREAK = float(ccm_optical_nir([OPT_BREAK])[1][0])   # 3.53402...


class BaseMCExtinction:
    """Base class: A(x)/A(V) = a(x) + b(x)/R(V) with a CCM-form UV segment.

    Subclasses set the class attributes A1..GAMMA.  Input ``x`` may be a plain
    array in micron^-1, or an astropy Quantity with length or wavenumber units.
    """

    NAME = "base"
    #: (a1, a2, a3, fa1, fa2)
    A_PARS: tuple = ()
    #: (b1, b2, b3, fb1, fb2)
    B_PARS: tuple = ()
    GAMMA: float = 0.0
    #: default R(V): the value at which this curve reproduces the published average
    RV_DEFAULT: float = 3.1
    #: range over which the prescription is calibrated
    X_RANGE = (0.3, 8.7)
    RV_RANGE = (2.0, 5.2)

    def __init__(self, x, rv=None):
        self.x = self._to_invmicron(x)
        self.rv = float(self.RV_DEFAULT if rv is None else rv)

    # ------------------------------------------------------------------
    @staticmethod
    def _to_invmicron(x):
        """Accept a bare array (assumed micron^-1) or an astropy Quantity."""
        try:
            from astropy import units as u
        except ImportError:                                   # astropy optional
            return np.asarray(x, dtype=float)
        if isinstance(x, u.Quantity):
            ptype = str(u.get_physical_type(x.unit))
            if ptype == "length":
                return np.asarray((1.0 / x.to(u.micron)).value, dtype=float)
            if ptype == "wavenumber":
                return np.asarray(x.to(1.0 / u.micron).value, dtype=float)
            raise ValueError("x must have units of length or wavenumber")
        return np.asarray(x, dtype=float)

    # ------------------------------------------------------------------
    @classmethod
    def _uv(cls, x):
        a1, a2, a3, fa1, fa2 = cls.A_PARS
        b1, b2, b3, fb1, fb2 = cls.B_PARS
        g = cls.GAMMA
        d = 1.0 / ((x - 4.6) ** 2 + g)
        fuv = x >= 5.9
        dx = np.where(fuv, x - 5.9, 0.0)
        fa = np.where(fuv, fa1 * dx ** 2 + fa2 * dx ** 3, 0.0)
        fb = np.where(fuv, fb1 * dx ** 2 + fb2 * dx ** 3, 0.0)
        return a1 + a2 * x + a3 * d + fa, b1 + b2 * x + b3 * d + fb

    @classmethod
    def ab(cls, x):
        """Return the a(x), b(x) coefficient arrays over the full range."""
        x = np.asarray(x, dtype=float)
        a, b = ccm_optical_nir(x)
        m = x > OPT_BREAK
        if np.any(m):
            a_uv, b_uv = cls._uv(x[m])
            a[m], b[m] = a_uv, b_uv
        return a, b

    @classmethod
    def axav(cls, x, rv):
        """A(x)/A(V) at wavenumbers ``x`` (micron^-1) for a given R(V)."""
        a, b = cls.ab(x)
        return a + b / float(rv)

    def evaluate(self):
        """A(lambda)/A(V) for the stored x and R(V)."""
        return self.axav(self.x, self.rv)

    __call__ = evaluate

    # ------------------------------------------------------------------
    @classmethod
    def continuity_residual(cls):
        """(a, b) mismatch at the 3.3 micron^-1 breakpoint. Should be ~0."""
        a_uv, b_uv = cls._uv(np.array([OPT_BREAK]))
        return float(a_uv[0] - A_BREAK), float(b_uv[0] - B_BREAK)


# ----------------------------------------------------------------------
# Milky Way reference curve, used only as a null model in the paper's tests.
# ----------------------------------------------------------------------
def ccm89(x, rv):
    """Cardelli, Clayton & Mathis (1989) Milky Way A(x)/A(V)."""
    x = np.asarray(x, dtype=float)
    a, b = ccm_optical_nir(x)
    m = x > OPT_BREAK
    xx = x[m]
    fuv = xx >= 5.9
    dx = np.where(fuv, xx - 5.9, 0.0)
    fa = np.where(fuv, -0.04473 * dx ** 2 - 0.009779 * dx ** 3, 0.0)
    fb = np.where(fuv, 0.2130 * dx ** 2 + 0.1207 * dx ** 3, 0.0)
    a[m] = 1.752 - 0.316 * xx - 0.104 / ((xx - 4.67) ** 2 + 0.341) + fa
    b[m] = -3.090 + 1.825 * xx + 1.206 / ((xx - 4.62) ** 2 + 0.263) + fb
    return a + b / float(rv)
