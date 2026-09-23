"""make_tables.py -- generate every table in the paper from the code and data.

No number in the manuscript should be typed by hand.  This script writes both a
machine-readable CSV and a LaTeX fragment for each table, into results/.
"""
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from paperlib import (FAMILIES, RESULTS, model_class, average_curve, red_chi2)
from _common import ccm89


def table1():
    """Best-fit coefficients of the UV segment."""
    rows = []
    for fam in FAMILIES:
        c = model_class(fam)
        a1, a2, a3, fa1, fa2 = c.A_PARS
        b1, b2, b3, fb1, fb2 = c.B_PARS
        rows.append(dict(extinction=fam, a1=a1, a2=a2, a3=a3, fa1=fa1, fa2=fa2,
                         b1=b1, b2=b2, b3=b3, fb1=fb1, fb2=fb2, gamma=c.GAMMA,
                         rv_default=c.RV_DEFAULT,
                         rv_min=c.RV_RANGE[0], rv_max=c.RV_RANGE[1]))
    t = pd.DataFrame(rows)
    t.to_csv(RESULTS / "table1_coefficients.csv", index=False)
    with open(RESULTS / "table1_coefficients.tex", "w") as f:
        f.write(t.drop(columns=["rv_default", "rv_min", "rv_max"]).to_latex(
            index=False, float_format="%.5g",
            caption=("Coefficients of the ultraviolet segment ($x>3.3\\,\\mu$m$^{-1}$) of "
                     "equations~(2)--(5). $a_1$ and $b_1$ are fixed by continuity at "
                     "$x=3.3\\,\\mu$m$^{-1}$ and are therefore not free parameters; "
                     "$\\gamma$ is shared between $a(x)$ and $b(x)$; for the LMC and LMC2 "
                     "the far-UV terms satisfy $F_a=F_b$."),
            label="tab:coeff"))
    return t


def table2():
    """Goodness of fit to the published average curves."""
    rows = []
    for fam in FAMILIES:
        c = model_class(fam)
        d = average_curve(fam)
        x, y, e = d.x.values, d.axav.values, d.err.values
        uv = x > 3.3
        p, cov = curve_fit(lambda xx, r: c.axav(xx, r), x, y, sigma=e,
                           absolute_sigma=True, p0=[3.0], bounds=(1, 6))
        rows.append(dict(
            extinction=fam, rv_nominal=c.RV_DEFAULT,
            rv_bestfit=p[0], rv_bestfit_err=float(np.sqrt(cov[0, 0])),
            redchi2_all=red_chi2(y, c.axav(x, c.RV_DEFAULT), e, 1),
            redchi2_uv=red_chi2(y[uv], c.axav(x[uv], c.RV_DEFAULT), e[uv], 1),
            redchi2_all_nolargest=_drop_worst(y, c.axav(x, c.RV_DEFAULT), e),
            worst_point_x=float(x[np.argmax(((y - c.axav(x, c.RV_DEFAULT)) / e) ** 2)]),
            worst_point_chi2frac=_worst_frac(y, c.axav(x, c.RV_DEFAULT), e)))
    t = pd.DataFrame(rows)
    t.to_csv(RESULTS / "table2_average_fits.csv", index=False)
    with open(RESULTS / "table2_average_fits.tex", "w") as f:
        f.write(t.to_latex(index=False, float_format="%.3f",
                           caption=("Reproduction of the published average curves. "
                                    "The final columns identify the single data point that "
                                    "dominates the $\\chi^2$ and the fraction of the total "
                                    "$\\chi^2$ it contributes."),
                           label="tab:avg"))
    return t


def _resid2(y, m, e):
    return ((y - m) / e) ** 2


def _drop_worst(y, m, e):
    r = _resid2(y, m, e)
    k = int(np.argmax(r))
    keep = np.ones_like(r, dtype=bool); keep[k] = False
    return float(np.sum(r[keep]) / (keep.sum() - 1))


def _worst_frac(y, m, e):
    r = _resid2(y, m, e)
    return float(r.max() / r.sum())


def table3():
    """Merged per-sightline table: published R(V), fitted R(V), MW null, chi2."""
    fixed = pd.read_csv(RESULTS / "fixed_rv_chi2.csv")
    rec = pd.read_csv(RESULTS / "rv_recovery.csv")
    t = fixed.merge(rec[["family", "sightline", "rv_uv", "rv_opt", "rv_uv_mw",
                         "d_uv", "d_opt"]], on=["family", "sightline"])
    cols = ["family", "sightline", "rv_pub", "rv_pub_err", "rv_uv", "rv_fit_err",
            "d_uv", "rv_opt", "d_opt", "rv_uv_mw", "chi2_fixed", "chi2_fitted",
            "rms_fixed", "rms_fitted"]
    t = t[[c for c in cols if c in t]]
    t.to_csv(RESULTS / "table3_sightlines.csv", index=False)
    with open(RESULTS / "table3_sightlines.tex", "w") as f:
        f.write(t.to_latex(index=False, float_format="%.3f", na_rep="\\ldots",
                           caption=("R(V) recovered for each sightline, compared with the "
                                    "published photometric value; the Milky Way CCM89 null "
                                    "model; and the goodness of fit with R(V) fixed to the "
                                    "published value (no free parameters)."),
                           label="tab:sightlines"), )
    return t


def run():
    RESULTS.mkdir(exist_ok=True)
    t1, t2 = table1(), table2()
    print("TABLE 1\n", t1.to_string(index=False), "\n")
    print("TABLE 2\n", t2.to_string(index=False, float_format=lambda v: f"{v:8.3f}"), "\n")
    try:
        t3 = table3()
        print(f"TABLE 3 written ({len(t3)} rows)")
    except FileNotFoundError:
        print("TABLE 3 skipped: run fixed_rv_chi2.py and rv_recovery.py first")
    print(f"\nAll tables written to {RESULTS}")


if __name__ == "__main__":
    run()
