"""fixed_rv_chi2.py -- the paper's primary, non-circular goodness-of-fit test.

For every sightline we evaluate the analytic curve at the *independently
published* R(V) (from NIR photometry), with NO fitted parameters at all, and
measure how well it reproduces the observed extinction.  The same is done for
the Milky Way CCM89 relation as a null model, and for the one-parameter fit as
a reference.

Outputs
-------
results/fixed_rv_chi2.csv     one row per sightline
results/fixed_rv_chi2_summary.csv
figures/fixed_rv_chi2.png
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paperlib import (FAMILIES, RESULTS, FIGURES, model_class, sightlines,
                      fit_rv, red_chi2, rms)
from _common import ccm89

UV_MIN = 3.3          # the UV segment is the only part refitted in this work


def run():
    rows = []
    for fam in FAMILIES:
        cls = model_class(fam)
        for name, d, rv_pub, rv_pub_err in sightlines(fam):
            m = d.x.values > UV_MIN
            x, y, e = d.x.values[m], d.axav.values[m], d.err.values[m]
            has_err = np.all(np.isfinite(e)) and np.all(e > 0)

            rv_fit, rv_fit_err, hit = fit_rv(lambda xx, r: cls.axav(xx, r), x, y,
                                             e if has_err else None)
            rv_mw, _, _ = fit_rv(lambda xx, r: ccm89(xx, r), x, y, e if has_err else None)

            row = dict(family=fam, sightline=name, n_uv=int(m.sum()),
                       rv_pub=rv_pub, rv_pub_err=rv_pub_err,
                       rv_fit=rv_fit, rv_fit_err=rv_fit_err, fit_hit_bound=hit,
                       rv_fit_mw=rv_mw, has_uncertainties=has_err)
            if has_err:
                row.update(
                    chi2_fixed=red_chi2(y, cls.axav(x, rv_pub), e, 0),
                    chi2_fitted=red_chi2(y, cls.axav(x, rv_fit), e, 1),
                    chi2_mw_fixed=red_chi2(y, ccm89(x, rv_pub), e, 0),
                    chi2_mw_fitted=red_chi2(y, ccm89(x, rv_mw), e, 1))
            else:
                row.update(
                    rms_fixed=rms(y, cls.axav(x, rv_pub)),
                    rms_fitted=rms(y, cls.axav(x, rv_fit)),
                    rms_mw_fixed=rms(y, ccm89(x, rv_pub)),
                    rms_mw_fitted=rms(y, ccm89(x, rv_mw)))
            rows.append(row)

    t = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    t.to_csv(RESULTS / "fixed_rv_chi2.csv", index=False)

    summ = []
    for fam, g in t.groupby("family", sort=False):
        s = dict(family=fam, n=len(g))
        for c in ["chi2_fixed", "chi2_fitted", "chi2_mw_fixed", "chi2_mw_fitted",
                  "rms_fixed", "rms_fitted", "rms_mw_fixed", "rms_mw_fitted"]:
            if c in g and g[c].notna().any():
                s["median_" + c] = float(g[c].median())
        summ.append(s)
    summ = pd.DataFrame(summ)
    summ.to_csv(RESULTS / "fixed_rv_chi2_summary.csv", index=False)
    print(t.to_string(index=False, float_format=lambda v: f"{v:8.3f}"))
    print("\nSUMMARY (medians)\n" + summ.to_string(index=False, float_format=lambda v: f"{v:8.3f}"))

    _figure(t)
    return t, summ


def _figure(t):
    smc = t[t.family == "SMC"].sort_values("rv_pub")
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    idx = np.arange(len(smc))
    ax.plot(idx, smc.chi2_fixed, "o-", color="tab:red", label="This work, R(V) fixed to published")
    ax.plot(idx, smc.chi2_mw_fixed, "s--", color="tab:grey",
            label="CCM89 Milky Way, R(V) fixed to published")
    ax.axhline(1.0, color="k", lw=0.8, ls=":")
    ax.set_yscale("log")
    ax.set_xticks(idx)
    ax.set_xticklabels(smc.sightline, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel(r"reduced $\chi^2$  (UV, no free parameters)")
    ax.legend(frameon=False, fontsize=8)
    ax.tick_params(direction="in", which="both")
    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / "fixed_rv_chi2.png", dpi=200)
    print(f"\nwrote {FIGURES / 'fixed_rv_chi2.png'}")


if __name__ == "__main__":
    run()
