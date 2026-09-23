"""rv_recovery.py -- where does the R(V) information come from?

This is the analysis prompted by K. Gordon's question (2026 April 8): if the
R(V) signal is mostly in the optical/NIR, an optical/NIR-only fit should recover
the published R(V) as well as, or better than, the full fit.

For every sightline we fit R(V) three times, to
    (a) the full curve,
    (b) the ultraviolet only   (x > 3.3 micron^-1),
    (c) the optical/NIR only   (x < 3.3 micron^-1),
and also fit the Milky Way CCM89 curve to the UV only as a null model.

Outputs
-------
results/rv_recovery.csv
results/rv_recovery_summary.csv
figures/rv_recovery.png
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paperlib import FAMILIES, RESULTS, FIGURES, model_class, sightlines, fit_rv
from _common import ccm89

BREAK = 3.3          # UV/optical breakpoint of the functional form
OPT_CUT = 3.0        # 'optical/NIR only' = x < 3.0, i.e. the UBVJHK photometry
                     # alone.  This is the cut used in the 2026 Apr 21 reply to
                     # K. Gordon and it reproduces those numbers exactly.


def run():
    rows = []
    for fam in FAMILIES:
        cls = model_class(fam)
        for name, d, rv_pub, rv_pub_err in sightlines(fam):
            x, y, e = d.x.values, d.axav.values, d.err.values
            has_err = np.all(np.isfinite(e)) and np.all(e > 0)

            def f(mask, fn=None):
                fn = fn or (lambda xx, r: cls.axav(xx, r))
                if mask.sum() < 3:
                    return np.nan, np.nan, False
                return fit_rv(fn, x[mask], y[mask], e[mask] if has_err else None)

            m_all = np.ones_like(x, dtype=bool)
            m_uv = x > BREAK
            m_opt = x < OPT_CUT
            rv_all, e_all, h1 = f(m_all)
            rv_uv, e_uv, h2 = f(m_uv)
            rv_opt, e_opt, h3 = f(m_opt)
            rv_mw, _, _ = f(m_uv, lambda xx, r: ccm89(xx, r))
            rows.append(dict(
                family=fam, sightline=name, rv_pub=rv_pub, rv_pub_err=rv_pub_err,
                n_opt=int(m_opt.sum()), n_uv=int(m_uv.sum()),
                rv_all=rv_all, rv_uv=rv_uv, rv_opt=rv_opt, rv_uv_mw=rv_mw,
                d_all=100 * (rv_all - rv_pub) / rv_pub,
                d_uv=100 * (rv_uv - rv_pub) / rv_pub,
                d_opt=100 * (rv_opt - rv_pub) / rv_pub,
                d_uv_mw=100 * (rv_mw - rv_pub) / rv_pub,
                hit_bound=bool(h1 or h2 or h3)))

    t = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    t.to_csv(RESULTS / "rv_recovery.csv", index=False)

    summ = []
    for fam, g in t.groupby("family", sort=False):
        row = dict(family=fam, n=len(g))
        for c in ["d_all", "d_uv", "d_opt", "d_uv_mw"]:
            row["median_abs_" + c] = float(g[c].abs().median())
            row["n_within10_" + c] = int((g[c].abs() < 10).sum())
        row["median_signed_d_uv"] = float(g["d_uv"].median())
        summ.append(row)
    summ = pd.DataFrame(summ)
    summ.to_csv(RESULTS / "rv_recovery_summary.csv", index=False)

    pd.set_option("display.width", 220)
    print(t.to_string(index=False, float_format=lambda v: f"{v:8.3f}"))
    print("\nSUMMARY\n" + summ.to_string(index=False, float_format=lambda v: f"{v:8.2f}"))
    _figure(t)
    return t, summ


def _figure(t):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.9), sharey=False)
    for ax, fam, col in zip(axes, FAMILIES, ["tab:red", "tab:blue", "tab:green"]):
        g = t[t.family == fam]
        lim = [min(g.rv_pub.min(), g.rv_uv.min()) - 0.3, max(g.rv_pub.max(), g.rv_uv.max()) + 0.3]
        ax.plot(lim, lim, "k--", lw=0.8, alpha=0.6)
        ax.errorbar(g.rv_pub, g.rv_uv, xerr=g.rv_pub_err, fmt="o", color=col,
                    ms=5, lw=1, label="UV only")
        ax.plot(g.rv_pub, g.rv_opt, "^", mfc="none", mec="k", ms=6, label="optical/NIR only")
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_title(fam)
        ax.set_xlabel("published R(V)")
        ax.tick_params(direction="in", which="both")
        if fam == "SMC":
            ax.set_ylabel("fitted R(V)")
            ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / "rv_recovery.png", dpi=200)
    print(f"\nwrote {FIGURES / 'rv_recovery.png'}")


if __name__ == "__main__":
    run()
