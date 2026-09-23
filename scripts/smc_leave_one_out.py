"""smc_leave_one_out.py -- out-of-sample test of the SMC prescription.

The SMC coefficients in the paper were derived from all 16 steep sightlines, so
the quoted R(V) recovery is an in-sample statistic.  This script repeats the
whole calibration 16 times, each time holding one sightline out, and then tests
the held-out sightline against the model built without it.

Calibration recipe (the CCM 1989 recipe, reimplemented here so it can be run
programmatically):
  1. bin each training sightline's UV extinction onto a common x grid;
  2. at every grid point, weighted linear regression of A(x)/A(V) against
     1/R(V) across the training sightlines, using the *published* (NIR
     photometric) R(V) values -> a_i, b_i;
  3. fit the analytic a(x), b(x) forms to those a_i, b_i, with a1 and b1 fixed
     by the continuity condition at x = 3.3 micron^-1 and a single shared gamma.

Information leakage: none from the held-out star's spectrum, which enters
neither step 2 nor step 3.  Its published R(V) is used only afterwards, for
comparison.  The optical/NIR segment is the CCM89 Milky Way form, adopted a
priori and never fitted, so it carries no information from any sightline.

Outputs
-------
results/smc_leave_one_out.csv
figures/smc_leave_one_out.png
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from paperlib import RESULTS, FIGURES, sightlines, fit_rv, red_chi2
from _common import A_BREAK, B_BREAK, OPT_BREAK

GRID = np.arange(3.35, 8.65, 0.10)     # bin centres, micron^-1
BIN = 0.10
MIN_PER_BIN = 3


# ----------------------------------------------------------------- model
def _uv(x, pars):
    a2, a3, fa1, fa2, b2, b3, fb1, fb2, g = pars
    a1 = A_BREAK - a2 * OPT_BREAK - a3 / ((OPT_BREAK - 4.6) ** 2 + g)
    b1 = B_BREAK - b2 * OPT_BREAK - b3 / ((OPT_BREAK - 4.6) ** 2 + g)
    d = 1.0 / ((x - 4.6) ** 2 + g)
    dx = np.where(x >= 5.9, x - 5.9, 0.0)
    fa = np.where(x >= 5.9, fa1 * dx ** 2 + fa2 * dx ** 3, 0.0)
    fb = np.where(x >= 5.9, fb1 * dx ** 2 + fb2 * dx ** 3, 0.0)
    return a1 + a2 * x + a3 * d + fa, b1 + b2 * x + b3 * d + fb


def _joint(x2, *pars):
    """a(x) stacked on b(x), for a doubled x vector."""
    n = len(x2) // 2
    a, b = _uv(x2[:n], pars)
    return np.concatenate([a, b])


P0 = np.array([0.03266, 0.007102, 0.05234, -1e-4, 2.1618, 0.007156, 0.04273, 1e-4, 0.0714])
LO = np.array([-1.0, -2.0, -0.5, -0.05, 0.5, -2.0, -0.5, -0.05, 0.005])
HI = np.array([1.0, 2.0, 0.5, 0.05, 4.0, 2.0, 0.5, 0.05, 2.0])


# ----------------------------------------------------------------- data
def load():
    out = []
    for name, d, rv, rverr in sightlines("SMC"):
        m = d.x.values > OPT_BREAK
        out.append(dict(name=name, x=d.x.values[m], y=d.axav.values[m],
                        e=d.err.values[m], rv=rv, rverr=rverr))
    return out


def binned(s):
    """Weighted mean of one sightline on the common grid."""
    y = np.full(GRID.size, np.nan)
    e = np.full(GRID.size, np.nan)
    for i, xc in enumerate(GRID):
        m = np.abs(s["x"] - xc) < BIN / 2
        if m.sum() >= MIN_PER_BIN:
            w = 1.0 / s["e"][m] ** 2
            y[i] = np.sum(w * s["y"][m]) / np.sum(w)
            e[i] = np.sqrt(1.0 / np.sum(w))
    return y, e


def calibrate(train):
    """Steps 2 and 3: return fitted parameter vector."""
    B = [binned(s) for s in train]
    invrv = np.array([1.0 / s["rv"] for s in train])
    a_i, b_i, ok = [], [], []
    for i in range(GRID.size):
        yy = np.array([b[0][i] for b in B])
        ee = np.array([b[1][i] for b in B])
        good = np.isfinite(yy) & np.isfinite(ee)
        if good.sum() < 5:
            a_i.append(np.nan); b_i.append(np.nan); ok.append(False); continue
        w = 1.0 / ee[good] ** 2
        X = np.vstack([np.ones(good.sum()), invrv[good]]).T
        W = np.diag(w)
        beta = np.linalg.solve(X.T @ W @ X, X.T @ W @ yy[good])
        a_i.append(beta[0]); b_i.append(beta[1]); ok.append(True)
    a_i, b_i, ok = np.array(a_i), np.array(b_i), np.array(ok)
    xg = GRID[ok]
    target = np.concatenate([a_i[ok], b_i[ok]])
    pars, _ = curve_fit(_joint, np.concatenate([xg, xg]), target,
                        p0=P0, bounds=(LO, HI), maxfev=200000)
    return pars, xg, a_i[ok], b_i[ok]


def model_fn(pars):
    def f(x, rv):
        a, b = _uv(np.asarray(x, dtype=float), pars)
        return a + b / rv
    return f


# ----------------------------------------------------------------- run
def run():
    data = load()
    rows = []
    pars_all, _, _, _ = calibrate(data)          # in-sample reference
    for k, held in enumerate(data):
        train = [s for j, s in enumerate(data) if j != k]
        pars, *_ = calibrate(train)
        f_loo, f_in = model_fn(pars), model_fn(pars_all)
        rv_loo, err_loo, hit = fit_rv(f_loo, held["x"], held["y"], held["e"])
        rv_in, _, _ = fit_rv(f_in, held["x"], held["y"], held["e"])
        rows.append(dict(
            sightline=held["name"], rv_pub=held["rv"], rv_pub_err=held["rverr"],
            rv_loo=rv_loo, rv_loo_err=err_loo, rv_insample=rv_in,
            d_loo=100 * (rv_loo - held["rv"]) / held["rv"],
            d_insample=100 * (rv_in - held["rv"]) / held["rv"],
            chi2_loo_fixed=red_chi2(held["y"], f_loo(held["x"], held["rv"]), held["e"], 0),
            chi2_loo_fitted=red_chi2(held["y"], f_loo(held["x"], rv_loo), held["e"], 1),
            hit_bound=hit))
        print(f"  held out {held['name']:12s} -> R(V)_LOO = {rv_loo:5.2f} "
              f"(published {held['rv']:5.2f}, {rows[-1]['d_loo']:+6.1f}%)")

    t = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    t.to_csv(RESULTS / "smc_leave_one_out.csv", index=False)
    print("\n" + t.to_string(index=False, float_format=lambda v: f"{v:8.3f}"))
    print(f"\nLOO   median |dR(V)| = {t.d_loo.abs().median():.2f}%   "
          f"within 10%: {(t.d_loo.abs() < 10).sum()}/{len(t)}")
    print(f"in-sample median |dR(V)| = {t.d_insample.abs().median():.2f}%   "
          f"within 10%: {(t.d_insample.abs() < 10).sum()}/{len(t)}")
    print(f"LOO median reduced chi2 at published R(V) = {t.chi2_loo_fixed.median():.3f}")

    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    lim = [1.7, 6.2]
    ax.plot(lim, lim, "k--", lw=0.8, alpha=0.6)
    ax.errorbar(t.rv_pub, t.rv_loo, xerr=t.rv_pub_err, fmt="o", color="tab:red", ms=5, lw=1,
                label="leave-one-out")
    ax.plot(t.rv_pub, t.rv_insample, "^", mfc="none", mec="k", ms=6, label="in-sample")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("published R(V)  (Gordon et al. 2024)")
    ax.set_ylabel("R(V) recovered from the UV")
    ax.legend(frameon=False, fontsize=9)
    ax.tick_params(direction="in", which="both")
    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / "smc_leave_one_out.png", dpi=200)
    print(f"wrote {FIGURES / 'smc_leave_one_out.png'}")
    return t


if __name__ == "__main__":
    run()
