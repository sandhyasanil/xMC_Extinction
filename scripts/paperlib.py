"""Small shared helpers for the analysis scripts."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
sys.path.insert(0, str(ROOT))

FAMILIES = ("SMC", "LMC", "LMC2")


def require(path):
    p = Path(path)
    if not p.exists():
        raise SystemExit(
            f"\nERROR: required file not found:\n    {p}\n"
            "Run  `python scripts/build_data.py --source \"<path to MC Extinction "
            "Codes and Data>\"`  first to create the data/ directory.\n")
    return p


def model_class(family):
    from SMC import SMCExtinction
    from LMC import LMCExtinction
    from LMC2 import LMC2Extinction
    return {"SMC": SMCExtinction, "LMC": LMCExtinction, "LMC2": LMC2Extinction}[family]


def sightlines(family):
    """Yield (name, DataFrame with x, axav and (SMC only) err, published R(V), err)."""
    sub = {"SMC": "smc_sightlines", "LMC": "lmc_sightlines", "LMC2": "lmc2_sightlines"}[family]
    pub = pd.read_csv(require(DATA / {"SMC": "smc_published_rv.csv",
                                      "LMC": "lmc_published_rv.csv",
                                      "LMC2": "lmc2_published_rv.csv"}[family]))
    pub["key"] = pub["sightline"].astype(str).str.strip()
    for _, row in pub.iterrows():
        fname = DATA / sub / (row["key"].replace(" ", "_") + ".csv")
        if not fname.exists():
            continue
        d = pd.read_csv(fname)
        if "err" not in d:
            d["err"] = np.nan
        yield row["key"], d, float(row["rv_pub"]), float(row["rv_pub_err"])


def average_curve(family):
    f = {"SMC": "smc_average_g24.csv", "LMC": "lmc_average_g03.csv",
         "LMC2": "lmc2_average_g03.csv"}[family]
    return pd.read_csv(require(DATA / f))


def fit_rv(model_fn, x, y, err=None, bounds=(1.0, 8.0), p0=3.0):
    """Single-parameter R(V) fit.  Returns (rv, rv_err, hit_bound)."""
    from scipy.optimize import curve_fit
    kw = {}
    if err is not None and np.all(np.isfinite(err)) and np.all(err > 0):
        kw = dict(sigma=err, absolute_sigma=True)
    p, c = curve_fit(model_fn, x, y, p0=[p0], bounds=bounds, **kw)
    hit = bool(np.isclose(p[0], bounds[0], atol=1e-3) or np.isclose(p[0], bounds[1], atol=1e-3))
    return float(p[0]), float(np.sqrt(c[0, 0])), hit


def red_chi2(y, model, err, n_par):
    return float(np.sum(((y - model) / err) ** 2) / (len(y) - n_par))


def rms(y, model):
    return float(np.sqrt(np.mean((y - model) ** 2)))
