#!/usr/bin/env python
"""reproduce_paper.py -- regenerate every number and figure in the paper.

Usage
-----
    python reproduce_paper.py                 # everything
    python reproduce_paper.py --stage tables  # one stage only
    python reproduce_paper.py --list          # show the stages

Every stage writes machine-readable output to results/ and figures to figures/.
Nothing is hard-coded: if a required input file is missing the script stops with
a message telling you how to create it.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

REQUIRED = [
    DATA / "smc_average_g24.csv",
    DATA / "lmc_average_g03.csv",
    DATA / "lmc2_average_g03.csv",
    DATA / "smc_published_rv.csv",
    DATA / "lmc_published_rv.csv",
    DATA / "lmc2_published_rv.csv",
    DATA / "smc_sightlines",
    DATA / "lmc_sightlines",
    DATA / "lmc2_sightlines",
]


def preflight():
    missing = [p for p in REQUIRED if not p.exists()]
    if missing:
        print("ERROR: the following inputs are missing:")
        for m in missing:
            print("   ", m.relative_to(ROOT))
        print("\nCreate them with:\n"
              '    python scripts/build_data.py --source "<path to MC Extinction '
              'Codes and Data>"\n')
        raise SystemExit(1)
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)


# ----------------------------------------------------------------- stages
def check_models():
    """Sanity checks on the analytic curves (continuity, LMC b1, Table 3 values)."""
    import check_model
    check_model.main()


def average_curves():
    """Fit each analytic curve to its published average curve (Table 2)."""
    import make_tables
    make_tables.table1()
    make_tables.table2()


def fixed_rv_tests():
    """Primary test: goodness of fit at the published R(V), zero free parameters."""
    import fixed_rv_chi2
    fixed_rv_chi2.run()


def rv_recovery():
    """Where does the R(V) signal come from: full vs UV-only vs optical-only."""
    import rv_recovery as m
    m.run()


def leave_one_out():
    """Out-of-sample test of the SMC calibration."""
    import smc_leave_one_out
    smc_leave_one_out.run()


def residuals():
    """Residual of each sightline from the model, as a function of wavelength."""
    import residuals as m
    m.run()


def tables():
    """Write every table as CSV and LaTeX."""
    import make_tables
    make_tables.run()


STAGES = {
    "models": check_models,
    "averages": average_curves,
    "fixed_rv": fixed_rv_tests,
    "recovery": rv_recovery,
    "loo": leave_one_out,
    "residuals": residuals,
    "tables": tables,
}
ORDER = ["models", "averages", "fixed_rv", "recovery", "loo", "residuals", "tables"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=ORDER, help="run a single stage")
    ap.add_argument("--list", action="store_true", help="list the stages and exit")
    a = ap.parse_args()
    if a.list:
        for s in ORDER:
            print(f"  {s:10s} {STAGES[s].__doc__.strip()}")
        return
    preflight()
    todo = [a.stage] if a.stage else ORDER
    for s in todo:
        print("\n" + "=" * 72)
        print(f"STAGE: {s} -- {STAGES[s].__doc__.strip()}")
        print("=" * 72)
        t0 = time.time()
        STAGES[s]()
        print(f"[{s} finished in {time.time() - t0:.1f} s]")
    print("\nAll done.  Results in results/, figures in figures/.")


if __name__ == "__main__":
    main()
