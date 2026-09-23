"""check_model.py -- diagnostics for the analytic curves.

Run:  python scripts/check_model.py
Checks (all must pass before the paper is submitted):
  1. continuity of a(x) and b(x) at the 3.3 micron^-1 breakpoint
  2. absence of a visible jump in A(x)/A(V) across the breakpoint
  3. the LMC coefficient b1 is the continuity-preserving value
  4. reproduction of the paper's reported LMC sightline R(V) values
"""
import numpy as np
from paperlib import DATA, ROOT, model_class, sightlines, fit_rv, require   # noqa: E402

TOL = 1e-3
FAIL = []


def check(label, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{'  ' + detail if detail else ''}")
    if not ok:
        FAIL.append(label)


def main():
    from _common import A_BREAK, B_BREAK, OPT_BREAK
    print(f"CCM breakpoint values: a(3.3) = {A_BREAK:.5f}, b(3.3) = {B_BREAK:.5f}\n")

    print("1/2. Continuity and smoothness at x = 3.3 micron^-1")
    for fam in ("SMC", "LMC", "LMC2"):
        cls = model_class(fam)
        da, db = cls.continuity_residual()
        check(f"{fam}: |da| < {TOL}, |db| < {TOL}", abs(da) < TOL and abs(db) < TOL,
              f"(da = {da:+.2e}, db = {db:+.2e})")
        lo = cls.axav(np.array([OPT_BREAK - 1e-6]), 3.1)[0]
        hi = cls.axav(np.array([OPT_BREAK + 1e-6]), 3.1)[0]
        check(f"{fam}: |jump in A/A(V)| < {TOL} at R(V)=3.1", abs(hi - lo) < TOL,
              f"(jump = {hi - lo:+.2e})")

    print("\n3. LMC b1 is the continuity-preserving value")
    from LMC import LMCExtinction
    b1, b2, b3 = LMCExtinction.B_PARS[:3]
    b1_req = B_BREAK - b2 * OPT_BREAK - b3 / ((OPT_BREAK - 4.6) ** 2 + LMCExtinction.GAMMA)
    check("LMC b1 == required value", abs(b1 - b1_req) < 1e-3,
          f"(table value {b1:.4f}, required {b1_req:.4f}; the superseded "
          f"value -3.5340 gives a residual of {abs(-3.534 - b1_req):.4f})")

    print("\n4. Reproduction of the LMC sightline R(V) values reported in the paper")
    paper = {"Sk 66 19": 3.65, "Sk 66 88": 3.63, "Sk 67 2": 2.82, "Sk 68 23": 3.70,
             "Sk 68 26": 3.32, "Sk 68 129": 3.09, "Sk 69 108": 3.24,
             "Sk 69 210": 3.26, "Sk 69 213": 3.16}
    worst = 0.0
    for name, d, _rvp, _e in sightlines("LMC"):
        if name not in paper:
            continue
        rv, _, _ = fit_rv(lambda xx, r: LMCExtinction.axav(xx, r), d.x.values, d.axav.values)
        worst = max(worst, abs(rv - paper[name]))
    check("max |fit - published table| < 0.02", worst < 0.02, f"(worst = {worst:.4f})")

    print()
    if FAIL:
        raise SystemExit(f"{len(FAIL)} check(s) FAILED: {FAIL}")
    print("All model checks passed.")


if __name__ == "__main__":
    main()
