# xMC_Extinction

Analytic, R(V)-dependent extinction curves for **SMC**, **LMC** and **LMC2 (LMC
supershell)** dust, from the far-ultraviolet to the near-infrared, in the
functional form of Cardelli, Clayton & Mathis (1989, CCM).

The standard Magellanic Cloud extinction curves (Gordon et al. 2003, 2024) are
parameterised with FM90, which is defined only in the ultraviolet, so they exist
as *fixed templates*: they can be adopted, but not fitted to a new sightline.
This package provides continuous curves over 1200–20000 Å in which R(V) acts as
a single shape parameter, so that a Magellanic-type curve can be fitted to an
individual sightline.

Accompanying paper: Sanil Kumar & Khaire (2026), *The Open Journal of
Astrophysics* (link and DOI to be added on acceptance).

---

## Installation

No installation is required. Clone the repository and work inside it:

```bash
git clone https://github.com/sandhyasanil/xMC_Extinction.git
cd xMC_Extinction
pip install -r requirements.txt
```

Dependencies: `numpy`, `scipy`, `matplotlib`, `pandas`.
`astropy` is optional (only if you want to pass `Quantity` inputs);
`openpyxl` is needed only by `scripts/build_data.py`.

---

## Quick start

```python
import numpy as np
from SMC import SMCExtinction

x = np.linspace(0.3, 8.7, 500)              # wavenumber, micron^-1
curve = SMCExtinction(x, rv=3.1)            # A(lambda)/A(V)
axav  = curve.evaluate()
```

Bare arrays are interpreted as inverse microns. An astropy `Quantity` with
length or wavenumber units is also accepted:

```python
import astropy.units as u
SMCExtinction(np.linspace(1200, 20000, 500) * u.AA, rv=3.1).evaluate()
```

A class method is available if you do not want to build an object:

```python
SMCExtinction.axav(x, rv=3.1)
```

`python examples/sample.py` plots all three families.

---

## The three curves

| class | file | default R(V) | R(V) range calibrated | notes |
|---|---|---|---|---|
| `SMCExtinction`  | `SMC.py`  | 3.02 | 2.0 – 5.2 | 16 steep/bumpless sightlines, Gordon et al. (2024) |
| `LMCExtinction`  | `LMC.py`  | 3.41 | 2.8 – 4.0 | 9 sightlines, Gordon et al. (2003) |
| `LMC2Extinction` | `LMC2.py` | 2.76 | 1.7 – 3.5 | 8 supershell sightlines, Gordon et al. (2003) |

The default R(V) of each class is the value at which it reproduces the
corresponding *published average curve*, not the nominal R(V) of some other
sample. For the SMC that value, 3.02, is also the R(V) quoted by Gordon et al.
(2024) for their SMC Average curve.

**Wavelength range.** 0.3 ≤ x ≤ 8.7 μm⁻¹ (≈ 1150–33000 Å). The prescription is
calibrated against data over roughly 3.3–8.6 μm⁻¹ in the ultraviolet; beyond
x ≈ 8 μm⁻¹ the sightline-to-sightline scatter about the relation becomes
comparable to the extinction itself, and results there should be treated as
extrapolation.

**R(V) range.** Use the ranges in the table. The relation is linear in 1/R(V)
by construction and has no physical content outside the range spanned by the
calibrating sample.

**What is and is not refitted.** Only the ultraviolet segment (x > 3.3 μm⁻¹)
differs between families. The near-infrared power law and the optical polynomial
are the CCM89 Milky Way forms, adopted unchanged; they are shared by all three
classes and live in `_common.py`.

---

## Reproducing the paper

```bash
python scripts/build_data.py --source "/path/to/MC Extinction Codes and Data"
python reproduce_paper.py
```

The first command converts the original Excel analysis tree into flat CSV files
under `data/` (run once; the CSVs are committed, so most users can skip it).
The second regenerates every table and figure:

| stage | what it does | output |
|---|---|---|
| `models` | continuity and self-consistency checks | console (all must PASS) |
| `averages` | fits to the published average curves | `results/table1_*`, `table2_*` |
| `fixed_rv` | goodness of fit at the published R(V), **zero free parameters** | `results/fixed_rv_chi2*.csv`, `figures/fixed_rv_chi2.png` |
| `recovery` | R(V) from the full curve vs UV-only vs optical-only, plus the Milky Way null model | `results/rv_recovery*.csv`, `figures/rv_recovery.png` |
| `loo` | leave-one-out recalibration of the SMC | `results/smc_leave_one_out.csv`, `figures/smc_leave_one_out.png` |
| `residuals` | residual versus wavelength | `results/residual_profiles.csv`, `figures/residuals_vs_wavelength.png` |
| `tables` | every table as CSV and LaTeX | `results/table*.csv`, `results/table*.tex` |

Run a single stage with `python reproduce_paper.py --stage fixed_rv`.

---

## Data provenance

| file | origin |
|---|---|
| `data/smc_average_g24.csv` | SMC Average curve, Gordon et al. (2024) |
| `data/smc_sightlines/*.csv` | Gordon et al. (2024) extinction curves (Zenodo 11187458), converted to A(x)/A(V) and quality-cut at 15 per cent fractional uncertainty. The cut removes the J, H and K photometry, whose fractional errors are large |
| `data/lmc_average_g03.csv`, `data/lmc2_average_g03.csv` | Gordon et al. (2003), Tables 4 and 5 |
| `data/lmc_sightlines/*.csv`, `data/lmc2_sightlines/*.csv` | **digitised from the figures of Gordon et al. (2003)**; no uncertainties are available, so these fits are unweighted |
| `data/*_published_rv.csv` | R(V) from Gordon et al. (2003) (χ² fit of the RIJHK extinction to Rieke & Lebofsky 1985) and Gordon et al. (2024) |

---

## Limitations

- The SMC calibration uses 16 sightlines and is **in-sample**; `reproduce_paper.py --stage loo` quantifies how much of the performance survives leave-one-out.
- The LMC and LMC2 coefficients were obtained by fitting the average curve and then adjusting the coefficients so that the fitted sightline R(V) values matched the published ones. Their R(V) comparisons are therefore calibration residuals, **not independent validation**.
- The LMC sample spans only R(V) ≈ 3.15–3.96, which is too narrow to constrain an R(V) dependence.
- Fitted LMC2 sightline R(V) values are systematically low with respect to the Gordon et al. (2003) photometric scale.
- Two SMC sightlines (AzV 214, MR12 Star 09) are not described by the relation; MR12 Star 09 has a published R(V) uncertainty of ±1.9 and should not be used as a test case.

---

## Citation

Please cite the paper and the software (see `CITATION.cff`).

## License

MIT (see `LICENSE`).
