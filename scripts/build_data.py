"""
build_data.py -- one-off converter: turns the original analysis tree (Excel files)
into flat, versionable CSV files under data/.

Run once:
    python scripts/build_data.py --source "/path/to/MC Extinction Codes and Data"

After that, nothing in the repository depends on Excel or on absolute paths.
"""
import argparse
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

SMC_STEEP = [  # 16 G24 "steep / weak-or-absent bump" sightlines
    "2dfs0413", "2dfs0626", "2dfs0662", "2dfs0699", "2dfs3014", "2dfs3171",
    "azv18", "azv214", "azv218", "azv23", "azv398", "bbb-smc280",
    "mr12-star09", "smc5-000398", "smc5-003739", "smc5-079264",
]
LMC_STARS = ["Sk 66 19","Sk 66 88","Sk 67 2","Sk 68 23","Sk 68 26",
             "Sk 68 129","Sk 69 108","Sk 69 210","Sk 69 213"]
# NOTE: Gordon et al. (2003) call this star Sk -68 155, not Sk -68 115.
LMC2_STARS = ["Sk 68 140","Sk 68 155","Sk 69 228","Sk 69 265","Sk 69 270",
              "Sk 69 279","Sk 69 280","Sk 70 116"]

LMC_AVG = dict(
 x=[0.455,0.606,0.8,1.818,2.273,2.703,3.375,3.625,3.875,4.125,4.375,4.625,4.875,
    5.125,5.375,5.625,5.875,6.125,6.375,6.625,6.875,7.125,7.375,7.625,7.875,8.125,8.375],
 axav=[0.030,0.186,0.257,1.000,1.293,1.518,1.786,1.969,2.149,2.391,2.771,2.967,2.846,
    2.646,2.565,2.566,2.589,2.607,2.668,2.787,2.874,2.983,3.118,3.231,3.374,3.366,3.467],
 err=[0.003,0.020,0.013,0.048,0.113,0.046,0.127,0.123,0.095,0.093,0.093,0.095,0.099,
    0.102,0.104,0.105,0.107,0.112,0.115,0.119,0.124,0.131,0.135,0.142,0.148,0.153,0.160])


def main(source):
    src = Path(source)
    for sub in ["", "smc_sightlines", "lmc_sightlines", "lmc2_sightlines"]:
        (DATA / sub).mkdir(parents=True, exist_ok=True)
    RN = src / "Research Note"

    d = pd.read_excel(RN / "SMC codes" / "New SMC averages.xlsx", sheet_name="Sheet1")
    d.rename(columns={"ext": "axav"}).to_csv(DATA / "smc_average_g24.csv", index=False)

    pd.DataFrame(LMC_AVG).to_csv(DATA / "lmc_average_g03.csv", index=False)

    for k in SMC_STEEP:
        t = pd.read_excel(RN / "SMC QC" / "Steep" / f"{k}_ext_forecor_FM90.xlsx")
        pd.DataFrame({"x": 1.0 / t["wl"].to_numpy(),
                      "axav": t["A(x)"].to_numpy(),
                      "err": t["Err"].to_numpy()}).sort_values("x").to_csv(
            DATA / "smc_sightlines" / f"{k}.csv", index=False)

    p = pd.read_excel(RN / "Test for CCM Criteria" / "SMC" / "smc_fit_params.xlsx",
                      sheet_name="Sheet1")
    p["sightline"] = p["Sightline"].str.replace("_ext_forecor_FM90", "", regex=False)
    p[["sightline", "RV", "RV_err"]].rename(
        columns={"RV": "rv_pub", "RV_err": "rv_pub_err"}).to_csv(
        DATA / "smc_published_rv.csv", index=False)

    for stars, sub, folder in [
            (LMC_STARS, "lmc_sightlines", RN / "LMC Fitting Codes" / "LMC plot digitizer"),
            (LMC2_STARS, "lmc2_sightlines", RN / "LMC2 Fitting Codes" / "LMC2 plot digitizer")]:
        for s in stars:
            t = pd.read_excel(folder / f"{s}.xlsx")
            pd.DataFrame({"x": t["x"].to_numpy(), "axav": t["A(x)"].to_numpy()}
                         ).sort_values("x").to_csv(
                DATA / sub / f"{s.replace(' ', '_')}.csv", index=False)

    xl = pd.ExcelFile(RN / "Test for CCM Criteria" / "LMC" / "LMC ext params.xlsx")
    for sheet, name in [("LMC", "lmc_published_rv.csv"), ("LMC2", "lmc2_published_rv.csv")]:
        t = xl.parse(sheet)
        t[["Name", "RV", "RV_err"]].rename(
            columns={"Name": "sightline", "RV": "rv_pub", "RV_err": "rv_pub_err"}
            ).to_csv(DATA / name, index=False)

    print(f"Wrote CSV data under {DATA}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    main(ap.parse_args().source)
