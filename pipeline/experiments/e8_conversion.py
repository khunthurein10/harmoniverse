"""
E8 (step 8): which lookup conversion predicts held-out years best?

Target: S-NPP hotspots without low confidence (A2). Input: Aqua hotspots, all pixels (E6: V1).
Choosing years 2015 to 2019, leave one year out (as E2/E6). All ratios are ratio of sums (GFED5NRT Sect. 3.1).

  B       one ratio per day/night x region x calendar month                         (E2 winner)
  B-3m    same, summed over a centred 3-month window (Dec-Jan-Feb for Jan, ...)     (MRBA60 Methods)
  B-fb    B, but a group with < MIN_AQUA training detections falls back to B-3m, then day/night x region,
          then day/night                                                            (MRBA60 thin-group rule, adapted)
  B-3m-fb B-3m with the same fallback
  C       each box's own ratio (day/night, 3-month window) if it has >= MIN_BOX training Aqua detections,
          else B-3m-fb                                                              (Volpe et al. 2019 per-pixel ratio, adapted)
Thresholds MIN_AQUA and MIN_BOX are our choice.

Usage: python pipeline/experiments/e8_conversion.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harmonizer.config import load_settings  # noqa: E402
from harmonizer.convert import window3  # noqa: E402
from e6_viewing_angle import load, YEARS  # noqa: E402

MIN_AQUA, MIN_BOX = 200, 50


def ratio_table(t, keys):
    t = t.copy()
    t["ratio"] = np.where(t["aqua"] > 0, t["snpp"] / t["aqua"].where(t["aqua"] > 0, 1), np.nan)
    return t


def lookup(test, table, keys, name, min_aqua=0):
    tb = table[table["aqua"] >= max(min_aqua, 1)][keys + ["ratio"]].rename(columns={"ratio": name})
    return test.merge(tb, on=keys, how="left")


def main():
    s = load_settings()
    out = Path(__file__).resolve().parent / "results" / "e8"
    out.mkdir(parents=True, exist_ok=True)
    w = load(s["paths"]["grid"]).rename(columns={"n_det": "aqua"})[["row", "col", "region", "daynight", "year", "cmonth", "month", "aqua", "snpp"]]

    RK, BK = ["daynight", "region"], ["row", "col", "daynight"]
    per_year_reg = w.groupby(RK + ["cmonth", "year"], as_index=False)[["snpp", "aqua"]].sum()
    per_year_box = w.groupby(BK + ["cmonth", "year"], as_index=False)[["snpp", "aqua"]].sum()

    preds = []
    for y in YEARS:
        tr_reg = per_year_reg[per_year_reg["year"] != y].groupby(RK + ["cmonth"], as_index=False)[["snpp", "aqua"]].sum()
        tr_box = per_year_box[per_year_box["year"] != y].groupby(BK + ["cmonth"], as_index=False)[["snpp", "aqua"]].sum()
        B = ratio_table(tr_reg, RK)
        B3 = ratio_table(window3(tr_reg, RK), RK)
        DR = ratio_table(tr_reg.groupby(RK, as_index=False)[["snpp", "aqua"]].sum(), RK)
        D = ratio_table(tr_reg.groupby(["daynight"], as_index=False)[["snpp", "aqua"]].sum(), ["daynight"])
        C = ratio_table(window3(tr_box, BK), BK)

        te = w[w["year"] == y].copy()
        te = lookup(te, B, RK + ["cmonth"], "r_B")
        te = lookup(te, B3, RK + ["cmonth"], "r_B3")
        te = lookup(te, B, RK + ["cmonth"], "r_Bthick", MIN_AQUA)
        te = lookup(te, B3, RK + ["cmonth"], "r_B3thick", MIN_AQUA)
        te = lookup(te, DR, RK, "r_DR", MIN_AQUA)
        te = lookup(te, D, ["daynight"], "r_D")
        te = lookup(te, C, BK + ["cmonth"], "r_C", MIN_BOX)
        fb_tail = te["r_DR"].fillna(te["r_D"])
        te["r_B3fb"] = te["r_B3thick"].fillna(fb_tail)
        te["r_Bfb"] = te["r_Bthick"].fillna(te["r_B3fb"])
        te["r_B"] = te["r_B"].fillna(te["r_Bfb"])
        te["r_B3"] = te["r_B3"].fillna(te["r_B3fb"])
        te["r_C"] = te["r_C"].fillna(te["r_B3fb"])
        for m in ["B", "B3", "Bfb", "B3fb", "C"]:
            te[f"p_{m}"] = te["aqua"] * te[f"r_{m}"]
        preds.append(te)
    p = pd.concat(preds, ignore_index=True)
    p.to_parquet(out / "predictions.parquet", index=False)

    names = {"B": "B (1 month)", "B3": "B-3m", "Bfb": "B-fb", "B3fb": "B-3m-fb", "C": "C per-box map"}
    rows, per_region = [], []
    g = p.groupby(["region", "month", "daynight"])[["snpp"] + [f"p_{m}" for m in names]].sum().reset_index()
    g = g[g["snpp"] >= 100]
    a = p.assign(y=p["year"]).groupby(["region", "y"])[["snpp"] + [f"p_{m}" for m in names]].sum()
    b = p[(p["snpp"] >= 20) & (p["aqua"] > 0)]
    for m, nm in names.items():
        e = (g[f"p_{m}"] - g["snpp"]).abs()
        rel = e / g["snpp"]
        be = (b[f"p_{m}"] - b["snpp"]).abs() / b["snpp"]
        rows.append({"method": nm,
                     "regional_median_err": round(100 * rel.median(), 1),
                     "regional_fire_weighted_err": round(100 * e.sum() / g["snpp"].sum(), 1),
                     "regional_day_median": round(100 * rel[g["daynight"] == "D"].median(), 1),
                     "regional_night_median": round(100 * rel[g["daynight"] == "N"].median(), 1),
                     "annual_region_median": round(100 * ((a[f"p_{m}"] - a["snpp"]).abs() / a["snpp"]).median(), 1),
                     "box_median_err": round(100 * be.median(), 1),
                     "box_night_median": round(100 * be[b["daynight"] == "N"].median(), 1)})
        pr = g.assign(e=e).groupby("region").apply(lambda x: x["e"].sum() / x["snpp"].sum() * 100, include_groups=False)
        per_region.append(pr.rename(nm))
    t = pd.DataFrame(rows)
    t.to_csv(out / "methods.csv", index=False)
    pr = pd.concat(per_region, axis=1).round(1)
    pr["winner"] = pr.idxmin(axis=1)
    pr.to_csv(out / "per_region_fire_weighted_err.csv")
    print(t.to_string(index=False))
    print("\nFire-weighted regional error by region (%)\n", pr.to_string())


if __name__ == "__main__":
    main()
