"""
E2 (step 6): does the S-NPP / Aqua ratio vary, and which splits are worth keeping?

Uses only the choosing years 2015 to 2019 (2012 to 2014 stay untouched for the final test).
Paper basis for the splits tested: GFED5NRT splits by region x land type (Sect. 2.4) and by
day/night and month for its viewing tables (Sect. 2.3.3); Li et al. 2018 report large boreal fires differ.

Test (our choice): leave one year out. Learn S-NPP/Aqua ratios (ratio of sums) from 4 years, predict the
5th year's S-NPP counts from its Aqua counts, score at region x month x day/night totals (the regional
calendar). A split is worth it if it lowers the error on held-out years.
  S0 one global ratio | S1 day/night | S2 day/night x region | S3 + land type | S4 + calendar month (season)
Also: how steady each ratio is from year to year (hotspots vs fire-days), and VIIRS low confidence kept vs dropped.

Usage: python pipeline/experiments/e2_ratio_varies.py
Writes pipeline/experiments/results/e2/*.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harmonizer.config import load_settings  # noqa: E402

YEARS = range(2015, 2020)
SPLITS = {
    "S0 global": [],
    "S1 day/night": ["daynight"],
    "S2 day/night x region": ["daynight", "region"],
    "S3 day/night x region x land": ["daynight", "region", "land"],
    "S4 day/night x region x month": ["daynight", "region", "cmonth"],
}
EVAL = ["region", "month", "daynight"]


def load(grid_root):
    cols = ["month", "satellite", "daynight", "region", "land", "n_det", "n_det_low", "n_firedays", "n_firedays_nolow"]
    parts = [pd.read_parquet(grid_root / "monthly" / f"{s}_{y}.parquet", columns=cols)
             for s in ["modis", "viirs_snpp"] for y in YEARS]
    m = pd.concat(parts, ignore_index=True)
    m = m[m["satellite"].isin(["AQUA", "SNPP"]) & (m["region"] != "OCEAN")]
    bad = pd.read_parquet(grid_root / "days_observed.parquet")
    bad = set(pd.to_datetime(bad.loc[bad["satellite"].isin(["AQUA", "SNPP"]) & (bad["days_missing"] > 2), "month"]))
    m["month"] = pd.to_datetime(m["month"])
    m = m[~m["month"].isin(bad)]
    m["year"], m["cmonth"] = m["month"].dt.year, m["month"].dt.month
    m["n_det_nolow"] = m["n_det"] - m["n_det_low"]
    keys = ["year", "month", "cmonth", "daynight", "region", "land"]
    w = m.pivot_table(index=keys, columns="satellite",
                      values=["n_det", "n_det_nolow", "n_firedays", "n_firedays_nolow"], aggfunc="sum", fill_value=0)
    w.columns = [f"{v}_{s}" for v, s in w.columns]
    return w.reset_index()


def loyo_error(w, by, snpp_col="n_det_SNPP", aqua_col="n_det_AQUA"):
    """Median absolute % error of held-out region x month x day/night S-NPP totals."""
    errs = []
    for y in YEARS:
        train, test = w[w["year"] != y], w[w["year"] == y].copy()
        if by:
            r = train.groupby(by)[[snpp_col, aqua_col]].sum()
            r = (r[snpp_col] / r[aqua_col]).replace(np.inf, np.nan).rename("ratio")
            test = test.join(r, on=by)
            # a group never seen with Aqua fire in training falls back to the global ratio
            test["ratio"] = test["ratio"].fillna(train[snpp_col].sum() / train[aqua_col].sum())
        else:
            test["ratio"] = train[snpp_col].sum() / train[aqua_col].sum()
        test["pred"] = test[aqua_col] * test["ratio"]
        e = test.groupby(EVAL)[["pred", snpp_col]].sum()
        e = e[e[snpp_col] >= 100]          # score only cells of the calendar with real fire (our choice)
        errs.append(((e["pred"] - e[snpp_col]).abs() / e[snpp_col]).rename(y))
    allerr = pd.concat(errs)
    return round(100 * allerr.median(), 1), round(100 * allerr.mean(), 1), len(allerr)


def steadiness(w, by, snpp_col, aqua_col):
    """Year-to-year wobble of each group's ratio (coefficient of variation, %), median over groups."""
    y = w.groupby(by + ["year"])[[snpp_col, aqua_col]].sum()
    y = y[y[aqua_col] >= 50]
    r = (y[snpp_col] / y[aqua_col]).unstack("year")
    cv = (r.std(axis=1) / r.mean(axis=1) * 100).dropna()
    return round(cv.median(), 1)


def main():
    s = load_settings()
    out = Path(__file__).resolve().parent / "results" / "e2"
    out.mkdir(parents=True, exist_ok=True)
    w = load(s["paths"]["grid"])

    # 1) which split predicts held-out years best
    rows = []
    for name, by in SPLITS.items():
        med, mean, n = loyo_error(w, by)
        rows.append({"split": name, "median_abs_err_pct": med, "mean_abs_err_pct": mean, "scored_cells": n})
    t1 = pd.DataFrame(rows)
    t1.to_csv(out / "split_errors.csv", index=False)
    print("1) Held-out error by split (hotspots, low confidence kept)\n", t1.to_string(index=False))

    # 2) ratios by day/night x region, with year-to-year range
    g = w.groupby(["daynight", "region", "year"])[["n_det_SNPP", "n_det_AQUA"]].sum()
    r = (g["n_det_SNPP"] / g["n_det_AQUA"]).unstack("year")
    tot = w.groupby(["daynight", "region"])[["n_det_SNPP", "n_det_AQUA"]].sum()
    t2 = pd.DataFrame({"ratio_2015_19": (tot["n_det_SNPP"] / tot["n_det_AQUA"]).round(2),
                       "min_year": r.min(axis=1).round(2), "max_year": r.max(axis=1).round(2),
                       "aqua_hotspots": tot["n_det_AQUA"]})
    t2.to_csv(out / "ratio_daynight_region.csv")
    print("\n2) Ratio by day/night x region (min/max over single years)\n", t2.to_string())

    # 3) land within region: day only, ratio per land group
    gl = w[w["daynight"] == "D"].groupby(["region", "land"])[["n_det_SNPP", "n_det_AQUA"]].sum()
    t3 = (gl["n_det_SNPP"] / gl["n_det_AQUA"]).round(2).unstack("land")
    t3.to_csv(out / "ratio_day_region_land.csv")
    print("\n3) Day ratio by region x land\n", t3.to_string())

    # 4) season: day ratio by calendar month, world
    gm = w[w["daynight"] == "D"].groupby("cmonth")[["n_det_SNPP", "n_det_AQUA"]].sum()
    t4 = (gm["n_det_SNPP"] / gm["n_det_AQUA"]).round(2)
    t4.to_csv(out / "ratio_day_month.csv")
    print("\n4) Day ratio by calendar month (world)\n", t4.to_string())

    # 5) hotspots vs fire-days, low confidence kept vs dropped
    rows = []
    for label, sn, aq in [("hotspots, low kept", "n_det_SNPP", "n_det_AQUA"),
                          ("hotspots, low dropped", "n_det_nolow_SNPP", "n_det_AQUA"),
                          ("fire-days, low kept", "n_firedays_SNPP", "n_firedays_AQUA"),
                          ("fire-days, low dropped", "n_firedays_nolow_SNPP", "n_firedays_AQUA")]:
            med, mean, n = loyo_error(w, SPLITS["S2 day/night x region"], sn, aq)
            rows.append({"measure": label, "held_out_median_err_pct": med, "held_out_mean_err_pct": mean,
                         "year_to_year_wobble_pct": steadiness(w, ["daynight", "region"], sn, aq)})
    t5 = pd.DataFrame(rows)
    t5.to_csv(out / "measures.csv", index=False)
    print("\n5) Hotspots vs fire-days, VIIRS low confidence kept vs dropped (split S2)\n", t5.to_string(index=False))

    # low-confidence share by region (context for sun glint, VIIRS UG 3.1.2)
    lc = w.groupby(["daynight", "region"])[["n_det_SNPP", "n_det_nolow_SNPP"]].sum()
    print("\nVIIRS low-confidence share by day/night x region (%)\n",
          ((1 - lc["n_det_nolow_SNPP"] / lc["n_det_SNPP"]) * 100).round(1).unstack("daynight").to_string())


if __name__ == "__main__":
    main()
