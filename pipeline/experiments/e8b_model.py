"""
E8b (step 8, stretch): does a tree model beat the lookup table?

MRBA60 (Torres-Vazquez et al. 2026, Methods "Harmonisation model") uses Random Forest (ntree = 250) per
region-month with a centred 3-month window, predictors incl. MODIS active fire counts, FRP sum/median,
lat/lon, NDVI, climate; Leave-One-Year-Out for predictor selection. Our simplified adaptation (our choice):
one model per day/night, LightGBM in random-forest mode (rf) and, for comparison, gradient boosting (gbdt),
Poisson objective for boosting, plain averaging (regression) for the forest. Only predictors that also exist for 2003 to 2011 are used:
  Aqua count, Aqua fire-days, Aqua mean FRP, Aqua mean pixel area, Terra count (same box, month, day/night),
  region, land group, calendar month, cell lat/lon.
Target: S-NPP hotspots without low confidence, per box x month x day/night x land.
Scored exactly like e8 (choosing years 2015 to 2019, leave one year out) against the chosen table
(B-fb everywhere, B-3m-fb in BONA and TENA), read from results/e8/predictions.parquet.

Usage: python pipeline/experiments/e8b_model.py            (full run, about 20 to 40 min)
       python pipeline/experiments/e8b_model.py --quick    (one region, to check it works)
"""
import argparse
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harmonizer.config import load_settings  # noqa: E402
from harmonizer.convert import FEATS, TABLE_3M  # noqa: E402

YEARS = range(2015, 2020)
KEYS = ["row", "col", "month", "daynight", "land", "region"]
MODELS = {
    "RF (MRBA60-like)": dict(boosting="rf", bagging_fraction=0.632, bagging_freq=1, feature_fraction=0.7,
                             num_leaves=255, min_data_in_leaf=20, objective="regression", verbose=-1, num_threads=0),
    "Gradient boosting": dict(boosting="gbdt", learning_rate=0.05, num_leaves=127, min_data_in_leaf=50,
                              feature_fraction=0.8, objective="poisson", verbose=-1, num_threads=0),
}
ROUNDS = {"RF (MRBA60-like)": 250, "Gradient boosting": 400}


def load(grid_root, quick):
    cols = KEYS + ["satellite", "cell_lat", "cell_lon", "n_det", "n_det_low", "n_firedays", "frp_sum", "pix_area_mean"]
    m = pd.concat([pd.read_parquet(grid_root / "monthly" / f"{s}_{y}.parquet", columns=cols)
                   for s in ["modis", "viirs_snpp"] for y in YEARS], ignore_index=True)
    m = m[m["region"] != "OCEAN"]
    if quick:
        m = m[m["region"] == "SHSA"]
    a, t, v = m[m.satellite == "AQUA"], m[m.satellite == "TERRA"], m[m.satellite == "SNPP"]
    a = a.assign(aqua_frp_mean=a.frp_sum / a.n_det)[KEYS + ["cell_lat", "cell_lon", "n_det", "n_firedays", "aqua_frp_mean", "pix_area_mean"]]
    a = a.rename(columns={"n_det": "aqua", "n_firedays": "aqua_days", "pix_area_mean": "aqua_pix"})
    t = t[KEYS + ["n_det"]].rename(columns={"n_det": "terra"})
    v = v.assign(snpp=v.n_det - v.n_det_low)[KEYS + ["cell_lat", "cell_lon", "snpp"]]
    w = a.merge(t, on=KEYS, how="outer").merge(v, on=KEYS, how="outer", suffixes=("", "_v"))
    w["cell_lat"] = w["cell_lat"].fillna(w["cell_lat_v"])
    w["cell_lon"] = w["cell_lon"].fillna(w["cell_lon_v"])
    w = w.drop(columns=["cell_lat_v", "cell_lon_v"])
    for c in ["aqua", "aqua_days", "terra", "snpp"]:
        w[c] = w[c].fillna(0)
    w["month"] = pd.to_datetime(w["month"])
    w["year"], w["cmonth"] = w["month"].dt.year, w["month"].dt.month
    w["region_c"] = w["region"].astype("category").cat.codes
    w["land_c"] = w["land"].astype("category").cat.codes
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    s = load_settings()
    here = Path(__file__).resolve().parent
    out = here / "results" / "e8"
    w = load(s["paths"]["grid"], args.quick)
    print(f"{len(w):,} rows (box x month x day/night x land), {int(w.snpp.sum()):,} S-NPP hotspots")

    years = [2017] if args.quick else list(YEARS)
    if args.quick:
        w = w[w["year"].isin([2016, 2017])]
    for name, params in MODELS.items():
        w[f"p_{name}"] = np.nan
        rounds = 20 if args.quick else ROUNDS[name]
        for y in years:
            for dn in ["D", "N"]:
                tr = (w.year != y) & (w.daynight == dn)
                te = (w.year == y) & (w.daynight == dn)
                ds = lgb.Dataset(w.loc[tr, FEATS], w.loc[tr, "snpp"], categorical_feature=["region_c", "land_c"])
                mdl = lgb.train(params, ds, num_boost_round=rounds)
                w.loc[te, f"p_{name}"] = mdl.predict(w.loc[te, FEATS])
            print(f"  {name}: held-out {y} done", flush=True)

    # compare at the same level as e8 (box x month x day/night, land summed)
    bk = ["row", "col", "month", "daynight", "region"]
    w = w[w["year"].isin(years)]
    mp = w.groupby(bk, as_index=False)[["snpp"] + [f"p_{n}" for n in MODELS]].sum()
    tb = pd.read_parquet(out / "predictions.parquet", columns=bk + ["aqua", "p_Bfb", "p_B3fb"])
    tb["month"] = pd.to_datetime(tb["month"])
    tb = tb[tb["month"].dt.year.isin(years)]
    tb["p_Table (chosen)"] = np.where(tb["region"].isin(TABLE_3M), tb["p_B3fb"], tb["p_Bfb"])
    p = mp.merge(tb[bk + ["aqua", "p_Table (chosen)"]], on=bk, how="left").fillna({"p_Table (chosen)": 0, "aqua": 0})

    methods = ["Table (chosen)"] + list(MODELS)
    g = p.groupby(["region", "month", "daynight"])[["snpp"] + [f"p_{m}" for m in methods]].sum().reset_index()
    g = g[g["snpp"] >= 100]
    a = p.assign(y=p["month"].dt.year).groupby(["region", "y"])[["snpp"] + [f"p_{m}" for m in methods]].sum()
    b = p[(p["snpp"] >= 20) & (p["aqua"] > 0)]
    rows, per_region = [], []
    for m in methods:
        e = (g[f"p_{m}"] - g["snpp"]).abs(); rel = e / g["snpp"]
        be = (b[f"p_{m}"] - b["snpp"]).abs() / b["snpp"]
        rows.append({"method": m, "regional_median_err": round(100 * rel.median(), 1),
                     "regional_fire_weighted_err": round(100 * e.sum() / g["snpp"].sum(), 1),
                     "regional_day_median": round(100 * rel[g.daynight == "D"].median(), 1),
                     "regional_night_median": round(100 * rel[g.daynight == "N"].median(), 1),
                     "annual_region_median": round(100 * ((a[f"p_{m}"] - a["snpp"]).abs() / a["snpp"]).median(), 1),
                     "box_median_err": round(100 * be.median(), 1),
                     "box_night_median": round(100 * be[b.daynight == "N"].median(), 1),
                     "total_pred_vs_actual_pct": round(100 * (p[f"p_{m}"].sum() / p["snpp"].sum() - 1), 1)})
        per_region.append(g.assign(e=e).groupby("region").apply(lambda x: x["e"].sum() / x["snpp"].sum() * 100,
                                                                 include_groups=False).rename(m))
    t = pd.DataFrame(rows)
    pr = pd.concat(per_region, axis=1).round(1)
    tag = "_quick" if args.quick else ""
    t.to_csv(out / f"model_vs_table{tag}.csv", index=False)
    pr.to_csv(out / f"model_vs_table_per_region{tag}.csv")
    print(t.to_string(index=False))
    print("\nFire-weighted regional error by region (%)\n", pr.to_string())


if __name__ == "__main__":
    main()
