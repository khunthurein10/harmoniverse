"""
Step 3: build the harmonized record (S-NPP scale) with uncertainty.

  2003 to 2011 : converted from MODIS with the gradient boosting models (data/interim/models, trained on
                 2015 to 2019), falling back to the table (B-fb; B-3m-fb in BONA, TENA) when a row's Aqua
                 count is above its region x day/night training max or Terra has an outage that month
  2012 to 2026 : real S-NPP counts (2026 to October, from the FIRMS API) (low confidence dropped). Missing and
                 partial S-NPP days (harmonizer/fill.py): from April 2018 replaced by NOAA-20 x same-month ratio
                 (GFED5NRT Sect. 2.5; E10), before that the good days are scaled up to the month (E9); with fewer
                 than 2/3 good days before April 2018, converted from MODIS (in practice January 2012 only)
Leak fix (Nov 12): the model only predicts boxes with MODIS evidence (Aqua > 0 or Terra > 0); fires S-NPP sees
in boxes without MODIS evidence are added back at the regional level: regional = sum of box predictions / (1 - m),
m = VIIRS-only share per region x day/night x calendar month learned on 2015 to 2019 (e8c_fixed.py; MRBA60 Fire
Mask idea). Box-level values therefore cover MODIS-evidence boxes only.
Uncertainty (MRBA60 Methods "Uncertainty", adapted): prediction/observation pairs binned into 10 prediction
quantiles, RMSE per bin, linear model RMSE = a + b x prediction. MRBA60 fitted this on its training years;
we fit it on 2012 to 2014, which the models never saw (our choice). Two fits: per box (per region x
day/night) and per regional monthly total (per day/night). Regional totals use the RMSE per size bin instead of
the line (E11: the line was too wide for small months, too narrow for the largest). The 90% band assumes normal
errors (our choice). Filled S-NPP months take their relative error from E9 / E10.

Writes data/processed/harmonized_box/<year>.parquet (box x month x day/night x land)
       data/processed/harmonized_region_month.parquet
       data/processed/uncertainty_fits.csv
Usage: python pipeline/scripts/03_harmonize.py [--years 2026]
       --years rebuilds only those years and swaps them into the regional table (others are kept)
"""
import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harmonizer.config import load_settings  # noqa: E402
from harmonizer.groups import LAND_GROUPS, REGION_NAMES  # noqa: E402
from harmonizer.fill import day_status, filled_counts, relative_error  # noqa: E402
from harmonizer.grid import cell_centre  # noqa: E402
from harmonizer.convert import FEATS, apply_table, table_ratios  # noqa: E402

KEYS = ["row", "col", "month", "daynight", "land", "region"]
TRAIN, CALIB, OBSERVED, CONVERT = range(2015, 2020), range(2012, 2015), range(2012, 2027), range(2003, 2012)
TERRA_OK_UNTIL = 2019       # our choice: after this, conversions use the table (Terra orbit drift)
REGIONS = sorted(v for v in REGION_NAMES.values() if v != "OCEAN")   # same codes as in training (alphabetical)
LANDS = sorted(LAND_GROUPS)
Z90 = 1.645


def load_year(grid, year):
    cols = KEYS + ["satellite", "cell_lat", "cell_lon", "n_det", "n_det_low", "n_firedays", "frp_sum", "pix_area_mean"]
    mf = grid / "monthly" / f"modis_{year}.parquet"
    parts = [pd.read_parquet(mf, columns=cols)] if mf.exists() else []   # 2026: no MODIS (orbit drift), S-NPP only
    vf = grid / "monthly" / f"viirs_snpp_{year}.parquet"
    if vf.exists():
        parts.append(pd.read_parquet(vf, columns=cols))
    m = pd.concat(parts, ignore_index=True)
    m = m[m["region"] != "OCEAN"]
    a, t, v = m[m.satellite == "AQUA"], m[m.satellite == "TERRA"], m[m.satellite == "SNPP"]
    a = a.assign(aqua_frp_mean=a.frp_sum / a.n_det)[KEYS + ["cell_lat", "cell_lon", "n_det", "n_firedays", "aqua_frp_mean", "pix_area_mean"]]
    a = a.rename(columns={"n_det": "aqua", "n_firedays": "aqua_days", "pix_area_mean": "aqua_pix"})
    t = t[KEYS + ["cell_lat", "cell_lon", "n_det"]].rename(columns={"n_det": "terra", "cell_lat": "lat_t", "cell_lon": "lon_t"})
    w = a.merge(t, on=KEYS, how="outer")
    if len(v):
        v = v.assign(snpp_obs=v.n_det - v.n_det_low)[KEYS + ["cell_lat", "cell_lon", "snpp_obs"]].rename(
            columns={"cell_lat": "lat_v", "cell_lon": "lon_v"})
        w = w.merge(v, on=KEYS, how="outer")
    else:
        w["snpp_obs"], w["lat_v"], w["lon_v"] = np.nan, np.nan, np.nan
    w["cell_lat"] = w["cell_lat"].fillna(w["lat_t"]).fillna(w["lat_v"])
    w["cell_lon"] = w["cell_lon"].fillna(w["lon_t"]).fillna(w["lon_v"])
    w = w.drop(columns=["lat_t", "lon_t", "lat_v", "lon_v"])
    for c in ["aqua", "aqua_days", "terra"]:
        w[c] = w[c].fillna(0)
    if vf.exists():
        w["snpp_obs"] = w["snpp_obs"].fillna(0)
    w["month"] = pd.to_datetime(w["month"])
    w["year"], w["cmonth"] = w["month"].dt.year, w["month"].dt.month
    w["region_c"] = pd.Categorical(w["region"], categories=REGIONS).codes
    w["land_c"] = pd.Categorical(w["land"], categories=LANDS).codes
    return w


def predict(w, models, tabs, tmax, terra_bad):
    w = w.copy()
    w["evidence"] = (w["aqua"] > 0) | (w["terra"] > 0)
    w["p_table"], _ = apply_table(w, tabs)
    w["p_model"] = np.nan
    for dn, mdl in models.items():
        sel = w["daynight"] == dn
        if sel.any():
            w.loc[sel, "p_model"] = mdl.predict(w.loc[sel, FEATS])
    limit = np.array([tmax.get(f"{d}|{r}", 0) for d, r in zip(w["daynight"], w["region"])])
    w["used_table"] = (w["aqua"].to_numpy() > limit) | w["month"].isin(terra_bad).to_numpy()
    w["used_table"] = w["used_table"] & w["evidence"]
    w["pred"] = np.where(w["evidence"], np.where(w["used_table"], w["p_table"], w["p_model"]), 0.0)
    return w


def fit_error(pred, obs, nbins=10, min_points=50):
    """MRBA60-style: RMSE in 10 prediction-quantile bins, then a linear fit RMSE = a + b * prediction."""
    d = pd.DataFrame({"pred": pred, "obs": obs})
    d = d[d["pred"] > 0]
    if len(d) < min_points:
        return None
    nbins = min(nbins, max(len(d) // 5, 2))
    d["bin"] = pd.qcut(d["pred"], nbins, labels=False, duplicates="drop")
    g = d.groupby("bin").apply(lambda x: pd.Series({"pred": x["pred"].mean(),
                                                    "rmse": np.sqrt(((x["pred"] - x["obs"]) ** 2).mean())}),
                               include_groups=False)
    b, a = np.polyfit(g["pred"], g["rmse"], 1)
    return float(a), float(b)


def fit_binned(pred, obs, nbins=10):
    """RMSE in hotspots per prediction-size bin (10 quantile bins), no line: RMSE = rmse(bin of the prediction).
    Chosen for regional totals after E11 (leave-one-year-out on 2012 to 2014): the single line covered 100% of small
    months and 86% of the largest day months; a relative error per bin gave tiny months +/-300 to 440%; this holds
    about 88 to 93% coverage at every size."""
    d = pd.DataFrame({"pred": np.asarray(pred, float), "obs": np.asarray(obs, float)})
    d = d[d["pred"] > 0]
    edges = np.unique(np.quantile(d["pred"], np.linspace(0, 1, nbins + 1)))
    d["bin"] = np.clip(np.searchsorted(edges, d["pred"], side="right") - 1, 0, len(edges) - 2)
    r = d.groupby("bin").apply(lambda x: np.sqrt(((x["pred"] - x["obs"]) ** 2).mean()), include_groups=False)
    return edges, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", help="only rebuild these years, e.g. 2026 or 2020-2026")
    args = ap.parse_args()
    only = None
    if args.years:
        a, _, b = args.years.partition("-")
        only = set(range(int(a), int(b or a) + 1))
    s = load_settings()
    grid = s["paths"]["grid"]
    out = s["paths"]["processed"]
    (out / "harmonized_box").mkdir(parents=True, exist_ok=True)
    mdir = grid.parent / "models"
    models = {"D": lgb.Booster(model_file=str(mdir / "gb_day.txt")), "N": lgb.Booster(model_file=str(mdir / "gb_night.txt"))}
    tmax = json.loads((mdir / "fallback_train_max.json").read_text())
    obs = pd.read_parquet(grid / "days_observed.parquet")
    obs["month"] = pd.to_datetime(obs["month"])
    bad = obs[obs["days_missing"] > 2]
    terra_bad = set(bad.loc[bad["satellite"] == "TERRA", "month"])
    snpp_bad = set(bad.loc[bad["satellite"] == "SNPP", "month"])
    aqua_bad = set(bad.loc[bad["satellite"] == "AQUA", "month"])

    train_years, calib_years, convert_years, observed_years = TRAIN, CALIB, CONVERT, OBSERVED

    tr = pd.concat([load_year(grid, y) for y in train_years]).rename(columns={"snpp_obs": "snpp"})
    tabs = table_ratios(tr[(tr["aqua"] > 0) | (tr["terra"] > 0)])      # evidence-only table for the fallback
    mshare = pd.read_parquet(mdir / "viirs_only_share.parquet").set_index(["daynight", "region", "cmonth"])["m"]
    del tr

    # 1) uncertainty from the untouched years 2012 to 2014
    cal = predict(pd.concat([load_year(grid, y) for y in calib_years]), models, tabs, tmax, terra_bad)
    cal_bad = set()
    for y in calib_years:
        cal_bad |= set(day_status(grid, y, True)["month"])
    cal = cal[~cal["month"].isin(snpp_bad | aqua_bad | cal_bad)]
    box = cal.groupby(["row", "col", "month", "daynight", "region"], as_index=False)[["pred", "snpp_obs"]].sum()
    fits = []
    for (dn, reg), x in box.groupby(["daynight", "region"]):
        f = fit_error(x["pred"], x["snpp_obs"])
        if f:
            fits.append({"level": "box", "daynight": dn, "region": reg, "a": f[0], "b": f[1]})
    for dn, x in box.groupby("daynight"):
        f = fit_error(x["pred"], x["snpp_obs"])
        fits.append({"level": "box", "daynight": dn, "region": "ALL", "a": f[0], "b": f[1]})
    rm = box.groupby(["region", "month", "daynight"], as_index=False)[["pred", "snpp_obs"]].sum()
    rm["cmonth"] = rm["month"].dt.month
    rm = rm.join(mshare, on=["daynight", "region", "cmonth"])
    rm["pred"] = rm["pred"] / (1 - rm["m"].fillna(0))
    for dn, x in rm.groupby("daynight"):
        f = fit_error(x["pred"], x["snpp_obs"], min_points=10)
        fits.append({"level": "region_month", "daynight": dn, "region": "ALL", "a": f[0], "b": f[1]})
    fits = pd.DataFrame(fits)
    fits.to_csv(out / "uncertainty_fits.csv", index=False)
    relfit, relrows = {}, []
    for dn, x in rm.groupby("daynight"):
        edges, r = fit_binned(x["pred"], x["snpp_obs"])
        relfit[dn] = (edges, r)
        for k in range(len(edges) - 1):
            relrows.append({"daynight": dn, "bin": k, "pred_from": edges[k], "pred_to": edges[k + 1],
                            "rmse": float(r.get(k, np.nan))})
    pd.DataFrame(relrows).to_csv(out / "uncertainty_region_bins.csv", index=False)
    print("regional RMSE by size bin:\n", pd.DataFrame(relrows).round(3).to_string(index=False))
    print("uncertainty fits (RMSE = a + b x prediction):\n", fits.round(3).to_string(index=False))
    bx = fits[fits["level"] == "box"].set_index(["daynight", "region"])[["a", "b"]]

    def box_sigma(w):
        k = pd.MultiIndex.from_arrays([w["daynight"], w["region"]])
        ab = bx.reindex(k)
        allab = bx.xs("ALL", level="region").reindex(w["daynight"])
        a = ab["a"].fillna(pd.Series(allab["a"].to_numpy(), index=ab.index)).to_numpy()
        b = ab["b"].fillna(pd.Series(allab["b"].to_numpy(), index=ab.index)).to_numpy()
        return np.maximum(a + b * w["pred"].to_numpy(), 0)

    # 2) build every year
    FILL = ("noaa20", "scale")
    LABEL = {"noaa20": "S-NPP observed, filled with NOAA-20", "scale": "S-NPP observed, gap-filled"}
    regional = []
    build_years = sorted(set(convert_years) | set(observed_years))
    if only:
        build_years = [y for y in build_years if y in only]
    for y in build_years:
        modis_exists = (grid / "monthly" / f"modis_{y}.parquet").exists()
        st = day_status(grid, y, modis_exists) if y in observed_years else pd.DataFrame(columns=["month", "method", "n_bad"])
        if len(st):
            print(f"{y} months with missing or partial S-NPP days: "
                  + ", ".join(f"{m:%m} {k} ({n})" for m, k, n in zip(st["month"], st["method"], st["n_bad"])))
        w = predict(load_year(grid, y), models, tabs, tmax, terra_bad)
        if y > TERRA_OK_UNTIL:                       # Terra drift: conversions after 2019 use the table
            w["used_table"] = w["evidence"]
            w["pred"] = np.where(w["evidence"], w["p_table"], 0.0)
        fills = filled_counts(grid, s["paths"]["ancillary"], y, st)
        w = w.merge(fills, on=KEYS, how="outer")
        new = w["cell_lat"].isna()
        if new.any():                                # boxes only NOAA-20 saw on the filled days
            w.loc[new, "cell_lat"], w.loc[new, "cell_lon"] = cell_centre(w.loc[new, "row"], w.loc[new, "col"], 0.25)
            for c in ["aqua", "aqua_days", "terra", "pred", "p_table", "snpp_obs"]:
                w.loc[new, c] = 0.0
            w.loc[new, ["evidence", "used_table"]] = False
            w.loc[new, "year"], w.loc[new, "cmonth"] = y, w.loc[new, "month"].dt.month
        w["evidence"] = w["evidence"].astype(bool)
        w["used_table"] = w["used_table"].astype(bool)
        w["pred_sigma"] = box_sigma(w)
        meth = w["month"].map(st.set_index("month")["method"]) if len(st) else pd.Series(np.nan, index=w.index)
        nbad = w["month"].map(st.set_index("month")["n_bad"]).fillna(0).astype(int) if len(st) else pd.Series(0, index=w.index)
        in_obs = np.full(len(w), y in observed_years)
        fill = in_obs & meth.isin(FILL).to_numpy()
        conv_outage = in_obs & (meth == "convert").to_numpy()
        observed = in_obs & meth.isna().to_numpy()
        w["snpp_filled"] = w["snpp_filled"].fillna(0)
        w["harmonized"] = np.where(observed, w["snpp_obs"].fillna(0), np.where(fill, w["snpp_filled"], w["pred"]))
        w["days_filled"] = np.where(fill, nbad, 0)
        w["source"] = np.select(
            [observed, fill & (meth == "noaa20").to_numpy(), fill, conv_outage & w["used_table"], conv_outage, w["used_table"]],
            ["S-NPP observed", LABEL["noaa20"], LABEL["scale"], "converted (S-NPP outage, table)",
             "converted (S-NPP outage, model)", "converted (table fallback)"], default="converted (model)")
        for k in FILL:                               # filled boxes: relative error from E10 / E9
            sel = fill & (meth == k).to_numpy()
            if sel.any():
                rel = np.asarray(relative_error(k, w.loc[sel, "daynight"].to_numpy(), nbad[sel].astype(int).to_numpy()), dtype=float)
                w.loc[sel, "pred_sigma"] = rel * w.loc[sel, "harmonized"].astype(float).to_numpy()
        w["outage_aqua"] = w["month"].isin(aqua_bad)
        w["outage_terra"] = w["month"].isin(terra_bad)
        w["outage_snpp"] = w["month"].isin(snpp_bad) & (w["year"] >= 2012)
        w["above_training_range"] = w["used_table"] & ~w["outage_terra"] & (w["year"] <= TERRA_OK_UNTIL)
        w["land_map_year"] = min(max(y, 2012), 2019)
        cols = KEYS + ["cell_lat", "cell_lon", "aqua", "terra", "snpp_obs", "pred", "pred_sigma", "used_table",
                       "harmonized", "days_filled", "source", "outage_aqua", "outage_terra", "outage_snpp",
                       "above_training_range", "land_map_year"]
        w[cols].to_parquet(out / "harmonized_box" / f"{y}.parquet", index=False)
        r = w.groupby(["region", "month", "daynight"], as_index=False).agg(
            aqua=("aqua", "sum"), terra=("terra", "sum"), snpp_obs=("snpp_obs", "sum"), pred_boxes=("pred", "sum"),
            harmonized_boxes=("harmonized", "sum"), share_table=("used_table", "mean"))
        r["cmonth"] = r["month"].dt.month
        r = r.join(mshare, on=["daynight", "region", "cmonth"])
        r["pred"] = r["pred_boxes"] / (1 - r["m"].fillna(0))
        rmeth = r["month"].map(st.set_index("month")["method"]) if len(st) else pd.Series(np.nan, index=r.index)
        r["days_filled"] = r["month"].map(st.set_index("month")["n_bad"]).fillna(0).astype(int) if len(st) else 0
        rin = np.full(len(r), y in observed_years)
        rfill = rin & rmeth.isin(FILL).to_numpy()
        robs = rin & rmeth.isna().to_numpy()
        r.loc[~rfill, "days_filled"] = 0
        r["fill_method"] = np.where(rfill, rmeth.astype(object).fillna(""), "")
        r["harmonized"] = np.where(robs | rfill, r["harmonized_boxes"], r["pred"])
        r["viirs_only_added"] = np.where(robs | rfill, 0.0, r["pred"] - r["pred_boxes"])
        r["observed"] = robs
        r["filled"] = rfill
        r["source"] = np.select([robs, rfill & (rmeth == "noaa20").to_numpy(), rfill, rin],
                                ["S-NPP observed", LABEL["noaa20"], LABEL["scale"], "converted (S-NPP outage)"],
                                default="converted")
        if y not in observed_years:
            r["snpp_obs"] = np.nan
        regional.append(r.drop(columns=["harmonized_boxes"]))
        print(f"{y}: {len(w):,} rows, regional harmonized {r['harmonized'].sum():,.0f}, "
              f"sources {w['source'].value_counts().to_dict()}", flush=True)

    reg = pd.concat(regional, ignore_index=True)
    sig = np.zeros(len(reg))
    for dn, (edges, r) in relfit.items():
        sel = (reg["daynight"] == dn).to_numpy()
        p = reg.loc[sel, "pred"].to_numpy(dtype=float)
        k = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
        sig[sel] = r.reindex(k).to_numpy()
    reg["sigma"] = np.where(reg["observed"], 0.0, sig)
    for k in FILL:                                   # filled months: relative error from E10 / E9
        sel = (reg["fill_method"] == k).to_numpy()
        if sel.any():
            rel = np.asarray(relative_error(k, reg.loc[sel, "daynight"].to_numpy(),
                                            reg.loc[sel, "days_filled"].astype(int).to_numpy()), dtype=float)
            reg.loc[sel, "sigma"] = rel * reg.loc[sel, "harmonized"].astype(float).to_numpy()
    reg["low90"] = np.maximum(reg["harmonized"] - Z90 * reg["sigma"], 0)
    reg["high90"] = reg["harmonized"] + Z90 * reg["sigma"]
    reg["outage_aqua"] = reg["month"].isin(aqua_bad)
    reg["outage_terra"] = reg["month"].isin(terra_bad)
    reg["outage_snpp"] = reg["month"].isin(snpp_bad) & (reg["month"].dt.year >= 2012)
    reg_path = out / "harmonized_region_month.parquet"
    if only and reg_path.exists():   # keep the other years, replace only the rebuilt ones
        old_reg = pd.read_parquet(reg_path)
        old_reg = old_reg[~old_reg["month"].dt.year.isin(only)]
        reg = pd.concat([old_reg, reg], ignore_index=True).sort_values(["region", "month", "daynight"])
    reg.to_parquet(reg_path, index=False)
    print(f"\nWrote {out / 'harmonized_region_month.parquet'} ({len(reg):,} rows) and harmonized_box/<year>.parquet")


if __name__ == "__main__":
    main()
