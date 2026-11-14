"""
Missing and partial S-NPP days: which days are bad, and the filled monthly counts per box.

A month's S-NPP count is short when the satellite sent nothing on some days (missing) or only some orbits
(partial). FIRMS lists only fully missing dates (GFED5NRT Table S4), so partial days are found here.

Bad days
  missing : S-NPP has no hotspot anywhere on Earth that day.
  partial : where NOAA-20 data exists, S-NPP's world total that day is below PARTIAL_VS_NOAA20 x NOAA-20's
            (normal days: median 1.01, 5th percentile 0.82, 2021 to 2026); elsewhere, below
            PARTIAL_VS_NEIGHBOURS x the median of the 15 surrounding S-NPP days (our choices, from real data).

Filling a month with bad days (method per month)
  noaa20  : from April 2018 (NOAA-20 standard data starts 2018-04-01, FIRMS). Bad days are replaced by NOAA-20
            counts x a ratio learned that month per region x day/night on the days both saw (GFED5NRT Sect. 2.5:
            "If observations from one satellite are unavailable, the combined product defaults to the remaining
            instrument"; same-month ratio is our choice). Tested in E10: 0.85% error vs 5.8% for scaling.
  scale   : no NOAA-20 and at least MIN_GOOD_SHARE of days good: good days scaled up to the whole month
            (Giglio et al. 2006 MODIS CMG normalize to a complete month; GFED5NRT temporal interpolation). E9.
  convert : before April 2018, fewer good days than MIN_GOOD_SHARE and MODIS exists: convert from MODIS
            (in practice only January 2012, when S-NPP data starts on the 20th).
Uncertainty of a filled month: relative RMSE from E10 (noaa20, per day/night) or E9 (scale, per day/night and
number of bad days) times the filled value.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from .groups import LAND_GROUPS, REGION_NAMES, region_of

NOAA20_FROM = pd.Timestamp("2018-04-01")
PARTIAL_VS_NOAA20 = 0.8
PARTIAL_VS_NEIGHBOURS = 0.5
MIN_GOOD_SHARE = 2 / 3
RESULTS = Path(__file__).resolve().parents[1] / "experiments" / "results"


def _cell_days(grid, sat, year):
    f = grid / "cell_days" / f"{sat}_{year}.parquet"
    if not f.exists():
        return None
    c = pd.read_parquet(f, columns=["row", "col", "date", "daynight", "land", "n_det", "n_det_low"])
    c["date"] = pd.to_datetime(c["date"]).dt.normalize()
    c["n"] = c["n_det"] - c["n_det_low"]
    return c.drop(columns=["n_det", "n_det_low"])


def day_status(grid, year, modis_exists):
    """One row per S-NPP month of `year` that has bad days: method, bad days, good share."""
    sn = _cell_days(grid, "viirs_snpp", year)
    if sn is None:
        return pd.DataFrame(columns=["month", "method", "bad_days", "n_bad", "good_share"])
    n2 = _cell_days(grid, "viirs_noaa20", year)
    s_day = sn.groupby("date")["n"].sum()
    n_day = n2.groupby("date")["n"].sum() if n2 is not None else pd.Series(dtype=float)
    first = s_day.index.min()
    allx = pd.date_range(pd.Timestamp(year, 1, 1), pd.Timestamp(year, 12, 31))
    s_full = s_day.reindex(allx)
    neigh = s_full.rolling(15, center=True, min_periods=5).median()
    rows = []
    for m in pd.date_range(pd.Timestamp(year, 1, 1), periods=12, freq="MS"):
        days = pd.date_range(m, periods=m.days_in_month)
        if days[-1] < first or not s_full.reindex(days).notna().any():
            continue
        has_n2 = m >= NOAA20_FROM and n_day.reindex(days).notna().any()
        bad = []
        for d in days:
            v = s_full.get(d)
            if pd.isna(v):
                bad.append(d)
            elif has_n2 and pd.notna(n_day.get(d)) and n_day[d] > 0:
                if v / n_day[d] < PARTIAL_VS_NOAA20:
                    bad.append(d)
            elif pd.notna(neigh.get(d)) and v < PARTIAL_VS_NEIGHBOURS * neigh[d]:
                bad.append(d)
        if not bad:
            continue
        good_share = 1 - len(bad) / len(days)
        if has_n2:
            method = "noaa20"
        elif good_share < MIN_GOOD_SHARE and modis_exists:
            method = "convert"
        else:
            method = "scale"
        rows.append({"month": m, "method": method, "bad_days": [d.day for d in bad], "n_bad": len(bad),
                     "good_share": good_share})
    return pd.DataFrame(rows, columns=["month", "method", "bad_days", "n_bad", "good_share"])


def filled_counts(grid, ancillary, year, status):
    """Box counts (S-NPP scale, low confidence dropped) for months filled by 'noaa20' or 'scale'.
    Returns row, col, month, daynight, land, region, snpp_filled."""
    todo = status[status["method"].isin(["noaa20", "scale"])]
    if todo.empty:
        return pd.DataFrame(columns=["row", "col", "month", "daynight", "land", "region", "snpp_filled"])
    from .groups import load_regions
    region_grid = load_regions(ancillary, grid / "lookup")
    sn = _cell_days(grid, "viirs_snpp", year)
    n2 = _cell_days(grid, "viirs_noaa20", year)
    out = []
    for _, st in todo.iterrows():
        m = st["month"]
        days = pd.date_range(m, periods=m.days_in_month)
        bad = {pd.Timestamp(m.year, m.month, d) for d in st["bad_days"]}
        good = [d for d in days if d not in bad]
        s = sn[sn["date"].isin(good)]
        keys = ["row", "col", "daynight", "land"]
        s_box = s.groupby(keys)["n"].sum().rename("s")
        if st["method"] == "noaa20" and n2 is not None:
            n2_days = set(n2["date"].unique())
            both = [d for d in good if d in n2_days]
            covered = [d for d in bad if d in n2_days]
            n_both = n2[n2["date"].isin(both)]
            n_cov = n2[n2["date"].isin(covered)]
            # ratio per region x day/night from the days both satellites saw this month
            reg = lambda df: np.vectorize(REGION_NAMES.get)(region_of(df["row"], df["col"], region_grid))
            rs = sn[sn["date"].isin(both)].assign(region=lambda d: reg(d)).groupby(["region", "daynight"])["n"].sum()
            rn = n_both.assign(region=lambda d: reg(d)).groupby(["region", "daynight"])["n"].sum()
            ratio = (rs / rn).replace([np.inf], np.nan)
            c_box = n_cov.groupby(keys)["n"].sum().rename("c")
            b = pd.concat([s_box, c_box], axis=1).fillna(0).reset_index()
            b["region"] = reg(b)
            r = ratio.reindex(pd.MultiIndex.from_arrays([b["region"], b["daynight"]])).to_numpy()
            r = np.where(np.isfinite(r), r, 1.0)   # no shared fires in that region x day/night: 1.0 (median ratio 0.99)
            seen = len(good) + len(covered)
            b["snpp_filled"] = (b["s"] + r * b["c"]) * len(days) / seen
        else:
            b = s_box.reset_index()
            b["region"] = np.vectorize(REGION_NAMES.get)(region_of(b["row"], b["col"], region_grid))
            b["snpp_filled"] = b["s"] * len(days) / len(good)
        b["month"] = m
        out.append(b[["row", "col", "month", "daynight", "land", "region", "snpp_filled"]])
    f = pd.concat(out, ignore_index=True)
    f["land"] = np.asarray(LAND_GROUPS, dtype=object)[f["land"].astype(int)]
    f = f.groupby(["row", "col", "month", "daynight", "land", "region"], as_index=False)["snpp_filled"].sum()
    return f[f["region"] != "OCEAN"]


def relative_error(method, daynight, n_bad):
    """Relative RMSE (share of the month's value) of a filled month, from E10 (noaa20) or E9 (scale)."""
    if method == "noaa20":
        t = pd.read_csv(RESULTS / "e10" / "compare_by_daynight.csv").set_index("daynight")
        return t["noaa20_relative_rmse_pct"].reindex(daynight).to_numpy() / 100
    t = pd.read_csv(RESULTS / "e9" / "gapfill_error.csv")
    k = np.clip(n_bad, t["k"].min(), t["k"].max())
    t = t.set_index(["daynight", "k"])["relative_rmse_pct"]
    return t.reindex(pd.MultiIndex.from_arrays([daynight, k])).to_numpy() / 100
