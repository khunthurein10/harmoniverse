"""
MODIS -> S-NPP conversion pieces shared by the pipeline (03_harmonize.py) and the experiments (E8, E8b, E8c).

Model inputs (FEATS): chosen in E8b. Table: S-NPP / Aqua ratio per region x day/night x calendar month,
trained 2015 to 2019 (E8, "B-fb"); a group with fewer than MIN_AQUA training detections falls back to the
centred 3-month window, then region x day/night, then day/night. BONA and TENA use the 3-month window
first (B-3m-fb, E8: fewer fires in winter months). MIN_AQUA and TABLE_3M are our choice.
"""
import numpy as np
import pandas as pd

FEATS = ["aqua", "aqua_days", "aqua_frp_mean", "aqua_pix", "terra", "region_c", "land_c", "cmonth", "cell_lat", "cell_lon"]
TABLE_3M = {"BONA", "TENA"}
MIN_AQUA = 200
RK = ["daynight", "region"]


def window3(t, keys):
    """Sum over the centred 3-month window (wraps Dec/Jan)."""
    parts = []
    for shift in (-1, 0, 1):
        s = t.copy()
        s["cmonth"] = (s["cmonth"] - 1 + shift) % 12 + 1
        parts.append(s)
    return pd.concat(parts).groupby(keys + ["cmonth"], as_index=False)[["snpp", "aqua"]].sum()


def table_ratios(tr):
    """Ratio tables from training rows (columns daynight, region, cmonth, snpp, aqua)."""
    reg = tr.groupby(RK + ["cmonth"], as_index=False)[["snpp", "aqua"]].sum()
    b1 = reg.assign(r1=reg.snpp / reg.aqua.where(reg.aqua >= MIN_AQUA))[RK + ["cmonth", "r1"]]
    w3 = window3(reg, RK)
    b3 = w3.assign(r3=w3.snpp / w3.aqua.where(w3.aqua >= MIN_AQUA))[RK + ["cmonth", "r3"]]
    dr = tr.groupby(RK, as_index=False)[["snpp", "aqua"]].sum()
    dr = dr.assign(rdr=dr.snpp / dr.aqua.where(dr.aqua >= MIN_AQUA))[RK + ["rdr"]]
    d = tr.groupby("daynight", as_index=False)[["snpp", "aqua"]].sum()
    d = d.assign(rd=d.snpp / d.aqua)[["daynight", "rd"]]
    return b1, b3, dr, d


def apply_table(te, tabs):
    """Converted count and the ratio used, per row of te."""
    b1, b3, dr, d = tabs
    x = te.merge(b1, on=RK + ["cmonth"], how="left").merge(b3, on=RK + ["cmonth"], how="left")
    x = x.merge(dr, on=RK, how="left").merge(d, on="daynight", how="left")
    tail = x["rdr"].fillna(x["rd"])
    r3fb = x["r3"].fillna(tail)
    rfb = x["r1"].fillna(r3fb)
    ratio = np.where(x["region"].isin(TABLE_3M), r3fb, rfb)
    return (x["aqua"] * ratio).to_numpy(), ratio
