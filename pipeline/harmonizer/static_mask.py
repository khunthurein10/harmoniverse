"""
VIIRS static-source mask (gas flares, volcanoes, industry).

Why: FIRMS VIIRS rows are version 2, where the type field often labels static
sources as type 0 (VIIRS C2 375 m User Guide, Section 6.1). Keeping type 0 only
is therefore not enough for VIIRS.

A 0.01 degree cell is masked for a calendar year if either rule holds:
  Rule 1 (GFED5NRT, Chen et al. 2026, Section 2.3.2): more than five detections
         classified as non-vegetation (type != 0).
  Rule 2 (our choice, switchable): detected on 16+ days (MODIS C6 User Guide
         Table 10 definition of a static source) spread over 7+ months
         (our choice, set from real 2012 FIRMS cells already caught by rule 1).
  Rule 2b (our choice, switchable): detected on 60+ days in the year, whatever the
         months. Catches sources that switch on mid-year (e.g. a 2012 Algeria cell:
         105 days within Sep to Dec). Real seasonal fire cells in 2012 stayed under 45 days.
Counts use every row of the cleaned year, whatever its flags.
"""
import numpy as np
import pandas as pd


def fine_cell(lat, lon, cell_deg=0.01):
    """Integer id of the 0.01 degree cell holding each point."""
    r = np.floor((np.asarray(lat, dtype="float64") + 90.0) / cell_deg).astype("int64")
    c = np.floor((np.asarray(lon, dtype="float64") + 180.0) / cell_deg).astype("int64")
    return r * 100000 + c


def build_static_mask(files, cell_deg=0.01, min_nonveg=6, use_persistence=True, min_days=16, min_months=7,
                      use_long_run=True, long_run_days=60):
    """Return one row per masked cell: cell, n_nonveg, n_days, n_months, rule1, rule2, rule2b."""
    parts = []
    for f in files:
        d = pd.read_parquet(f, columns=["lat", "lon", "acq_utc", "type"])
        d = d.dropna(subset=["lat", "lon", "acq_utc"])
        g = pd.DataFrame({
            "cell": fine_cell(d["lat"], d["lon"], cell_deg),
            "date": d["acq_utc"].dt.floor("D"),
            "nonveg": (d["type"].fillna(0) != 0).astype("int32"),
        })
        parts.append(g.groupby(["cell", "date"], sort=False)["nonveg"].sum().reset_index())
    if not parts:
        return pd.DataFrame(columns=["cell", "n_nonveg", "n_days", "n_months", "rule1", "rule2", "rule2b"])
    cd = pd.concat(parts, ignore_index=True).groupby(["cell", "date"], sort=False)["nonveg"].sum().reset_index()
    cd["month"] = cd["date"].dt.month
    cells = cd.groupby("cell").agg(n_nonveg=("nonveg", "sum"), n_days=("date", "nunique"), n_months=("month", "nunique"))
    cells["rule1"] = cells["n_nonveg"] >= min_nonveg
    cells["rule2"] = use_persistence & (cells["n_days"] >= min_days) & (cells["n_months"] >= min_months)
    cells["rule2b"] = use_long_run & (cells["n_days"] >= long_run_days)
    return cells[cells["rule1"] | cells["rule2"] | cells["rule2b"]].reset_index()


def in_mask(df, mask_cells, cell_deg=0.01):
    """Boolean Series: True where a row falls in a masked cell."""
    return pd.Series(np.isin(fine_cell(df["lat"], df["lon"], cell_deg), mask_cells), index=df.index)
