"""
Grid cells and monthly counts (problem P1).

Cell ids come from one function, cell_index(), so the grid type can change
(0.25 degree squares now, hexagons in test E9) without touching other code.

Counting runs in two passes to keep memory low on a global year:
  1. cell_day_counts(): per country file, collapse hotspots to cell x day x satellite x day/night
  2. monthly_counts(): combine those small tables and roll up to months
Fire-days are counted after combining, so cells on a country border are not double counted.
"""
import numpy as np
import pandas as pd

from .angle import edge_weight, is_centre
from .groups import LAND_GROUPS, REGION_NAMES, land_of, region_of

EARTH_RADIUS_KM = 6371.0
DAY_KEYS = ["row", "col", "date", "sensor", "satellite", "daynight", "land"]
MONTH_KEYS = ["row", "col", "month", "sensor", "satellite", "daynight", "land"]


def cell_index(lat, lon, cell_deg=0.25, grid_type="latlon"):
    """Row/column of the grid cell containing each point."""
    if grid_type != "latlon":
        raise NotImplementedError("Only the lat/lon grid exists so far (hexagons are test E9).")
    row = np.floor((np.asarray(lat, dtype="float64") + 90.0) / cell_deg).astype("int32")
    col = np.floor((np.asarray(lon, dtype="float64") + 180.0) / cell_deg).astype("int32")
    n_rows, n_cols = int(round(180 / cell_deg)), int(round(360 / cell_deg))
    return np.clip(row, 0, n_rows - 1), np.clip(col, 0, n_cols - 1)


def cell_centre(row, col, cell_deg=0.25):
    lat = -90.0 + (np.asarray(row) + 0.5) * cell_deg
    lon = -180.0 + (np.asarray(col) + 0.5) * cell_deg
    return lat, lon


def cell_area_km2(row, cell_deg=0.25):
    """True area of a lat/lon cell. Shrinks toward the poles (about 770 km2 at the equator)."""
    lat_s = np.radians(-90.0 + np.asarray(row) * cell_deg)
    lat_n = lat_s + np.radians(cell_deg)
    return EARTH_RADIUS_KM ** 2 * np.radians(cell_deg) * (np.sin(lat_n) - np.sin(lat_s))


def cell_day_counts(clean, cell_deg=0.25, max_angle_deg=30.0, use_filters=True, date_basis="utc", land_grid=None,
                    edge_tables=None):
    """Pass 1: collapse cleaned hotspots to cell x day x satellite x day/night."""
    df = clean
    if use_filters:
        # MODIS: drop confidence < 30% (MRBA60). VIIRS: keep low confidence for now and count it
        # separately (n_det_low), so the target can leave it out (decision A2, tested in E2).
        df = df[df["keep_type"] & (df["keep_conf"] | (df["sensor"] == "VIIRS"))]
    df = df.dropna(subset=["lat", "lon", "acq_utc", "satellite"])
    # Day basis (decided: UTC, the FIRMS acq_date/acq_time; local solar date kept in the clean files as a clue)
    day = df["solar_date"] if date_basis == "local_solar" else df["acq_utc"].dt.tz_convert(None).dt.normalize()
    if df.empty:
        return pd.DataFrame(columns=DAY_KEYS + ["n_det", "n_det_low", "n_det_centre", "n_det_w", "frp_sum", "pix_area_sum"])

    row, col = cell_index(df["lat"], df["lon"], cell_deg)
    land = (land_of(df["lat"], df["lon"], land_grid) if land_grid is not None
            else np.full(len(df), LAND_GROUPS.index("other"), dtype="uint8"))
    # E6 option V3: MODIS edge weights (1.0 for VIIRS or when no table is given)
    w = (edge_weight(df, edge_tables[0], edge_tables[1], land, day.dt.month.to_numpy())
         if edge_tables is not None else np.ones(len(df)))
    work = pd.DataFrame({
        "row": row, "col": col,
        "date": day.to_numpy(),
        "sensor": df["sensor"].to_numpy(),
        "satellite": df["satellite"].to_numpy(),
        "daynight": df["daynight"].to_numpy(),
        # land type of the 0.05 degree MCD12C1 cell holding each hotspot
        "land": land,
        "w": w,
        "centre": is_centre(df["sensor"].to_numpy(), df["scan"].to_numpy(), max_angle_deg),
        "low": (df["conf_class"] == "low").fillna(False).to_numpy(dtype="int64"),
        "frp": df["frp"].to_numpy(dtype="float64"),
        "pix_area": (df["scan"] * df["track"]).to_numpy(dtype="float64"),
    })
    return (work.groupby(DAY_KEYS, observed=True, dropna=False)
            .agg(n_det=("frp", "size"), n_det_low=("low", "sum"), n_det_centre=("centre", "sum"), n_det_w=("w", "sum"),
                 frp_sum=("frp", "sum"), pix_area_sum=("pix_area", "sum"))
            .reset_index())


def monthly_counts(cell_days, cell_deg=0.25, region_grid=None):
    """Pass 2: combine cell-day tables (any number of countries/years) and roll up to months."""
    cd = (cell_days.groupby(DAY_KEYS, observed=True, dropna=False)
          [["n_det", "n_det_low", "n_det_centre", "n_det_w", "frp_sum", "pix_area_sum"]].sum().reset_index())
    cd["day_nolow"] = (cd["n_det"] - cd["n_det_low"]) > 0
    cd["month"] = pd.to_datetime(cd["date"]).dt.to_period("M").dt.to_timestamp()
    m = (cd.groupby(MONTH_KEYS, observed=True, dropna=False)
         .agg(n_det=("n_det", "sum"), n_det_low=("n_det_low", "sum"), n_det_centre=("n_det_centre", "sum"),
              n_det_w=("n_det_w", "sum"),
              n_firedays=("date", "nunique"), n_firedays_nolow=("day_nolow", "sum"), frp_sum=("frp_sum", "sum"),
              pix_area_sum=("pix_area_sum", "sum"))
         .reset_index())
    m["pix_area_mean"] = m["pix_area_sum"] / m["n_det"]
    m["cell_lat"], m["cell_lon"] = cell_centre(m["row"], m["col"], cell_deg)
    m["land"] = np.asarray(LAND_GROUPS, dtype=object)[m["land"].astype(int)]
    if region_grid is not None:
        m["region"] = np.vectorize(REGION_NAMES.get)(region_of(m["row"], m["col"], region_grid))
    m["area_km2"] = cell_area_km2(m["row"], cell_deg)
    m["det_per_1000km2"] = 1000.0 * m["n_det"] / m["area_km2"]
    for c in ["n_det", "n_det_low", "n_det_centre", "n_firedays", "n_firedays_nolow"]:
        m[c] = m[c].astype("int64")
    return m.drop(columns="pix_area_sum")
