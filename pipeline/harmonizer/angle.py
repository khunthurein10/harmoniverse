"""
MODIS viewing angle from the FIRMS 'scan' field (problem P6, test E6).

FIRMS gives each pixel's along-scan size in km: 1.0 straight down, about 4.8
at the edge of the strip. Inverting the satellite geometry turns that into the
scan angle. VIIRS keeps its pixels nearly even across the strip (Roman 2024),
so this only applies to MODIS.

Check: scan angle 30 degrees (the MRBA60 cut-off) is FIRMS scan of about 1.42 km.
"""
import numpy as np

EARTH_RADIUS_KM = 6371.0
MODIS_ALTITUDE_KM = 705.0

_theta = np.radians(np.linspace(0.0, 55.0, 1101))
_k = (EARTH_RADIUS_KM + MODIS_ALTITUDE_KM) / EARTH_RADIUS_KM
_scan_growth = EARTH_RADIUS_KM * (
    _k * np.cos(_theta) / np.sqrt(1 - (_k * np.sin(_theta)) ** 2) - 1
) / MODIS_ALTITUDE_KM


def modis_scan_angle(scan_km):
    """Scan angle in degrees for MODIS pixels, from FIRMS 'scan' (km)."""
    s = np.clip(np.asarray(scan_km, dtype="float64"), 1.0, _scan_growth[-1])
    return np.degrees(np.interp(s, _scan_growth, _theta))


def is_centre(sensor, scan_km, max_angle_deg=30.0):
    """True for MODIS pixels within max_angle_deg of straight down, and for all VIIRS pixels."""
    sensor = np.asarray(sensor)
    angle = modis_scan_angle(scan_km)
    return (sensor != "MODIS") | (angle <= max_angle_deg)


# ---------------------------------------------------------------------------
# Edge weights (test E6, option V3), adapted from GFED5NRT supplement Sect. S2:
#   wgt(s) = beta / FireCount(s), s = viewing position across the swath, per day/night x month x land.
# Over many years fires fall evenly across the swath, so a position that sees fewer detections is
# under-counting and gets a larger weight. GFED5NRT did this for VIIRS (sample number 0 to 6399) and
# kept total FRP fixed. Our adaptation (our choice): MODIS, position = scan angle from FIRMS 'scan'
# (verified: edge pixel ground area about 9.7 x nadir, Giglio et al. 2025 RSE Sect. 3.1), each 0.1 km
# scan bin weighted by the ground width it covers, beta keeps the total COUNT fixed (we convert counts).
# Thin strata or bins borrow the satellite x day/night table.
# ---------------------------------------------------------------------------
import pandas as pd

MIN_STRATUM, MIN_BIN = 2000, 20     # our choice: below these, use the coarser table


def scan_bin(scan_km):
    return np.round(np.asarray(scan_km, dtype="float64") * 10).astype("int64")   # FIRMS scan has 0.1 km steps


def ground_distance_km(angle_deg):
    """Distance on the ground from the track to a pixel seen at this scan angle (spherical Earth)."""
    t = np.radians(np.asarray(angle_deg, dtype="float64"))
    return EARTH_RADIUS_KM * (np.arcsin(np.clip(_k * np.sin(t), -1, 1)) - t)


def _bin_width(bins):
    """Width of ground (km across the swath) covered by each 0.1 km scan bin. GFED5NRT flattens counts per
    VIIRS sample, whose ground size is nearly constant; the MODIS equivalent is per km of ground (our choice)."""
    lo = ground_distance_km(modis_scan_angle(np.maximum(bins / 10 - 0.05, 1.0)))
    hi = ground_distance_km(modis_scan_angle(bins / 10 + 0.05))
    return np.maximum(hi - lo, 1e-6)


def _weights(counts, keys):
    """counts: DataFrame with keys + bin + n. Returns keys + bin + weight (count-conserving)."""
    c = counts.copy()
    c["width"] = _bin_width(c["bin"].to_numpy())
    tot = c.groupby(keys)[["n", "width"]].transform("sum")
    c["weight"] = (tot["n"] / tot["width"]) * c["width"] / c["n"]
    c["stratum_n"] = tot["n"]
    return c


def build_edge_weights(frames):
    """frames: iterable of DataFrames with satellite, daynight, cmonth, land, scan (MODIS kept rows).
    Returns lookup table: satellite, daynight, cmonth, land, bin, weight."""
    n = (pd.concat([f.assign(bin=scan_bin(f["scan"]))[["satellite", "daynight", "cmonth", "land", "bin"]] for f in frames])
         .value_counts().rename("n").reset_index())
    fine_keys, coarse_keys = ["satellite", "daynight", "cmonth", "land"], ["satellite", "daynight"]
    fine = _weights(n, fine_keys)
    coarse = _weights(n.groupby(coarse_keys + ["bin"], as_index=False)["n"].sum(), coarse_keys)
    fine = fine.merge(coarse[coarse_keys + ["bin", "weight"]].rename(columns={"weight": "w_coarse"}),
                      on=coarse_keys + ["bin"], how="left")
    thin = (fine["stratum_n"] < MIN_STRATUM) | (fine["n"] < MIN_BIN)
    fine.loc[thin, "weight"] = fine.loc[thin, "w_coarse"]
    fine["source"] = np.where(thin, "satellite x day/night", "fine")
    return fine[fine_keys + ["bin", "n", "weight", "source"]], coarse[coarse_keys + ["bin", "n", "weight"]]


def edge_weight(df, fine, coarse, land_codes, cmonth):
    """Weight per MODIS row (1.0 for VIIRS or anything missing from the tables)."""
    w = np.ones(len(df))
    is_modis = (df["sensor"] == "MODIS").to_numpy()
    if not is_modis.any():
        return w
    k = pd.DataFrame({"satellite": df["satellite"].to_numpy(), "daynight": df["daynight"].to_numpy(),
                      "cmonth": cmonth, "land": land_codes, "bin": scan_bin(df["scan"])})
    k = k.merge(fine[["satellite", "daynight", "cmonth", "land", "bin", "weight"]], how="left",
                on=["satellite", "daynight", "cmonth", "land", "bin"])
    k = k.merge(coarse.rename(columns={"weight": "wc"})[["satellite", "daynight", "bin", "wc"]], how="left",
                on=["satellite", "daynight", "bin"])
    out = k["weight"].fillna(k["wc"]).fillna(1.0).to_numpy()
    return np.where(is_modis, out, 1.0)
