"""
GFED5.1 burned area by fire type, on the same 0.25 degree grid as our boxes.

Source: GFED5.1 ecosystem files (van der Werf et al. 2025), GFED5.1_ecosystem_<year>.nc, 2002 to 2022,
group burned_area_partitioning, variables BA_01_Tundra ... BA_16_Deforestation (m2 per month), shape
(12, 720, 1440), lat ascending (row 0 = south), lon from -180 (same rows and columns as grid.cell_index()).

The 16 classes are put in the 9 groups of GFED5NRT (Chen et al. 2026, Fig. 3 caption).
This is burned area, not hotspots: it says what kind of land burned in a box, as modelled by GFED.
"""
import zipfile
from pathlib import Path

import numpy as np

GFED5_ZIP = "GFED5.1_ecosystem.zip"
GFED5_YEARS = range(2002, 2023)

# GFED5NRT Fig. 3 caption: BoreFor = Tundra + Sparse boreal forest + Boreal forest; Shrub = temperate shrubland,
# temperate mosaic, tropical shrubland, woody savanna; Grass = temperate grassland, tropical grassland, open savanna
FIRE_GROUPS = {
    "boreal_forest": [1, 2, 3],
    "temperate_forest": [7],
    "tropical_forest": [12],
    "deforestation": [16],
    "peat": [15],
    "shrub": [5, 6, 9, 11],
    "grass": [4, 8, 10],
    "cropland": [14],
    "other": [13],
}


def _open(ancillary_dir, year):
    import netCDF4
    z = zipfile.ZipFile(Path(ancillary_dir) / GFED5_ZIP)
    data = z.read(f"GFED5.1_ecosystem_{year}.nc")
    return netCDF4.Dataset("inmemory.nc", memory=data)


def yearly_groups(ancillary_dir, year):
    """Burned area per fire group for one year: dict group -> (12, 720, 1440) float32 km2."""
    d = _open(ancillary_dir, year)
    part = d["burned_area_partitioning"]
    names = {int(v[3:5]): v for v in part.variables if v.startswith("BA_")}
    out = {}
    for g, classes in FIRE_GROUPS.items():
        out[g] = sum(np.asarray(part[names[k]][:], dtype="float32") for k in classes) / 1e6
    d.close()
    return out


def basis_regions(ancillary_dir, year=2003):
    d = _open(ancillary_dir, year)
    r = np.asarray(d["basis_regions"][:]).astype("int16")
    d.close()
    return r


def static_groups(ancillary_dir, cache_dir, years=range(2003, 2023)):
    """Burned area per fire group summed over years: (9, 720, 1440) float32 km2, order of FIRE_GROUPS.
    Cached; a partial sum is kept so a stopped run resumes."""
    cache = Path(cache_dir) / f"gfed5_groups_{years[0]}_{years[-1]}.npy"
    if cache.exists():
        return np.load(cache)
    part = Path(cache_dir) / "gfed5_groups_partial.npz"
    total, done = np.zeros((len(FIRE_GROUPS), 720, 1440), dtype="float32"), []
    if part.exists():
        z = np.load(part)
        total, done = z["total"], list(z["done"])
    for y in years:
        if y in done:
            continue
        g = yearly_groups(ancillary_dir, y)
        total += np.stack([g[k].sum(0) for k in FIRE_GROUPS])
        done.append(y)
        np.savez(part, total=total, done=np.array(done))
        print(f"  GFED5.1 {y} read", flush=True)
    tmp = cache.with_name(cache.stem + "_writing.npy")
    np.save(tmp, total)          # write then rename, so a stopped run never leaves a broken cache
    tmp.replace(cache)
    try:
        part.unlink(missing_ok=True)
    except OSError:
        pass
    return total


# GFED5NRT (Chen et al. 2026), reprocessed daily ecosystem files, combined S-NPP + NOAA-20 ("CMB": simple average
# of the two), 2023 and 2024. Variable BA (16 classes, 1, 720, 1440), same 16 classes in the same order and the same
# grid as GFED5.1. Extract only Reprocessed/Daily/Ecosystem/CMB from GFED5NRT.tar.gz into data/ancillary/gfed5nrt/.
NRT_DIR = "gfed5nrt/Reprocessed/Daily/Ecosystem/CMB"
NRT_YEARS = range(2023, 2025)


def _nrt_year(ancillary_dir, year):
    """dict group -> (12, 720, 1440) km2, summed from the daily files."""
    import netCDF4
    files = sorted((Path(ancillary_dir) / NRT_DIR / str(year)).glob("GFED5NRTeco_CMB_*.nc"))
    if not files:
        raise FileNotFoundError(f"No GFED5NRT CMB files for {year} in {Path(ancillary_dir) / NRT_DIR}")
    month_ba = np.zeros((12, 16, 720, 1440), dtype="float32")
    for f in files:
        m = int(f.stem.split("_")[-1][5:7])
        with netCDF4.Dataset(f) as d:
            ba = np.asarray(d["BA"][:, 0], dtype="float32")
        month_ba[m - 1] += np.nan_to_num(ba)
    return {g: month_ba[:, [k - 1 for k in ks]].sum(1) / 1e6 for g, ks in FIRE_GROUPS.items()}


def monthly_by_box(ancillary_dir, cache_dir, year):
    """Burned area per fire group for every box-month with burned area in one year.
    Returns a DataFrame (row, col, month, <9 groups in km2>) or None when the year is not covered.
    2003 to 2022: GFED5.1; 2023 and 2024: GFED5NRT combined. Cached per year (small: burning boxes only)."""
    import pandas as pd
    if year not in GFED5_YEARS and year not in NRT_YEARS:
        return None
    cache = Path(cache_dir) / f"gfed5_box_month_{year}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    g = yearly_groups(ancillary_dir, year) if year in GFED5_YEARS else _nrt_year(ancillary_dir, year)
    stack = np.stack([g[k] for k in FIRE_GROUPS])              # (9, 12, 720, 1440)
    m, r, c = np.nonzero(stack.sum(0) > 0)
    df = pd.DataFrame({"row": r.astype("int16"), "col": c.astype("int16"), "month": (m + 1).astype("int8")})
    for i, k in enumerate(FIRE_GROUPS):
        df[k] = stack[i, m, r, c].astype("float32")
    tmp = cache.with_name(cache.stem + "_writing.parquet")
    df.to_parquet(tmp, index=False)
    tmp.replace(cache)
    return df
