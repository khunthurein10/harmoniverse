"""
Region and land type for every hotspot.

Region: GFED basis regions, 14 continental regions + ocean, 0.25 degree map.
  Source file: GFED4.1s_<year>.hdf5, dataset /ancill/basis_regions (same map in every year).
  GFED5NRT (Chen et al. 2026, Sect. 2.4) uses "the same 14 continental-scale regions" as GFED5.

Land type: MODIS MCD12C1 v061, Majority_Land_Cover_Type_1 (IGBP), 0.05 degree, one map per year.
  GFED5NRT tags each detection with MCD12Q1 (500 m) and 16 classes (Sect. 2.3.1). We use the
  coarser MCD12C1 and 6 groups (decision L2, our choice), built from GFED5 Table 2
  (van der Werf et al. 2025). Climate (boreal / temperate / tropical) comes from the region.
"""
from pathlib import Path

import numpy as np

LAND_GROUPS = ["forest", "woody_savanna", "savanna_grass_shrub", "cropland", "wetland", "other"]

# IGBP class (MCD12C1 file attributes) -> our group
IGBP_TO_GROUP = {
    1: "forest", 2: "forest", 3: "forest", 4: "forest", 5: "forest",   # GFED5 forest classes, merged (our choice)
    8: "woody_savanna",                                                # GFED5 keeps woody savanna separate
    6: "savanna_grass_shrub", 7: "savanna_grass_shrub",                # GFED5 "Savanna & grassland" group
    9: "savanna_grass_shrub", 10: "savanna_grass_shrub",
    12: "cropland", 14: "cropland",                                    # GFED5 Table 2 footnote 2
    11: "wetland",                                                     # our choice: closest IGBP class to peatland
    0: "other", 13: "other", 15: "other", 16: "other",                 # GFED5 Table 2 footnote 1
    255: "other",                                                      # unclassified fill value (our choice)
}

REGION_NAMES = {0: "OCEAN", 1: "BONA", 2: "TENA", 3: "CEAM", 4: "NHSA", 5: "SHSA", 6: "EURO", 7: "MIDE",
                8: "NHAF", 9: "SHAF", 10: "BOAS", 11: "CEAS", 12: "SEAS", 13: "EQAS", 14: "AUST"}


def load_regions(ancillary_dir, cache_dir):
    """GFED regions as a (720, 1440) array with row 0 at the south pole, like grid.cell_index()."""
    cache = Path(cache_dir) / "gfed_regions.npy"
    if cache.exists():
        return np.load(cache)
    import h5py
    files = sorted(Path(ancillary_dir).glob("GFED4.1s_*.hdf5"))
    if not files:
        raise FileNotFoundError(f"No GFED4.1s_*.hdf5 in {ancillary_dir}")
    with h5py.File(files[0], "r") as f:
        reg = f["ancill/basis_regions"][:]
        if f["lat"][0, 0] > 0:          # file rows run north to south
            reg = reg[::-1]
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, reg)
    return reg


def load_land(ancillary_dir, cache_dir, year):
    """Land group codes (index into LAND_GROUPS) as a (3600, 7200) array, row 0 at the south pole.
    Uses that year's MCD12C1 map, or the nearest year available (GFED5NRT Sect. 2.3.4 uses the latest
    available year for years without a map)."""
    files = {int(p.name.split(".")[1][1:5]): p for p in Path(ancillary_dir).glob("MCD12C1.A*.hdf")}
    cached = {int(p.stem.rsplit("_", 1)[1]): p for p in Path(cache_dir).glob("mcd12c1_groups_*.npy")}
    years = set(files) | set(cached)       # the cached maps are enough once built (the .hdf files can be removed)
    if not years:
        raise FileNotFoundError(f"No MCD12C1 files in {ancillary_dir} and no cached maps in {cache_dir}")
    use = min(years, key=lambda y: (abs(y - year), -y))
    cache = Path(cache_dir) / f"mcd12c1_groups_{use}.npy"
    if cache.exists():
        return np.load(cache)
    from pyhdf.SD import SD, SDC
    igbp = SD(str(files[use]), SDC.READ).select("Majority_Land_Cover_Type_1").get()[::-1]   # file rows run north to south
    lut = np.full(256, LAND_GROUPS.index("other"), dtype="uint8")
    for k, g in IGBP_TO_GROUP.items():
        lut[k] = LAND_GROUPS.index(g)
    groups = lut[igbp]
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, groups)
    return groups


def land_of(lat, lon, land_grid, cell_deg=0.05):
    """Land group code for each point."""
    r = np.clip(np.floor((np.asarray(lat, dtype="float64") + 90.0) / cell_deg).astype("int64"), 0, land_grid.shape[0] - 1)
    c = np.clip(np.floor((np.asarray(lon, dtype="float64") + 180.0) / cell_deg).astype("int64"), 0, land_grid.shape[1] - 1)
    return land_grid[r, c]


def region_of(row, col, region_grid):
    """GFED region code for 0.25 degree grid cells (row/col from grid.cell_index)."""
    return region_grid[np.asarray(row), np.asarray(col)]
