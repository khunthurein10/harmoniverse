"""
Country and province/state label for every 0.25 degree box.

Source: Natural Earth 1:10m "Admin 1 - States, Provinces" (public domain, naturalearthdata.com).
Each province shape also carries its country, so the two labels always agree.
Natural Earth draws disputed borders as controlled on the ground (for example Crimea is labelled Russia).

How a box is labelled (our choice):
  1. the province whose shape contains the box centre;
  2. if the centre is in the sea (coastal box), the nearest province within one box width (0.25 degree);
  3. otherwise left empty (offshore, e.g. gas platforms).
crosses_border is True when the box square touches more than one country, so the label covers only part of it.

Check (run on 2019 S-NPP): the label agrees with the FIRMS country file the detections came from for
98.9% of boxes (99.5% weighted by hotspots); the rest are border boxes and disputed areas.
"""
from pathlib import Path

import numpy as np
import pandas as pd

NE_FILE = "ne_10m_admin_1_states_provinces.zip"


def _read_provinces(ancillary_dir):
    import geopandas as gpd
    src = Path(ancillary_dir) / NE_FILE
    if not src.exists():   # also accept the unzipped folder
        src = Path(ancillary_dir) / "natural_earth" / "ne_10m_admin_1_states_provinces.shp"
    path = f"zip://{src.as_posix()}" if src.suffix == ".zip" else str(src)
    ne = gpd.read_file(path)[["admin", "adm0_a3", "name_en", "name", "geometry"]]
    ne["province"] = ne["name_en"].fillna(ne["name"])
    return ne.rename(columns={"admin": "country", "adm0_a3": "country_code"})[
        ["country", "country_code", "province", "geometry"]]


def label_boxes(lat, lon, ancillary_dir, cell_deg=0.25):
    """Return a DataFrame (latitude, longitude, country, country_code, province, crosses_border)."""
    import warnings
    import geopandas as gpd
    from shapely.geometry import box as square
    ne = _read_provinces(ancillary_dir)
    b = pd.DataFrame({"latitude": lat, "longitude": lon}).drop_duplicates().reset_index(drop=True)
    pts = gpd.GeoDataFrame(b, geometry=gpd.points_from_xy(b.longitude, b.latitude), crs=ne.crs)
    j = gpd.sjoin(pts, ne, how="left", predicate="within")
    j = j[~j.index.duplicated()]
    miss = j["country"].isna()
    if miss.any():
        with warnings.catch_warnings():   # distance in degrees is fine for a 0.25 degree cut-off
            warnings.simplefilter("ignore")
            n = gpd.sjoin_nearest(pts[miss], ne, how="left", max_distance=cell_deg)
        n = n[~n.index.duplicated()]
        j.loc[miss, ["country", "country_code", "province"]] = n[["country", "country_code", "province"]]
    h = cell_deg / 2
    sq = gpd.GeoDataFrame(b[[]], geometry=[square(x - h, y - h, x + h, y + h)
                                            for x, y in zip(b.longitude, b.latitude)], crs=ne.crs)
    touch = gpd.sjoin(sq, ne[["country", "geometry"]], how="inner", predicate="intersects")
    n_countries = touch.groupby(level=0)["country"].nunique()
    out = b.copy()
    out[["country", "country_code", "province"]] = j[["country", "country_code", "province"]].values
    out["crosses_border"] = n_countries.reindex(out.index).fillna(0).to_numpy() > 1
    return out


def load_place_labels(boxes_lat, boxes_lon, ancillary_dir, cache_dir):
    """Cached labels; rebuilt when new boxes appear."""
    cache = Path(cache_dir) / "place_labels.parquet"
    want = pd.DataFrame({"latitude": np.round(boxes_lat, 3), "longitude": np.round(boxes_lon, 3)}).drop_duplicates()
    if cache.exists():
        have = pd.read_parquet(cache)
        new = want.merge(have[["latitude", "longitude"]], how="left", indicator=True).query("_merge == 'left_only'")
        if new.empty:
            return have
        have = pd.concat([have, label_boxes(new.latitude, new.longitude, ancillary_dir)], ignore_index=True)
    else:
        have = label_boxes(want.latitude, want.longitude, ancillary_dir)
    cache.parent.mkdir(parents=True, exist_ok=True)
    have.to_parquet(cache, index=False)
    return have
