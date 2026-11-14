"""
Step 4: export the harmonized record as CSV for the web app (and for download).

Reads  data/processed/harmonized_region_month.parquet, data/processed/harmonized_box/<year>.parquet
Writes data/processed/export/
         harmonized_region_month.csv          one row per GFED region x month x day/night (D, N, ALL)
         harmonized_box_month_<year>.csv      one row per 0.25 degree box x month (day and night as columns)
         harmonized_country_month.csv         one row per country x month (sum of boxes)
         harmonized_province_month.csv        one row per province/state x month (sum of boxes)
         conversion_lookup_table.csv          S-NPP / Aqua ratio per region x day/night x month (the explainable table)
         grid_cells.csv                       fixed labels per box for map colouring and filters
         filled_days.csv                      S-NPP months with missing or partial days and how they were filled
         README.md                            files, columns, fields, method, caveats, credits
       data/processed/export_zip/  the same files zipped (one zip per CSV), small enough to commit
All counts are hotspots on the S-NPP VIIRS scale (375 m, low confidence dropped, static sources masked).
Needs the GFED5.1 zip and the GFED5NRT CMB folder in data/ancillary for the fire-type columns (see the main README).
Usage: python pipeline/scripts/04_export.py [--years 2026]
       --years writes only those years' box CSVs (the regional CSV and the zips are always rebuilt)
"""
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harmonizer.config import load_settings  # noqa: E402
from harmonizer.places import load_place_labels  # noqa: E402
from harmonizer.groups import LAND_GROUPS, REGION_NAMES as REGION_CODES, load_land  # noqa: E402

REGION_NAMES = {"BONA": "Boreal North America", "TENA": "Temperate North America", "CEAM": "Central America",
                "NHSA": "Northern Hemisphere South America", "SHSA": "Southern Hemisphere South America",
                "EURO": "Europe", "MIDE": "Middle East", "NHAF": "Northern Hemisphere Africa",
                "SHAF": "Southern Hemisphere Africa", "BOAS": "Boreal Asia", "CEAS": "Central Asia",
                "SEAS": "Southeast Asia", "EQAS": "Equatorial Asia", "AUST": "Australia and New Zealand"}
Z90 = 1.645
NO_MODIS_FROM = 2026   # MODIS not used from 2026 (Terra/Aqua orbit drift); raw MODIS columns left blank
FLAGS = ["outage_aqua", "outage_terra", "outage_snpp"]


def region_csv(src, dst):
    r = pd.read_parquet(src)
    r["year"], r["month_num"] = r["month"].dt.year, r["month"].dt.month
    base = r[["region", "year", "month_num", "daynight", "aqua", "terra", "snpp_obs", "harmonized", "sigma",
              "viirs_only_added", "source", "days_filled"] + FLAGS].copy()
    agg = {"aqua": "sum", "terra": "sum", "snpp_obs": lambda s: s.sum(min_count=1), "harmonized": "sum",
           "viirs_only_added": "sum", "source": lambda s: " / ".join(sorted(set(s))), "days_filled": "max"}
    agg.update({f: "max" for f in FLAGS})
    tot = base.groupby(["region", "year", "month_num"], as_index=False).agg(
        {**agg, "sigma": lambda s: np.sqrt((s.fillna(0) ** 2).sum()) if s.notna().any() else np.nan})
    tot["daynight"] = "ALL"
    out = pd.concat([base, tot], ignore_index=True)
    out["low90"] = np.maximum(out["harmonized"] - Z90 * out["sigma"].fillna(0), 0)
    out["high90"] = out["harmonized"] + Z90 * out["sigma"].fillna(0)
    out["region_name"] = out["region"].map(REGION_NAMES)
    out = out.rename(columns={"month_num": "month", "aqua": "aqua_raw", "terra": "terra_raw",
                              "snpp_obs": "snpp_observed", "sigma": "uncertainty_rmse"})
    cols = ["region", "region_name", "year", "month", "daynight", "harmonized", "low90", "high90", "uncertainty_rmse",
            "aqua_raw", "terra_raw", "snpp_observed", "viirs_only_added", "source", "days_filled"] + FLAGS
    out.loc[out["year"] >= NO_MODIS_FROM, ["aqua_raw", "terra_raw"]] = np.nan
    out = out[cols].sort_values(["region", "year", "month", "daynight"])
    for c in ["harmonized", "low90", "high90", "uncertainty_rmse", "viirs_only_added"]:
        out[c] = out[c].round(1)
    out.to_csv(dst, index=False)
    return len(out)


def join_labels(df, keys, col):
    """Per group, the distinct labels in `col` joined as "a / b" (sorted). Fast path: most groups have one label."""
    u = df[keys + [col]].drop_duplicates()
    cnt = u.groupby(keys)[col].transform("size")
    single = u[cnt == 1].set_index(keys)[col]
    multi = u[cnt > 1].sort_values(col).groupby(keys)[col].agg(" / ".join)
    return pd.concat([single, multi])


def sum_min1(df, keys, col):
    """Group sum that stays blank when every value in the group is blank (like sum(min_count=1)), vectorized."""
    return df.groupby(keys)[col].sum(min_count=1)


def add_fire_type(g, year, places_dir, cell_deg=0.25):
    """GFED fire type for each box-month (2003 to 2024): the group with most GFED burned area in that box and month,
    its share, and the box's GFED burned area. Blank outside 2003 to 2024 or where GFED has no burned area."""
    from harmonizer.gfed5 import FIRE_GROUPS, monthly_by_box
    g["fire_type"], g["fire_type_share"], g["gfed_burned_km2"] = "", np.nan, np.nan
    fb = monthly_by_box(*places_dir, year)
    if fb is None:
        return g
    names = list(FIRE_GROUPS)
    vals = fb[names].to_numpy()
    tot = vals.sum(1)
    fb = pd.DataFrame({"row": fb["row"].astype(int), "col": fb["col"].astype(int), "month": fb["month"].astype(int),
                       "ft": np.array(names)[vals.argmax(1)], "fs": (vals.max(1) / tot).round(2), "ba": tot.round(3)})
    key = pd.DataFrame({"row": np.floor((g["latitude"].to_numpy() + 90) / cell_deg).astype(int),
                        "col": np.floor((g["longitude"].to_numpy() + 180) / cell_deg).astype(int),
                        "month": g["month"].astype(int).to_numpy()})
    m = key.merge(fb, on=["row", "col", "month"], how="left")
    g["fire_type"] = m["ft"].fillna("").to_numpy()
    g["fire_type_share"] = m["fs"].to_numpy()
    g["gfed_burned_km2"] = m["ba"].to_numpy()
    return g


def box_csv(src, dst, mshare, places_dir):
    """mshare: VIIRS-only share m per (daynight, region, cmonth). places_dir: (ancillary, lookup cache) folders."""
    b = pd.read_parquet(src)
    # Converted rows: spread the fires MODIS never located (share m, same add-back as the regional file,
    # MRBA60 Fire Mask idea) over the converted boxes in proportion, so boxes add up to the region total and a
    # drawn area or country has no step at 2012 (our choice; exact for large areas, approximate for single boxes).
    conv = b["source"].str.contains("converted")
    key = pd.MultiIndex.from_arrays([b["daynight"], b["region"], b["month"].dt.month])
    m = pd.Series(mshare.reindex(key).to_numpy(), index=b.index).fillna(0)
    factor = np.where(conv, 1 / (1 - m), 1.0)
    b["viirs_only_added"] = b["harmonized"] * (factor - 1)
    b["harmonized"] = b["harmonized"] * factor
    b["pred_sigma"] = b["pred_sigma"] * factor
    for c in FLAGS + ["above_training_range"]:
        if c not in b:
            b[c] = False
    if "land_map_year" not in b:
        b["land_map_year"] = np.nan
    b["year"], b["month_num"] = b["month"].dt.year, b["month"].dt.month
    keys = ["row", "col", "year", "month_num"]
    # main land type: the land group holding most harmonized hotspots in this box-month (ties: most Aqua)
    lt = b.sort_values(["harmonized", "aqua"], ascending=False).drop_duplicates(keys)[keys + ["land"]]
    dn = b.pivot_table(index=keys, columns="daynight", values="harmonized", aggfunc="sum", fill_value=0)
    dn.columns = [f"harmonized_{'day' if c == 'D' else 'night'}" for c in dn.columns]
    b["sig2"] = b["pred_sigma"] ** 2
    g = b.groupby(keys).agg(latitude=("cell_lat", "first"), longitude=("cell_lon", "first"), region=("region", "first"),
                            aqua_raw=("aqua", "sum"), terra_raw=("terra", "sum"),
                            harmonized=("harmonized", "sum"), viirs_only_added=("viirs_only_added", "sum"),
                            uncertainty_rmse=("sig2", "sum"),
                            days_filled=("days_filled", "max"),
                            outage_aqua=("outage_aqua", "max"), outage_terra=("outage_terra", "max"),
                            outage_snpp=("outage_snpp", "max"), above_training_range=("above_training_range", "max"),
                            land_map_year=("land_map_year", "first"))
    g["uncertainty_rmse"] = np.sqrt(g["uncertainty_rmse"])
    g["snpp_observed"] = sum_min1(b, keys, "snpp_obs")
    g["source"] = join_labels(b, keys, "source")
    g = g.join(dn).reset_index().merge(lt, on=keys).rename(columns={"month_num": "month", "land": "land_type"})
    observed = g["source"].str.startswith("S-NPP observed") & ~g["source"].str.contains("converted|filled")
    g.loc[observed, "uncertainty_rmse"] = 0.0
    for c in ["latitude", "longitude"]:
        g[c] = g[c].round(3)
    g = add_fire_type(g, int(g["year"].iloc[0]), places_dir)
    pl = load_place_labels(g["latitude"], g["longitude"], *places_dir)
    g = g.merge(pl, on=["latitude", "longitude"], how="left")
    for c in ["harmonized", "harmonized_day", "harmonized_night", "uncertainty_rmse", "viirs_only_added"]:
        if c in g:
            g[c] = g[c].round(2)
    cols = ["latitude", "longitude", "year", "month", "country", "country_code", "province", "crosses_border",
            "region", "land_type", "fire_type", "fire_type_share", "gfed_burned_km2", "harmonized", "harmonized_day", "harmonized_night", "uncertainty_rmse",
            "aqua_raw", "terra_raw", "snpp_observed", "viirs_only_added", "source", "days_filled",
            "outage_aqua", "outage_terra", "outage_snpp", "above_training_range", "land_map_year"]
    g.loc[g["year"] >= NO_MODIS_FROM, ["aqua_raw", "terra_raw"]] = np.nan
    g = g[[c for c in cols if c in g.columns]].sort_values(["year", "month", "latitude", "longitude"])
    g.to_csv(dst, index=False)
    return len(g)


def lookup_csv(grid, dst, years=range(2015, 2020), min_aqua=200, table_3m=("BONA", "TENA")):
    """The lookup table (ratio of sums, GFED5NRT Sect. 3.1): S-NPP hotspots / Aqua hotspots, 2015 to 2019,
    per region x day/night x calendar month. Same rules as the pipeline's table: groups with fewer than
    min_aqua Aqua detections fall back to a 3-month window, then to region x day/night (MRBA60-style thin-group
    rule, our choice); BONA and TENA use the 3-month window (E8)."""
    cols = ["region", "month", "daynight", "satellite", "n_det", "n_det_low"]
    m = pd.concat([pd.read_parquet(grid / "monthly" / f"{s}_{y}.parquet", columns=cols)
                   for s in ["modis", "viirs_snpp"] for y in years], ignore_index=True)
    m = m[m["satellite"].isin(["AQUA", "SNPP"]) & (m["region"] != "OCEAN")]
    m["value"] = np.where(m["satellite"] == "SNPP", m["n_det"] - m["n_det_low"], m["n_det"])
    m["month"] = pd.to_datetime(m["month"]).dt.month
    t = m.pivot_table(index=["region", "daynight", "month"], columns="satellite", values="value",
                      aggfunc="sum", fill_value=0).reset_index().rename(columns={"AQUA": "aqua", "SNPP": "snpp"})
    shift = []
    for k in (-1, 0, 1):
        x = t[["region", "daynight", "month", "aqua", "snpp"]].copy()
        x["month"] = (x["month"] - 1 + k) % 12 + 1
        shift.append(x)
    w3 = pd.concat(shift).groupby(["region", "daynight", "month"], as_index=False)[["aqua", "snpp"]].sum()
    t = t.merge(w3.rename(columns={"aqua": "aqua_3m", "snpp": "snpp_3m"}), on=["region", "daynight", "month"])
    yr = t.groupby(["region", "daynight"], as_index=False)[["aqua", "snpp"]].sum().rename(
        columns={"aqua": "aqua_year", "snpp": "snpp_year"})
    t = t.merge(yr, on=["region", "daynight"])
    t["ratio_1month"] = (t["snpp"] / t["aqua"].where(t["aqua"] > 0)).round(2)
    t["ratio_3month"] = (t["snpp_3m"] / t["aqua_3m"].where(t["aqua_3m"] > 0)).round(2)
    t["ratio_year"] = (t["snpp_year"] / t["aqua_year"].where(t["aqua_year"] > 0)).round(2)
    use3 = t["region"].isin(table_3m)
    ok1, ok3 = t["aqua"] >= min_aqua, t["aqua_3m"] >= min_aqua
    t["ratio_used"] = np.select([~use3 & ok1, ok3], [t["ratio_1month"], t["ratio_3month"]], default=t["ratio_year"])
    t["ratio_level"] = np.select([~use3 & ok1, ok3], ["1 month", "3 month window"], default="whole year (thin data)")
    t["region_name"] = t["region"].map(REGION_NAMES)
    t = t.rename(columns={"aqua": "aqua_hotspots_2015_2019", "snpp": "snpp_hotspots_2015_2019"})
    cols = ["region", "region_name", "daynight", "month", "ratio_used", "ratio_level", "ratio_1month", "ratio_3month",
            "ratio_year", "aqua_hotspots_2015_2019", "snpp_hotspots_2015_2019"]
    t[cols].sort_values(["region", "daynight", "month"]).to_csv(dst, index=False)
    return len(t)


def filled_days_csv(grid, dst, years):
    """One row per S-NPP month with missing or partial days: which days, and how the month was filled (fill.py)."""
    from harmonizer.fill import day_status
    label = {"noaa20": "filled with NOAA-20 x same-month ratio", "scale": "scaled up from the month's good days",
             "convert": "converted from MODIS (too few S-NPP days)"}
    rows = []
    for y in years:
        st = day_status(grid, y, (grid / "monthly" / f"modis_{y}.parquet").exists())
        for _, r in st.iterrows():
            rows.append({"year": r["month"].year, "month": r["month"].month, "days_missing_or_partial": r["n_bad"],
                         "days": ",".join(str(d) for d in r["bad_days"]), "share_of_days_good": round(r["good_share"], 3),
                         "method": label[r["method"]]})
    pd.DataFrame(rows).to_csv(dst, index=False)
    return len(rows)


def place_csvs(box_files, out, region_file):
    """Country x month and province x month totals, summed from the box CSVs (all years).
    Uncertainty (our choice): each box takes its region's relative error for that month (day and night separately,
    from the region file), and errors are added up as if fully correlated. Errors inside a region come from the
    same conversion, so they move together; adding them gives a cautious (wide) range, never a falsely narrow one."""
    use = ["year", "month", "country", "country_code", "province", "region", "harmonized", "harmonized_day",
           "harmonized_night", "aqua_raw", "terra_raw", "snpp_observed", "source", "days_filled"]
    b = pd.concat([pd.read_csv(f, usecols=use) for f in box_files], ignore_index=True)
    b = b[b["country"].notna()]
    rg = pd.read_csv(region_file, usecols=["region", "year", "month", "daynight", "harmonized", "uncertainty_rmse"])
    rg = rg[rg["daynight"].isin(["D", "N"])]
    rg["rel"] = np.where(rg["harmonized"] > 0, rg["uncertainty_rmse"].fillna(0) / rg["harmonized"], 0.0)
    rel = rg.pivot_table(index=["region", "year", "month"], columns="daynight", values="rel", fill_value=0)
    rel.columns = [f"rel_{c}" for c in rel.columns]
    b = b.merge(rel.reset_index(), on=["region", "year", "month"], how="left")
    b["err"] = b["harmonized_day"] * b["rel_D"].fillna(0) + b["harmonized_night"] * b["rel_N"].fillna(0)
    b["kind"] = np.select([b["source"].str.contains("converted"), b["source"].str.contains("NOAA-20"),
                           b["source"].str.contains("gap-filled")],
                          ["converted", "S-NPP observed, filled with NOAA-20", "S-NPP observed, gap-filled"],
                          default="S-NPP observed")
    n = {}
    for level, keys, name in [("country", ["country", "country_code"], "harmonized_country_month.csv"),
                              ("province", ["country", "country_code", "province"], "harmonized_province_month.csv")]:
        gk = keys + ["year", "month"]
        bb = b.copy()
        bb[keys] = bb[keys].fillna("")             # province can be blank: keep those rows as their own group
        g = bb.groupby(gk).agg(
            harmonized=("harmonized", "sum"), harmonized_day=("harmonized_day", "sum"),
            harmonized_night=("harmonized_night", "sum"), n_boxes=("harmonized", "size"),
            uncertainty_rmse=("err", "sum"), days_filled=("days_filled", "max"))
        for c in ["aqua_raw", "terra_raw", "snpp_observed"]:
            g[c] = sum_min1(bb, gk, c)
        g["source"] = join_labels(bb, gk, "kind")
        g = g.reset_index()
        g[keys] = g[keys].replace("", np.nan)
        g["low90"] = np.maximum(g["harmonized"] - Z90 * g["uncertainty_rmse"], 0)
        g["high90"] = g["harmonized"] + Z90 * g["uncertainty_rmse"]
        first = keys + ["year", "month", "harmonized", "low90", "high90", "uncertainty_rmse"]
        rest = ["harmonized_day", "harmonized_night", "aqua_raw", "terra_raw", "snpp_observed", "n_boxes",
                "days_filled", "source"]
        g = g[first + rest]
        for c in ["harmonized", "harmonized_day", "harmonized_night", "low90", "high90", "uncertainty_rmse"]:
            g[c] = g[c].round(1)
        g.sort_values(keys + ["year", "month"]).to_csv(out / name, index=False)
        n[name] = len(g)
    return n


README = """# Harmonized burning activity data (S-NPP VIIRS scale), {y0} to {y1}

Every count is a number of active-fire hotspots expressed on the S-NPP VIIRS 375 m scale, so the satellite of origin
no longer matters. Before 2012 the values are converted from MODIS (Aqua + Terra); from 2012 they are real S-NPP counts.

## Files
One CSV per table (each also zipped). Columns are listed here; every column is explained under Fields.

### harmonized_region_month.csv
One row per GFED region x month x day/night (D, N, ALL = day + night). Use for regional calendars and charts.
Columns: region, region_name, year, month, daynight, harmonized, low90, high90, uncertainty_rmse, aqua_raw,
terra_raw, snpp_observed, viirs_only_added, source, days_filled, outage_aqua, outage_terra, outage_snpp

### harmonized_box_month_<year>.csv
One row per 0.25 degree box (about 28 km at the equator) x month, for maps and user-drawn areas.
Columns: latitude, longitude, year, month, country, country_code, province, crosses_border, region, land_type,
fire_type, fire_type_share, gfed_burned_km2, harmonized, harmonized_day, harmonized_night, uncertainty_rmse,
aqua_raw, terra_raw, snpp_observed, viirs_only_added, source, days_filled, outage_aqua, outage_terra, outage_snpp,
above_training_range, land_map_year

Drawn areas: sum the boxes whose centre is inside the area. Range for a drawn area (same rule as the country and
province files): for each box, error = harmonized_day x rel_D + harmonized_night x rel_N, where
rel = uncertainty_rmse / harmonized of the box's region that month in the region file (D and N rows); add the box
errors up, then low90 = max(sum - 1.645 x error, 0) and high90 = sum + 1.645 x error. Do not add up the box files'
own uncertainty_rmse: it is a single-box error.

### harmonized_country_month.csv, harmonized_province_month.csv
The boxes summed per country, and per province or state, per month. A box counts fully toward the place containing
its centre. low90/high90 use each box's region error for that month, added up as fully correlated (a cautious range).
Columns: country, country_code, (province), year, month, harmonized, low90, high90, uncertainty_rmse,
harmonized_day, harmonized_night, aqua_raw, terra_raw, snpp_observed, n_boxes, days_filled, source

### grid_cells.csv
One row per box that has hotspots anywhere in the data: the fixed labels a map needs to colour or filter it.
Join to the box files on latitude + longitude.
Columns: latitude, longitude, country, country_code, province, crosses_border, region, region_name, land_type,
land_type_share, fire_type, fire_type_share, gfed_burned_km2_per_year

### filled_days.csv
Every S-NPP month (2012 on) with missing or partial days. Matches the days_filled column.
Columns: year, month, days_missing_or_partial, days (the day numbers), share_of_days_good, method
(filled with NOAA-20 x same-month ratio / scaled up from the month's good days / converted from MODIS)

### conversion_lookup_table.csv
How many S-NPP hotspots one Aqua hotspot corresponds to, per region x day/night x calendar month (2015 to 2019).
The explainable version of the conversion: the main conversion is a model; this table is its baseline and fallback.
Columns: region, region_name, daynight, month, ratio_used (what the table applies), ratio_level (which pooling:
1 month, 3-month window or whole year), ratio_1month, ratio_3month, ratio_year, aqua_hotspots_2015_2019,
snpp_hotspots_2015_2019

## Fields
| field | meaning |
|---|---|
| latitude, longitude | centre of the 0.25 degree box |
| year, month | calendar month (UTC dates) |
| country, country_code, province | place containing the box centre (Natural Earth 1:10m Admin 1, public domain; ISO 3166-1 alpha-3 style code). Coastal boxes take the nearest province within 0.25 degree; a few offshore boxes are blank. Disputed borders follow Natural Earth (as controlled on the ground) |
| crosses_border | the box also covers part of another country |
| n_boxes | place files: number of boxes with hotspots that month |
| region, region_name | GFED basis region (14 regions), from the GFED4.1s map that the harmonization used. GFED5.1's own map differs for about 2,000 inland boxes along region borders and about 13,000 coastal boxes (0.4% of GFED5.1 burned area); we keep GFED4.1s so every file uses one map |
| land_type | forest, woody_savanna, savanna_grass_shrub, cropland, wetland, other (MODIS MCD12C1 IGBP, grouped after GFED5 Table 2); in box files the group with most hotspots that month |
| land_type_share | grid_cells.csv: share of the box area in land_type (25 MCD12C1 pixels per box, 2019 map). In grid_cells.csv land_type is the group covering most of the box; in box files it is the group with most hotspots that month, so the two can differ (they agree for about 87 to 91% of hotspots). "other" includes water, so coastal boxes can show "other" |
| fire_type | the kind of land that burned most, from GFED burned area, in the 9 groups of GFED5NRT (Chen et al. 2026, Fig. 3): boreal_forest (incl. tundra), temperate_forest, tropical_forest, deforestation, peat, shrub (incl. woody savanna), grass (incl. open savanna), cropland, other. Box files: that box and month, 2003 to 2022 from GFED5.1 (van der Werf et al. 2025), 2023 and 2024 from GFED5NRT (S-NPP + NOAA-20 combined); blank from 2025, and blank where GFED has no burned area that month (in 2022: about half of box-months, but only about 6% of hotspots). grid_cells.csv: the whole 2003 to 2022 GFED5.1 record per box. Modelled burned area, not hotspots: it describes what kind of land burns in the box, it does not label each hotspot. From 2023 far fewer box-months are blank (about 94% of box-months have a fire type, against about 50% in 2012 to 2022): GFED5NRT records burned area in more boxes than GFED5.1. This is a difference between the two products, not a change in fires. Across the switch the main type of a box stays the same as often as between other years (83% from 2022 to 2023, 85% within GFED5.1). |
| fire_type_share | share of the GFED burned area in fire_type (box-month, or 2003 to 2022 in grid_cells.csv) |
| gfed_burned_km2 | box files: GFED burned area in the box that month (km2) |
| gfed_burned_km2_per_year | mean yearly GFED5.1 burned area in the box, 2003 to 2022 (km2) |
| daynight | D (afternoon / 13:30 pass), N (night / 01:30 pass), ALL (sum) |
| harmonized, harmonized_day, harmonized_night | hotspot count on the S-NPP scale |
| low90, high90 | 90% range (assumes normal errors); equal to harmonized when fully observed |
| uncertainty_rmse | typical error (RMSE): converted months from the 2012 to 2014 test; filled months from E9 (scaling) or E10 (NOAA-20); 0 when fully observed |
| days_filled | number of missing or partial S-NPP days in that month that were filled (0 = none) |
| aqua_raw, terra_raw | original MODIS hotspot counts (Aqua, Terra) after cleaning |
| snpp_observed | real S-NPP count (blank before 2012) |
| viirs_only_added | converted months: fires added back for areas where MODIS saw nothing. Regional file: added at region level. Box file: the same share spread over the region's boxes in proportion (already included in harmonized) |
| source | how the value was made: S-NPP observed; S-NPP observed, filled with NOAA-20; S-NPP observed, gap-filled (scaled); converted (box files add model or table fallback); converted (S-NPP outage, ...). Place files list every kind present that month |
| outage_aqua, outage_terra, outage_snpp | that satellite had more than 2 days without data that month |
| above_training_range | count larger than the model learned from; the simpler table conversion was used |
| land_map_year | year of the land cover map used (maps cover 2012 to 2019; other years use the nearest) |

## How values were made
Short version; the full method, experiments and results are in the project report.
- Cleaning (NASA FIRMS standard processing): vegetation fires only (type 0); MODIS confidence below 30% removed;
  VIIRS low confidence removed; VIIRS static sources (flares, volcanoes, industry) masked per 0.01 degree cell.
- Conversion (2003 to 2011): gradient boosting models (one day, one night) trained on 2015 to 2019, using Aqua count,
  Aqua fire-days, Aqua mean FRP, Aqua pixel size, Terra count, region, land type, month and location. Tested on
  untouched years 2012 to 2014: regional monthly error 5.4% (fire-weighted), 12.6% at night, yearly regional totals 2.0%,
  single box 26%. Fires in boxes MODIS missed are added back per region and spread over its boxes in proportion.
- Uncertainty: error measured on 2012 to 2014 (years the models never saw), as RMSE in 10 prediction bins (method
  after MRBA60, Torres-Vazquez et al. 2026): a linear fit per box; for regional totals the RMSE per size bin (no line),
  which keeps 90% ranges at about 88 to 93% coverage for small and large months alike (leave-one-year-out test, E11).

## Caveats
- Before 2012, fires only VIIRS can see (MODIS never located them) are spread over each region's boxes in proportion, so
  boxes add up to the region row. This is right on average for large areas; for one box or a small area it is an
  approximation (subtract viirs_only_added to get the MODIS-located part only).
- Country and province totals assign each 28 km box to one place, so values near borders are approximate
  (see crosses_border). The labels agree with the FIRMS country files for 98.9% of boxes (2019 check).
- Single boxes are much less certain than regional or yearly totals. Prefer areas of several boxes.
- Monthly, not daily: MODIS does not see every place every day in the tropics (Giglio et al. 2025, RSE 331:115006).
- 2003 to 2011 includes larger fire months than the training years; uncertainty may be underestimated there.
- Missing and partial S-NPP days (partial = S-NPP saw under 0.8 of NOAA-20's fires worldwide that day, or under
  half of the surrounding days where NOAA-20 is not available) are filled: from April 2018 with NOAA-20 x a ratio
  learned that month per region x day/night (GFED5NRT Sect. 2.5; tested error 0.6% vs 3.6% for scaling), before
  that by scaling the good days up to the month (assumes the missing days were typical). See days_filled.
- S-NPP stopped on Nov 2 2026 (NOAA NESDIS); this file ends in {y1} (2026: January to October, from the FIRMS API).
  January to July 2026 are standard processing (VIIRS_SNPP_SP); August to October 2026 are near-real-time
  (VIIRS_SNPP_NRT), the only version FIRMS offers for those months.
- From 2026, MODIS is not used (Terra and Aqua are drifting: Copernicus CAMS); aqua_raw and terra_raw are blank.

## Credits
NASA FIRMS (MODIS MCD14ML, VIIRS VNP14IMGML), MODIS MCD12C1 land cover (NASA LP DAAC), GFED regions
(GFED4.1s, van der Werf et al. 2017; GFED5, van der Werf et al. 2025), GFED5.1 burned area by ecosystem
(van der Werf et al. 2025), fire type groups from GFED5NRT (Chen et al. 2026).
"""



def grid_cells_csv(grid, ancillary, dst, land_year=2019, cell_deg=0.25):
    """One row per box that has hotspots in the data: every fixed label a map needs to colour or filter it.
    Region: GFED4.1s basis regions (the map used to harmonize). Country/province: Natural Earth (places.py).
    Land type: MCD12C1 land groups, share of the box area (0.05 degree pixels, 25 per box), map year land_year.
    Fire type: GFED5.1 burned area 2003 to 2022 in the 9 GFED5NRT groups (gfed5.py)."""
    from harmonizer.gfed5 import FIRE_GROUPS, static_groups
    lookup = grid / "lookup"
    pl = pd.read_parquet(lookup / "place_labels.parquet")
    row = np.floor((pl["latitude"].to_numpy() + 90) / cell_deg).astype(int)
    col = np.floor((pl["longitude"].to_numpy() + 180) / cell_deg).astype(int)
    reg = np.load(lookup / "gfed_regions.npy")[row, col]
    pl["region"] = pd.Series(reg).map(REGION_CODES).to_numpy()
    pl = pl[pl["region"] != "OCEAN"].copy()
    row, col = row[reg != 0], col[reg != 0]
    pl.insert(pl.columns.get_loc("region") + 1, "region_name", pl["region"].map(REGION_NAMES))
    # land type: count the 25 MCD12C1 pixels of each group inside the box
    land = load_land(ancillary, lookup, land_year)
    k = int(round(cell_deg / 0.05))
    blocks = land.reshape(land.shape[0] // k, k, land.shape[1] // k, k)[row, :, col, :].reshape(len(row), -1)
    counts = np.stack([(blocks == i).sum(1) for i in range(len(LAND_GROUPS))], axis=1)
    pl["land_type"] = np.array(LAND_GROUPS)[counts.argmax(1)]
    pl["land_type_share"] = (counts.max(1) / blocks.shape[1]).round(2)
    # fire type: GFED5.1 burned area, 2003 to 2022 summed
    ba = static_groups(ancillary, lookup)[:, row, col].T
    tot = ba.sum(1)
    names = np.array(list(FIRE_GROUPS))
    pl["fire_type"] = np.where(tot > 0, names[ba.argmax(1)], "")
    pl["fire_type_share"] = np.where(tot > 0, ba.max(1) / np.where(tot > 0, tot, 1), np.nan).round(2)
    pl["gfed_burned_km2_per_year"] = (tot / 20).round(2)
    pl = pl.sort_values(["latitude", "longitude"])
    pl.to_csv(dst, index=False)
    return len(pl)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", help="only write these years' box CSVs, e.g. 2026")
    args = ap.parse_args()
    only = None
    if args.years:
        a, _, b = args.years.partition("-")
        only = set(range(int(a), int(b or a) + 1))
    s = load_settings()
    proc = s["paths"]["processed"]
    out = proc / "export"
    out.mkdir(parents=True, exist_ok=True)
    n = region_csv(proc / "harmonized_region_month.parquet", out / "harmonized_region_month.csv")
    print(f"harmonized_region_month.csv: {n:,} rows", flush=True)
    n = lookup_csv(s["paths"]["grid"], out / "conversion_lookup_table.csv")
    print(f"conversion_lookup_table.csv: {n:,} rows", flush=True)
    mshare = pd.read_parquet(s["paths"]["grid"].parent / "models" / "viirs_only_share.parquet") \
        .set_index(["daynight", "region", "cmonth"])["m"]
    places_dir = (s["paths"]["ancillary"], s["paths"]["grid"] / "lookup")
    years = sorted(int(p.stem) for p in (proc / "harmonized_box").glob("*.parquet"))
    for y in years:
        if only and y not in only:
            continue
        n = box_csv(proc / "harmonized_box" / f"{y}.parquet", out / f"harmonized_box_month_{y}.csv", mshare, places_dir)
        size = (out / f"harmonized_box_month_{y}.csv").stat().st_size / 1e6
        print(f"harmonized_box_month_{y}.csv: {n:,} rows, {size:.0f} MB", flush=True)
    for name, n in place_csvs(sorted(out.glob("harmonized_box_month_*.csv")), out, out / "harmonized_region_month.csv").items():
        print(f"{name}: {n:,} rows", flush=True)
    n = grid_cells_csv(s["paths"]["grid"], s["paths"]["ancillary"], out / "grid_cells.csv")
    print(f"grid_cells.csv: {n:,} boxes", flush=True)
    n = filled_days_csv(s["paths"]["grid"], out / "filled_days.csv", [y for y in years if y >= 2012])
    print(f"filled_days.csv: {n:,} months", flush=True)
    (out / "README.md").write_text(README.format(y0=years[0], y1=years[-1]), encoding="utf-8")
    zdir = proc / "export_zip"
    zdir.mkdir(parents=True, exist_ok=True)
    for f in sorted(out.glob("*.csv")):
        with zipfile.ZipFile(zdir / f"{f.stem}.zip", "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            z.write(f, arcname=f.name)
    (zdir / "README.md").write_text((out / "README.md").read_text(encoding="utf-8"), encoding="utf-8")
    total = sum(z.stat().st_size for z in zdir.glob("*.zip")) / 1e6
    print(f"\nDone: {out}\nZipped: {zdir} ({total:.0f} MB in {len(list(zdir.glob('*.zip')))} zips)")


if __name__ == "__main__":
    main()
