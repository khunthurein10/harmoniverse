"""
Step 2: count hotspots per 0.25 degree cell per month, and find outage days.

Usage (from the project root):
    python pipeline/scripts/02_grid.py
    python pipeline/scripts/02_grid.py --years 2012-2019

Reads   data/interim/clean/<satellite>/<year>/*.parquet
Writes  data/interim/grid/cell_days/<satellite>_<year>.parquet   (cell x day, small)
        data/interim/grid/monthly/<satellite>_<year>.parquet     (cell x month, what the methods use)
        data/interim/grid/days_observed.parquet                  (outage check, only meaningful for global runs)

Each (satellite, year) is done separately so memory stays low. Re-running skips finished ones.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harmonizer.config import load_settings  # noqa: E402
from harmonizer.grid import cell_day_counts, monthly_counts  # noqa: E402
from harmonizer.gaps import days_observed, gap_months  # noqa: E402
from harmonizer.static_mask import build_static_mask, in_mask  # noqa: E402
from harmonizer.groups import load_land, load_regions, land_of  # noqa: E402
from harmonizer.angle import build_edge_weights  # noqa: E402


def parse_years(text):
    if not text:
        return None
    a, _, b = text.partition("-")
    return set(range(int(a), int(b or a) + 1))


def load_edge_weights(s, clean_root, grid_root):
    """MODIS edge weights (E6 option V3), built once from the choosing years only (2015 to 2019)."""
    fine_p, coarse_p = grid_root / "lookup" / "modis_edge_weights.parquet", grid_root / "lookup" / "modis_edge_weights_coarse.parquet"
    if fine_p.exists() and coarse_p.exists():
        return pd.read_parquet(fine_p), pd.read_parquet(coarse_p)
    frames = []
    for year in range(2015, 2020):
        land_grid = load_land(s["paths"]["ancillary"], grid_root / "lookup", year)
        for f in sorted((clean_root / "modis" / str(year)).glob("*.parquet")):
            d = pd.read_parquet(f, columns=["satellite", "daynight", "acq_utc", "lat", "lon", "scan", "keep_type", "keep_conf"])
            d = d[d["keep_type"] & d["keep_conf"]].dropna(subset=["scan", "lat", "lon", "acq_utc"])
            frames.append(pd.DataFrame({"satellite": d["satellite"].to_numpy(), "daynight": d["daynight"].to_numpy(),
                                        "cmonth": d["acq_utc"].dt.month.to_numpy(), "scan": d["scan"].to_numpy(),
                                        "land": land_of(d["lat"], d["lon"], land_grid)}))
    if not frames:
        print("no MODIS 2015 to 2019 clean files: edge weights skipped (n_det_w = n_det)")
        return None
    fine, coarse = build_edge_weights(frames)
    fine_p.parent.mkdir(parents=True, exist_ok=True)
    fine.to_parquet(fine_p, index=False)
    coarse.to_parquet(coarse_p, index=False)
    print(f"edge weights built: {len(fine):,} rows, {(fine['source'] == 'fine').mean():.0%} from fine strata")
    return fine, coarse


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--years", help="e.g. 2012-2019")
    p.add_argument("--force", action="store_true", help="redo years that are already gridded")
    args = p.parse_args()

    s = load_settings()
    clean_root, grid_root = s["paths"]["clean"], s["paths"]["grid"]
    cell_deg = s["grid"]["cell_deg"]
    max_angle = s["viewing_angle"]["modis_max_scan_angle_deg"]
    years = parse_years(args.years)

    jobs = sorted((d.parent.name, int(d.name), d) for d in clean_root.glob("*/*") if d.is_dir() and d.name.isdigit())
    jobs = [j for j in jobs if years is None or j[1] in years]
    if not jobs:
        sys.exit(f"No cleaned data under {clean_root}. Run 01_clean.py first.")

    region_grid = load_regions(s["paths"]["ancillary"], grid_root / "lookup")
    edge_tables = load_edge_weights(s, clean_root, grid_root)
    for sat_folder, year, folder in jobs:
        tag = f"{sat_folder}_{year}"
        cd_path = grid_root / "cell_days" / f"{tag}.parquet"
        mo_path = grid_root / "monthly" / f"{tag}.parquet"
        if mo_path.exists() and not args.force:
            print(f"skip  {tag}")
            continue
        files = sorted(folder.glob("*.parquet"))
        land_grid = load_land(s["paths"]["ancillary"], grid_root / "lookup", year)
        mask_cells = None
        if sat_folder.startswith("viirs"):
            c = s["cleaning"]
            mask = build_static_mask(files, c["static_mask_cell_deg"], c["static_mask_min_detections"],
                                     c.get("static_mask_persistence", True), c.get("static_mask_min_days", 16),
                                     c.get("static_mask_min_months", 7), c.get("static_mask_long_run", True),
                                     c.get("static_mask_long_run_days", 60))
            mask_path = grid_root / "static_mask" / f"{tag}.parquet"
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            mask.to_parquet(mask_path, index=False)
            has_nrt = any(pd.read_parquet(f, columns=["is_nrt"])["is_nrt"].any() for f in files)
            prev = grid_root / "static_mask" / f"{sat_folder}_{year - 1}.parquet"
            if has_nrt and prev.exists():   # NRT rows have no 'type': add last year's standard-data mask (GFED5NRT 2.3.2)
                mask = pd.concat([mask, pd.read_parquet(prev)]).drop_duplicates("cell")
                mask.to_parquet(mask_path, index=False)
                print(f"mask  {tag}: NRT rows found, added {year - 1} mask cells")
            snpp_mask = grid_root / "static_mask" / f"viirs_snpp_{year}.parquet"
            if sat_folder != "viirs_snpp" and snpp_mask.exists():
                # our choice: flares and volcanoes sit in fixed places, so the S-NPP mask of the same year also
                # applies to NOAA-20 (same VIIRS instrument). Needed because NOAA-20 is only downloaded for a few
                # months, too few for the 7-month persistence rule to find static sources on its own.
                mask = pd.concat([mask, pd.read_parquet(snpp_mask)]).drop_duplicates("cell")
                mask.to_parquet(mask_path, index=False)
                print(f"mask  {tag}: added viirs_snpp_{year} mask cells")
            mask_cells = mask["cell"].to_numpy()
            print(f"mask  {tag}: {int(mask['rule1'].sum()):,} cells rule 1, "
                  f"{int(((mask['rule2'] | mask['rule2b']) & ~mask['rule1']).sum()):,} extra cells rule 2/2b")
        parts = []
        for f in files:
            df = pd.read_parquet(f)
            if mask_cells is not None:
                df = df[~in_mask(df, mask_cells, s["cleaning"]["static_mask_cell_deg"])]
            parts.append(cell_day_counts(df, cell_deg, max_angle, date_basis=s["grid"]["date_basis"], land_grid=land_grid,
                                         edge_tables=edge_tables))
        parts = [x for x in parts if len(x)]
        if not parts:
            print(f"empty {tag}")
            continue
        cell_days = pd.concat(parts, ignore_index=True)
        monthly = monthly_counts(cell_days, cell_deg, region_grid)
        cd_path.parent.mkdir(parents=True, exist_ok=True)
        mo_path.parent.mkdir(parents=True, exist_ok=True)
        cell_days.to_parquet(cd_path, index=False)
        monthly.to_parquet(mo_path, index=False)
        print(f"done  {tag}: {int(monthly['n_det'].sum()):,} hotspots in {len(monthly):,} cell-months")

    all_cd = [pd.read_parquet(f, columns=["satellite", "date"]) for f in sorted((grid_root / "cell_days").glob("*.parquet"))]
    days = days_observed(pd.concat(all_cd, ignore_index=True))
    days.to_parquet(grid_root / "days_observed.parquet", index=False)
    gaps = gap_months(days)
    print(f"\nOutage check: {len(gaps)} satellite-months with more than 2 days missing")
    if len(gaps):
        print(gaps.to_string(index=False, max_rows=30))


if __name__ == "__main__":
    main()
