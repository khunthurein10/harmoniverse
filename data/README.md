# Data

Only our own outputs are committed: the zipped export (`processed/export_zip/`) and the trained models
(`interim/models/`). Everything else is downloaded or rebuilt by the pipeline; the steps are in the main README.

## raw/ (downloaded FIRMS hotspots, never edited)
| Folder | Product | Used for |
|---|---|---|
| `modis/` | MODIS C6.1 (Terra + Aqua), 1 km, 2003 to 2025 | Input to the conversion |
| `viirs_snpp/` | VIIRS S-NPP 375 m, Jan 2012 to Oct 2026 | Target scale |
| `viirs_noaa20/` | VIIRS NOAA-20 375 m, only months with missing or partial S-NPP days | Filling those days |

`pipeline/scripts/00_download_firms.py` (yearly files) and `00b_download_firms_api.py` (2026 and NOAA-20).

## ancillary/
| File | Used for | Source |
|---|---|---|
| `MCD12C1.A<year>001.061.*.hdf`, 2012 to 2019 | Land groups | NASA LP DAAC (Earthdata login) |
| `GFED4.1s_<year>.hdf5` (any year) | GFED basis regions | globalfiredata.org |
| `GFED5.1_ecosystem.zip` | Fire type, 2003 to 2022 | globalfiredata.org |
| `gfed5nrt/Reprocessed/Daily/Ecosystem/CMB/` (from `GFED5NRT.tar.gz`) | Fire type, 2023 and 2024 | globalfiredata.org |
| `ne_10m_admin_1_states_provinces.zip` | Country and province | naturalearthdata.com |

After the first run the maps are cached in `interim/grid/lookup/`, and the source files are no longer read.

## interim/ (made by the pipeline)
- `clean/`: one Parquet per downloaded file, one schema for MODIS and VIIRS, with `_summary.csv` (rows removed per filter)
- `grid/`: box x day and box x month counts, days observed, static masks, cached maps (`lookup/`)
- `models/`: trained conversion models and the VIIRS-only share (committed)

## processed/
- `harmonized_box/`, `harmonized_region_month.parquet`: the harmonized record (step 3)
- `export/`: CSV files for the frontend and for download (step 4); files, columns and fields in `export/README.md` (copied to `export_zip/README.md`)
- `export_zip/`: the same files zipped (committed)
