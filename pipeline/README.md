# Pipeline

Turns raw FIRMS hotspots into the harmonized record in `data/processed/`.

## How it is organised
- `harmonizer/` is the library: all real logic lives here, as functions.
- `scripts/` are short numbered steps that call the library in order. Run them from the project root.
- `experiments/` are the tests that chose the method (E1 to E11). They use the same library.
- Settings come from `config/settings.yaml`.

## Steps (scripts/)
| Step | Script | Does | Library modules |
|---|---|---|---|
| 0 | `00_download_firms.py` | FIRMS yearly archive zips (MODIS, S-NPP) | (standalone) |
| 0b | `00b_download_firms_api.py` | FIRMS API: recent S-NPP months and NOAA-20 months (needs `FIRMS_MAP_KEY` in `.env`) | `io` |
| 1 | `01_clean.py` | One schema, flags, local dates | `clean`, `io` |
| 2 | `02_grid.py` | Box x day counts on the 0.25 degree grid, outage days, static (non-fire) mask | `grid`, `angle`, `gaps`, `static_mask`, `groups` |
| 3 | `03_harmonize.py` | MODIS to S-NPP conversion (2003 to 2011), S-NPP missing/partial day fill, uncertainty | `convert`, `fill`, `groups`, `grid` |
| 4 | `04_export.py` | CSV files and zips for the frontend, with the data README (files, columns, fields) | `places`, `groups`, `gfed5` |
| 5 | `05_seam_chart.py` | Before/after chart of the 2012 seam | |

## Library (harmonizer/)
| Module | Does |
|---|---|
| `config.py` | reads `config/settings.yaml` and folder paths |
| `io.py` | FIRMS file reading, source folders |
| `clean.py` | cleaning rules and flags |
| `grid.py` | 0.25 degree box index and centres |
| `angle.py` | MODIS swath position weights (E6) |
| `gaps.py` | days without data per satellite |
| `static_mask.py` | boxes with persistent non-fire heat sources |
| `groups.py` | GFED regions and MCD12C1 land groups |
| `convert.py` | model inputs and the ratio table (shared by step 3 and E8) |
| `fill.py` | missing and partial S-NPP days: NOAA-20 fill or scaling (E9, E10) |
| `places.py` | country and province per box (Natural Earth) |
| `gfed5.py` | GFED5.1 burned area by fire type |

## Run order
```
python pipeline/scripts/01_clean.py
python pipeline/scripts/02_grid.py
python pipeline/scripts/03_harmonize.py
python pipeline/scripts/04_export.py
```
Tests: `pytest pipeline/tests -q`
