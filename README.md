# Harmoniverse

**One consistent record of global fire hotspots, 2003 to 2026.**
Built for the NASA Space Apps Challenge 2026: *Harmonization of MODIS and VIIRS Hot Spots*.

MODIS (since 2000, 1 km pixels) and VIIRS (since 2012, 375 m pixels) see the same fires differently: for the same
fires in 2012 to 2015, S-NPP VIIRS counted about 6 times more hotspots than Aqua MODIS. Joining the two records as
they are creates a fake jump in 2012. Harmoniverse converts the MODIS years to the S-NPP VIIRS scale, so every month
from January 2003 to October 2026 can be compared directly, each with a 90% range.

The repository has three parts:

| Part | Folder | What it is |
|---|---|---|
| Pipeline | `pipeline/` | Python code that downloads NASA FIRMS hotspots, cleans them, puts them on a 0.25° grid and harmonizes them |
| Data | `data/` | The harmonized dataset (zipped CSVs), the trained conversion models and the data guide |
| Website | `web/` | A static landing page (`index.html`) and the Fire Explorer (`app.html`) |

## The dataset

Every count is a number of active-fire hotspots on the **S-NPP VIIRS 375 m scale**.

- **January 2003 to January 2012:** converted from MODIS (Aqua and Terra) by two gradient boosting models, one for
  day passes and one for night passes, trained on 2015 to 2019.
- **February 2012 onward:** real S-NPP counts. Missing or partial S-NPP days are filled with NOAA-20 (from April
  2018) or by scaling up the month's good days (before that).
- **Cleaning:** vegetation fires only (FIRMS type 0), low-confidence detections removed, and persistent non-fire
  heat sources (gas flares, volcanoes, industry) masked.
- **Levels:** 0.25° boxes (about 28 km), provinces, countries and the 14 GFED regions, per month, with day and night
  parts.

The files are in `data/processed/export_zip/`. Every file, column and field is explained in
[`data/processed/export_zip/README.md`](data/processed/export_zip/README.md).

### How accurate it is

The models were trained on 2015 to 2019 and tested once on 2012 to 2014, years when both satellites flew but the
models never saw. Their estimates were compared with what S-NPP really counted:

| Level | Typical error |
|---|---|
| A region's yearly total | 2.0% |
| A region's monthly total (fire-weighted) | 5.4% (the simple lookup table: 7.2%) |
| Night months | 12.6% |
| A single 28 km box | 26% |

The 90% ranges held 88 to 93% of the true values in testing. Each experiment behind these choices is listed in
[`pipeline/experiments/README.md`](pipeline/experiments/README.md), with its results in
`pipeline/experiments/results/`.

### What it is not

- **A hotspot is not a fire.** One large fire can produce hundreds of hotspots across days and overpasses.
- **Hotspots are not burned area.** They show where heat was seen, not how much land burned.
- **It is not a forecast or an alert system.** It describes past activity for planning and research. For active
  fires, follow local authorities and official alerts.
- Single boxes and months before 2012 that are larger than anything in the training years are the least certain.
  Prefer areas of several boxes.

## The website

`web/` is a static site with no build step:

- **`index.html`:** the landing page. It explains the method in four chapters, shows the accuracy results and
  example fire seasons, and offers the conversion lookup table as a download.
- **`app.html`:** the Fire Explorer. Search any country, province or region, or draw an area. Then read its burning
  calendar, map, comparison, "Is it normal?" verdict and trends, and export what you see as CSV or images.

To run it locally (pages must be served over HTTP; opening the file directly does not work):

```sh
cd web
python3 -m http.server 8000
# open http://localhost:8000
```

The site reads small files in `web/data/`, made from the zipped dataset by a standard-library script:

```sh
python3 web/build_data.py
```

## Rebuilding the dataset

```sh
python3 -m venv .venv && source .venv/bin/activate
pip install -r pipeline/requirements.txt
cp .env.example .env    # add a free FIRMS map key for the API download step
```

Then download the inputs and run the steps from the project root:

```sh
python pipeline/scripts/00_download_firms.py      # FIRMS yearly archives (MODIS, S-NPP)
python pipeline/scripts/00b_download_firms_api.py # recent S-NPP months and NOAA-20 fill months
python pipeline/scripts/01_clean.py
python pipeline/scripts/02_grid.py
python pipeline/scripts/03_harmonize.py
python pipeline/scripts/04_export.py
```

Some inputs are not downloaded by the scripts: MCD12C1 land cover, GFED regions and fire types, and Natural Earth
borders. Where to get each one is listed in [`data/README.md`](data/README.md). Settings are in
`config/settings.yaml`. What each step and library module does is in [`pipeline/README.md`](pipeline/README.md).

Tests:

```sh
python -m pytest pipeline/tests -q
```

## Repository layout

```
config/settings.yaml      pipeline settings
data/                     dataset, trained models, data guide (raw inputs are not committed)
pipeline/harmonizer/      the library: cleaning, gridding, conversion, filling, labels
pipeline/scripts/         numbered steps that run the library in order
pipeline/experiments/     the experiments (E1 to E11) that chose the method, with result tables
pipeline/tests/           tests for cleaning and gridding
web/                      landing page, Fire Explorer, site data and its build script
```

## Data sources

- NASA FIRMS active fire detections: MODIS Collection 6.1 (Terra and Aqua), VIIRS S-NPP and NOAA-20
- MCD12C1 land cover (NASA LP DAAC)
- GFED4.1s basis regions, GFED5.1 and GFED5NRT fire types and burned area (globalfiredata.org)
- Natural Earth admin boundaries (naturalearthdata.com)
- Map imagery on the website: NASA GIBS / EOSDIS

## License

Made by **BladeX & Beyond** for the NASA Space Apps Challenge 2026.

- **Code:** [MIT License](LICENSE).
- **Dataset** (`data/processed/`, `web/downloads/`): [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Use it
  freely, crediting "Harmoniverse harmonized fire hotspot record, BladeX & Beyond".

If you use the dataset, please also cite the sources it is built on, listed above. GFED in particular asks to be
cited.
