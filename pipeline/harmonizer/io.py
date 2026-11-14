"""
Read FIRMS files and write/read Parquet tables.

FIRMS yearly zips hold one CSV per country, e.g. modis_2012_Canada.csv.
Reading them one country at a time keeps memory low, even for a full VIIRS year.
"""
import re
import zipfile
from pathlib import Path

import pandas as pd

# FIRMS source name in file names -> our folder name under data/raw/
SOURCE_FOLDERS = {"modis": "modis", "viirs-snpp": "viirs_snpp", "viirs-jpss1": "viirs_noaa20", "viirs-jpss2": "viirs_noaa21"}

_ZIP_NAME = re.compile(r"^(?P<source>[a-z0-9-]+)_(?P<year>\d{4})_all_countries\.zip$")
_CSV_NAME = re.compile(r"^(?:.*/)?(?P<source>[a-z0-9-]+)_(?P<year>\d{4})_(?P<country>.+)\.csv$")


def list_raw_zips(raw_dir):
    """All FIRMS yearly zips under data/raw/<satellite>/zips/, as (source, year, path)."""
    found = []
    for path in sorted(Path(raw_dir).glob("*/zips/*.zip")):
        m = _ZIP_NAME.match(path.name)
        if m:
            found.append((m["source"], int(m["year"]), path))
    return found


def iter_country_csvs(zip_path):
    """Yield (country, DataFrame) for each country CSV inside one FIRMS yearly zip."""
    with zipfile.ZipFile(zip_path) as z:
        for name in sorted(z.namelist()):
            m = _CSV_NAME.match(name)
            if not m:
                continue
            with z.open(name) as f:
                df = pd.read_csv(f, low_memory=False)
            if len(df):
                yield m["country"], df


def write_parquet(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a .tmp file first and rename only when complete, so a stopped run
    # never leaves a half-written .parquet that later runs would skip (our choice).
    tmp = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def read_parquet_folder(folder, columns=None):
    """Read every Parquet file under a folder (recursively) into one DataFrame."""
    files = sorted(Path(folder).rglob("*.parquet"))
    if not files:
        return pd.DataFrame(columns=columns or [])
    return pd.concat([pd.read_parquet(f, columns=columns) for f in files], ignore_index=True)
