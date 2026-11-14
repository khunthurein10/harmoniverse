"""
Step 0: download FIRMS yearly hotspot files (standard processing, whole world).

Usage (from the project folder):
    python pipeline/scripts/00_download_firms.py                 # everything 2000-2025
    python pipeline/scripts/00_download_firms.py --years 2012-2019 --sources modis viirs-snpp

What it downloads:
    FIRMS "Country Yearly Summary" zips, one per satellite per year, e.g.
    https://firms.modaps.eosdis.nasa.gov/data/country/zips/modis_2012_all_countries.zip
    Each zip holds one CSV per country. Standard processing, includes the 'type' field.

    source name    satellite            years    approx total (zipped)
    modis          Terra + Aqua MODIS   2000-    2.0 GB
    viirs-snpp     S-NPP VIIRS          2012-    5.0 GB
    viirs-jpss1    NOAA-20 VIIRS        2018-    2.8 GB

    2026 is not in these yearly files yet; it comes from a separate step.

Safe to stop and re-run: finished files are skipped, half-finished ones restart.
Uses only the Python standard library.
"""
import argparse
import sys
import time
import urllib.request
from pathlib import Path

BASE = "https://firms.modaps.eosdis.nasa.gov/data/country"
LIST_URL = f"{BASE}/yearly_summary_files.txt"
ZIP_URL = BASE + "/zips/{name}"

# FIRMS source name -> our folder under data/raw/
FOLDERS = {"modis": "modis", "viirs-snpp": "viirs_snpp", "viirs-jpss1": "viirs_noaa20"}


def fetch_file_list():
    """Return {(source, year): (filename, size_bytes)} for the all_countries zips."""
    with urllib.request.urlopen(LIST_URL, timeout=60) as r:
        lines = r.read().decode("utf-8").splitlines()
    files = {}
    for line in lines[1:]:
        parts = line.split(",")
        if len(parts) != 3 or not parts[0].endswith("_all_countries.zip"):
            continue
        name, size = parts[0], int(parts[1])
        source, year = name.split("_")[0], int(name.split("_")[1])
        files[(source, year)] = (name, size)
    return files


def download(url, dest, expected_size):
    if dest.exists() and dest.stat().st_size == expected_size:
        print(f"  skip  {dest.name} (already done)")
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    start = time.time()
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        done = 0
        while True:
            chunk = r.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            pct = 100 * done / expected_size if expected_size else 0
            print(f"\r  get   {dest.name}  {done/1e6:7.1f} MB  {pct:5.1f}%", end="", flush=True)
    secs = time.time() - start
    if tmp.stat().st_size != expected_size:
        print(f"\n  WARNING {dest.name}: got {tmp.stat().st_size} bytes, expected {expected_size}. Re-run to retry.")
        return
    tmp.replace(dest)
    print(f"\r  done  {dest.name}  {expected_size/1e6:7.1f} MB in {secs/60:4.1f} min      ")


def parse_years(text):
    if "-" in text:
        a, b = text.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(text)]


def main():
    p = argparse.ArgumentParser(description="Download FIRMS yearly hotspot zips")
    p.add_argument("--sources", nargs="+", default=list(FOLDERS), choices=list(FOLDERS))
    p.add_argument("--years", default="2000-2025", help="e.g. 2012-2019 or 2015")
    p.add_argument("--out", default="data/raw", help="raw data folder")
    args = p.parse_args()

    years = parse_years(args.years)
    files = fetch_file_list()
    todo = [(s, y) for s in args.sources for y in years if (s, y) in files]
    total = sum(files[k][1] for k in todo)
    print(f"{len(todo)} files, {total/1e9:.2f} GB to check/download into {args.out}/\n")

    failed = []
    for source, year in todo:
        name, size = files[(source, year)]
        folder = Path(args.out) / FOLDERS[source] / "zips"
        folder.mkdir(parents=True, exist_ok=True)
        try:
            download(ZIP_URL.format(name=name), folder / name, size)
        except Exception as e:  # network hiccup: note it and keep going
            print(f"\n  FAILED {name}: {e}")
            failed.append(name)

    print("\nFinished.")
    if failed:
        print(f"{len(failed)} file(s) failed, re-run the same command to retry:")
        for n in failed:
            print("  ", n)
        sys.exit(1)


if __name__ == "__main__":
    main()
