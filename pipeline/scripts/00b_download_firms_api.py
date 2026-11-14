"""
Step 0b: download recent months that are not yet in the FIRMS yearly files, through the FIRMS API.

Used for 2026 (S-NPP stopped Nov 2 2026, NOAA NESDIS; the record ends with October 2026, the last full month).
API: /api/area/csv/[MAP_KEY]/[SOURCE]/[AREA]/[DAY_RANGE]/[DATE], DAY_RANGE 1..5 (server limit; docs page says 1..10).
Standard processing (*_SP, same kind as the yearly files) is tried first; where it is not out yet, near-real-time
(*_NRT) is used. NRT rows have no 'type' column; the static mask covers them (see 02_grid.py).

The chunks are packed into data/raw/<satellite>/zips/<source>_<year>_all_countries.zip, one CSV per 5-day chunk,
so 01_clean.py, 02_grid.py and 03_harmonize.py work unchanged (the 'country' label becomes e.g. world-2026-01-01).

Needs FIRMS_MAP_KEY in .env (never commit .env).
Also used for NOAA-20 (viirs-jpss1) in the S-NPP outage months only, to fill the missing days (see 03_harmonize.py).

Usage: python pipeline/scripts/00b_download_firms_api.py --start 2026-01-01 --end 2026-10-31
       python pipeline/scripts/00b_download_firms_api.py --sources viirs-jpss1 --start 2022-07-01 --end 2022-08-31
"""
import argparse
import sys
import time
import urllib.error
import urllib.request
import zipfile
from datetime import date, timedelta
from pathlib import Path

API = "https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key}/{product}/-180,-90,180,90/{days}/{start}"
PRODUCTS = {"viirs-snpp": ("VIIRS_SNPP_SP", "VIIRS_SNPP_NRT"), "modis": ("MODIS_SP", "MODIS_NRT"),
            "viirs-jpss1": ("VIIRS_NOAA20_SP", "VIIRS_NOAA20_NRT")}
FOLDERS = {"viirs-snpp": "viirs_snpp", "modis": "modis", "viirs-jpss1": "viirs_noaa20"}
CHUNK_DAYS = 5   # FIRMS server answers "Invalid day range. Expects [1..5]" (its docs page says 1..10)


def read_key(root):
    env = root / ".env"
    if not env.exists():
        sys.exit("No .env file. Copy .env.example to .env and put your FIRMS_MAP_KEY in it.")
    for line in env.read_text().splitlines():
        if line.strip().startswith("FIRMS_MAP_KEY="):
            key = line.split("=", 1)[1].strip().strip('"').strip("'")
            if key and key != "your_key_here":
                return key
    sys.exit("FIRMS_MAP_KEY missing in .env")


def fetch(url, tries=6):
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=300) as r:
                return r.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")[:300].replace("\n", " ")
            if "transaction limit" in body.lower():
                print("    FIRMS transaction limit reached (5000 per 10 minutes): waiting 10 minutes, then retrying", flush=True)
                time.sleep(610)
                continue
            print(f"    HTTP {e.code} from FIRMS: {body}")
            return None
        except (urllib.error.URLError, TimeoutError) as e:
            if i == tries - 1:
                print(f"    failed: {e}")
                return None
            time.sleep(5 * (i + 1))


def check(key):
    """Print what the API says about the key and which dates each product covers."""
    status = fetch(f"https://firms.modaps.eosdis.nasa.gov/mapserver/mapkey_status/?MAP_KEY={key}")
    print("key status:", (status or "no answer")[:200])
    avail = fetch(f"https://firms.modaps.eosdis.nasa.gov/api/data_availability/csv/{key}/all")
    print("data availability:")
    for line in (avail or "").splitlines():
        if "SNPP" in line or "NOAA20" in line or line.startswith("data_id") or "MODIS" in line:
            print("   ", line)


def has_rows(text):
    return text is not None and text.startswith("latitude") and text.count("\n") > 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-01-01")
    ap.add_argument("--end", default="2026-10-31")
    ap.add_argument("--sources", nargs="+", default=["viirs-snpp", "modis"], choices=list(PRODUCTS))
    ap.add_argument("--check", action="store_true", help="only print key status and data availability")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[2]
    key = read_key(root)
    if args.check:
        check(key)
        return
    start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)

    avail = {}
    for line in (fetch(f"https://firms.modaps.eosdis.nasa.gov/api/data_availability/csv/{key}/all") or "").splitlines()[1:]:
        parts = line.split(",")
        if len(parts) == 3:
            avail[parts[0]] = (date.fromisoformat(parts[1]), date.fromisoformat(parts[2]))

    for source in args.sources:
        sp, nrt = PRODUCTS[source]
        if sp not in avail and nrt not in avail:
            sys.exit(f"FIRMS lists neither {sp} nor {nrt}; run with --check")
        cache = root / "data" / "raw" / FOLDERS[source] / "api"
        cache.mkdir(parents=True, exist_ok=True)
        d = start
        while d <= end:
            # standard processing where it exists (same kind as the yearly files), otherwise near-real-time
            if sp in avail and avail[sp][0] <= d <= avail[sp][1]:
                product, stop = sp, min(end, avail[sp][1])
            elif nrt in avail and avail[nrt][0] <= d <= avail[nrt][1]:
                product, stop = nrt, min(end, avail[nrt][1])
            else:
                print(f"{source} {d}: not covered by {sp} or {nrt}, skipped")
                d += timedelta(days=1)
                continue
            days = min(CHUNK_DAYS, (stop - d).days + 1)
            out = cache / f"{source}_{d.year}_world-{d.isoformat()}.csv"
            if not out.exists():
                text = fetch(API.format(key=key, product=product, days=days, start=d.isoformat()))
                if not has_rows(text):
                    print(f"{source} {d} +{days}d from {product}: no rows (first line: {(text or '').splitlines()[:1]})")
                    d += timedelta(days=days)
                    continue
                out.write_text(text, encoding="utf-8")
                print(f"{source} {d} +{days}d: {text.count(chr(10)) - 1:,} rows from {product}", flush=True)
                time.sleep(2)
            d += timedelta(days=days)

        for year in range(start.year, end.year + 1):
            files = sorted(cache.glob(f"{source}_{year}_world-*.csv"))
            if not files:
                continue
            zpath = root / "data" / "raw" / FOLDERS[source] / "zips" / f"{source}_{year}_all_countries.zip"
            zpath.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
                for f in files:
                    z.write(f, arcname=f.name)
            print(f"packed {len(files)} chunks into {zpath.name}")


if __name__ == "__main__":
    main()
