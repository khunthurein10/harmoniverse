"""
Step 1: clean every downloaded FIRMS zip into one shared schema.

Usage (from the project root):
    python pipeline/scripts/01_clean.py                       # everything downloaded
    python pipeline/scripts/01_clean.py --years 2012-2019      # just these years

Reads   data/raw/<satellite>/zips/*_all_countries.zip
Writes  data/interim/clean/<satellite>/<year>/<country>.parquet
        data/interim/clean/_summary.csv   (rows in, kept, dropped: proves nothing was lost)

Safe to stop and re-run: countries already written are skipped.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harmonizer.config import load_settings  # noqa: E402
from harmonizer.io import SOURCE_FOLDERS, iter_country_csvs, list_raw_zips, write_parquet  # noqa: E402
from harmonizer.clean import clean_frame, summary  # noqa: E402


def parse_years(text):
    if not text:
        return None
    a, _, b = text.partition("-")
    return set(range(int(a), int(b or a) + 1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--years", help="e.g. 2012-2019")
    p.add_argument("--sources", nargs="+", choices=list(SOURCE_FOLDERS))
    args = p.parse_args()

    s = load_settings()
    raw, out_root = s["paths"]["raw"], s["paths"]["clean"]
    years = parse_years(args.years)
    zips = [z for z in list_raw_zips(raw)
            if (years is None or z[1] in years) and (args.sources is None or z[0] in args.sources)]
    if not zips:
        sys.exit(f"No FIRMS zips found under {raw}. Run 00_download_firms.py first.")

    log_path = out_root / "_summary.csv"
    log = pd.read_csv(log_path) if log_path.exists() else pd.DataFrame()

    for source, year, zpath in zips:
        folder = out_root / SOURCE_FOLDERS[source] / str(year)
        totals = {"rows": 0, "kept": 0, "dropped_type": 0, "dropped_conf": 0, "missing_core": 0}
        n_new = 0
        for country, raw_df in iter_country_csvs(zpath):
            dest = folder / f"{country}.parquet"
            if dest.exists():
                continue
            clean = clean_frame(raw_df, country=country)
            write_parquet(clean, dest)
            stats = summary(clean)
            for k in totals:
                totals[k] += stats[k]
            log = pd.concat([log, pd.DataFrame([{"source": source, "year": year, "country": country, **stats}])],
                            ignore_index=True)
            n_new += 1
        print(f"{zpath.name}: {n_new} new countries, {totals['rows']:,} rows, "
              f"{totals['kept']:,} kept, {totals['dropped_type']:,} non-vegetation, "
              f"{totals['dropped_conf']:,} low confidence")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log.drop_duplicates(["source", "year", "country"], keep="last").to_csv(log_path, index=False)

    print(f"\nDone. Summary: {log_path}")


if __name__ == "__main__":
    main()
