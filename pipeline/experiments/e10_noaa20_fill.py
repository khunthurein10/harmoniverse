"""
E10: fill S-NPP outage days with NOAA-20, or scale up the days S-NPP saw (E9)? Which is closer to the truth?

NOAA-20 carries the same VIIRS instrument as S-NPP, half an orbit apart, so its counts differ by a ratio.
In each S-NPP outage month (more than 2 days missing, 2018 onward, when NOAA-20 exists), the ratio is learned per
region x day/night from the days both satellites saw that month (our choice: same month, so season and fire
conditions match).
Partial S-NPP days (only some orbits delivered) are found by comparing world totals with NOAA-20 and treated as missing.

Test (our choice), on the days S-NPP did see (their total is the truth): hide every run of k consecutive seen days,
k = the month's real number of missing days (at most half the seen days), then rebuild the total two ways:
  scale : seen-minus-hidden x n_seen / (n_seen - k)                (E9 method, after Giglio et al. 2006 CMG)
  noaa20: seen-minus-hidden + ratio x NOAA-20 on the hidden days   (ratio learned without the hidden days)
Scored like E9 and the main results: fire-weighted error = sum |estimate - truth| / sum truth, region x month x
day/night. Also writes the real filled values both ways for each outage month.

Needs NOAA-20 for the outage months: 00b_download_firms_api.py --sources viirs-jpss1, then 01_clean.py, 02_grid.py.
Usage: python pipeline/experiments/e10_noaa20_fill.py
Writes pipeline/experiments/results/e10/*.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harmonizer.config import load_settings  # noqa: E402

FIRST_YEAR = 2018   # NOAA-20 standard data starts 2018-04-01 (FIRMS data availability)
PARTIAL_SHARE = 0.8 # our choice: S-NPP day below 0.8 of NOAA-20 worldwide = partial day (normal days: median 1.01, 5th pct 0.82)


def daily(grid, sat, year, months):
    """Hotspots per region x day/night x date for one satellite-year, low confidence dropped, plus the dates seen."""
    f = grid / "cell_days" / f"{sat}_{year}.parquet"
    if not f.exists():
        sys.exit(f"Missing {f}. Download, clean and grid NOAA-20 first (see docstring).")
    cd = pd.read_parquet(f, columns=["row", "col", "date", "daynight", "n_det", "n_det_low"])
    cd["date"] = pd.to_datetime(cd["date"]).dt.normalize()
    cd["month"] = cd["date"].dt.to_period("M").dt.to_timestamp()
    cd = cd[cd["month"].isin(months)]
    seen = cd.groupby("month")["date"].unique().to_dict()
    reg = pd.read_parquet(grid / "monthly" / f"{sat}_{year}.parquet", columns=["row", "col", "region"]).drop_duplicates(["row", "col"])
    cd = cd.merge(reg, on=["row", "col"], how="left")
    cd = cd[cd["region"].notna() & (cd["region"] != "OCEAN")]
    cd["n"] = cd["n_det"] - cd["n_det_low"]
    d = cd.groupby(["region", "daynight", "month", "date"], as_index=False)["n"].sum()
    d = pd.concat([d, d.groupby(["region", "month", "date"], as_index=False)["n"].sum().assign(daynight="ALL")])
    return d, seen


def score(x, col):
    err = x[col] - x["truth"]
    return pd.Series({"fire_weighted_error_pct": 100 * err.abs().sum() / x["truth"].sum(),
                      "median_error_pct": 100 * (err.abs() / x["truth"]).median(),
                      "bias_pct": 100 * err.sum() / x["truth"].sum(),
                      "relative_rmse_pct": 100 * np.sqrt(((err / x["truth"]) ** 2).mean())})


def main():
    s = load_settings()
    grid = Path(s["paths"]["grid"])
    res = Path(__file__).resolve().parent / "results" / "e10"
    res.mkdir(parents=True, exist_ok=True)

    days = pd.read_parquet(grid / "days_observed.parquet")
    # every month NOAA-20 was downloaded for (the months with missing or partial S-NPP days)
    out = days[(days["satellite"] == "NOAA20") & (days["month"].dt.year >= FIRST_YEAR)]
    print("Months tested:", [m.strftime("%Y-%m") for m in out["month"]])

    tests, fills = [], []
    for year, g in out.groupby(out["month"].dt.year):
        months = list(g["month"])
        sn, sn_seen = daily(grid, "viirs_snpp", year, months)
        n2, n2_seen = daily(grid, "viirs_noaa20", year, months)
        for m in months:
            ndays = m.days_in_month
            allm = pd.date_range(m, periods=ndays, freq="D")
            n_days = set(n2_seen.get(m, []))
            # partial days (our choice, after GFED5NRT Sect. 2.5 "verifies global data completeness"): S-NPP sent only
            # some orbits, so its world total is far below NOAA-20's (normal days 0.92 to 1.09, partial about 0.15).
            # A day counts as missing when S-NPP sees less than PARTIAL_SHARE of NOAA-20's world total.
            st = sn[(sn["month"] == m) & (sn["daynight"] == "ALL")].groupby("date")["n"].sum()
            nt = n2[(n2["month"] == m) & (n2["daynight"] == "ALL")].groupby("date")["n"].sum()
            share = (st / nt.reindex(st.index)).dropna()
            partial = set(share[share < PARTIAL_SHARE].index)
            s_days = [x for x in allm if x in set(sn_seen.get(m, [])) and x not in partial]
            if partial:
                print(f"{m:%Y-%m}: partial S-NPP days counted as missing: "
                      + ", ".join(f"{d.day} ({share[d]:.2f})" for d in sorted(partial)))
            miss = [x for x in allm if x not in set(s_days)]
            print(f"{m:%Y-%m}: S-NPP saw {len(s_days)} days, NOAA-20 saw {len(n_days)} days, "
                  f"NOAA-20 covers {sum(x in n_days for x in miss)} of {len(miss)} S-NPP missing days")
            both = [x for x in s_days if x in n_days]
            k = min(len(miss), len(both) // 2)
            for (region, dn), a in sn[sn["month"] == m].groupby(["region", "daynight"]):
                b = n2[(n2["month"] == m) & (n2["region"] == region) & (n2["daynight"] == dn)]
                sv = a.set_index("date")["n"].reindex(allm, fill_value=0)
                nv = b.set_index("date")["n"].reindex(allm, fill_value=0)
                # real fill for the output
                rs, rn = sv[both].sum(), nv[both].sum()
                ratio = rs / rn if rn > 0 else np.nan
                seen_total = sv[s_days].sum()
                n_miss = [x for x in miss if x in n_days]
                fills.append({"month": m.strftime("%Y-%m"), "region": region, "daynight": dn, "ratio": ratio,
                              "snpp_seen": seen_total,
                              "scaled": seen_total * ndays / len(s_days),
                              "noaa20_filled": seen_total + (ratio * nv[n_miss].sum() if rn > 0 else 0)
                              + (seen_total / len(s_days)) * (len(miss) - len(n_miss))})
                # test: hide runs of k seen days that NOAA-20 also saw
                truth = sv[both].sum()
                if truth <= 0 or k < 1:
                    continue
                for i in range(len(both) - k + 1):
                    hid = both[i:i + k]
                    keep = both[:i] + both[i + k:]
                    kept_s, kept_n = sv[keep].sum(), nv[keep].sum()
                    r = kept_s / kept_n if kept_n > 0 else np.nan
                    tests.append({"month": m, "region": region, "daynight": dn, "k": k, "truth": truth,
                                  "scale": kept_s * len(both) / len(keep),
                                  "noaa20": kept_s + (r * nv[hid].sum() if kept_n > 0 else kept_s / len(keep) * k)})
    t = pd.DataFrame(tests)
    pd.DataFrame(fills).to_csv(res / "fill_values.csv", index=False)

    def both_methods(x):
        a, b = score(x, "scale"), score(x, "noaa20")
        return pd.Series({"cases": len(x), **{f"scale_{c}": v for c, v in a.items()}, **{f"noaa20_{c}": v for c, v in b.items()}})

    for name, keys in [("by_daynight", ["daynight"]), ("by_month", ["month", "daynight"]), ("by_region", ["region", "daynight"])]:
        r = t.groupby(keys).apply(both_methods).reset_index()
        r.to_csv(res / f"compare_{name}.csv", index=False)
        if name != "by_region":
            print(f"\nScale vs NOAA-20 fill, {name}:")
            print(r.round(2).to_string(index=False))
    r = t[t["daynight"] == "ALL"].groupby("region").apply(both_methods).reset_index()
    print("\nBy region (D + N):")
    print(r.round(2).to_string(index=False))
    print(f"\nWrote {res}")


if __name__ == "__main__":
    main()
