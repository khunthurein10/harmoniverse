"""
E9: how accurate is gap-filling an S-NPP month?

When S-NPP misses more than 2 days in a month (an outage), 03_harmonize.py keeps the real count and scales it
up for the missing days: filled = observed count x days in month / days observed (our choice; it assumes the
missing days were typical). This test measures that error where the truth is known.

Truth: S-NPP 2015 to 2019, where every month is complete (0 days missing, from days_observed.parquet).
Test (our choice): for each region x month x day/night (and D + N), hide a block of k consecutive days
(outages are usually runs of days), scale the rest back up, compare with the real monthly total. Every start
day is tried. k = 3 (smallest outage) up to a third of the month (the 2/3 rule used for real gap-filled months).
Scored like the main results: fire-weighted error = sum |filled - truth| / sum truth, at region x month x
day/night. relative_rmse is the typical error as a share of the count, usable as the uncertainty of a filled month.
Also lists the real gap-filled months (days missing, and whether the missing days are one run) to compare k.

Usage: python pipeline/experiments/e9_gapfill_test.py
Writes pipeline/experiments/results/e9/gapfill_error.csv, gapfill_error_by_region.csv, real_gap_months.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harmonizer.config import load_settings  # noqa: E402

YEARS = range(2015, 2020)
K_MIN = 3


def daily_region_counts(grid):
    """Hotspots per region x day/night x day, S-NPP 2015 to 2019 (low confidence dropped, static mask already applied)."""
    reg = pd.concat([pd.read_parquet(grid / "monthly" / f"viirs_snpp_{y}.parquet", columns=["row", "col", "region"])
                     for y in YEARS]).drop_duplicates(["row", "col"])
    parts = []
    for y in YEARS:
        cd = pd.read_parquet(grid / "cell_days" / f"viirs_snpp_{y}.parquet",
                             columns=["row", "col", "date", "daynight", "n_det", "n_det_low"])
        cd["n"] = cd["n_det"] - cd["n_det_low"]
        cd = cd.merge(reg, on=["row", "col"], how="left")
        parts.append(cd.groupby(["region", "daynight", "date"], as_index=False)["n"].sum())
        print(f"loaded {y}", flush=True)
    d = pd.concat(parts, ignore_index=True)
    d = d[d["region"].notna() & (d["region"] != "OCEAN")]
    both = d.groupby(["region", "date"], as_index=False)["n"].sum().assign(daynight="ALL")
    d = pd.concat([d, both], ignore_index=True)
    d["date"] = pd.to_datetime(d["date"])
    d["month"] = d["date"].dt.to_period("M").dt.to_timestamp()
    d["day"] = d["date"].dt.day
    return d


def simulate(d):
    """Hide every run of k consecutive days, scale back up, keep truth and filled value."""
    out = []
    for (region, dn, month), g in d.groupby(["region", "daynight", "month"]):
        ndays = month.days_in_month
        daily = np.zeros(ndays)
        np.add.at(daily, g["day"].to_numpy() - 1, g["n"].to_numpy())
        truth = daily.sum()
        if truth <= 0:
            continue
        cs = np.concatenate([[0.0], np.cumsum(daily)])
        for k in range(K_MIN, ndays // 3 + 1):
            starts = np.arange(0, ndays - k + 1)
            hidden = cs[starts + k] - cs[starts]
            filled = (truth - hidden) * ndays / (ndays - k)
            out.append(pd.DataFrame({"region": region, "daynight": dn, "month": month, "k": k,
                                     "truth": truth, "filled": filled}))
    return pd.concat(out, ignore_index=True)


def score(x):
    err = x["filled"] - x["truth"]
    rel = err / x["truth"]
    return pd.Series({"cases": len(x),
                      "fire_weighted_error_pct": 100 * err.abs().sum() / x["truth"].sum(),
                      "median_error_pct": 100 * rel.abs().median(),
                      "p90_error_pct": 100 * rel.abs().quantile(0.9),
                      "relative_rmse_pct": 100 * np.sqrt((rel ** 2).mean()),
                      "bias_pct": 100 * err.sum() / x["truth"].sum()})


def real_gap_months(grid):
    """Real S-NPP months with more than 2 days missing, and whether the missing days form one run."""
    days = pd.read_parquet(grid / "days_observed.parquet")
    g = days[(days["satellite"] == "SNPP") & (days["days_missing"] > 2)].copy()
    rows = []
    for _, r in g.iterrows():
        y = r["month"].year
        f = grid / "cell_days" / f"viirs_snpp_{y}.parquet"
        seen = pd.to_datetime(pd.read_parquet(f, columns=["date"])["date"]).dt.normalize().unique() if f.exists() else []
        allm = pd.date_range(r["month"], periods=r["days_in_month"], freq="D")
        miss = allm[~allm.isin(seen)]
        runs = int((np.diff(miss.day) != 1).sum() + 1) if len(miss) else 0
        rows.append({"month": r["month"].strftime("%Y-%m"), "days_observed": r["days_observed"],
                     "days_missing": r["days_missing"], "share_seen": round(r["days_observed"] / r["days_in_month"], 2),
                     "missing_runs": runs, "missing_days": ",".join(str(x) for x in miss.day)})
    return pd.DataFrame(rows)


def main():
    s = load_settings()
    grid = Path(s["paths"]["grid"])
    res = Path(__file__).resolve().parent / "results" / "e9"
    res.mkdir(parents=True, exist_ok=True)

    real = real_gap_months(grid)
    print("\nReal S-NPP months with more than 2 days missing:")
    print(real.to_string(index=False))
    real.to_csv(res / "real_gap_months.csv", index=False)

    sim = simulate(daily_region_counts(grid))
    by_k = sim.groupby(["daynight", "k"]).apply(score).reset_index()
    by_k.to_csv(res / "gapfill_error.csv", index=False)
    print("\nGap-fill error by day/night and days hidden (k), region x month level, 2015 to 2019:")
    print(by_k.round(2).to_string(index=False))

    by_reg = sim[sim["daynight"] == "ALL"].groupby("region").apply(score).reset_index()
    by_reg.to_csv(res / "gapfill_error_by_region.csv", index=False)
    print("\nBy region (D + N, all k):")
    print(by_reg.round(2).to_string(index=False))
    print(f"\nWrote {res}")


if __name__ == "__main__":
    main()
