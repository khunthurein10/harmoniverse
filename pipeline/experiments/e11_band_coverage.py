"""
E11: do the 90% ranges hold about 90% of the true values? (band coverage)

The regional uncertainty (03_harmonize.py) is one straight line per day/night, RMSE = a + b x prediction, fitted on
2012 to 2014 region-months (MRBA60-style binned RMSE). Checked here on the same region-months, where the truth
(real S-NPP) is known, without reusing a year's own errors: leave one year out (fit on two years, check the third).
Coverage = share of region-months whose truth lies within prediction +/- 1.645 x RMSE; it should be near 90%.
Split by size of the month, because one line may be too wide for small months and too narrow for large ones.

Compared (our choice of candidates):
  linear   : RMSE = a + b x prediction (current)
  relative : RMSE = r x prediction, r = RMS relative error per size bin (10 bins of prediction)
  binned   : RMSE per size bin in hotspots (piecewise constant)
Usage: python pipeline/experiments/e11_band_coverage.py
Writes pipeline/experiments/results/e11/coverage.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harmonizer.config import load_settings  # noqa: E402

Z90 = 1.645
YEARS = [2012, 2013, 2014]


def fit_linear(d, nbins=10):
    d = d[d["pred"] > 0].copy()
    d["bin"] = pd.qcut(d["pred"], nbins, labels=False, duplicates="drop")
    g = d.groupby("bin").apply(lambda x: pd.Series({"pred": x["pred"].mean(),
                                                    "rmse": np.sqrt(((x["pred"] - x["obs"]) ** 2).mean())}),
                               include_groups=False)
    b, a = np.polyfit(g["pred"], g["rmse"], 1)
    return lambda p: np.maximum(a + b * p, 0)


def fit_relative(d, nbins=10):
    d = d[d["pred"] > 0].copy()
    edges = np.unique(np.quantile(d["pred"], np.linspace(0, 1, nbins + 1)))
    d["bin"] = np.clip(np.searchsorted(edges, d["pred"], side="right") - 1, 0, len(edges) - 2)
    r = d.groupby("bin").apply(lambda x: np.sqrt((((x["pred"] - x["obs"]) / x["pred"]) ** 2).mean()), include_groups=False)
    def f(p):
        k = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
        return r.reindex(k).to_numpy() * p
    return f


def fit_binned(d, nbins=10):
    """Absolute RMSE per size bin, no line (piecewise constant)."""
    d = d[d["pred"] > 0].copy()
    edges = np.unique(np.quantile(d["pred"], np.linspace(0, 1, nbins + 1)))
    d["bin"] = np.clip(np.searchsorted(edges, d["pred"], side="right") - 1, 0, len(edges) - 2)
    r = d.groupby("bin").apply(lambda x: np.sqrt(((x["pred"] - x["obs"]) ** 2).mean()), include_groups=False)
    def f(p):
        k = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
        return r.reindex(k).to_numpy()
    return f


def main():
    s = load_settings()
    out = Path(__file__).resolve().parent / "results" / "e11"
    out.mkdir(parents=True, exist_ok=True)
    r = pd.read_parquet(s["paths"]["processed"] / "harmonized_region_month.parquet")
    r = r[(r["month"].dt.year.isin(YEARS)) & (r["source"] == "S-NPP observed") & (r["pred"] > 0)]
    d = r.rename(columns={"snpp_obs": "obs"})[["region", "month", "daynight", "pred", "obs"]].copy()
    d["year"] = d["month"].dt.year
    rows = []
    for dn, x in d.groupby("daynight"):
        for y in YEARS:
            train, test = x[x["year"] != y], x[x["year"] == y].copy()
            for name, fit in [("linear", fit_linear), ("relative", fit_relative), ("binned", fit_binned)]:
                sig = fit(train)(test["pred"].to_numpy())
                test[name] = np.abs(test["obs"] - test["pred"]) <= Z90 * sig
                test[name + "_width_pct"] = 100 * Z90 * sig / test["obs"].clip(lower=1)
            rows.append(test)
    t = pd.concat(rows)
    t["size"] = pd.qcut(t["obs"], 5, labels=["smallest 20%", "small", "middle", "large", "largest 20%"])
    summ = t.groupby(["daynight", "size"], observed=True).agg(
        months=("obs", "size"), median_true=("obs", "median"),
        linear_coverage_pct=("linear", lambda v: 100 * v.mean()), linear_half_width_pct=("linear_width_pct", "median"),
        relative_coverage_pct=("relative", lambda v: 100 * v.mean()), relative_half_width_pct=("relative_width_pct", "median"),
        binned_coverage_pct=("binned", lambda v: 100 * v.mean()), binned_half_width_pct=("binned_width_pct", "median"))
    allr = t.groupby("daynight").agg(linear_coverage_pct=("linear", lambda v: 100 * v.mean()),
                                     relative_coverage_pct=("relative", lambda v: 100 * v.mean()),
                                     binned_coverage_pct=("binned", lambda v: 100 * v.mean()))
    summ.round(1).to_csv(out / "coverage.csv")
    print("Coverage of the 90% range, leave one year out on 2012 to 2014 (target about 90%):")
    print(allr.round(1).to_string())
    print("\nBy size of the month (half width = 1.645 x RMSE as % of the true value, median):")
    print(summ.round(1).to_string())


if __name__ == "__main__":
    main()
