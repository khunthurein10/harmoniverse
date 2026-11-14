"""
E6 (step 7): how should MODIS swath-edge pixels be handled?

  V1 none       : all Aqua detections count 1 (n_det)
  V2 drop edges : only Aqua detections with scan angle <= 30 degrees (n_det_centre); MRBA60 Methods
  V3 weight     : Aqua detections weighted by position across the swath (n_det_w); GFED5NRT supplement S2,
                  adapted to MODIS (harmonizer/angle.py)
Target: S-NPP hotspots without low confidence (decision A2). Ratio split day/night x region x month (E2).
Test: choosing years 2015 to 2019, leave one year out. Scored two ways (our choice):
  regional  = region x month x day/night totals with >= 100 S-NPP hotspots (as E2)
  box       = 0.25 degree box x month x day/night with >= 20 S-NPP hotspots and Aqua > 0
Usage: python pipeline/experiments/e6_viewing_angle.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harmonizer.config import load_settings  # noqa: E402

YEARS = range(2015, 2020)
BY = ["daynight", "region", "cmonth"]
VARIANTS = {"V1 none": "n_det", "V2 drop edges (MRBA60)": "n_det_centre", "V3 weight edges (GFED5NRT-style)": "n_det_w"}


def load(grid_root):
    keys = ["row", "col", "month", "daynight", "region"]
    a = pd.concat([pd.read_parquet(grid_root / "monthly" / f"modis_{y}.parquet",
                                   columns=keys + ["satellite", "n_det", "n_det_centre", "n_det_w"]) for y in YEARS])
    a = a[a["satellite"] == "AQUA"].groupby(keys, as_index=False)[["n_det", "n_det_centre", "n_det_w"]].sum()
    v = pd.concat([pd.read_parquet(grid_root / "monthly" / f"viirs_snpp_{y}.parquet",
                                   columns=keys + ["n_det", "n_det_low"]) for y in YEARS])
    v["snpp"] = v["n_det"] - v["n_det_low"]
    v = v.groupby(keys, as_index=False)["snpp"].sum()
    w = a.merge(v, on=keys, how="outer").fillna({"n_det": 0, "n_det_centre": 0, "n_det_w": 0, "snpp": 0})
    w = w[w["region"] != "OCEAN"]
    w["month"] = pd.to_datetime(w["month"])
    w["year"], w["cmonth"] = w["month"].dt.year, w["month"].dt.month
    return w


def score(w, aqua_col):
    reg, box = [], []
    for y in YEARS:
        tr, te = w[w["year"] != y], w[w["year"] == y].copy()
        r = tr.groupby(BY)[["snpp", aqua_col]].sum()
        r = (r["snpp"] / r[aqua_col]).replace(np.inf, np.nan).rename("ratio")
        te = te.join(r, on=BY)
        te["ratio"] = te["ratio"].fillna(tr["snpp"].sum() / tr[aqua_col].sum())
        te["pred"] = te[aqua_col] * te["ratio"]
        g = te.groupby(["region", "month", "daynight"])[["pred", "snpp"]].sum()
        g = g[g["snpp"] >= 100]
        reg.append((g["pred"] - g["snpp"]).abs() / g["snpp"])
        b = te[(te["snpp"] >= 20) & (te[aqua_col] > 0)]
        box.append(pd.DataFrame({"err": (b["pred"] - b["snpp"]).abs() / b["snpp"],
                                 "logerr": np.abs(np.log(b["pred"] / b["snpp"])), "daynight": b["daynight"]}))
    reg, box = pd.concat(reg), pd.concat(box)
    return {"regional_median_err_pct": round(100 * reg.median(), 1),
            "box_median_err_pct": round(100 * box["err"].median(), 1),
            "box_day_median_err_pct": round(100 * box.loc[box["daynight"] == "D", "err"].median(), 1),
            "box_night_median_err_pct": round(100 * box.loc[box["daynight"] == "N", "err"].median(), 1),
            "box_typical_factor_off": round(float(np.exp(box["logerr"].median())), 2),
            "boxes_scored": len(box)}


def main():
    s = load_settings()
    out = Path(__file__).resolve().parent / "results" / "e6"
    out.mkdir(parents=True, exist_ok=True)
    w = load(s["paths"]["grid"])
    rows = [{"variant": k, "aqua_detections_used": int(w[c].sum()), **score(w, c)} for k, c in VARIANTS.items()]
    t = pd.DataFrame(rows)
    t.to_csv(out / "viewing_angle.csv", index=False)
    print(t.to_string(index=False))


if __name__ == "__main__":
    main()
