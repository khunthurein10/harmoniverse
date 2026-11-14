"""
Final test (step 12, run ONCE): train on the choosing years 2015 to 2019, predict the untouched 2012 to 2014.

Pre-set rule (Nov 12, before running): main method = A (gradient boosting, day and night models). A is kept
if it beats the chosen table on 2012 to 2014; otherwise the table is used. B (table by day, boosting at night)
and the table are reported alongside.

  Table : B-fb (B-3m-fb in BONA and TENA), ratio of sums, see e8_conversion.py
  A     : gradient boosting per day/night (e8b_model.py settings, unchanged), falling back to the table when a
          row's Aqua count is above its region x day/night training maximum (MRBA60 BAmax-style rule) or Terra
          has an outage that month
  B     : table by day, A at night
Outage months (> 2 days missing) are left out of scoring. Also saves the trained models and table for step 11,
and prints which inputs the models rely on (gain importance).

Usage: python pipeline/experiments/final_test.py [--quick]
"""
import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from harmonizer.config import load_settings  # noqa: E402
from e8b_model import MODELS, ROUNDS, load  # noqa: E402
from harmonizer.convert import FEATS, apply_table, table_ratios  # noqa: E402
import e8b_model  # noqa: E402

TRAIN, TEST = list(range(2015, 2020)), list(range(2012, 2015))
GB = "Gradient boosting"
RK = ["daynight", "region"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    s = load_settings()
    grid = s["paths"]["grid"]
    out = HERE / "results" / "final_test"
    model_dir = s["paths"]["grid"].parent / ("models_quick" if args.quick else "models")
    out.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    global TRAIN, TEST
    if args.quick:
        TRAIN, TEST = [2016], [2013]
    e8b_model.YEARS = TRAIN + TEST
    w = load(grid, args.quick)
    obs = pd.read_parquet(grid / "days_observed.parquet")
    bad = obs[obs["days_missing"] > 2]
    bad_months = {(r.satellite, pd.Timestamp(r.month)) for r in bad.itertuples()}
    w["terra_outage"] = [("TERRA", m) in bad_months for m in w["month"]]
    w["scored"] = [not (("AQUA", m) in bad_months or ("SNPP", m) in bad_months) for m in w["month"]]
    tr, te = w[w["year"].isin(TRAIN)].copy(), w[w["year"].isin(TEST)].copy()
    print(f"train {len(tr):,} rows (2015 to 2019), test {len(te):,} rows (2012 to 2014)")

    # table
    tabs = table_ratios(tr)
    te["p_Table"], _ = apply_table(te, tabs)

    # gradient boosting, one model per day/night
    te["p_GBraw"] = np.nan
    imp = {}
    rounds = 20 if args.quick else ROUNDS[GB]
    for dn in ["D", "N"]:
        t = tr[tr["daynight"] == dn]
        mdl = lgb.train(MODELS[GB], lgb.Dataset(t[FEATS], t["snpp"], categorical_feature=["region_c", "land_c"]),
                        num_boost_round=rounds)
        mdl.save_model(str(model_dir / f"gb_{'day' if dn == 'D' else 'night'}.txt"))
        sel = te["daynight"] == dn
        te.loc[sel, "p_GBraw"] = mdl.predict(te.loc[sel, FEATS])
        g = pd.Series(mdl.feature_importance("gain"), index=FEATS)
        imp[dn] = (100 * g / g.sum()).round(1)

    # fallback rule
    tmax = tr.groupby(RK)["aqua"].max().rename("train_max")
    te = te.join(tmax, on=RK)
    use_table = (te["aqua"] > te["train_max"].fillna(0)) | te["terra_outage"]
    te["p_A"] = np.where(use_table, te["p_Table"], te["p_GBraw"])
    te["p_B"] = np.where(te["daynight"] == "D", te["p_Table"], te["p_A"])
    print(f"rows falling back to the table: {int(use_table.sum()):,} ({use_table.mean() * 100:.3f}%)")

    # scoring at box x month x day/night (land summed), outage months excluded
    sc = te[te["scored"]]
    bk = ["row", "col", "month", "daynight", "region"]
    methods = {"Table": "p_Table", "A (boosting day+night)": "p_A", "B (table day, boosting night)": "p_B"}
    p = sc.groupby(bk, as_index=False)[["snpp", "aqua"] + list(methods.values())].sum()
    g = p.groupby(["region", "month", "daynight"])[["snpp"] + list(methods.values())].sum().reset_index()
    g = g[g["snpp"] >= 100]
    a = p.assign(y=p["month"].dt.year).groupby(["region", "y"])[["snpp"] + list(methods.values())].sum()
    b = p[(p["snpp"] >= 20) & (p["aqua"] > 0)]
    rows, per_region = [], []
    for name, c in methods.items():
        e = (g[c] - g["snpp"]).abs(); rel = e / g["snpp"]
        be = (b[c] - b["snpp"]).abs() / b["snpp"]
        rows.append({"method": name, "regional_median_err": round(100 * rel.median(), 1),
                     "regional_fire_weighted_err": round(100 * e.sum() / g["snpp"].sum(), 1),
                     "regional_day_median": round(100 * rel[g.daynight == "D"].median(), 1),
                     "regional_night_median": round(100 * rel[g.daynight == "N"].median(), 1),
                     "annual_region_median": round(100 * ((a[c] - a["snpp"]).abs() / a["snpp"]).median(), 1),
                     "box_median_err": round(100 * be.median(), 1),
                     "box_night_median": round(100 * be[b.daynight == "N"].median(), 1),
                     "total_pred_vs_actual_pct": round(100 * (p[c].sum() / p["snpp"].sum() - 1), 1)})
        per_region.append(g.assign(e=e).groupby("region").apply(lambda x: x["e"].sum() / x["snpp"].sum() * 100,
                                                                include_groups=False).rename(name))
    t = pd.DataFrame(rows)
    pr = pd.concat(per_region, axis=1).round(1)
    imp_t = pd.DataFrame(imp).rename(columns={"D": "day_model_%", "N": "night_model_%"}).sort_values("day_model_%", ascending=False)
    tag = "_quick" if args.quick else ""
    t.to_csv(out / f"final_test{tag}.csv", index=False)
    pr.to_csv(out / f"final_test_per_region{tag}.csv")
    imp_t.to_csv(out / f"importance{tag}.csv")
    (model_dir / "fallback_train_max.json").write_text(json.dumps({f"{k[0]}|{k[1]}": int(v) for k, v in tmax.items()}))
    print(t.to_string(index=False))
    print("\nFire-weighted regional error by region (%)\n", pr.to_string())
    print("\nWhat the models rely on (share of gain, %)\n", imp_t.to_string())


if __name__ == "__main__":
    main()
