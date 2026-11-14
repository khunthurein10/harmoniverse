"""
E8c (step 8 + final test, re-run after fixing a leak found on Nov 12).

The leak: in 2012+ the data has rows for boxes where only S-NPP saw fire (Aqua = 0 and Terra = 0). Those rows
exist only because S-NPP saw something, so a model trained and scored on them is told "fire here" by the
row's existence. Before 2012 such rows cannot exist. Fix (MRBA60 Methods: "Harmonisation was restricted to cells
with Fire Mask = 1, preventing artificial BA generation in areas without fire evidence"):
  * the model is trained and applied only on rows with MODIS fire evidence (Aqua > 0 or Terra > 0)
  * fires S-NPP saw in boxes without MODIS evidence ("VIIRS-only share" m) are added back at the regional level:
    regional estimate = sum of box predictions / (1 - m), m learned per region x day/night x calendar month
    from the training years (falls back to region x day/night when the group has < 500 S-NPP hotspots; our choice)
  * the table needs no change at the regional level: its ratio (all S-NPP / Aqua) already includes those fires

Scored as before: validation = 2015 to 2019 leave one year out; final test = train 2015 to 2019, predict 2012 to
2014 (second use of these years, disclosed: the first run had the leak). Box scores use rows with MODIS evidence.

Usage: python pipeline/experiments/e8c_fixed.py [--quick]
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
import e8b_model  # noqa: E402
from e8b_model import MODELS, ROUNDS  # noqa: E402
from harmonizer.convert import FEATS, apply_table, table_ratios  # noqa: E402

GB = "Gradient boosting"
RK = ["daynight", "region"]
VAL, FINAL_TRAIN, FINAL_TEST = list(range(2015, 2020)), list(range(2015, 2020)), list(range(2012, 2015))
MIN_SNPP_M = 500


def viirs_only_share(tr):
    """m = share of S-NPP hotspots in rows without MODIS evidence, per region x day/night x calendar month."""
    tr = tr.assign(snpp_noev=np.where(tr["evidence"], 0, tr["snpp"]))
    g = tr.groupby(RK + ["cmonth"])[["snpp", "snpp_noev"]].sum()
    c = tr.groupby(RK)[["snpp", "snpp_noev"]].sum()
    m = (g["snpp_noev"] / g["snpp"]).where(g["snpp"] >= MIN_SNPP_M)
    coarse = (c["snpp_noev"] / c["snpp"]).reindex(g.index.droplevel("cmonth")).to_numpy()
    m = m.fillna(pd.Series(coarse, index=g.index)).clip(0, 0.9)
    return m.rename("m")


def fit_predict(tr, te, rounds):
    """Train GB per day/night on evidence rows of tr; predict evidence rows of te (non-evidence rows -> 0)."""
    te = te.copy()
    te["p_gb"] = 0.0
    models = {}
    for dn in ["D", "N"]:
        t = tr[(tr["daynight"] == dn) & tr["evidence"]]
        mdl = lgb.train(MODELS[GB], lgb.Dataset(t[FEATS], t["snpp"], categorical_feature=["region_c", "land_c"]),
                        num_boost_round=rounds)
        models[dn] = mdl
        sel = (te["daynight"] == dn) & te["evidence"]
        te.loc[sel, "p_gb"] = mdl.predict(te.loc[sel, FEATS])
    # fallback rule (unchanged): above training max or Terra outage -> evidence-only table
    tabs_ev = table_ratios(tr[tr["evidence"]])
    te["p_tab_ev"], _ = apply_table(te, tabs_ev)
    tmax = tr.groupby(RK)["aqua"].max().rename("train_max")
    te = te.join(tmax, on=RK)
    fb = te["evidence"] & ((te["aqua"] > te["train_max"].fillna(0)) | te["terra_outage"])
    te["p_gb"] = np.where(fb, te["p_tab_ev"], te["p_gb"])
    te["fallback"] = fb
    # table baseline (as chosen before, all-rows ratio)
    te["p_table"], _ = apply_table(te, table_ratios(tr))
    return te, models, tmax


def score(te, m):
    """Regional (with VIIRS-only share added back for GB), annual, box metrics."""
    te = te[te["scored"]]
    g = te.groupby(["region", "month", "daynight", "cmonth"])[["snpp", "p_table", "p_gb"]].sum().reset_index()
    g = g.join(m, on=RK + ["cmonth"])
    g["p_gb_reg"] = g["p_gb"] / (1 - g["m"].fillna(0))
    rows, per_region = {}, {}
    gg = g[g["snpp"] >= 100]
    a = g.assign(y=g["month"].dt.year).groupby(["region", "y"])[["snpp", "p_table", "p_gb_reg"]].sum()
    bk = ["row", "col", "month", "daynight"]
    b = te[te["evidence"]].groupby(bk)[["snpp", "aqua", "p_table", "p_gb"]].sum()
    b = b[(b["snpp"] >= 20) & (b["aqua"] > 0)]
    for name, c, cb in [("Table", "p_table", "p_table"), ("Boosting (fixed)", "p_gb_reg", "p_gb")]:
        e = (gg[c] - gg["snpp"]).abs(); rel = e / gg["snpp"]
        be = (b[cb] - b["snpp"]).abs() / b["snpp"]
        night = b.index.get_level_values("daynight") == "N"
        rows[name] = {"regional_median_err": round(100 * rel.median(), 1),
                      "regional_fire_weighted_err": round(100 * e.sum() / gg["snpp"].sum(), 1),
                      "regional_day_median": round(100 * rel[gg["daynight"] == "D"].median(), 1),
                      "regional_night_median": round(100 * rel[gg["daynight"] == "N"].median(), 1),
                      "annual_region_median": round(100 * ((a[c] - a["snpp"]).abs() / a["snpp"]).median(), 1),
                      "box_median_err": round(100 * be.median(), 1),
                      "box_night_median": round(100 * be[night].median(), 1),
                      "total_pred_vs_actual_pct": round(100 * (g[c].sum() / g["snpp"].sum() - 1), 1)}
        per_region[name] = gg.assign(e=e).groupby("region").apply(
            lambda x: x["e"].sum() / x["snpp"].sum() * 100, include_groups=False)
    return pd.DataFrame(rows).T, pd.DataFrame(per_region).round(1)


def prep(w, grid):
    obs = pd.read_parquet(grid / "days_observed.parquet")
    obs["month"] = pd.to_datetime(obs["month"])
    bad = obs[obs["days_missing"] > 2]
    w["terra_outage"] = w["month"].isin(set(bad.loc[bad.satellite == "TERRA", "month"]))
    w["scored"] = ~w["month"].isin(set(bad.loc[bad.satellite.isin(["AQUA", "SNPP"]), "month"]))
    w["evidence"] = (w["aqua"] > 0) | (w["terra"] > 0)
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    s = load_settings()
    grid = s["paths"]["grid"]
    out = HERE / "results" / "e8c"
    out.mkdir(parents=True, exist_ok=True)
    rounds = 20 if args.quick else ROUNDS[GB]
    val_years = [2016, 2017] if args.quick else VAL
    tr_years, te_years = ([2016], [2013]) if args.quick else (FINAL_TRAIN, FINAL_TEST)

    e8b_model.YEARS = sorted(set(val_years) | set(tr_years) | set(te_years))
    w = prep(e8b_model.load(grid, args.quick), grid)
    print(f"{len(w):,} rows; rows without MODIS evidence hold {w.loc[~w.evidence, 'snpp'].sum() / w['snpp'].sum() * 100:.1f}% of S-NPP hotspots", flush=True)

    # validation
    parts = []
    for y in val_years:
        tr, te = w[w["year"].isin(val_years) & (w["year"] != y)], w[w["year"] == y]
        te, _, _ = fit_predict(tr, te, rounds)
        te = te.join(viirs_only_share(tr), on=RK + ["cmonth"])
        parts.append(te)
        print(f"  validation: held-out {y} done", flush=True)
    v = pd.concat(parts)
    vs, vpr = score(v.drop(columns="m"), viirs_only_share(w[w["year"].isin(val_years)]))
    # note: per-fold m was joined above; score() rejoins m from all validation years only for grouping keys,
    # so recompute per fold for honesty
    folds = []
    for y in val_years:
        tr = w[w["year"].isin(val_years) & (w["year"] != y)]
        folds.append(score(v[v["year"] == y].drop(columns="m"), viirs_only_share(tr))[0])
    print("\nVALIDATION (2015 to 2019, leave one year out; per-fold average)\n", (sum(folds) / len(folds)).round(1).to_string())

    # final test
    tr, te = w[w["year"].isin(tr_years)], w[w["year"].isin(te_years)]
    te, models, tmax = fit_predict(tr, te, rounds)
    m = viirs_only_share(tr)
    fs, fpr = score(te, m)
    print("\nFINAL TEST (train 2015 to 2019, predict 2012 to 2014; second use, disclosed)\n", fs.to_string())
    print("\nFinal test, fire-weighted regional error by region (%)\n", fpr.to_string())
    print(f"\nrows using table fallback: {te['fallback'].mean() * 100:.3f}%")
    imp = pd.DataFrame({dn: pd.Series(mdl.feature_importance("gain"), index=FEATS) for dn, mdl in models.items()})
    print("\nWhat the models rely on (share of gain, %)\n", (100 * imp / imp.sum()).round(1).to_string())

    tag = "_quick" if args.quick else ""
    (sum(folds) / len(folds)).round(1).to_csv(out / f"validation{tag}.csv")
    fs.to_csv(out / f"final_test{tag}.csv")
    fpr.to_csv(out / f"final_test_per_region{tag}.csv")
    if not args.quick:
        mdir = grid.parent / "models"
        for dn, mdl in models.items():
            mdl.save_model(str(mdir / f"gb_{'day' if dn == 'D' else 'night'}.txt"))
        (mdir / "fallback_train_max.json").write_text(json.dumps({f"{k[0]}|{k[1]}": int(x) for k, x in tmax.items()}))
        m.reset_index().to_parquet(mdir / "viirs_only_share.parquet", index=False)
        print(f"\nSaved fixed models, fallback limits and VIIRS-only shares in {mdir}")


if __name__ == "__main__":
    main()
