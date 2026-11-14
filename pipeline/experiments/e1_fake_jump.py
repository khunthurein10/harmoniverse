"""
E1 (step 5): how big is the fake jump if MODIS and VIIRS are simply joined at 2012?

Baseline only, no model, no paper behind it (our choice): every conversion method must beat it.

Joined record = Aqua up to 2011, S-NPP from 2012 (what a naive user would do).
Same-sensor record = Aqua all the way through.
For each GFED region, compare the 4 years before the seam (2008 to 2011) with the 4 after (2012 to 2015):
  true change   = Aqua after / Aqua before - 1      (what the fires really did, as Aqua sees them)
  joined change = S-NPP after / Aqua before - 1     (what the joined record shows)
  fake jump     = S-NPP after / Aqua after          (how much of the joined change is the sensor switch)
Counts use the final target (VIIRS low confidence left out, decision A2; an earlier run that kept it gave x6.8).
Means are per month, day and night together. Outage months are left out for both sensors
(S-NPP Jan 2012, see days_observed.parquet).

Usage (from the project root):
    python pipeline/experiments/e1_fake_jump.py
Writes pipeline/experiments/results/e1/fake_jump.csv and fake_jump.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harmonizer.config import load_settings  # noqa: E402

BEFORE, AFTER = (2008, 2011), (2012, 2015)
AQUA_COLOR, JOINED_COLOR = "#2a78d6", "#eb6834"   # dataviz reference palette, slots 1 and 2 (light)
INK, MUTED, GRID, SURFACE = "#1f1e1c", "#6b6a64", "#e4e3de", "#fcfcfb"
REGION_ORDER = ["BONA", "TENA", "CEAM", "NHSA", "SHSA", "EURO", "MIDE", "NHAF", "SHAF",
                "BOAS", "CEAS", "SEAS", "EQAS", "AUST"]


def load_monthly(grid_root):
    parts = []
    for f in sorted((grid_root / "monthly").glob("*.parquet")):
        m = pd.read_parquet(f, columns=["month", "satellite", "region", "n_det", "n_det_low"])
        m["n_det"] = m["n_det"] - m["n_det_low"]     # final target: VIIRS low confidence left out (decision A2)
        parts.append(m[m["satellite"].isin(["AQUA", "SNPP"])].drop(columns="n_det_low"))
    m = pd.concat(parts, ignore_index=True)
    m = m.groupby(["region", "satellite", "month"], as_index=False)["n_det"].sum()
    return m


def outage_months(grid_root, sats=("AQUA", "SNPP")):
    d = pd.read_parquet(grid_root / "days_observed.parquet")
    d = d[d["satellite"].isin(sats) & (d["days_missing"] > 2)]
    return set(pd.to_datetime(d["month"]))


def main():
    s = load_settings()
    grid_root = s["paths"]["grid"]
    out = Path(__file__).resolve().parent / "results" / "e1"
    out.mkdir(parents=True, exist_ok=True)

    m = load_monthly(grid_root)
    m["month"] = pd.to_datetime(m["month"])
    bad = outage_months(grid_root)
    m = m[~m["month"].isin(bad)]
    m = m[m["region"] != "OCEAN"]
    world = m.groupby(["satellite", "month"], as_index=False)["n_det"].sum().assign(region="WORLD")
    m = pd.concat([m, world], ignore_index=True)
    m["year"] = m["month"].dt.year

    # every region x satellite x month present (a month with no fires counts as 0)
    months = sorted(m["month"].unique())
    idx = pd.MultiIndex.from_product([m["region"].unique(), ["AQUA", "SNPP"], months], names=["region", "satellite", "month"])
    m = m.set_index(["region", "satellite", "month"])["n_det"].reindex(idx, fill_value=0).reset_index()
    m["year"] = m["month"].dt.year
    m = m[~((m["satellite"] == "SNPP") & (m["year"] < 2012))]

    def per_month(sat, yrs):
        x = m[(m["satellite"] == sat) & m["year"].between(*yrs)]
        return x.groupby("region")["n_det"].mean()

    aq_b, aq_a, sn_a = per_month("AQUA", BEFORE), per_month("AQUA", AFTER), per_month("SNPP", AFTER)
    res = pd.DataFrame({
        "aqua_per_month_2008_11": aq_b.round(0), "aqua_per_month_2012_15": aq_a.round(0),
        "snpp_per_month_2012_15": sn_a.round(0),
        "true_change_pct": ((aq_a / aq_b - 1) * 100).round(1),
        "joined_change_pct": ((sn_a / aq_b - 1) * 100).round(1),
        "fake_jump_factor": (sn_a / aq_a).round(2),
    })
    res = res.loc[[r for r in ["WORLD"] + REGION_ORDER if r in res.index]]
    res.to_csv(out / "fake_jump.csv")
    print(res.to_string())

    # small multiples: hotspots per month, yearly average
    yearly = m.groupby(["region", "satellite", "year"])["n_det"].mean().unstack("satellite")
    regions = list(res.index)
    fig, axes = plt.subplots(3, 5, figsize=(17, 9.5), facecolor=SURFACE)
    for ax, reg in zip(axes.flat, regions):
        y = yearly.loc[reg]
        ax.set_facecolor(SURFACE)
        ax.plot(y.index, y["AQUA"], color=AQUA_COLOR, lw=2, label="Aqua MODIS (same sensor)")
        joined = pd.concat([y["AQUA"][y.index == 2011], y["SNPP"][y.index >= 2012]])
        ax.plot(joined.index, joined.values, color=JOINED_COLOR, lw=2, ls="--", label="Joined record (S-NPP VIIRS from 2012)")
        ax.axvline(2011.5, color=MUTED, lw=1, ls=":")
        ax.set_title(f"{reg}   fake jump x{res.loc[reg, 'fake_jump_factor']:.1f}", fontsize=10, color=INK, loc="left")
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_ylim(bottom=0)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        for sp in ["left", "bottom"]:
            ax.spines[sp].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.set_xticks([2003, 2007, 2011, 2015, 2019])
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v/1000:.0f}k" if v >= 1000 else f"{v:.0f}"))
    for ax in list(axes.flat)[len(regions):]:
        ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 0.955), ncol=2, frameon=False, fontsize=10, labelcolor=INK)
    fig.suptitle("E1: joining MODIS and VIIRS at 2012 creates a fake jump (hotspots per month, yearly average, GFED regions)",
                 x=0.01, ha="left", fontsize=13, color=INK)
    fig.text(0.01, 0.005, "Data: NASA FIRMS standard processing, cleaned (type 0, MODIS conf >= 30%, VIIRS static mask). "
             "Outage month S-NPP Jan 2012 left out for both sensors. Dotted line = sensor switch.", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 0.92))
    fig.savefig(out / "fake_jump.png", dpi=150, facecolor=SURFACE)
    print(f"\nWrote {out / 'fake_jump.csv'} and {out / 'fake_jump.png'}")


if __name__ == "__main__":
    main()
