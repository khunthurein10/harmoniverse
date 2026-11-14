"""
Pitch figure: the 2012 seam before and after harmonization (world, yearly hotspots).

Left : what a naive record shows (Aqua MODIS up to 2011, S-NPP VIIRS from 2012), plus Aqua continuing (same sensor).
Right: the harmonized record (S-NPP scale): converted before 2012 with its 90% range, real S-NPP from 2012.
Reads data/processed/harmonized_region_month.parquet. Writes reports/figures/seam_before_after.png
Usage: python pipeline/scripts/05_seam_chart.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harmonizer.config import load_settings  # noqa: E402

BLUE, ORANGE, INK, MUTED, GRID, SURFACE = "#2a78d6", "#eb6834", "#1f1e1c", "#6b6a64", "#e4e3de", "#fcfcfb"
Z90 = 1.645


def main():
    s = load_settings()
    r = pd.read_parquet(s["paths"]["processed"] / "harmonized_region_month.parquet")
    r["year"] = r["month"].dt.year
    full = r.groupby("year")["month"].nunique()
    years = full[full == 12].index          # whole years only, so partial years do not look like drops
    y = r[r["year"].isin(years)].groupby("year").agg(aqua=("aqua", "sum"), snpp=("snpp_obs", "sum"),
                                                      harm=("harmonized", "sum"),
                                                      var=("sigma", lambda v: float((v.fillna(0) ** 2).sum())))
    y.loc[y.index < 2012, "snpp"] = np.nan
    y.loc[y.index >= 2026, "aqua"] = np.nan
    naive = y["aqua"].where(y.index < 2012, y["snpp"])
    band = Z90 * np.sqrt(y["var"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), facecolor=SURFACE, sharey=True)
    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.axvline(2011.5, color=MUTED, lw=1, ls=":")
        ax.grid(axis="y", color=GRID, lw=0.8)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        for sp in ["left", "bottom"]:
            ax.spines[sp].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(5e6))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1e6:.0f}M"))

    a0 = axes[0]
    a0.plot(y.index, y["aqua"], color=BLUE, lw=2, label="Aqua MODIS (same sensor throughout)")
    a0.plot(y.index[y.index >= 2011], naive[y.index >= 2011], color=ORANGE, lw=2, ls="--",
            label="Joined record: MODIS, then VIIRS from 2012")
    # same definition as E1: S-NPP / Aqua over 2012 to 2015, months with an Aqua or S-NPP outage left out
    ok = r[r["year"].between(2012, 2015)]
    bad = ok.groupby("month")[["outage_aqua", "outage_snpp"]].max().any(axis=1)
    ok = ok[~ok["month"].isin(bad[bad].index)]
    jump = ok["snpp_obs"].sum() / ok["aqua"].sum()
    a0.set_title(f"Before: switching satellites fakes a x{jump:.1f} jump in 2012", loc="left", fontsize=12, color=INK)
    a0.set_ylabel("Hotspots per year (world)", color=MUTED)
    a0.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper left")

    a1 = axes[1]
    conv = y.index < 2012
    a1.fill_between(y.index[conv], (y["harm"] - band)[conv], (y["harm"] + band)[conv], color=ORANGE, alpha=0.18,
                    lw=0, label="90% range of converted years")
    a1.plot(y.index, y["harm"], color=ORANGE, lw=2, label="Harmonized record (S-NPP scale)")
    gaps = r[(r["days_filled"] > 0) | r["source"].str.contains("converted")]["year"].unique()   # filled or converted months
    full_obs = (~conv) & ~y.index.isin(gaps)
    a1.plot(y.index[full_obs], y["snpp"][full_obs], color=INK, lw=0, marker="o", ms=3.5,
            label="Real S-NPP VIIRS (years without data gaps)")
    a1.set_title("After: one consistent record, no jump at the switch", loc="left", fontsize=12, color=INK)
    a1.legend(frameon=False, fontsize=9, labelcolor=INK, loc="lower left")
    top = max(naive.max(), (y["harm"] + band).max()) * 1.08
    for ax in axes:
        ax.set_xticks([y.index.min(), 2007, 2012, 2017, 2022, y.index.max()])
        ax.set_ylim(0, top)

    fig.text(0.01, 0.01, f"NASA FIRMS standard processing, cleaned; converted 2003 to 2011 with gradient boosting (tested on "
             f"unseen 2012 to 2014: 5.4% regional error). Whole years shown up to {years.max()}. Dotted line = satellite switch.",
             fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out = ROOT.parent / "reports" / "figures" / "seam_before_after.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
