"""
Data gaps (problem P12): which days each satellite actually delivered data.

A satellite with zero hotspots anywhere in the world on a day almost certainly
had an outage that day (somewhere on Earth is always burning). Months with
missing days get flagged, so a quiet-looking month is not mistaken for a
quiet fire month.
"""
import pandas as pd


def days_observed(cell_days):
    """Distinct solar dates per satellite, from the cell-day table (all filters already applied)."""
    d = cell_days[["satellite", "date"]].drop_duplicates()
    d["month"] = pd.to_datetime(d["date"]).dt.to_period("M").dt.to_timestamp()
    out = d.groupby(["satellite", "month"]).agg(days_observed=("date", "nunique")).reset_index()
    out["days_in_month"] = out["month"].dt.days_in_month
    out["days_missing"] = out["days_in_month"] - out["days_observed"]
    return out


def gap_months(days_table, max_missing=2):
    """Satellite-months with more than max_missing days without any data."""
    return days_table[days_table["days_missing"] > max_missing].reset_index(drop=True)
