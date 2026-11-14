"""
Checks for cleaning, viewing angle and gridding, on small fake data where the
right answer is known. Run from the project root:  pytest pipeline/tests -q
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harmonizer.clean import clean_frame, summary  # noqa: E402
from harmonizer.angle import modis_scan_angle  # noqa: E402
from harmonizer.grid import cell_area_km2, cell_day_counts, cell_index, monthly_counts  # noqa: E402

MODIS_HEADER = "latitude,longitude,brightness,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_t31,frp,daynight,type"
VIIRS_HEADER = "latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_ti5,frp,daynight,type"


def modis_rows():
    rows = [
        "-15.2,132.5,330,1.0,1.0,2013-08-01,435,Aqua,MODIS,85,6.2,300,45.3,D,0",    # kept
        "-15.3,132.6,315,2.4,1.5,2013-08-01,1340,Terra,MODIS,22,6.2,298,8.1,N,0",   # low confidence
        "37.75,14.99,350,1.0,1.0,2013-08-02,0035,Aqua,MODIS,90,6.2,301,120,N,1",    # volcano
        "-15.21,132.51,320,4.8,2.0,2013-08-03,0500,Aqua,MODIS,70,6.2,300,30,D,0",   # kept, swath edge
    ]
    return pd.read_csv(pd.io.common.StringIO("\n".join([MODIS_HEADER, *rows])))


def viirs_rows():
    rows = [
        "-15.21,132.51,340,0.4,0.37,2013-08-01,0420,N,VIIRS,n,2,295,5.2,D,0",
        "-15.22,132.52,367,0.4,0.37,2013-08-01,0420,N,VIIRS,h,2,296,12,D,0",
        "-15.22,132.52,367,0.4,0.37,2013-08-02,0420,N20,VIIRS,l,2,296,1.2,D,0",
    ]
    return pd.read_csv(pd.io.common.StringIO("\n".join([VIIRS_HEADER, *rows])))


def test_clean_keeps_every_row_and_flags_correctly():
    c = clean_frame(modis_rows(), country="Australia")
    s = summary(c)
    assert s["rows"] == 4 and s["missing_core"] == 0
    assert s["kept"] == 2 and s["dropped_type"] == 1 and s["dropped_conf"] == 1
    assert list(c["satellite"]) == ["AQUA", "TERRA", "AQUA", "AQUA"]
    assert c["country"].eq("Australia").all()


def test_viirs_codes_and_confidence():
    c = clean_frame(viirs_rows())
    assert list(c["satellite"]) == ["SNPP", "SNPP", "NOAA20"]
    assert list(c["conf_class"]) == ["nominal", "high", "low"]
    assert c["sensor"].eq("VIIRS").all()


def test_solar_date_shifts_with_longitude():
    # 23:00 UTC at 150E is already the next morning locally
    df = modis_rows().iloc[:1].copy()
    df["longitude"], df["acq_time"], df["acq_date"] = 150.0, 2300, "2013-08-01"
    c = clean_frame(df)
    assert c["solar_date"].iloc[0] == pd.Timestamp("2013-08-02")


def test_scan_angle_matches_geometry():
    a = modis_scan_angle([1.0, 1.42, 4.8])
    assert a[0] == pytest.approx(0, abs=0.5)
    assert a[1] == pytest.approx(30, abs=1.0)    # MRBA60 cut-off
    assert a[2] == pytest.approx(55, abs=1.0)    # MODIS swath edge


def test_cell_index_and_area():
    row, col = cell_index([-89.99, 0.1, 89.99], [-179.99, 0.1, 179.99])
    assert list(row) == [0, 360, 719] and list(col) == [0, 720, 1439]
    eq = cell_area_km2(360)
    bc = cell_area_km2(int((50 + 90) / 0.25))
    assert eq == pytest.approx(770, rel=0.02)
    assert bc / eq == pytest.approx(np.cos(np.radians(50.125)), rel=0.01)
    total = cell_area_km2(np.arange(720)).sum() * 1440
    assert total == pytest.approx(510.07e6, rel=0.001)   # Earth's surface


def test_grid_counts_add_up():
    clean = pd.concat([clean_frame(modis_rows()), clean_frame(viirs_rows())], ignore_index=True)
    # VIIRS low-confidence rows stay in n_det and are also counted in n_det_low (decision A3), so
    # n_det - n_det_low is what the pipeline uses: rows with keep_type and keep_conf
    kept = int((clean["keep_type"] & clean["keep_conf"]).sum())
    kept_with_low = int((clean["keep_type"] & (clean["keep_conf"] | (clean["sensor"] == "VIIRS"))).sum())
    m = monthly_counts(cell_day_counts(clean))
    assert int(m["n_det"].sum()) == kept_with_low
    assert int((m["n_det"] - m["n_det_low"]).sum()) == kept
    aqua = m[m["satellite"] == "AQUA"].iloc[0]
    assert aqua["n_det"] == 2 and aqua["n_det_centre"] == 1 and aqua["n_firedays"] == 2


def test_border_cells_not_double_counted():
    # Same cell and day split across two country files must count as one fire-day
    a = clean_frame(viirs_rows().iloc[:1], country="A")
    b = clean_frame(viirs_rows().iloc[1:2], country="B")
    cd = pd.concat([cell_day_counts(a), cell_day_counts(b)], ignore_index=True)
    m = monthly_counts(cd)
    assert len(m) == 1 and m["n_det"].iloc[0] == 2 and m["n_firedays"].iloc[0] == 1
