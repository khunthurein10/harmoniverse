"""
One schema for MODIS and VIIRS hotspots.

Nothing is deleted here. Rows are only flagged (keep_type, keep_conf), so later
steps can always count what was removed and why.

Sources for the rules:
- Confidence classes: MODIS C6 Active Fire User Guide (low <30, nominal 30-79, high 80+)
- type field: 0 vegetation, 1 volcano, 2 other static land source, 3 offshore (FIRMS)
- Local solar date: our addition, so "a day" means the same thing everywhere
"""
import pandas as pd

# FIRMS uses different satellite codes across products and versions
SAT_MAP = {
    "TERRA": "TERRA", "T": "TERRA",
    "AQUA": "AQUA", "A": "AQUA",
    "N": "SNPP", "NPP": "SNPP", "SUOMI NPP": "SNPP",
    "N20": "NOAA20", "1": "NOAA20", "J1": "NOAA20",
    "N21": "NOAA21", "2": "NOAA21", "J2": "NOAA21",
}

COLUMNS = [
    "sensor", "satellite", "lat", "lon", "acq_utc", "solar_date", "daynight",
    "frp", "bright_mir", "bright_tir", "conf_class", "scan", "track", "type", "version", "is_nrt",
    "country", "keep_type", "keep_conf",
]


def detect_sensor(columns):
    cols = set(columns)
    if "bright_ti4" in cols:
        return "VIIRS"
    if "brightness" in cols and "bright_t31" in cols:
        return "MODIS"
    raise ValueError(f"Cannot tell sensor from columns: {sorted(cols)}")


def confidence_class(conf, sensor):
    """Map MODIS 0-100 or VIIRS l/n/h to low / nominal / high."""
    if sensor == "MODIS":
        c = pd.to_numeric(conf, errors="coerce")
        return pd.cut(c, bins=[-1, 29, 79, 100], labels=["low", "nominal", "high"]).astype("string")
    first = conf.astype("string").str.strip().str.lower().str[0]
    return first.map({"l": "low", "n": "nominal", "h": "high"}).astype("string")


def solar_date(acq_utc, lon):
    """Local solar date: UTC time shifted by longitude/15 hours."""
    solar = acq_utc + pd.to_timedelta(lon / 15.0, unit="h")
    return solar.dt.tz_localize(None).dt.normalize()


def clean_frame(raw, country=None, min_conf_class="nominal"):
    """Turn one raw FIRMS table (MODIS or VIIRS) into the shared schema."""
    df = raw.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    sensor = detect_sensor(df.columns)

    out = pd.DataFrame(index=df.index)
    out["sensor"] = pd.Series(sensor, index=df.index, dtype="string")
    out["satellite"] = (df["satellite"].astype("string").str.strip().str.upper()
                        .map(SAT_MAP).astype("string"))
    out["lat"] = pd.to_numeric(df["latitude"], errors="coerce").astype("float32")
    out["lon"] = pd.to_numeric(df["longitude"], errors="coerce").astype("float32")

    hhmm = df["acq_time"].astype("string").str.replace(":", "", regex=False).str.zfill(4)
    out["acq_utc"] = pd.to_datetime(df["acq_date"].astype("string") + " " + hhmm,
                                    format="%Y-%m-%d %H%M", errors="coerce", utc=True)
    out["solar_date"] = solar_date(out["acq_utc"], out["lon"].astype("float64"))
    out["daynight"] = df["daynight"].astype("string").str.strip().str.upper()
    out["frp"] = pd.to_numeric(df["frp"], errors="coerce").astype("float32")
    # Brightness temperatures (Kelvin). Kept as clues and for the CSV, never harmonized:
    # MODIS 'brightness' = channel 21/22, 'bright_t31' = channel 31;
    # VIIRS 'bright_ti4' = I-4, 'bright_ti5' = I-5 (FIRMS attribute page). The 'sensor' column says which.
    mir, tir = ("brightness", "bright_t31") if sensor == "MODIS" else ("bright_ti4", "bright_ti5")
    out["bright_mir"] = pd.to_numeric(df[mir], errors="coerce").astype("float32")
    out["bright_tir"] = pd.to_numeric(df[tir], errors="coerce").astype("float32")
    out["conf_class"] = confidence_class(df["confidence"], sensor)
    out["scan"] = pd.to_numeric(df["scan"], errors="coerce").astype("float32")
    out["track"] = pd.to_numeric(df["track"], errors="coerce").astype("float32")
    if "type" in df.columns:
        out["type"] = pd.to_numeric(df["type"], errors="coerce").astype("Int8")
    else:  # NRT rows have no type; the static mask handles them later
        out["type"] = pd.Series(pd.NA, index=df.index, dtype="Int8")
    out["version"] = df["version"].astype("string")
    out["is_nrt"] = out["version"].str.upper().str.contains("NRT", na=False)
    out["country"] = pd.Series(country, index=df.index, dtype="string")

    out["keep_type"] = out["type"].isna() | (out["type"] == 0)
    keep = {"nominal": ["nominal", "high"], "high": ["high"], "low": ["low", "nominal", "high"]}[min_conf_class]
    out["keep_conf"] = out["conf_class"].isin(keep)
    return out[COLUMNS].reset_index(drop=True)


def summary(df):
    """Small dict of counts, for printing and for checking nothing was lost."""
    return {
        "rows": len(df),
        "kept": int((df["keep_type"] & df["keep_conf"]).sum()),
        "dropped_type": int((~df["keep_type"]).sum()),
        "dropped_conf": int((~df["keep_conf"]).sum()),
        "missing_core": int(df[["lat", "lon", "acq_utc", "satellite"]].isna().any(axis=1).sum()),
    }
