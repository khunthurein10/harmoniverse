#!/usr/bin/env python3
"""Turn the harmonized export (data/processed/export_zip/*.zip) into small files the website loads.

Run from anywhere:  python3 web/build_data.py
Reads only data/processed/export_zip/. Writes web/data/. Standard library only.

Every monthly series below is 288 values (Jan 2003 .. Dec 2026); null = no data that month.
  H harmonized, D day part, N night part, L / U the 90% range, M raw MODIS (Aqua + Terra), V raw S-NPP VIIRS

  meta.json              years, last month, month source codes, countries (code, name, bounds), regions,
                         provinces, land types, burn types
  series.json            the world (GL) and every country: H D N L U M V
  regions.json           the 14 GFED regions: H D N L U M V, plus relD / relN (error / value, day and night),
                         which drawn areas need for their 90% range
  provinces/<ISO3>.json  every province of one country: H D N L U M V
  grid.bin               10-byte records: uint32 box id, uint16 country, uint16 province, uint8 region,
                         uint8 land type (low 7 bits) + crosses_border (high bit)
  grid-metrics.bin       matching grid order: float32 land share, float32 mean GFED km²/year (NaN = unavailable)
  map/<year>.bin         uint32[12] rows per month; per month: uint32 box ids, float32 harmonized,
                         uint8 day share (0..255), uint8 burn type (0 = none), padded to 4 bytes
  detail/<year>.bin      same rows in the same order: uint16 aqua, terra, snpp (65535 = none),
                         uint16 GFED burned km2 x 10 (65535 = none), uint16 box uncertainty_rmse x 10,
                         uint8 burn-type share x 255,
                         uint8 flags (1 = above the training range: table conversion used), padded to 4
  viirs-added/<year>.bin float32 VIIRS-only addition for each map row, for years with additions
  tiles/<t>.bin          one 10 x 10 degree square, for drawn areas: uint32 n, uint16 box-in-square[n],
                         uint16 month index[n], float32 harmonized[n], uint8 day share[n]
  tiles/index.json       which squares exist
"""
import csv
import io
import json
import struct
import sys
import zipfile
from array import array
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "processed" / "export_zip"
OUT = Path(__file__).resolve().parent / "data"
YEARS = list(range(2003, 2027))
NMONTH = len(YEARS) * 12
LAST = (2026, 10)                 # S-NPP ended 2 Nov 2026: the record runs to Oct 2026
FIRST_SNPP = (2012, 2)            # S-NPP began 20 Jan 2012: Jan 2012 is still converted from MODIS
NRT = [(2026, 8), (2026, 9), (2026, 10)]   # near-real-time months, may change when FIRMS reprocesses
TILE, SIDE = 10, 40               # degrees per drawn-area square, boxes per square side
PER_ROW = 360 // TILE
SERIES_KEYS = ("H", "D", "N", "L", "U", "M", "V")
SOURCE_CODES = ["observed", "filled with NOAA-20", "gap-filled (scaled)", "converted from MODIS"]


def rows(name):
    with zipfile.ZipFile(SRC / f"{name}.zip") as z:
        member = next(n for n in z.namelist() if n.endswith(".csv"))
        with z.open(member) as f:
            yield from csv.DictReader(io.TextIOWrapper(f, encoding="utf-8", newline=""))


def num(v):
    return float(v) if v not in ("", None) else None


def box_id(lat, lon):
    return int(round((89.875 - lat) / 0.25)) * 1440 + int(round((lon + 179.875) / 0.25))


def mi(y, m):
    return (y - YEARS[0]) * 12 + (m - 1)


def in_range(y, m):
    return (y, m) <= LAST


def new_series():
    s = {k: [None] * NMONTH for k in SERIES_KEYS}
    for y in YEARS:
        for m in range(1, 13):
            if not in_range(y, m):
                continue
            k = mi(y, m)
            for key in ("H", "D", "N", "L", "U"):
                s[key][k] = 0.0
            s["M"][k] = 0.0 if y < 2026 else None          # MODIS not used from 2026
            s["V"][k] = 0.0 if y >= 2012 else None         # S-NPP starts 2012
    return s


def add_row(s, r):
    y, m = int(r["year"]), int(r["month"])
    if not in_range(y, m):
        return
    k = mi(y, m)
    h = float(r["harmonized"])
    day = num(r.get("harmonized_day"))
    night = num(r.get("harmonized_night"))
    s["H"][k] += h
    s["D"][k] += day if day is not None else 0.0
    s["N"][k] += night if night is not None else 0.0
    s["L"][k] += float(r["low90"])
    s["U"][k] += float(r["high90"])
    a, t = num(r["aqua_raw"]), num(r["terra_raw"])
    if s["M"][k] is not None and (a is not None or t is not None):
        s["M"][k] += (a or 0) + (t or 0)
    v = num(r["snpp_observed"])
    if s["V"][k] is not None and v is not None:
        s["V"][k] += v


def rounded(s):
    return {key: [None if v is None else round(v) for v in arr] for key, arr in s.items()}


def write_json(path, obj):
    path.write_text(json.dumps(obj, separators=(",", ":"), ensure_ascii=False))


def main():
    if not SRC.exists():
        sys.exit(f"Not found: {SRC}")
    for d in ("map", "detail", "tiles", "provinces", "filters", "viirs-added"):
        (OUT / d).mkdir(parents=True, exist_ok=True)

    # ---------- month source codes (the same for every place in a month) ----------
    source = [None] * NMONTH
    filled = [0] * NMONTH
    fill_details = {}
    for y in YEARS:
        for m in range(1, 13):
            if in_range(y, m):
                source[mi(y, m)] = 3 if (y, m) < FIRST_SNPP else 0
    for r in rows("filled_days"):
        k = mi(int(r["year"]), int(r["month"]))
        meth = r["method"].lower()
        source[k] = 1 if "noaa-20" in meth else 2 if "scaled" in meth else 3
        filled[k] = int(r["days_missing_or_partial"])
        fill_details[k] = {"days": [int(day) for day in r["days"].split(",")], "method": r["method"]}

    # ---------- countries and provinces ----------
    print("countries ...", flush=True)
    names, series = {}, defaultdict(new_series)
    for r in rows("harmonized_country_month"):
        names[r["country_code"]] = r["country"]
        add_row(series[r["country_code"]], r)
    print("provinces ...", flush=True)
    prov_series = defaultdict(lambda: defaultdict(new_series))
    for r in rows("harmonized_province_month"):
        add_row(prov_series[r["country_code"]][r["province"]], r)
    for code, provs in prov_series.items():
        write_json(OUT / "provinces" / f"{code}.json", {p: rounded(s) for p, s in provs.items()})

    # ---------- regions and the world ----------
    print("regions ...", flush=True)
    reg = defaultdict(new_series)
    rel = defaultdict(lambda: {"relD": [None] * NMONTH, "relN": [None] * NMONTH})
    region_names = {}
    world = new_series()
    outage = [0] * NMONTH          # per month: 1 = Aqua, 2 = Terra, 4 = S-NPP had more than 2 days without data
    for r in rows("harmonized_region_month"):
        y, m = int(r["year"]), int(r["month"])
        if not in_range(y, m):
            continue
        code, dn, k = r["region"], r["daynight"], mi(y, m)
        region_names[code] = r["region_name"]
        outage[k] |= (r.get("outage_aqua") == "True") | (r.get("outage_terra") == "True") << 1 | (r.get("outage_snpp") == "True") << 2
        h = float(r["harmonized"])
        if dn == "ALL":
            row = dict(r, harmonized_day="0", harmonized_night="0")
            add_row(reg[code], row)
            add_row(world, row)
        else:
            key = "D" if dn == "D" else "N"
            reg[code][key][k] += h
            world[key][k] += h
            rel[code]["rel" + key][k] = round(float(r["uncertainty_rmse"]) / h, 4) if h > 0 else 0
    out_reg = {}
    for code, s in reg.items():
        out_reg[code] = dict(rounded(s), name=region_names[code], **rel[code])
    write_json(OUT / "regions.json", out_reg)
    out_series = {"GL": rounded(world)}
    out_series.update({c: rounded(s) for c, s in series.items()})
    write_json(OUT / "series.json", out_series)

    # ---------- grid ----------
    print("grid ...", flush=True)
    codes = sorted(names, key=lambda c: names[c])
    cidx = {c: i for i, c in enumerate(codes)}
    reg_codes = sorted(region_names)
    ridx = {c: i for i, c in enumerate(reg_codes)}
    provs, pidx, lands, lidx, grid = [], {}, [], {}, []
    for r in rows("grid_cells"):
        b = box_id(float(r["latitude"]), float(r["longitude"]))
        code, pkey = r["country_code"], (r["country_code"], r["province"])
        if r["province"] and pkey not in pidx:
            pidx[pkey] = len(provs)
            provs.append([cidx.get(code, 65535), r["province"]])
        if r["land_type"] not in lidx:
            lidx[r["land_type"]] = len(lands)
            lands.append(r["land_type"])
        flags = lidx[r["land_type"]] | (128 if r["crosses_border"] == "True" else 0)
        metrics = (num(r["land_type_share"]), num(r["gfed_burned_km2_per_year"]))
        grid.append((b, cidx.get(code, 65535), pidx.get(pkey, 65535), ridx.get(r["region"], 255), flags, *metrics))
    grid.sort()
    with open(OUT / "grid.bin", "wb") as f, open(OUT / "grid-metrics.bin", "wb") as metrics:
        for b, country, province, region, flags, land_share, burned in grid:
            f.write(struct.pack("<IHHBB", b, country, province, region, flags))
            metrics.write(struct.pack("<ff", *(float("nan") if v is None else v for v in (land_share, burned))))
    box_country = {g[0]: g[1] for g in grid}
    box_region = {g[0]: g[3] for g in grid}
    box_land = {g[0]: g[4] & 127 for g in grid}
    # land-type and burn-type series per scope (world, region, country): harmonized H and day part D
    fser = defaultdict(lambda: (array("d", bytes(8 * NMONTH)), array("d", bytes(8 * NMONTH))))

    # ---------- boxes ----------
    burn, bidx = [], {}

    def burn_code(v):
        if not v:
            return 0
        if v not in bidx:
            bidx[v] = len(burn) + 1
            burn.append(v)
        return bidx[v]

    def u16(v, scale=1):
        return 65535 if v is None else min(65534, int(round(v * scale)))

    def pad(f):
        f.write(b"\0" * (-f.tell() % 4))

    tiles = defaultdict(lambda: (array("H"), array("H"), array("f"), array("B")))
    box_total = defaultdict(float)
    addition_years = []
    for y in YEARS:
        print(f"boxes {y} ...", flush=True)
        cols = [[array("I"), array("f"), array("B"), array("B"), array("H"), array("H"), array("H"), array("H"), array("H"), array("B"), array("B")] for _ in range(12)]
        additions = [array("f") for _ in range(12)]
        has_additions = False
        for r in rows(f"harmonized_box_month_{y}"):
            m = int(r["month"])
            v = float(r["harmonized"])
            if v <= 0 or not in_range(y, m):
                continue
            b = box_id(float(r["latitude"]), float(r["longitude"]))
            day = num(r["harmonized_day"]) or 0.0
            share = max(0, min(255, round(255 * day / v)))
            c = cols[m - 1]
            c[0].append(b); c[1].append(v); c[2].append(share); c[3].append(burn_code(r["fire_type"]))
            c[4].append(u16(num(r["aqua_raw"]))); c[5].append(u16(num(r["terra_raw"]))); c[6].append(u16(num(r["snpp_observed"])))
            c[7].append(u16(num(r["gfed_burned_km2"]), 10))
            c[8].append(u16(num(r["uncertainty_rmse"]), 10))     # single-box typical error (0 when measured)
            fs = num(r["fire_type_share"])
            c[9].append(0 if fs is None else max(0, min(255, round(255 * fs))))
            c[10].append(1 if r["above_training_range"] == "True" else 0)
            extra = num(r["viirs_only_added"]) or 0
            additions[m - 1].append(extra)
            has_additions = has_additions or extra > 0
            box_total[b] += v
            k, bt, dv = mi(y, m), c[3][-1], v * share / 255
            ci, ri, li = box_country.get(b, 65535), box_region.get(b, 255), box_land.get(b)
            scopes = [("GL",)] + ([("r", ri)] if ri != 255 else []) + ([("c", ci)] if ci != 65535 else [])
            for sc in scopes:
                if li is not None:
                    a = fser[sc + ("L", li)]; a[0][k] += v; a[1][k] += dv
                if bt:
                    a = fser[sc + ("B", bt - 1)]; a[0][k] += v; a[1][k] += dv
                if li is not None and bt:   # land type and burn type together
                    a = fser[sc + ("X", li * 100 + bt - 1)]; a[0][k] += v; a[1][k] += dv
            row, col = divmod(b, 1440)
            tl = tiles[(row // SIDE) * PER_ROW + col // SIDE]
            tl[0].append((row % SIDE) * SIDE + col % SIDE); tl[1].append(mi(y, m)); tl[2].append(v); tl[3].append(share)
        with open(OUT / "map" / f"{y}.bin", "wb") as f:
            f.write(struct.pack("<12I", *(len(c[0]) for c in cols)))
            for c in cols:
                for a in c[:4]:
                    f.write(a.tobytes())
                pad(f)
        with open(OUT / "detail" / f"{y}.bin", "wb") as f:
            for c in cols:
                for a in c[4:]:
                    f.write(a.tobytes())
                pad(f)
        if has_additions:
            addition_years.append(y)
            with open(OUT / "viirs-added" / f"{y}.bin", "wb") as f:
                for a in additions:
                    f.write(a.tobytes())

    print("filters ...", flush=True)
    files = defaultdict(lambda: {"land": {}, "burn": {}, "both": {}})
    for key, (h, d) in fser.items():
        sc, kind, idx = key[:-2], key[-2], key[-1]
        name = "GL" if sc == ("GL",) else ("r_" + reg_codes[sc[1]] if sc[0] == "r" else "c_" + codes[sc[1]])
        label = lands[idx] if kind == "L" else burn[idx] if kind == "B" else lands[idx // 100] + "|" + burn[idx % 100]
        out = lambda arr: [None if not in_range(YEARS[0] + i // 12, i % 12 + 1) else round(arr[i]) for i in range(NMONTH)]
        files[name]["land" if kind == "L" else "burn" if kind == "B" else "both"][label] = {"H": out(h), "D": out(d)}
    for name, obj in files.items():
        write_json(OUT / "filters" / f"{name}.json", obj)
    print("tiles ...", flush=True)
    for t, (loc, mon, val, day) in tiles.items():
        with open(OUT / "tiles" / f"{t}.bin", "wb") as f:
            f.write(struct.pack("<I", len(loc)))
            for a in (loc, mon, val, day):
                f.write(a.tobytes())
    write_json(OUT / "tiles" / "index.json", sorted(tiles))

    # ---------- meta: country bounds from where their hotspots are (2nd..98th percentile) ----------
    print("meta ...", flush=True)
    pts = defaultdict(list)
    for b, w in box_total.items():
        c = box_country.get(b)
        if c is not None and c != 65535:
            row, col = divmod(b, 1440)
            pts[c].append((89.875 - row * 0.25, -179.875 + col * 0.25, w))

    def wq(items, q):
        items = sorted(items)
        tot, acc = sum(w for _, w in items), 0.0
        for v, w in items:
            acc += w
            if acc >= q * tot:
                return v
        return items[-1][0]

    countries = []
    for i, code in enumerate(codes):
        p = pts.get(i)
        bbox = None
        if p:
            la, lo = [(a, w) for a, _, w in p], [(o, w) for _, o, w in p]
            bbox = [round(wq(la, .02) - .125, 2), round(wq(la, .98) + .125, 2), round(wq(lo, .02) - .125, 2), round(wq(lo, .98) + .125, 2)]
        countries.append({"code": code, "name": names[code], "bbox": bbox, "boxes": len(p or []), "provinces": code in prov_series})

    write_json(OUT / "meta.json", {
        "years": [YEARS[0], YEARS[-1]],
        "last": {"y": LAST[0], "m": LAST[1] - 1},
        "firstSnpp": {"y": FIRST_SNPP[0], "m": FIRST_SNPP[1] - 1},
        "nrt": [mi(y, m) for y, m in NRT],
        "sourceNames": SOURCE_CODES,
        "source": source,
        "daysFilled": filled,
        "fillDetails": fill_details,
        "additionYears": addition_years,
        "outage": outage,
        "countries": countries,
        "regions": [[c, region_names[c]] for c in reg_codes],
        "provinces": provs,
        "landTypes": lands,
        "burnTypes": burn,
        "tileSize": TILE,
    })
    print("done", flush=True)


if __name__ == "__main__":
    main()
