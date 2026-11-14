# Harmonized burning activity data (S-NPP VIIRS scale), 2003 to 2026

Every count is a number of active-fire hotspots expressed on the S-NPP VIIRS 375 m scale, so the satellite of origin
no longer matters. Before 2012 the values are converted from MODIS (Aqua + Terra); from 2012 they are real S-NPP counts.

## Files
One CSV per table (each also zipped). Columns are listed here; every column is explained under Fields.

### harmonized_region_month.csv
One row per GFED region x month x day/night (D, N, ALL = day + night). Use for regional calendars and charts.
Columns: region, region_name, year, month, daynight, harmonized, low90, high90, uncertainty_rmse, aqua_raw,
terra_raw, snpp_observed, viirs_only_added, source, days_filled, outage_aqua, outage_terra, outage_snpp

### harmonized_box_month_<year>.csv
One row per 0.25 degree box (about 28 km at the equator) x month, for maps and user-drawn areas.
Columns: latitude, longitude, year, month, country, country_code, province, crosses_border, region, land_type,
fire_type, fire_type_share, gfed_burned_km2, harmonized, harmonized_day, harmonized_night, uncertainty_rmse,
aqua_raw, terra_raw, snpp_observed, viirs_only_added, source, days_filled, outage_aqua, outage_terra, outage_snpp,
above_training_range, land_map_year

Drawn areas: sum the boxes whose centre is inside the area. Range for a drawn area (same rule as the country and
province files): for each box, error = harmonized_day x rel_D + harmonized_night x rel_N, where
rel = uncertainty_rmse / harmonized of the box's region that month in the region file (D and N rows); add the box
errors up, then low90 = max(sum - 1.645 x error, 0) and high90 = sum + 1.645 x error. Do not add up the box files'
own uncertainty_rmse: it is a single-box error.

### harmonized_country_month.csv, harmonized_province_month.csv
The boxes summed per country, and per province or state, per month. A box counts fully toward the place containing
its centre. low90/high90 use each box's region error for that month, added up as fully correlated (a cautious range).
Columns: country, country_code, (province), year, month, harmonized, low90, high90, uncertainty_rmse,
harmonized_day, harmonized_night, aqua_raw, terra_raw, snpp_observed, n_boxes, days_filled, source

### grid_cells.csv
One row per box that has hotspots anywhere in the data: the fixed labels a map needs to colour or filter it.
Join to the box files on latitude + longitude.
Columns: latitude, longitude, country, country_code, province, crosses_border, region, region_name, land_type,
land_type_share, fire_type, fire_type_share, gfed_burned_km2_per_year

### filled_days.csv
Every S-NPP month (2012 on) with missing or partial days. Matches the days_filled column.
Columns: year, month, days_missing_or_partial, days (the day numbers), share_of_days_good, method
(filled with NOAA-20 x same-month ratio / scaled up from the month's good days / converted from MODIS)

### conversion_lookup_table.csv
How many S-NPP hotspots one Aqua hotspot corresponds to, per region x day/night x calendar month (2015 to 2019).
The explainable version of the conversion: the main conversion is a model; this table is its baseline and fallback.
Columns: region, region_name, daynight, month, ratio_used (what the table applies), ratio_level (which pooling:
1 month, 3-month window or whole year), ratio_1month, ratio_3month, ratio_year, aqua_hotspots_2015_2019,
snpp_hotspots_2015_2019

## Fields
| field | meaning |
|---|---|
| latitude, longitude | centre of the 0.25 degree box |
| year, month | calendar month (UTC dates) |
| country, country_code, province | place containing the box centre (Natural Earth 1:10m Admin 1, public domain; ISO 3166-1 alpha-3 style code). Coastal boxes take the nearest province within 0.25 degree; a few offshore boxes are blank. Disputed borders follow Natural Earth (as controlled on the ground) |
| crosses_border | the box also covers part of another country |
| n_boxes | place files: number of boxes with hotspots that month |
| region, region_name | GFED basis region (14 regions), from the GFED4.1s map that the harmonization used. GFED5.1's own map differs for about 2,000 inland boxes along region borders and about 13,000 coastal boxes (0.4% of GFED5.1 burned area); we keep GFED4.1s so every file uses one map |
| land_type | forest, woody_savanna, savanna_grass_shrub, cropland, wetland, other (MODIS MCD12C1 IGBP, grouped after GFED5 Table 2); in box files the group with most hotspots that month |
| land_type_share | grid_cells.csv: share of the box area in land_type (25 MCD12C1 pixels per box, 2019 map). In grid_cells.csv land_type is the group covering most of the box; in box files it is the group with most hotspots that month, so the two can differ (they agree for about 87 to 91% of hotspots). "other" includes water, so coastal boxes can show "other" |
| fire_type | the kind of land that burned most, from GFED burned area, in the 9 groups of GFED5NRT (Chen et al. 2026, Fig. 3): boreal_forest (incl. tundra), temperate_forest, tropical_forest, deforestation, peat, shrub (incl. woody savanna), grass (incl. open savanna), cropland, other. Box files: that box and month, 2003 to 2022 from GFED5.1 (van der Werf et al. 2025), 2023 and 2024 from GFED5NRT (S-NPP + NOAA-20 combined); blank from 2025, and blank where GFED has no burned area that month (in 2022: about half of box-months, but only about 6% of hotspots). grid_cells.csv: the whole 2003 to 2022 GFED5.1 record per box. Modelled burned area, not hotspots: it describes what kind of land burns in the box, it does not label each hotspot. From 2023 far fewer box-months are blank (about 94% of box-months have a fire type, against about 50% in 2012 to 2022): GFED5NRT records burned area in more boxes than GFED5.1. This is a difference between the two products, not a change in fires. Across the switch the main type of a box stays the same as often as between other years (83% from 2022 to 2023, 85% within GFED5.1). |
| fire_type_share | share of the GFED burned area in fire_type (box-month, or 2003 to 2022 in grid_cells.csv) |
| gfed_burned_km2 | box files: GFED burned area in the box that month (km2) |
| gfed_burned_km2_per_year | mean yearly GFED5.1 burned area in the box, 2003 to 2022 (km2) |
| daynight | D (afternoon / 13:30 pass), N (night / 01:30 pass), ALL (sum) |
| harmonized, harmonized_day, harmonized_night | hotspot count on the S-NPP scale |
| low90, high90 | 90% range (assumes normal errors); equal to harmonized when fully observed |
| uncertainty_rmse | typical error (RMSE): converted months from the 2012 to 2014 test; filled months from E9 (scaling) or E10 (NOAA-20); 0 when fully observed |
| days_filled | number of missing or partial S-NPP days in that month that were filled (0 = none) |
| aqua_raw, terra_raw | original MODIS hotspot counts (Aqua, Terra) after cleaning |
| snpp_observed | real S-NPP count (blank before 2012) |
| viirs_only_added | converted months: fires added back for areas where MODIS saw nothing. Regional file: added at region level. Box file: the same share spread over the region's boxes in proportion (already included in harmonized) |
| source | how the value was made: S-NPP observed; S-NPP observed, filled with NOAA-20; S-NPP observed, gap-filled (scaled); converted (box files add model or table fallback); converted (S-NPP outage, ...). Place files list every kind present that month |
| outage_aqua, outage_terra, outage_snpp | that satellite had more than 2 days without data that month |
| above_training_range | count larger than the model learned from; the simpler table conversion was used |
| land_map_year | year of the land cover map used (maps cover 2012 to 2019; other years use the nearest) |

## How values were made
Short version; the full method, experiments and results are in the project report.
- Cleaning (NASA FIRMS standard processing): vegetation fires only (type 0); MODIS confidence below 30% removed;
  VIIRS low confidence removed; VIIRS static sources (flares, volcanoes, industry) masked per 0.01 degree cell.
- Conversion (2003 to 2011): gradient boosting models (one day, one night) trained on 2015 to 2019, using Aqua count,
  Aqua fire-days, Aqua mean FRP, Aqua pixel size, Terra count, region, land type, month and location. Tested on
  untouched years 2012 to 2014: regional monthly error 5.4% (fire-weighted), 12.6% at night, yearly regional totals 2.0%,
  single box 26%. Fires in boxes MODIS missed are added back per region and spread over its boxes in proportion.
- Uncertainty: error measured on 2012 to 2014 (years the models never saw), as RMSE in 10 prediction bins (method
  after MRBA60, Torres-Vazquez et al. 2026): a linear fit per box; for regional totals the RMSE per size bin (no line),
  which keeps 90% ranges at about 88 to 93% coverage for small and large months alike (leave-one-year-out test, E11).

## Caveats
- Before 2012, fires only VIIRS can see (MODIS never located them) are spread over each region's boxes in proportion, so
  boxes add up to the region row. This is right on average for large areas; for one box or a small area it is an
  approximation (subtract viirs_only_added to get the MODIS-located part only).
- Country and province totals assign each 28 km box to one place, so values near borders are approximate
  (see crosses_border). The labels agree with the FIRMS country files for 98.9% of boxes (2019 check).
- Single boxes are much less certain than regional or yearly totals. Prefer areas of several boxes.
- Monthly, not daily: MODIS does not see every place every day in the tropics (Giglio et al. 2025, RSE 331:115006).
- 2003 to 2011 includes larger fire months than the training years; uncertainty may be underestimated there.
- Missing and partial S-NPP days (partial = S-NPP saw under 0.8 of NOAA-20's fires worldwide that day, or under
  half of the surrounding days where NOAA-20 is not available) are filled: from April 2018 with NOAA-20 x a ratio
  learned that month per region x day/night (GFED5NRT Sect. 2.5; tested error 0.6% vs 3.6% for scaling), before
  that by scaling the good days up to the month (assumes the missing days were typical). See days_filled.
- S-NPP stopped on Nov 2 2026 (NOAA NESDIS); this file ends in 2026 (2026: January to October, from the FIRMS API).
  January to July 2026 are standard processing (VIIRS_SNPP_SP); August to October 2026 are near-real-time
  (VIIRS_SNPP_NRT), the only version FIRMS offers for those months.
- From 2026, MODIS is not used (Terra and Aqua are drifting: Copernicus CAMS); aqua_raw and terra_raw are blank.

## Credits
NASA FIRMS (MODIS MCD14ML, VIIRS VNP14IMGML), MODIS MCD12C1 land cover (NASA LP DAAC), GFED regions
(GFED4.1s, van der Werf et al. 2017; GFED5, van der Werf et al. 2025), GFED5.1 burned area by ecosystem
(van der Werf et al. 2025), fire type groups from GFED5NRT (Chen et al. 2026).
