# Experiments

The tests that chose each part of the method. Choosing years are 2015 to 2019 (leave one year out);
2012 to 2014 were kept untouched for the final test. Result tables are in `results/<experiment>/`.

| # | Script | Question | Outcome |
|---|---|---|---|
| E1 | `e1_fake_jump.py` | How big is the fake jump if MODIS and VIIRS are simply joined at 2012? | About x6.8 worldwide: harmonization is needed |
| E2 | `e2_ratio_varies.py` | Does the S-NPP / Aqua ratio vary, and by what? | Day/night, region and month matter; land type not at regional level |
| E6 | `e6_viewing_angle.py` | Drop or weight MODIS swath-edge pixels? | Neither helped: all pixels are used |
| E8 | `e8_conversion.py` | Which lookup table predicts held-out years best? | Region x day/night x month with fallbacks (3-month window first in BONA, TENA) |
| E8b | `e8b_model.py` | Does a tree model beat the table? | Gradient boosting, one model per day/night |
| E8c | `e8c_fixed.py` | Re-run after the leak fix (model only where MODIS saw fire, VIIRS-only share added back) | Fix kept; writes the VIIRS-only share |
| Final | `final_test.py` | Train 2015 to 2019, predict 2012 to 2014 (run once) | Model kept: regional monthly error 5.4% (table 7.2%) |
| E9 | `e9_gapfill_test.py` | How accurate is scaling up a month with missing S-NPP days? | Error by number of missing days, used for filled months before April 2018 |
| E10 | `e10_noaa20_fill.py` | Fill missing S-NPP days with NOAA-20, or scale up? | NOAA-20: 0.6% error vs 3.6% for scaling; used from April 2018 |
| E11 | `e11_band_coverage.py` | Do the 90% ranges hold about 90% of true values? | RMSE per size bin: 88 to 93% at every size |
