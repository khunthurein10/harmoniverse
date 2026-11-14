# Tests

`test_clean_grid.py` checks the cleaning and gridding code on small hand-made rows: every row kept and flagged
correctly, box index and area, scan angle geometry, counts adding up (with and without VIIRS low confidence),
and border boxes not counted twice.

Run from the project root: `python -m pytest pipeline/tests -q`
