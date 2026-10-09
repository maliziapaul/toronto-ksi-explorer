# Toronto KSI Explorer

A single-file explorer of Toronto Police Service collisions where someone was killed or seriously injured (KSI), 2006 to present, modelled on the BC Road Data Explorer.

- `index.html` is the whole site (data inlined). It can be hosted as-is on GitHub Pages.
- To refresh the data, run from `build/`:
  1. Download `ksi.csv`, `nbhd.geojson`, `wards.geojson` and `centreline.geojson` from open.toronto.ca (see `build_data.py` for the datasets).
  2. `python3 build_data.py` (needs pandas and shapely), then `python3 assemble.py`.
