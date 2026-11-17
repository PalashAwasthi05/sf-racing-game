# Market Street elevation input

`data/points.npz` is the generator input: key `points`, float64 rows
`[EPSG:3857 x, EPSG:3857 y, source absolute elevation metres, LAS classification, intensity]`.
The generator handles conversion to its local frame. Class 2 is measured ground.

Rebuild from the final corridor context polygon:

```
work/geoenv/bin/python work/areas/market-street/fetch_lidar.py work/areas/market-street/context-boundary.geojson
```

The source is the public USGS `CA_SanFrancisco_1_B23` EPT survey. Nodes through
depth 8 are selected by intersection with the actual corridor polygon, including
all vertical nodes. This avoids the existing radius fetcher's 300 m height cap.
Tile data is cropped with the actual polygon, not its enclosing rectangle.

The fetcher decodes one tile at a time in 100,000-return chunks. Cropped tile
shards are written to disk and consolidated through a memory map, avoiding
holding the corridor cloud or an enclosing-radius cloud in RAM. Existing survey
caches are copied into this area's cache without changing other areas.

`data/lidar-provenance.json` records source, extent, classifications, elevation
range and the number of returns at or above 300 m. `data/lidar-nodes.json` records
the precise EPT node selection. `lidar-fetch.log` records acquisition progress.
Source-tile caches and cropped shards are retained to make reruns economical.
