# World compiler v0.3

The source adapters fetch geospatial inputs into a cache. `config.json` sets the study region and quality contract. `observations.json` contains visual-fit data independently of code. `facades.py`, `streets.py`, `vegetation.py`, `materials.py`, and `signs.py` compile shared components. No target address selects an architectural template in these modules.

Run from the bundle root with dependencies in `work/requirements-king-pilot.txt` installed:

1. `python work/fetch_geodata.py` (network; cache source GIS and OSM)
2. `python work/fetch_lidar.py` (network; cache the regional LiDAR subset)
3. `python work/build_world.py` (cached, deterministic geometry and materials)
4. `python work/pipeline/plan_observations.py` (prioritized missing observations)

The initial source adapters are configured for this San Francisco pilot. To change regions, provide the corresponding source datasets and region configuration. The geometry/compiler modules remain unchanged. City/provider adapter generalization is a separate layer and has not been validated in a second city.

`observation.schema.json` specifies packets for future calibrated image processing. The active pilot `observations.json` is a compact grammar-fit record, not a complete set of measured opening rectangles. `mapillary_manifest.py` is an optional metadata adapter requiring an authorized local `MAPILLARY_ACCESS_TOKEN`; it was not exercised in this run. It neither supplies credentials nor downloads imagery by default.

Material maps and physical scales live in `work/material-assets`, with CC0 licensing and original source metadata. Scanned material examples are not evidence of an actual building's material. Repeated window geometry, branches, blinds, fixtures and most facade layouts remain inferred or synthetic.

The output GLB uses real meter scale and local axes X east, Y up, Z south. Standard glTF PBR materials embed textures. The browser version transfers gzip-compressed 4 MiB chunks. These are transfer chunks, not HLOD or spatial streaming. The local GLB remains suitable for import into a 3D editor or engine, subject to ordinary engine material/import differences.

Full calibrated multi-view reconstruction, automatic image observation extraction, engine instancing/HLOD, road collision, and held-out image accuracy tests are not claimed as implemented.
