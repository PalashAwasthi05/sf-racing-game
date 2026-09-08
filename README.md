# 188 King Street — pilot 01

A generated 3D study around 188 King Street, San Francisco. Created September 8, 2026.

The study radius is 200 m (about 12.6 hectares), with context to 265 m. Buildings crossing the context boundary are cut there. This is the first geometry-and-evidence baseline for iterative refinement. It is not yet a verified 1:1 visual replica or a complete racing game.

## Explore

Use Orbit to inspect the neighborhood, Street to move at eye level, or Plan for an overhead view. Click a building to inspect its estimated height and data sources. Focus building moves the camera closer. In Street mode, drag to look and use W/A/S/D to move; Shift increases speed. Movement follows the sampled terrain but does not yet collide with buildings.

Turn facade details, trees, street objects, and markings on or off. Evidence colors distinguish LiDAR-derived shape, mapped placement, and inferred surface detail. Sources & remaining gaps gives the provenance inside the viewer.

## What was generated

- 64 building features, including 188 King Street and the edge of Oracle Park.
- Building roof tiers and terrain fitted from 4,624,404 LiDAR points.
- 96 mapped road segments, 45 mapped crossing features, and 4 rail segments. These are data features, not counts of unique streets, intersections, or rail lines.
- 174 inventory-based tree placements with reusable species-inspired geometry.
- 12 mapped signal nodes and 19 mapped lamp nodes, with approximate pole and fixture shapes.
- 247 nearby municipal sign records retained in the evidence file; exact sign positions and orientation remain unresolved.

The model is approximately 10.7 MB, with 202 meshes and 520,963 triangles. It uses vertex colors and procedural facade geometry; photographic facade textures are not included.

## Data and evidence

| Attribute | Source and method | Remaining uncertainty |
| --- | --- | --- |
| Address identity | [OpenStreetMap way 149167949](https://www.openstreetmap.org/way/149167949), tagged 188/190 King Street | Main footprint and adjacent annex remain separate features. |
| Building outlines | [DataSF footprints](https://data.sfgov.org/d/ynuv-fyni), reconciled with newer OSM footprints | Much municipal geometry derives from older mapping; changes since capture require checks. |
| Ground and roofs | [2023 San Francisco LiDAR](https://www.fisheries.noaa.gov/inport/item/73386/full-list), sampled through the [USGS public EPT index](https://s3-us-west-2.amazonaws.com/usgs-lidar-public/CA_SanFrancisco_1_B23/ept.json) | Roof heights are percentile fits; vegetation, fixtures, misalignment, and regularization can produce errors. |
| Roads, crossings and rail | [OpenStreetMap](https://www.openstreetmap.org/copyright), retrieved September 2026 | Lane counts where tagged; widths, stripe spacing, gauge placement, and many fixture dimensions are inferred. |
| Trees | [DataSF street-tree inventory](https://data.sfgov.org/d/tkzw-k3nq) | Inventory positions/species guide placement; exact height, canopy, age, and survival are unverified. |
| Signs | [DataSF sign inventory](https://data.sfgov.org/d/m48z-6ji4) | Coarse locations are retained as unresolved records, not rendered as exact signs. |
| Street-level reference | [Google Street View on the King Street frontage](https://www.google.com/maps/@37.7786899,-122.3913452,3a,75y,315h,110t/data=!3m4!1e1!3m2!1sS4hs-Kyh3P14aRv9CWeTvg!2e0), capture labeled January 2025 | Targeted visual inspection informed the frontage/median treatment. No Google pixels are embedded as textures. |
| Additional imagery | [Mapillary around King and Third](https://www.mapillary.com/app/?lat=37.7786&lng=-122.3918&z=18) | Dense capture coverage was observed. Individual captures, dates, and image-derived measurements have not been ingested. |

Google imagery is one targeted reference among the inputs. The entire neighborhood has not been checked against that January 2025 capture. A retrieval date does not mean every mapped feature is current as of that date.

## Coordinates and interchange

`king-street-pilot.glb` is in meters. X points east, Y points up, and Z points south. The horizontal origin is the centroid of the OSM address footprint, approximately latitude 37.7786228425823, longitude -122.39182032784258. The underlying projected coordinates use UTM zone 10N. Vertical values are relative to the median nearby ground sample; the absolute source offset is recorded as `verticalOrigin` in the evidence file. This is not a surveyed vertical control point.

Mesh names retain semantic layers and building IDs so later passes can replace facades, roofs, or props independently. The JSON records per-building sources and sample counts, terrain samples, source links, and unresolved sign records. Small terrain triangles can extend about 2 m beyond the nominal context edge.

## Refinement priorities

1. Calibrate 188 King Street against several dated street captures: building edges, window counts, projecting frames, penthouse setbacks, and material appearance. The current fitted height is about 36.0 m and should be checked against the architectural envelope rather than accepted as a survey measurement.
2. Fit the King/Third intersection: curb boundaries, transit platforms, lane widths, stop lines, turn arrows, crossing geometry, signal placement, and sign orientation.
3. Replace the most visible neighboring facades and simplify noisy roof tiers. Oracle Park especially needs a dedicated architectural treatment.
4. Compare measured errors on this small area, then rerun the same pipeline on another block before estimating citywide time and cost.

Water geometry is included only where mapped water intersects this area. Bridges and distant waterfront features outside the study are not modeled. No interiors, vehicles, crowds, exact sign lettering, or race physics are included.

## What this says about the heuristic

The geometric bulk came from cached data processing rather than asking a language model to describe every object. Stable feature IDs and separate facade/road-detail layers make targeted revision possible without regenerating the entire neighborhood. This pass used public data without paid purchases or a separate model API. It does not establish that full visual fidelity across San Francisco is achievable in a month; street-level reconstruction and measured error checks remain the work needed to test that target.

## Validation

The GLB was reloaded and checked for finite coordinates and valid triangle indices. A separate software projection was used to inspect the geometry. The viewer passed TypeScript checking and its production build. Interactive browser behavior and game-engine performance have not been comprehensively tested.

Attribution: © OpenStreetMap contributors, DataSF / City and County of San Francisco, and USGS. The evidence links preserve source attribution for subsequent review and refinement.
