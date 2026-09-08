# 188 King Street — detail pilot 02

[Open the rebuilt private viewer](https://king-street-world-pilot.reflex-4683.chatgpt.site).

This rebuild applies heuristic v0.3 across the same 200 m study radius, with context to 265 m. It improves the detail pipeline while retaining the evidence boundary: it does **not yet meet the supplied game image's realism or the requested 1:1 facade accuracy**.

## What changed everywhere

The compiler now builds walls around recessed openings rather than drawing flat window rectangles on opaque massing. Shared components generate metal frames, mullions, transoms, sills, canopies, handles, vents, coping and downpipes. Upper roof/setback walls use the same system. No address selects custom architectural geometry in the compiler; geographic observation records supply parameters.

Scanned CC0 brick, asphalt and concrete supply color, normal and roughness maps at physical scale. Glazing and metal have separate surface response. Sidewalks now have height, curb edges, joints, tree pits and tactile panels. Street semantics generate selected bus-lane treatments and turn markings. Inventory records supply candidate sign panels with readable legends; their curb snaps remain explicitly inferred. Shared tree templates generate branches, foliage and palm fronds instead of simple canopy lumps.

The viewer preserves the source inspector, layer controls and evidence colors. Green identifies meshes associated with a partial visual fit; it does not mean every opening on that mesh was measured. Amber identifies inferred detail. Cyan identifies terrain/roof fits. Material texture can be realistic while its assignment to a particular wall is still wrong.

## Generated result

| Item | This run |
| --- | ---: |
| Building features | 64 |
| Window assemblies | 15,702 |
| Fitted roof sections | 122 |
| Tree placements | 174 |
| Candidate sign panels | 20 |
| Generated sidewalk area | 15,219 m² |
| Wall sections, including setback sections | 996 |
| Sections with partial direct visual fits | 8 |
| Independently verified facade sections | 0 |
| Raw GLB | 207.0 MB |
| Compressed transfer | 52.8 MB |
| Triangles | 4,195,912 |
| Material/feature meshes | 677 |
| Cached generation before export/compression | 90.4 s |

Window assemblies and wall sections are compiler components, not surveyed counts of real windows or unique building facades. More subdivision creates more wall sections; the count alone is not an accuracy measure.

Generation uses cached point clouds, GIS and compact observation records with no language-model calls. The timing excludes research, source downloads, manual observation, asset preparation, export/compression, development, validation and publishing. It is not an end-to-end city-generation benchmark. The raw model is much heavier than pilot 01; transfer compression does not reduce GPU geometry or draw calls.

## Observed versus inferred

The two dated Google Street View viewpoints on King Street informed the projecting loft frames and adjoining bay-window/stone-band facade families. Source IDs, dates and observed/unknown attributes are recorded separately in `work/pipeline/observations.json`. No Google image textures are embedded. Most other wall layouts and attachment positions use architectural priors, and dimensions in the visual-fit records have not been independently calibrated.

The PBR material maps come from [ambientCG Bricks085](https://ambientcg.com/a/Bricks085), [Asphalt033](https://ambientcg.com/a/Asphalt033), and [Concrete034](https://ambientcg.com/a/Concrete034), under CC0. Those scans are reusable examples, not photographs of these exact surfaces. Tree branches and blind variations are synthetic.

Sign evidence retains the original inventory coordinate, curb-snap distance, direction text, date and an `imageVerified: false` flag. The 20 rendered candidates are not a complete or verified sign inventory. Original coarse records remain in the evidence JSON. The [SFMTA Third Street project](https://www.sfmta.com/projects/3rd-street-quick-build-project) supplies installed-project context; the compiler does not present a proposed drawing as exact current paint.

## Heuristic changes and portability

Read `world-generation-blueprint.md` for the revised heuristic. The bundle includes shared facade/street/material/vegetation/sign compilers, region settings, observation inputs, a future observation-packet schema and a missing-view planner. The Mapillary metadata adapter requires authorized local configuration and has not been run in this pass.

Geometry code is separate from evidence. New areas should add source data and observation records through the same schema, not address-specific code. The source adapters are still configured for this San Francisco pilot; a second-region run remains necessary to validate portability. The current GLB bakes repeated components, and the download chunks are not spatial streaming/HLOD. Both instancing and rendering budgets remain city-scale work.

## Remaining gates

- Sufficient dated images and independent views for every street-visible facade.
- Measured window positions/counts, frame depth, material identity and distinctive ornament.
- Calibrated curb geometry, lane/paint positions and sign poses.
- Cleaner roof reconstruction where the height fit still produces terrace artifacts.
- Engine-side spatial streaming, instancing, collision, and measured frame time/memory.
- Held-out image comparisons against the supplied quality target.

The v0.3 accuracy gate remains **not passed**. This prevents the more detailed but partly inferred model from being called a current 1:1 replica. The next bottleneck is observing and fitting the missing facade structure, followed by measured rendering optimization.

## Interchange and validation

GLB units are meters: X east, Y up, Z south. The horizontal origin is approximately 37.7786228425823, -122.39182032784258 in a local UTM zone 10N frame. Vertical values use the recorded local ground offset. These coordinate conventions match pilot 01.

The exported geometry and texture coordinates were checked for finite values and valid triangle indices. Compressed transfer chunks were reconstructed and compared byte-for-byte with the raw GLB. A CPU projection inspected geometry and texture scaling; this is not a browser or game-engine performance test. Type checking and the production build are recorded separately in the task.

Attribution: OpenStreetMap contributors; DataSF / City and County of San Francisco; USGS; ambientCG. All detailed source links and dates are retained in the evidence and material manifests.
