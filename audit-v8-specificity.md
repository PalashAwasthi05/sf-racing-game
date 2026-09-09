# Specificity dependency audit (v8)

## Finding

The reusable geometry operators are mostly location-blind, but the current build is not a zero-location-specific heuristic. Hand-authored records are loaded as first-class inputs and their geographic matches select geometry. A clean dependency cut is to keep raw geographic observations and a city/source adapter, while making every authored profile an optional, evidence-derived feature packet rather than a required landmark registry.

## Paths from authored knowledge to generated geometry

1. **Fixed region and coordinate adapter (`build_world.py:19-70`, `pipeline/config.json`).** `region.addressWayId` selects the origin way; `label`, fixed radii, EPSG:32610, source file names, and the local transform determine which OSM ways, footprints, LiDAR points, trees, signs and roads enter the scene. This is adapter/configuration state, but the current config is explicitly King Street/San Francisco and therefore location-bound.

2. **Facade profiles and layout fits (`build_world.py:21-24,180-215`; `facades.py:22-56`; `facade_grammar.py`).** `facade-profiles.json` matches OSM way IDs and supplies spans, normals, levels, materials, surface masks, grids, arbitrary apertures, components, depths and joins. `facade-layouts.json` supplies identity-matched opening lattices; `boundary-layouts.json` is appended directly to matched profiles. These records change wall ownership, roof replacement masks, apertures, material zones, projecting parts, and attached structures. They are manual location/building profiles, even though the compiler has no address branches.

3. **Envelope observations (`build_world.py:26,180-217`; `envelopes.py`; `spatial_rules.perimeter_strip`).** `envelope-observations.json` profiles match feature/OSM IDs and define manually annotated building parts, roof levels, depths, towers, open structures, and joins. `prepare()` resolves them; `replacement_mask()` removes the generic shell; `compile()` emits the articulated geometry. This is the strongest landmark-specific geometry path (Beacon/Glassworks/Oracle Park and named frontages).

4. **General observations (`build_world.py:27,210,214`; `facades.py`).** `observations.json` matches OSM IDs and provides per-wall material/opening/layout overrides, negative evidence and footprint-source decisions. A match changes style, opening admission, and which wall is visual-fit versus inferred. `footprintDecisions` can suppress a municipal candidate in favor of a specific OSM way.

5. **Lettering (`build_world.py:25,216`; `lettering.py`).** `lettering-observations.json` matches feature/OSM IDs and contains literal tenant, address and landmark text plus carrier/profile/node attachments. It generates raised facade geometry and can add a canopy. This is intentionally evidence-driven, but it is authored location-specific content and directly changes geometry.

6. **Street and vegetation inventories (`build_world.py:224-283`; `streets.py`, `mobility.py`, `vegetation.py`).** `osm.xml`, `row.json`, `trees.json`, and `signs.json` are raw geographic inputs; their mapped positions and tags generate roads, sidewalks, crossings, tracks, barriers, signs and trees. The generic rules (lane parsing, buffers, curb strips, tree templates) are reusable. The current adapter is location-specific because file schemas, local projection, source IDs, and the study anchor are embedded in the driver. `signs.py` converts inventory legends to geometry but does not invent text.

7. **Derived metadata/source text (`build_world.py:334-344`).** Named San Francisco sources and fixed URLs do not select mesh geometry, but they are another hard-coded regional registry and should move to adapter metadata. `rebuild_world.py` fingerprints every pipeline JSON, so authored profiles are mandatory build dependencies today.

## Generic operators versus adapters and manual inputs

Generic operators: `envelopes.py` (matching/frames/levels/nodes/returns/apertures), `facades.py` priors and wall compiler, `facade_grammar.py`, `fit_boundaries.py`, `infer_facades.py`, `surface_evidence.py`, `spatial_rules.py`, `mobility.py`, `streets.py`, `vegetation.py`, `lettering.py`, `signs.py`, materials, freshness, and the deterministic mesh helpers in `build_world.py`. They should accept normalized feature/evidence records and never inspect names, addresses, landmark IDs or city labels.

Location adapter code: the input discovery and normalization in `build_world.py`, fixed projection/origin, study clipping, OSM/DataSF/LiDAR/tree/sign/ROW loaders, source URLs, and the current `config.json`. Adapter output should be normalized geometry, tags, timestamps, IDs, and evidence references; IDs must remain opaque.

Manual observation inputs: all profile/layout/boundary/envelope/lettering/observation JSON, including normals, spans, graph nodes, level priors, material masks, opening rectangles, joins, source IDs, literal text and feature matches. They may remain as external runtime evidence packets, but the general heuristic must run without them and must not require named landmark profiles or overrides.

## Minimal dependency-cut plan

1. Replace `config.region` with a runtime `area/snapshot/adapter` object; move CRS, source paths, center selection, radii and source ledger out of the compiler.
2. Normalize raw sources before compilation. Remove direct file reads from pipeline modules; pass trees, ROW, signs, and observations through context/schema.
3. Make profiles/evidence optional packets keyed by opaque feature IDs. If absent, use the same generic LiDAR/footprint/tag-derived fallback for every feature. Do not special-case size, name, landmark class, or address; any venue prior must be a documented generic tag/measurement rule.
4. Delete driver-side direct profile matching and lettering/profile mutation; resolve packets in the adapter, validate their provenance, and let the generic compiler consume compatible evidence only. A packet may add geometry only when its observation evidence is present.
5. Keep `rebuild_world.py` as an input fingerprint/rebuild wrapper, but fingerprint normalized adapter outputs and generic schemas separately so a city does not become a code dependency.
6. Add a negative audit that fails if compiler code references address/landmark labels, hard-coded geographic IDs, or display names, and if a build with no manual packets cannot produce a generic scene.

## Four identity-blind/equal-treatment tests

1. **Opaque ID/name permutation:** permute feature IDs, OSM way IDs, names, addresses and landmark labels while preserving normalized geometry/tags/evidence links. Mesh vertices, semantic geometry layers, materials and counts must be identical after metadata-ID normalization.
2. **Translation/rotation replay:** feed one generic footprint, terrain sample and road/evidence packet at two translated/rotated locations with fresh opaque IDs. Transform the outputs back; geometry and material choices must match within tolerance.
3. **Paired equal inputs:** build two buildings with identical footprint, LiDAR heights, tags and evidence, but different names/address text. Their generated geometry must be congruent; only explicitly observed literal lettering may differ, and removing lettering packets must make outputs identical.
4. **No-profile fallback:** run two unrelated areas with all manual profile/layout/envelope/lettering packets absent. Both must complete through the same generic operators, produce no landmark-specific branches/geometry, and expose identical fallback status and rule paths for equivalent normalized inputs.

