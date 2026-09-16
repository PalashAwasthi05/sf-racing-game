# v15 facade profile: the smallest machine-readable input the grammar compiles

Purpose: one frontage photo, read by an automated observation step, becomes one profile record;
`pipeline/facade_grammar.py` (unchanged) compiles it into a detailed wall. Everything below is what the
existing code actually reads. Consumers: `facade_grammar.compile_wall` / `openings`,
`surface_evidence.admissible` / `resolve_material`, `pipeline/facades.py:134` (matching),
`generate_detailed_v14.py:24` (load) and `:320` (street views), `spatial_rules.perimeter_strip`.
Validator: `work/validate_profile_v15.py` (authoritative for ranges and names; `--materials` prints the
accepted vocabulary from the live source). JSON Schema for structured emitters: `profile.schema.json`
(generated from the validator's tables).

Profiles are data. Ids and match keys are join keys carried through from data; nothing in the grammar
selects a rule by identity.

## 1. Frame and units

- `normal` `n = [nx, ny]` is the outward unit normal of the frontage in the local metric frame
  (x east, y north; the `coordinateSystem` / `center` of `world.json`). Copy it from the target wall's
  `facadeWalls[].normal` in `king-street-evidence.json`.
- Tangent `u = [-ny, nx]`. Standing on the street facing the wall, `u` points to the viewer's right, so
  `x` fractions run left to right exactly as in a street photo of the frontage.
- `W` (frontage width) is the extent along `u` of every footprint edge parallel to `u` (|cos| > .96,
  about 16 degrees) that lies within `planeToleranceM` (default 2.5 m) of the outermost such edge.
  `x = 0` is the left end of that extent, `x = 1` the right end.
- `H` (frontage height) is measured by the pipeline from LiDAR (peak return cluster in the top 35% of
  the building, .85 quantile + .12 m, capped at the building height; wall level as fallback).
  `y = 0` is the wall base, `y = 1` its top.
- The emitter never supplies `W` or `H`. Every `region` / `rect` is `[x0, y0, x1, y1]` as fractions of
  `W` and `H`. Fields ending in `M` are metres. `mullionFractions` / `transomFractions` are fractions of
  the opening's own width / height.
- The grammar is anchored to the whole frontage and then clipped to each wall segment; openings thinner
  than .12 m after clipping are dropped. It is applied only to ground-level outer walls
  (`elevationOffset < .5`), never to roof tiers.

## 2. Minimal profile

```json
{
  "id": "example-minimal-frontage",
  "match": { "osmWayId": "0000000" },
  "normal": [0.0, 1.0],
  "sourceIds": ["photo-example-001"],
  "materialEvidence": { "family": "render", "finish": "gray", "sourceIds": ["photo-example-001"] },
  "groups": [
    { "region": [0, 0, 1, 0.3], "columns": 4, "rows": 1,
      "style": { "kind": "shopfront", "openingRatio": 0.8, "openingHeightRatio": 0.85,
                 "mullionFractions": [0.5], "transomFractions": [0.8] } },
    { "region": [0, 0.3, 1, 1], "columns": 6, "rows": 3,
      "style": { "kind": "window", "openingRatio": 0.55, "openingHeightRatio": 0.6,
                 "mullionFractions": [], "transomFractions": [] } }
  ],
  "unresolved": ["Region boundaries, counts and ratios read from one frontage photo; not survey dimensions."]
}
```

Compiled against the validator's synthetic 20 x 12 m wall this yields 22 openings, 31 wall pieces,
179 detail pieces, record status `visual-grammar`, material `render_gray`.

Strictly required by the code (KeyError otherwise): `id`, `match` with the `osmWayId` key, `normal`,
`sourceIds`, and for each group either `region` or a resolvable `fitId`. Everything else has a default.
The observation step should additionally emit `materialEvidence`, `style.kind`, `openingRatio`,
`openingHeightRatio`, empty `mullionFractions` / `transomFractions` when none are seen, and `unresolved`.

### 2.1 Top-level fields

| field | type / range | required | default when missing | notes |
|---|---|---|---|---|
| `id` | non-empty string, unique in the file | yes | KeyError | becomes `grammarId`, `archetype`, mesh layer suffix `__observed_<id>` |
| `match.osmWayId` | string (OSM way id from data) or `null` | key required | `generate_detailed_v14.py:320` KeyError | matched by `str()` equality with the building's OSM way; `null` never matches |
| `match.featureId` | string building id (`osm_...` / `sf_...`) | no | none | honoured by `envelopes.matches` and `perimeter_strip`, NOT by `facades.py:134` (see 6) |
| `normal` | `[nx, ny]`, unit length within .01 | yes | KeyError | matching uses `dot(n, normal) > .95` on the raw value (18 degrees); a non-unit normal matches too widely or never |
| `sourceIds` | list of strings | yes | KeyError in `compile_wall` | non-empty: openings get evidence `visual-grammar`; empty: `inferred` |
| `material` | accepted material name (section 4) | no | `stucco` | wall material when `materialEvidence` is absent |
| `materialEvidence` | `{family, finish, sourceIds}` | recommended | `material` is used as is | `family` must be one of the six families (ValueError otherwise); unknown `finish` falls to the family default; `sourceIds` must be non-empty (ValueError); the resolved recipe overrides `material` and is audited as `material-family` |
| `groups` | list of groups (2.2) | recommended | `[]` (blank wall) | one entry per repeat bank |
| `unresolved` | list of strings | recommended | `[]` | copied into the wall record; state what is an estimate |
| `panels` | list of `{rect, material}` | no | none | wall-material override zones; last match wins; `material` required (KeyError) |
| `bands` | list of `{height, thicknessM, depthM, projectionM, material}` | no | none | continuous horizontal band at `height` (fraction of H, required); defaults .16 m / .2 m / .05 m / `stone` |
| `openings` | list of explicit openings `{rect or polygon, style, sourceIds, id, evidence}` | no | none | always admitted; each vetoes repeat cells overlapping it by > 5% of the cell area |
| `surfaceEvidence` | list of `{rect or polygon, label, sourceIds, ...}` | no | none | labels `solid`, `opaque-panel`, `sign-carrier` veto any opening overlapping > 1.5% of its area; `unknown` and any other label are ignored (unknown is not solid) |
| `openingPolicy` | `bounded-repeat` or `observed-only` | no | `bounded-repeat` | `observed-only` suppresses every repeat cell; only explicit `openings` remain |
| `planeToleranceM` | (0, 10] m | no | 2.5 | which parallel edges count as the frontage plane |
| `copingMaterial` | accepted material name | no | `charcoal` | top coping, always emitted |

Unknown top-level keys are ignored by the pipeline (the validator warns).

### 2.2 Group fields

| field | type / range | required | default | notes |
|---|---|---|---|---|
| `region` | `[x0, y0, x1, y1]`, `0 <= x0 < x1 <= 1`, `0 <= y0 < y1 <= 1` | yes (unless `fitId` resolves) | KeyError | fractions of W and H |
| `columns`, `rows` | integer >= 1 | no | 1 | `0` gives ZeroDivisionError in `openings()` |
| `style` | object (2.3) | no | `{}` (plain window with defaults) | |
| `omit` | list of `[columnIndex, rowIndex]` | no | `[]` | cells skipped; out-of-range pairs are ignored |
| `xCenters`, `yCenters` | fractions of the region | no | uniform `(i + .5) / n` | override the grid centres; the count then comes from the list, `columns`/`rows` only set the default pitch |
| `spacingX`, `spacingY` | (0, 1] fraction of the region | no | `1/columns`, `1/rows` | cell pitch; opening width = `spacingX * (x1 - x0) * openingRatio` |
| `fitId` | id in `facade-layouts.json` | no | none | do NOT emit from a photo: needs a numerical fit; a missing fit with no `region` raises KeyError `'region'`; even with a `region` fallback the record status becomes `image-layout-fit` |

Regions of different groups may overlap (a narrow stack inside a wide bank is a legitimate pattern), but the
resulting opening rectangles must not: overlapping openings interpenetrate frames and glazing. The validator
warns on region overlap and errors on opening overlap (> 1% of the smaller opening).

### 2.3 Style fields

| field | type / range | default | notes |
|---|---|---|---|
| `kind` | one of section 3 | `window` | any other string compiles as a plain window |
| `openingRatio` | [0.05, 1] | .7 | opening width as a fraction of the cell pitch |
| `openingHeightRatio` | [0.05, 1] | .7 | opening height as a fraction of the row pitch |
| `frameWidthM` | (0, 0.5] m | .06 | |
| `frameMaterial` | accepted name | `charcoal` | |
| `glassMaterial` | accepted name | `glass` | |
| `recessM` | [0, 3] m | .2 | glazing plane behind the wall face |
| `projectionM` | [0, 1.5] m | .04 | frame projection; bay depth for `projected_bay` |
| `mullionFractions` | list of (0, 1) fractions of opening width | `[.25]` | absent key means ONE mullion; emit `[]` for none |
| `transomFractions` | list of (0, 1) fractions of opening height | `[.76]` | absent key means ONE transom; emit `[]` for none |
| `canopy` | boolean | false | plate + glass + two steel rods (cylinders) above the opening |
| `canopyDepthM` | (0, 4] m | 1.6 | |
| `shade` | boolean | false | horizontal louvre shade above the opening |
| `material` | accepted name | `zinc` | `projected_bay` cheeks and inter-floor cladding only |

Unknown style keys are ignored (the validator warns).

## 3. Style kinds the grammar knows

`window` (default), `entry`, `shopfront`, `rollup`, `terrace_frame`, `projected_bay`.

- `window`: recessed glazing, interior backing, frame, mullions/transoms, sill; blinds behind some cells
  (deterministic by index).
- `entry`, `shopfront`: as `window` plus pull handles; no blinds; typically with `canopy`.
- `rollup`: zinc door slab with horizontal slats; ignores frame, glass, mullion, transom, canopy and shade.
- `terrace_frame`: deep structural outside frame, stone sill, railing with `glass_light` infill; use a
  large `recessM` (about 1 m).
- `projected_bay`: bay projecting `projectionM` from the wall with cheeks and an inter-floor panel in
  `style.material`.

## 4. Accepted material names

Resolution order in `pipeline/materials.py Materials.get`: alias, then recipe, then scan, then flat
colour / palette. Any other name raises in `Materials.get` at mesh emission (after the profile compiled),
unless it happens to be six hex characters, in which case it is silently read as a colour.

- recipes (`material_recipes.RECIPES`, 15): `brick_dark brick_modular limestone panel_dark panel_silver
  render_gray render_salmon render_white stone stucco stucco_red terracotta_panel terracotta_ribbed
  timber_dark timber_tan`
- scans (`material-assets/manifest.json`, 3): `brick asphalt concrete` (note: `brick` is the scanned
  tile, not the brick recipe; use `materialEvidence.family: brick` for the recipe)
- flat colours (27): `aluminum bark charcoal copper curtain glass glass_blue glass_clear glass_dark
  glass_light greenpaint interior joint lamp leaf1 leaf2 leaf3 recess redpaint roofing silver soil steel
  terracotta verdigris white zinc`
- generator palette (`generate_detailed_v14.py:76`, 5 extra): `grass leaf trunk water yellow`
- aliases (8): `brick2->brick sidewalk->concrete frame->aluminum rail->steel metal->steel dark->recess
  paint->white roof->roofing`

`materialEvidence.family` / `finish` (in `surface_evidence.resolve_material`), the intended photo vocabulary:

| family | finishes -> recipe |
|---|---|
| `brick` | `default -> brick_modular`, `dark -> brick_dark` |
| `terracotta` | `default -> terracotta_panel`, `ribbed -> terracotta_ribbed` |
| `render` | `default -> stucco`, `red -> stucco_red`, `salmon -> render_salmon`, `gray -> render_gray`, `white -> render_white` |
| `timber` | `default -> timber_dark`, `dark -> timber_dark`, `tan -> timber_tan` |
| `pale-trim` | `default -> stone` |
| `metal-panel` | `default -> panel_silver`, `dark -> panel_dark` |

Materials the grammar emits on its own regardless of the profile: `interior`, `stone`, `charcoal`, `zinc`,
`silver`, `steel`, `glass_light`, `curtain`.

## 5. What happens on missing fields (summary)

- Missing `id`, `normal`, `sourceIds`, `match.osmWayId` key, group `region` (without fit), panel
  `material`, band `height`: exception in the generator (KeyError / TypeError), which aborts the whole
  build. The validator reports these as errors before compile.
- Missing `materialEvidence`: `material` is used verbatim; missing `material` too: `stucco`.
- Missing `style` or style keys: the defaults of 2.3, including one mullion at .25 and one transom at .76.
- Missing `columns` / `rows`: 1. Missing `omit`, `panels`, `bands`, `openings`, `surfaceEvidence`: none.
- Missing `unresolved`: `[]` in the record (the pipeline's honesty notes are lost; emit them).
- A profile whose normal does not match a footprint edge within 16 degrees, whose frontage is under 3 m,
  or whose wall run is under .3 m: `compile_wall` returns `None` and the wall silently falls back to the
  prior (status `inferred`).

## 6. Matching and building identity

`facades.py:134`:
`profile = next(p for p in profiles if c['osm'] and str(p['match'].get('osmWayId')) == str(c['osm']['id']) and dot(n, p['normal']) > .95)`
First matching profile in file order wins; two profiles with the same way id and normals within 18 degrees
are ambiguous (the validator warns).

Buildings that cannot match today: those with no OSM way attached (`c['osm']` is `None`). In
`world.json` they are the buildings with `footprintSource == "DataSF footprint"` AND
`outlineMetadata == {"captureAgeKnown": false}` (no `editedAt` / `version`); in the current world that is
2 of 58 (`sf_201006_0000003`, `sf_201006_0056871`). The other 7 `DataSF footprint` buildings do carry an
OSM way (the max-overlap OSM building way with area > 40 m2 in `data/osm.xml`, the rule at
`generate_detailed_v14.py:143-146`), but its id is not written to `world.json`; recover it from the data
with the same rule. Buildings with an OSM URL in `sourceUrl` (49 of 58) give the way id directly. Caution:
several `sf_` fragments can share one OSM way, so a profile keyed by that way applies to every fragment
wall with a matching normal, each anchored to its own frontage.

Match by building id without a code change: not possible in `facades.py` (it reads only `osmWayId`).
`envelopes.matches` already accepts `match.featureId == c['id']`, so `perimeter_strip` would honour it while
the grammar wall never compiles. The one-line general change (NOT made here) is to give `facades.py:134`
the `envelopes.matches` semantics:

```python
profile=next((p for p in profiles if (p['match'].get('featureId')==c['id'] or (c['osm'] and str(p['match'].get('osmWayId'))==str(c['osm']['id']))) and np.dot(n,p['normal'])>.95),None)
```

(`generate_detailed_v14.py:320` should then read `profile['match'].get('osmWayId')`; it only builds the
street-view list.) Until then, emit `match.osmWayId` as a string and keep the key present (`null`) with a
`featureId` for OSM-less buildings so the file loads.

## 7. Effect on outputs

A compiled profile wall is recorded in `facadeWalls` with `edge: "grammar"`, `status: "visual-grammar"`
(or `image-layout-fit` when any group carries a `fitId`), `material` = resolved recipe, `sourceIds`,
`unresolved`, `openingCount`, and the building's `facadeEvidence` becomes `Image layout / visual grammar`
with `fittedWalls` incremented. `materialEvidence` and every vetoed opening appear in the rule audit.

## 8. Validator

```
PYTHONPATH=work work/geoenv/bin/python work/validate_profile_v15.py PROFILES.json [--no-fits] [--json REPORT.json]
PYTHONPATH=work work/geoenv/bin/python work/validate_profile_v15.py --materials
```

Checks: types and ranges of every field above, `columns`/`rows >= 1`, unit-length normal, known material
names (from the live `materials.py` / `material_recipes.py` / generator palette), material family and
finish, region overlap (warning) and opening overlap (error), `omit` bounds, duplicate ids, ambiguous
matches, unknown keys and kinds (warnings). Then it compiles the profile with `facade_grammar.compile_wall`
against a synthetic 20 x 12 m frontage oriented by the profile's own normal, using a stub ctx
(`cube`, `cylinder`, `polys` callables), and reports openings, wall pieces, detail pieces, cylinders,
record status, resolved material, opening area fraction and any exception. Exit status 1 on any error or
exception. `--no-fits` reproduces the environment of the photo step (no `facade-layouts.json`).

Results on the five shipped profiles (with fits): all pass; openings 48 / 57 / 118 / 118 / 61,
wall pieces 293 / 187 / 421 / 411 / 309. Without fits one shipped profile fails with KeyError `'region'`
(a region-less `fitId` group), which is why `fitId` is excluded from the v15 emitter vocabulary.
`examples/minimal-profile.json` passes (22 openings); `examples/broken-profile.json` produces 18 + 10 errors
and two compile exceptions (`ValueError: Material evidence needs a source`, `KeyError: 'sourceIds'`).
Reports of these runs: `validate-existing-profiles.json`, `validate-existing-profiles-nofits.json`,
`validate-minimal.json`, `validate-broken.json` in this folder.
