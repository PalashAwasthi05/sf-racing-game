# King Street reconstruction pipeline v0.7

`spatial_rules.py` resolves overlapping outlines, sampled foundations, nested roof volumes and duplicate faces across the entire study. `mobility.py` owns cycleways, side-specific lanes, mapped stairs and source-backed barriers. `freshness.py` keeps capture age separate from edit/publication dates and excludes superseded or nonvisual sign text. Adjacent envelope faces generate only their exposed depth difference. New reusable components include arched apertures, pilasters, fire escapes and raised entry steps.

Twelve additional frontage observation packets are recorded under `broader-observations.json` and compiled by the common envelope grammar. They are manually interpreted observations with estimated dimensions, not an automatic vision result. Major heuristic updates regenerate the entire study.

Run `python work/validate_spatial.py` for unfamiliar-input physical/freshness regressions. The full generator needs all other checks listed below.

Every build writes `king-street-rule-audit.json`: rejected apertures, material corrections and semantic sign attachment corrections. `resolve_material` chooses an appearance recipe from source-linked family/finish evidence, including when a prior has the wrong material. `lettering.attachment` constrains signs to compiled supports and fails on missing support declarations. Names and addresses never choose these rules. Tests use unfamiliar names and deliberately incorrect inputs. Manual source annotation remains separate from automatic numeric inference.

`surface_evidence.py` applies observed solid/opaque masks before aperture creation. Unknown is distinct from solid. `fit_boundaries.py` rectifies four manually selected edges for each explicit opening; `boundary-layouts.json` is generated data. Repeat grids cannot overwrite explicit apertures or cross negative regions. Source conflicts still need review; the current compiler veto is not a source-arbitration model.

`material_recipes.py` separates small-joint brick, terracotta panels/ribs, stucco, pale smooth trim and metal panels. These are deterministic synthesized material families, not scanned materials from the buildings. CC0 road/concrete scans remain under `material-assets`.

`lettering-observations.json` contains literal photographed text and facade attachment estimates. `lettering.py` traces font contours into raised geometry. Arial/Georgia system fonts are used when installed, otherwise Pillow's bundled font is used; exact custom brand letterforms are not reproduced. No font files or source-photo pixels are embedded in the bundle.

Run boundary fitting before the scene build:

```sh
PYTHONPATH=work python -m pipeline.fit_boundaries
python work/build_world.py
python work/validate_surfaces.py
python work/validate_detail.py
```


`envelope-observations.json` stores building matches and image-derived hierarchical structure. `envelopes.py` fits bounded roof-family levels, joins explicitly corresponding corners/datums, resolves nested regions, and compiles replacement facade geometry. Geographic identifiers occur only in observation data. Source photos are research references and are not shipped as textures.

The graph is assistant-annotated, not an automatic vision output. Signed depths, fine dimensions and many glazing details remain estimates. `validate_envelopes.py` checks geometry and structural invariants on a synthetic unfamiliar building. It does not certify photographic likeness. Reuse requires new observation data for new buildings; do not add building-specific compiler branches.

Run from the bundle root with the geospatial Python requirements installed:

```sh
python work/validate_envelopes.py
python work/build_world.py
python work/validate_detail.py
```

The complete generator also needs the earlier geodata and LiDAR fetch steps below. Those cached source datasets are not included in the bundle.

---

# World compiler v0.4

The compiler prioritizes street-visible layout and driver-height inspection. Location-specific evidence lives in data; geometry code does not select templates by address.

Reproduce from a bundle root with `work/requirements-king-pilot.txt` dependencies installed:

1. `python work/fetch_geodata.py` and `python work/fetch_lidar.py` populate the source cache (network access).
2. `python work/pipeline/infer_facades.py` rectifies sparse image landmarks and fits repeated layout grids.
3. `python work/validate_inference.py` checks perspective/occlusion recovery and contradictory opening regions.
4. `python work/build_world.py` produces the scene and evidence from cached inputs.
5. `python work/pipeline/plan_observations.py` prioritizes missing evidence from approximate road visibility.
6. `python work/validate_detail.py` checks the exported geometry and transfer chunks.

`observations/image-landmarks.json` contains assistant-authored image landmarks and wall-region bounds. It is not automatic vision output. `facade-layouts.json` is the numerical result: rectification, lattice spacing, completed cells, same-image holdouts and input hashes. `facade-profiles.json` stores reusable region/component grammars; some use fitted grids and others are visual annotations or explicit same-building transfers. `observations.json` stores source identity and dates.

`facade_grammar.py` partitions walls around openings and compiles terraces, projecting bays, glazing divisions, shades, roll-up entries, canopies and material zones. `facades.py` anchors a profile to an outward frontage; other walls retain explicit priors. Local high-elevation LiDAR support determines frontage height independently of low courtyard roof levels. Footprint consolidation is based on identity and overlap, with decisions saved in the evidence file.

The fitter's simplicity penalty and residual scale are prototype hyperparameters. Synthetic test success and same-image held-out points do not establish independent 3D accuracy. The two live fits recover 5 × 3 and 3 × 7 repeat banks. Current dated street observations cover two building identities; hotel photo dates are unknown.

Road surfaces use densified triangulations against a common terrain interpolation. The 8 m road-station observation queue approximates visibility using 2D footprint occlusion, which can miss visible upper setbacks and cannot prove a clear image sightline.

The source adapters remain configured for San Francisco; second-city ingestion is not validated. A Mapillary metadata adapter is supplied but has not been run with credentials. Dense automatic image extraction, calibrated multi-view depth, exact materials, sculpted ornament, collision, instancing/HLOD and independent street-view accuracy validation remain unfinished.

The interchange GLB uses meters, X east / Y up / Z south and standard PBR materials. CC0 material scans include provenance and physical scales under `work/material-assets`; they are examples, not captures of these buildings. Image references are not embedded as facade textures. The browser adds raster sky/sun lighting, local shadows, GTAO and tone mapping; these renderer effects are not baked into the GLB. Four-MiB gzip parts are transfer chunks, not spatial streaming.

## Reconstruction contract: evidence owns parts, parts own surfaces

The reconstruction unit is a physical part and its surfaces, not one uniform
building style. A source reading is a measurement in a particular camera and
wall frame; it is not permission to repeat that layout on every nearby wall.

1. **Preserve the measurement frame.** Record source image hashes, capture date,
   camera placement, gravity correction and the exact crop transform. A newer
   image can supersede an older observation only for the region it actually sees.
   A gravity correction does not validate camera position. Changes to a frame
   invalidate dependent metric coordinates until they are transformed or reread.
2. **Resolve ownership before drawing.** Repeated observations of one physical
   window, bay or roof edge must resolve to one host. Use source-space agreement,
   visibility and independent views; shared panorama identity alone is not enough.
   Nearby parallel planes are not interchangeable. Keep genuine setbacks and the
   nonoverlapping portions of partial readings. Ambiguity remains explicit.
3. **Use independently measured part extents.** A bay can rise above a lower main
   roof. A loggia removes an owned wall volume and needs its back, returns, floor
   and soffit. A door leaf, its casing, glazing, recess and access route are separate
   parts. A decorative crown does not change a window's aperture shape. Do not
   change an entire wall height to accommodate one projecting part.
4. **Reconstruct roof topology before detail.** Fit locally supported planes,
   recover their measured intersections, preserve real steps and courtyard holes,
   then attach dormers and other observed parts. A bounded face recovered from
   independent photo rays can complement sparse nadir returns; propagate corner
   uncertainty and reject ill-conditioned triangulation. Clip a host only inside
   a complete replacement part. Never invent a whole-building pitch from a small
   visible fragment or regularize real height discontinuities into ramps.
5. **Separate structure, pigment and wear.** Select or synthesize a material by
   observed substrate, joint/bond pattern, physical module and optical finish.
   Reuse its maps for color variants. Create a new structure only when an existing
   structure cannot represent the observation; unknown module is not a measured
   module. A cavity inherits its local facade zone unless independently observed.
   Do not let procedural wear or a large-radius shading approximation repaint the
   source-selected material. Billboards and literal legible text require their own
   observations; a facade material classification must not absorb them.
6. **Check the actual emitted scene.** Preserve normals, UV phase, aperture
   boundaries and material identity through clipping, merging and export. Verify
   both front-face visibility and opaque support behind openings. A passing
   topology test does not establish photographic likeness. Compare source-matched
   and held-out cameras, then drive the complete world; isolate lighting/material
   effects when diagnosing an anomaly rather than masking it with extra geometry.
7. **Regenerate by dependency, then integrate.** Cache readings and reusable
   material structures by input and rule hashes. Rebuild affected parts to test a
   rule, then regenerate the full area after substantial rule changes. Freeze
   generation rules during a build and retain exact input hashes. Do not present
   an old full-area build as the newly reviewed isolated candidate.

Track acquired, aligned, read, compiled, rendered and visually verified coverage
separately. Suppressing an uncertain feature does not finish its reconstruction.
An unresolved visible feature stays in the work queue. Rules contain no geographic
identities; external evidence uses identities only to join measurements to output.
The automated steps are implemented incrementally: this contract is not a claim
that source completeness, camera position or visual fidelity has been achieved.
