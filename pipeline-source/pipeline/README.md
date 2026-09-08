# King Street reconstruction pipeline v0.6

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
