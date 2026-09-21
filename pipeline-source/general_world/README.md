# General reconstruction core

This package accepts normalized external measurements. It does not load authored facade profiles, geographic IDs, addresses, landmark templates, tenant lists, or example-building geometry. Opaque keys label outputs only.

Implemented: point-supported roof islands and held-out continuous roof-plane fitting; closed positive-volume solids; boundary-derived contiguous facade frames; image-evidence admission; coverage-first capture scheduling; supported stair profiles; strict object/support admission; constrained terrain surfaces; explicit failed appearance gate. Measured arbitrary aperture polygons compile to wall, frame, glazing and explicit door components. Sign carriers have independent ownership, support and collision checks. An external PBR inventory loader preserves provenance, metric scale and channel conventions.

Not implemented end to end: automatic dense image segmentation, calibrated multiview reconstruction, measured material assignment and complete automatic facade-detail reconstruction. A tested mesher is not an image extractor. Missing implementations must not silently fall back to fabricated windows or building templates.

`scene_roofs.py` applies the roof operator uniformly to a supplied detailed scene. It preserves unrelated input geometry; inherited facade content remains unverified. This compatibility pass is not the full automatic generator and does not certify an input made with authored profiles. It consumes opaque scene ownership keys only to connect parts, never as selection rules.

Run from a folder containing this package:

```
python -m general_world.run --input path/to/normalized-input.json --output path/to/result
```

The output is an explicitly incomplete geometry candidate when the image reconstruction branch is unavailable. A false appearance gate prohibits publishing it as an accuracy improvement. `policy.py` contains location-independent thresholds; they are prototype parameters, not validated universal tolerances.

Revision 0.11 connects admitted measured wall records to the host solid through
`detail_compiler.replace_measured_panel`: the old panel volume is removed before
new measured wall/opening parts are inserted. Image extraction, material fitting
and complete coverage remain unresolved; acceptance gates stay false.

`member_support` is a geometry-based compatibility correction for unverified
long elevated members with gross footprint overreach. Measured cantilevers are
exempt. It returns local replacement parts and preserves the original source
arrays. The detailed-scene adapter takes a complete external footprint registry;
a curated set of reference buildings is insufficient. `uv_transfer` preserves
physical repeat on replacement surfaces. `mesh_hygiene` removes empty faces
without filling real gaps. `review_route` produces connected road traversals
for repeatable driver-height inspection, independent of building names.

Revision 0.12 adds explicit planar surface coverage contracts before measured
wall subtraction. Missing, duplicate and reversed front faces cannot silently
replace the host. Owned openings have explicit cut domains. Roof coverage
recovery is bounded by adjacent planes, point support, neighboring surfaces,
and game-export precision checks. These are conservative compatibility repairs,
not a replacement for image-based facade reconstruction.

The imagery adapters discover complete area coverage, select spatially distributed
recent views, retrieve images and compare provider poses against reconstruction
records. World registration remains a separate required admission flag. Runtime
credentials are read without terminal echo and are never stored in source files.

`surface_orientation` corrects measured open horizontal caps in place, preserving UVs and material. It rejects closed solids, insufficient point support and plausible undersides. Coverage repair must follow orientation audit; nearby faces of either winding already own their area. New roof perimeters must meet existing roof or wall-top evidence at that height. A lower footprint cannot supply an upper-storey boundary. Sloped cap orientation, full facade coverage and all inherited architectural layouts remain unresolved.

The capture selector now targets visible street-level facade frames and independent sequences. It checks concave self-occlusion, limits height weighting to a twelve-metre planning band, and keeps unknown heights explicit. Photo adapters support full-resolution retrieval, bounded world alignment, paired metric crops, local obstruction masks and facade-part proposal diagnostics. These are implemented evidence-production stages; they do not yet supply validated facade replacements.

`sloped_surface_orientation` is an experimental read-only audit. It distinguishes plane-specific downward facets from supported undersides and preserves unresolved cases. It is not connected to automatic production face reversal.

`stereo_rays` tests depth hypotheses from paired camera rays. Shared reconstruction, source identity, baseline/parallax, positive depths, ray conditioning, residuals and bounded support domains are checked by the diagnostic adapter. It retains repeated-pattern ambiguity and cannot certify contours, absolute placement or opening geometry. The detailed world is not regenerated from these hypotheses until the missing reconstruction and validation stages are implemented.

Revision 0.13 adds `material_synthesis`, `material_periodicity` and
`scanned_surface_variant`. Explicit external descriptors produce shared metric
PBR structures; pigment variants preserve relief. Scan module estimation rejects
ambiguous half periods and inconsistent row phases before bounded rescaling.
`apply_materials_v13.py` applies these rules to externally joined surface
observations, preserving geometry, transforms and physical UV scale. This
compatibility pass does not infer surface assignments from photographs.
