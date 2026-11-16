# Roof and attic observation contract

Apply this same pass to every facade packet with visible upper structure. Identifiers join evidence only. Do not choose reconstruction rules by address, name, street, or ID.

Read the actual photographs alongside the metric facade frame. Separate the host roof from dormers, gable end walls, tower caps, chimneys, and upper-storey balconies. A small gabled dormer is not evidence that the whole building has a gable roof. Preserve unknown and occluded regions; a cropped roof is not a flat roof.

Output a JSON list of observations with `side`, `hostRoofKind` (hip, gable, mansard, flat, unknown), `confidence`, `eaveM`, `roofColor`, `features`, `unresolved`, and `reason`. Heights are metres above the building datum; x is metres from the source facade frame's left edge. Never silently mix pixel coordinates or normalized fractions with metres.

Each feature has `kind` (dormer or conical-cap), `x0M`, `x1M`, `bottomM`, `topM`, `roof`, `windowCount`, `trimColor`, `wallColor`, and `confidence`. Use null for unmeasured values. Distinguish glass from shutters and louvres. Keep unsupported curved caps, ornamental relief, hidden depth, and uncertain camera calibration in `unresolved`, rather than inventing a supported shape. Colors are provisional image observations, not calibrated reflectance.

Pass this external JSON to the generator with `--roof-observations`. The adapter checks the metric frame, part ownership, finite bounds and measured roof support. Width-based depth remains explicitly inferred. The continuous roof surface owns the main cap and its true perimeter; internal patch boundaries do not receive walls. Components must meet the host without burying openings or leaving exposed seams. Compare same-camera renders with the source images before admitting a candidate. Geometry tests establish continuity only, never resemblance.

This contract is a candidate workflow. Its current schema still cannot faithfully express every curved dormer, stepped roof, ornamental cap, or per-side eave. Those omissions must remain visible in the audit and require a representation change before claiming completion.

## Measurement ownership and validation

`eaveM` is the wall/roof junction, not the top of a gable end or the highest visible wall. `hostRidgeM` is separately measured and must exceed eave height for a pitched form; unknown/occluded ridges remain null. `ridgeConfidence` and `ridgeNote` explain visibility and calibration. Never recycle `wallTopM` without verifying its physical meaning.

Observed dormers, tower caps, and chimneys own their returns. Exclude their bounded source-frame strips from host-roof fitting; a tall rooftop component must not become the host ridge. A rejected analytic host is a reconstruction failure to resolve, not permission to display raw scan noise as architecture. Audit the facade-to-footprint registration independently of accepting the vertical observation.

Review cameras must preserve source position, target, vertical field of view, and aspect ratio. Position alone is not a matched-camera comparison. The local viewer supports `viewFrom`, `viewAt`, and `viewFov` for repeatable inspection.

Surface observations also need typed structure where visible: see `pipeline/material-observation-contract.md`. Color-only observations cannot establish roof substrate, board direction, brick bond, or measured repeats.

## Frontage outline is not a global roof form

For a front-facing gable, eave height describes its corners; it must never flatten the whole frontage to that height. Record a separate `frontageRoofline` with metric `[x,z]` points, source pixel segments, confidence, and uncertainty. Distinguish directly visible points from an apex derived by intersecting two visible straight slopes. Do not use the image crop edge as the apex. Occluded or disconnected outline segments remain unknown. This frontage outline constrains its own wall and roof seam; it does not establish the ridge height or shape throughout the building.

A visible roof edge can constrain a boundary even when a cornice hides the wall/roof junction. Keep its measured interval and role as `visible-roof-edge`; do not relabel it a confirmed eave. Register constraints to the relevant boundary chain, not every nearby boundary vertex.

A continuous mesh with large unsupported areas or steep interpolation artifacts does not pass. Compare supported regions, unresolved extent, source silhouettes and street renders. Full-area regeneration remains gated on visual acceptance across building types; passing mesh tests alone is insufficient.

## Source measurements are not model output

A reader's `confidence` label is not a measurement check. Store the actual source image, image dimensions, padding/scale, observed pixel segments, and pixel uncertainty. Derive metric points in code (`pipeline/roof_image_measurements.py`) and retain the result. Never copy a guessed metric apex back into plausible-looking pixel coordinates. The source segments must be visually checked against the source image before setting `sourcePixelsVerified`; numerical consistency alone cannot do this.

Attach these inputs as `frontageRoofline.measurement`. The roof adapter rejects standalone `pointsM` claims without that provenance, invalid segment geometry, contradictory supplied metric points, and inconsistent views. A derived apex may lie above a crop only when the visible slopes support that bounded extrapolation. Surface height, gable-wall outline, roof covering, and decorative trim are distinct observations.

Global skyline camera registration is only an initial camera estimate. Its accepted flag does not establish facade-specific alignment, and its competing-fit score is not a held-out residual. Compare corresponding facade features across rectified views. A visible shift means the dimensions remain uncertain until local pose/plane alignment is solved; do not average the views into a supposedly exact measurement. Keep image capture date and camera identity with each observation, including when registered source selection differs from the nominal source.

Qualitative `frontageKind: "gable-end"` can survive a rejected height measurement. A gable-end reconstruction fits both slopes and the ridge position from spatial scan support, validates against held-out cells on both slopes, and stops where measured structure changes. It must not force a centered ridge, infer full-building depth from a front view, or reuse a hidden eave as a hard constraint. `anchorUncertaintyM` is a fit prior, not permission to move a roof to an unsupported height.
