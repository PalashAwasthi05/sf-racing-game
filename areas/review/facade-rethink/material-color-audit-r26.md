# R26 material assignment audit

## Finding

The measured color swatches are reaching the generated GLB. I found no swatch discard, accidental second color tint, or second tone-map pass in the examined path. The concrete weakness is that evidence-backed surface textures cover very little of the R26 candidate, while most typed or unclassified surfaces keep deterministic, low-detail recipes or flat fallback materials.

## Swatch path checks

- `regen/specs.py:46-53` encodes a reading's six-digit hex and optional family/pattern/finish/module fields into the material key. `to_profile()` also carries `materialEvidence.colorRgb` from that source swatch.
- `pipeline/material_recipes.py:138-143, 299-340` decodes typed keys to the exact RGB bytes and builds the color texture with a neutral white `baseColorFactor`. `pipeline/materials.py:14-24` does the same for legacy `base#rrggbb` keys.
- In the R26 candidate GLB, representative texture medians were within one byte/channel of the input hex for plain stucco and typed horizontal-course timber. Examples: `stucco#98a384` -> median `[152,163,133]` for source bytes `[152,163,132]`; typed timber `#8e9aa6` -> `[141,153,165]` for `[142,154,166]`. Neutral factor was `[255,255,255,255]`. The rough brick sample `#984b3c` was `[150,80,66]`; the larger offset from `[152,75,60]` is expected from its recipe's mortar/joint pixels and noise, not a tint multiplication.
- Plain fallback keys such as `#5a6455` export linearized factors `[26,32,23]`, which is the expected sRGB-to-linear conversion for those hex bytes. That is not a darkened swatch when consumed as glTF PBR.
- `finish.ts:60-77` does not tint facade materials: authored tint is applied only to ground layers. Facade finish rules alter roughness/normal/probe properties, not base color.
- `world-view.ts:44,52` sets ACES on the renderer, but Three.js `WebGLRenderer.js:2197-2204` selects `NoToneMapping` whenever the render target is non-null. The scene target is a half-float target (`world-view.ts:52`); `grade.ts:135-166` applies the custom ACES in the final full-screen pass. Thus the materials do not receive ACES twice.

## Measured texture coverage and generic appearance

R26 output material reports show:

- v13: `620/620` meshes retained without a new observation; `0` new structure assets; `0` reused scan structures; the report's assignment is explicitly not image verified.
- v10: only `34/620` meshes updated (`5.5%`); `517` are protected/unclassified and `26` have incomplete typed descriptors. Its report also labels the assignments “inherited material intent; not image verified.”

The procedural recipes explain the remaining generic appearance. For example, plain stucco's color texture varies at about 1 RGB level around the sampled swatch (`noise: 1.3`); typed timber horizontal courses use 3 mm joints, about 1 mm relief, and only 2 RGB levels of variation, with an explicitly unmeasured default course pitch when `moduleM.z` is absent. These are deliberately restrained, repeatable recipes, not recovered source texture detail. The result can therefore preserve the observed hue yet still look pale/smooth or generic. Absolute brightness against source photography remains non-diagnostic without registered views and lighting.

## Candidate next step

A general coverage repair should make the current profile's typed surface evidence available to texture assignment, rather than relying only on the historical GLB material-fingerprint observation table. Keep texture assignment gated by verified family, pattern, finish, and measured module; let unsupported or incomplete zones abstain. This targets the demonstrated 586-mesh gap without changing lighting or forcing a material color correction that the evidence does not support.

## Why the protected counts are not a field-loss bug

I traced the current v10 categories back to source material names in the R26 raw GLB:

- The 26 incomplete entries are 22 horizontal-course timber variants, 3 running-bond brick variants, and 1 scalloped-shingle timber variant. They retain the observed family and pattern. The missing field is `finish`; the material key has no `~fin=` token. `selector_target()` (`pipeline/material_recipes.py:213-237`) correctly refuses scan-library matching without finish, while `observed_recipe()` still emits a procedural recipe from available structure and swatch. The scan selector needs finish to distinguish an otherwise matching asset safely.
- The 517 protected/unclassified entries are not 517 bare facade walls. They include opening/trim colors with no material family, glazing, curtains, recesses, interior, frames/metals, and already-assigned scan materials. The v10 adapter explicitly removes the painted `stucco`, `render_*`, and timber recipes from its scan targets (lines 80-81) because raw wood/plaster scans would add unobserved grain or knots. The remaining procedural stone/roof recipes are candidates for a scan only when required structure and scale evidence match.
- The profile compiler passes `pattern`, `finish`, and `moduleM` through `material()` into the self-contained GLB key (`regen/specs.py:46-53`). It uses that same adapter for primary surfaces, material panels, bay cladding, and bay material zones (`regen/specs.py:327-374`). Existing typed GLB names confirm those descriptors reach material creation. Across the current 1,446-profile input, 12 primary materials have a pattern, 3 have a finish, and none supplies a measured `moduleM`. No measured module was dropped; there was no measured scale for the selector to use.

Therefore I found no evidence-backed dispatch/compiler fix to make here. Removing the finish or scale gates would manufacture scan compatibility; adding more procedural geometry would invent surface structure. The safe concrete next step is to increase the reading coverage of finish class and measured module where the source actually supports them, then the existing selector can consume those values.
