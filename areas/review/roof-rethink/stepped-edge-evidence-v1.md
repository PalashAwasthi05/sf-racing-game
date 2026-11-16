# Generic stepped-edge evidence, v1

**Status:** initial pixel-space prototype. The discrimination follow-up is in
`stepped-edge-discrimination-v2.md`; use that report for current counts,
smooth-edge controls, and metric-position limitations. This remains an
experiment, not a production constraint.

## Evidence

Read the registered image records from all `sides-*.json` files under
`work/areas/pacific-heights/regen/sides-reg/` and resolved their rectified crops
from the same index's `img/` directory. All four crops were registered as
`skyline`, with `treesNearSightLine=0`.

| Role | Image | Distance | Incidence | Left steps | Right steps | Bright neutral coverage near contour |
|---|---|---:|---:|---:|---:|---|
| Target view 0 | `110228481918019`, `sf_201006_0017913__s3p0__v0_rect.jpg` | 17.51 m | 0.649 | 7 | 8 | 0.817 left, 0.318 right |
| Target view 1 | `210201111438676`, `sf_201006_0017913__s3p0__v1_rect.jpg` | 20.52 m | 0.557 | 8 | 7 | 0.871 left, 0.085 right |
| Flat-roof control view 0 | `1265570360904518`, `sf_201006_0015135__s2p0__v0_rect.jpg` | 13.26 m | 0.999 | 0 | 0 | 0.348 left, 0.189 right |
| Flat-roof control view 1 | `190848303518897`, `sf_201006_0015135__s2p0__v1_rect.jpg` | 14.43 m | 0.886 | 1 | 0 | 0.153 left, 0.140 right |

Step counts are horizontal contour runs 4–18 px long separated by 5–18 px
vertical changes, after a 5-pixel median filter and 3-pixel level quantization.
The contour comes from the largest dark connected mass entering the upper
quarter of the rectified crop. Bright neutral coverage counts columns with a
nearby neutral bright pixel; it is a proximity measure, not a trim classifier.

The target shows repeated steps in both views and both rakes. Neutral bright
coverage is consistently high along the left rake; it is weak on the right in
view 1. The flat control has at most one isolated transition. The counts alone
do not prove cross-view position agreement; see the v2 metric comparison.

## Reusable extractor

`work/regen/extract_stair_edge_evidence.py` accepts arbitrary rectified images
and reports contour run counts and nearby bright-neutral coverage. It contains
no feature IDs, location names, or trim keywords. The method is intentionally
bounded to a large dark upper mass and should abstain if none is found. It does
not alter source images, R31 inputs, or generated worlds.
