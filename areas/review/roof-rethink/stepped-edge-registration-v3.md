# Independent lower-wall registration, v3

**Decision:** abstain. The existing SIFT + held-out homography routine could
not fit from the lower planar wall features when the gable, bay front, and
neighbor were excluded. There is no accepted transform to use for held-out
step-position evaluation or control remapping.

## Mask and result

Used the exact registered pair from `sides-1.json`, at 416x539 px and 45 px/m.
The mask was derived from the registered profile for this side:

- Keep pixels below the measured wall top (`wallTopM=7.1`, rectified row 194).
- Exclude the bay-front x span from its recorded profile box (rectified x
  77–278 px).
- Exclude the adjacent frontage after the profile's noted neighbor boundary
  near x=7.2 m (rectified x=350 px).
- Keep only the remaining left and right lower-wall strips; this preserves
  18.9% of the crop and excludes all visible gable-step rows.

Called `regen.align_surface_photos.align` with that mask on both views. It
found 134/167 SIFT keypoints and only 7 mutual distinct matches; the helper
requires at least 24 before it fits a homography. It returned
`insufficient-independent-matches`; no train/held-out split, homography, or
warped target exists. A stricter lower-wall mask ending at row 230 produced 6
mutual distinct matches and the same abstention.

Because no transform was accepted, held-out rake-position improvement is
unmeasured and the smooth/flat false-positive controls were not remapped. The
prior v2 source-frame comparison remains the only measured position result:
median horizontal event offsets of 0.172–0.200 m, with registration
uncertainty still unresolved. Do not promote the extractor or use an inferred
per-view correction to force agreement.
