# Stepped-edge discrimination, v2

**Decision:** keep the extractor as an experiment. It separates J01 from the
synthetic smooth diagonals by step size, but the two registered J01 crops do
not agree closely enough on step positions in their shared metric frame to
accept a per-step source constraint.

## Smooth-edge checks

The actual smooth control is registered in `sides-1.json` as two `skyline`
views of `sf_201006_0012476__s5p0`. The independent roofline record marks both
gable slopes visible and supplies straight fitted segments in both views:
`[[30,132],[190,27],[336,132]]` and `[[26,133],[194,27],[375,133]]`. The
extractor abstains on both crops because its upper dark-mass requirement is not
met. This produces no false steps, but it is an abstention rather than a
successful smooth-edge classification.

I also rendered a synthetic smooth, straight gable edge at the target's
416x539 pixel scale, with a contrasting straight border, at four subpixel
phases and three border thicknesses (3, 6, and 9 px; 12 cases total). Each was
rendered at 8x resolution and area-downsampled. The extractor reported zero
step transitions on both sides in all 12 cases. Thus ordinary pixel staircasing
of these diagonal edges did not trigger the measured-step rule under the tested
phases or thicknesses.

## J01 two-view metric agreement

The source records are both `skyline`, both at 45 px/m, with `rectPad=26` and
no nearby trees. For local coordinates, the run used the registered crop origin
and profile height (`xM=(xPx-26)/45`; `zM=10.84-(yPx-26)/45`). Each event's
position is its horizontal tread center and vertical midpoint. Matched events
are paired by nearest vertical midpoint within 5 px; this deliberately does
not force lateral agreement.

| Rake | Events v0/v1 | Matched | Median lateral offset | Median vertical offset | Median run difference | Median rise difference |
|---|---:|---:|---:|---:|---:|---:|
| Left | 7 / 8 | 7 | 0.200 m | 0.033 m | 0.000 m | 0.067 m |
| Right | 8 / 7 | 6 | 0.172 m | 0.050 m | 0.022 m | 0.000 m |

One left event and three right events remain unmatched. Median run lengths are
about 0.27–0.29 m and rises about 0.27–0.33 m, so the repeated dimensions are
similar. The lateral offset is about 8–9 px (0.18–0.20 m), close to one
measured tread run, and is much larger than the one-pixel raster resolution.
The registered index records no per-view horizontal correction; its source
registration qualification says annotation and camera errors are unquantified.
Per-step positions therefore remain unresolved in rectified metric coordinates.

The flat-roof control remains low at 0/0 transitions in one view and 1/0 in the
other. The single isolated count shows why a repeated-step threshold and
cross-view agreement must both be required.

## Reusable extractor

`work/regen/extract_stair_edge_evidence.py` now emits individual pixel events
and optional metric coordinates when supplied `--pixels-per-metre`,
`--rect-pad-px`, and `--reference-height-m`. It contains no location IDs or
semantic trim keywords. The smooth synthetic and real-crop work was read-only;
no R31 inputs or generated worlds were changed.
