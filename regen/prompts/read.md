# Reading a building side from its photographs

You measure one side of a building from street photographs and write what you see as JSON. You do not guess what you
cannot see. Preserve clearly legible signage text verbatim with its source image and measured location. Do not guess
blurred or occluded characters. Wall-mounted signs use `signs` as described below; unsupported mounts remain in
`unmodelled` instead of silently disappearing or acquiring invented lettering.

## Wall signs
For each observed wall sign, record `x0,x1,z0,z1` in the same metric wall frame, `mount:"wall"`, and the actual
`sourceImages`. A board requires `board:{family,hex,thicknessM}`; omit it for painted or separately attached letters.
Readable text requires verbatim `text`, `textLegible:true`, and `textHex`. Unread text stays unresolved even when the
board is visible. `offsetM` and `letterDepthM` are optional measured depths, not cues to invent a projecting box.
Unknown lettering uses a generic font and remains explicitly approximate. Projecting, roof-mounted and freestanding
billboards need separately measured supports and stay `unmodelled` until those supports are represented.

## Window guards and balconies
Distinguish a walkable projecting balcony from a shallow guard fixed across a window. Record each separate guard
as its own `balconies` item with `hasDeck:false`, its measured width, projection and railing height. Leave the gaps
between separate panels; a row of panels does not become one continuous balcony. Use `hasDeck:true` only when a
platform is visible. Preserve small measured dimensions instead of replacing them with standard balcony dimensions.
Describe the visible infill pattern separately in `unmodelled` when the available fields cannot express it. A generic
bar pattern is not evidence that the real panel has vertical bars. Hidden repeated panels must be marked as inferred,
not as measured geometry. Use the perspective image to distinguish projections from shadows and painted trim.

## What you get, per side
- `rect` images: the wall rectified square-on. Red ticks along the top and bottom edges mark metres along the wall
  (labelled every 5 m, starting at 0 m at the left edge). Blue ticks on the left and right mark metres above the ground
  (labelled every 5 m, 0 at the bottom). The white margin is not part of the wall. All your numbers are in these metres.
- `persp` images: the same camera's ordinary view, for context (depth, what projects, what is behind a tree).
- The side's size: `lengthM` along, `heightM` up (the image frame; the building may stop lower, sky above it).
- Up to two views. They can disagree because a camera position is off by metres; a view that is blurred, taken from
  right under the wall, shows a different building, or is mostly hidden by trees or trucks is not usable. Say so.

## Coordinates
x: metres from the left edge (0) to the right edge (`lengthM`). z: metres from the ground (0) up. A rectangle is
x0 < x1, z0 < z1. Read positions off the tick marks; be as exact as the ticks allow (to about 0.2 m).

When a street-to-roof context view is included, its bottom and sides extend past the original frame.
Use the marked zero line as the original wall datum. On a slope an entrance or garage threshold can
be below zero. Check the wide perspective for stairs, garage mouths, landings, side returns and attic
windows missed by the main crop. A bright cornice or a terrace rail is not an eave: inspect the
surface immediately above and below. Record a break in roof slope separately from a wall-to-roof
junction. Where a tree or parked car hides the boundary, do not continue it through the obstruction
as measured. A recess visible in projection alone does not determine its metric depth; report that
uncertainty and distinguish the visible portal from a door leaf behind it.

## Write this JSON (one object per side) to the output path you are given
{
 "id": "<side id>",
 "views": [{"view": 0, "usable": true, "issue": ""}, {"view": 1, "usable": false, "issue": "camera under the canopy"}],
 "use": 0,                       // the view you measured from
 "thisBuilding": true,           // false if the usable view shows some other building or nothing (then stop here)
 "wallTopM": 17.2,               // top of this building's wall or parapet at this side (null if not visible)
 "storeys": [0, 4.6, 8.1, 11.6, 15.1],   // floor levels (m) you can see, ground first
 "materials": [                  // wall surfaces, the main one first; a zone is a rectangle
   {"x0": 0, "x1": 24.6, "z0": 0, "z1": 17.2, "family": "render", "hex": "#c9c5b8"}
 ],
 "banks": [                      // a regular grid of like openings: give the bank's outer rectangle and its counts
   {"x0": 1.0, "x1": 11.2, "z0": 10.3, "z1": 16.6, "columns": 4, "rows": 6, "kind": "window",
    "head": "flat", "openingWidthM": 2.2, "openingHeightM": 0.9, "frame": "#9aa0a6", "mullions": 1, "transoms": 0,
    "recessM": 0.15, "ribbon": false, "omit": [[0, 5]]}
 ],
 "openings": [                   // one-off openings not in a bank: doors, a big arch, a garage door
   {"x0": 11.4, "x1": 13.6, "z0": 0, "z1": 7.9, "kind": "entry", "head": "arch", "frame": "#2b2f33"}
 ],
 "bands": [{"zM": 9.9, "thicknessM": 0.4, "projectionM": 0.1, "hex": "#555a5e", "kind": "string"}],
 "piers": [{"xM": 11.1, "widthM": 0.5, "projectionM": 0.1, "hex": "#d8d6cf"}],
 "bays": [{"x0": 3.0, "x1": 6.0, "z0": 3.5, "z1": 11.0, "projectionM": 0.6,
            "rows": [[4.5, 6.5], [7.5, 9.5]],
            "front": {"count": 1, "widthM": 1.4, "head": "pointed", "springRatio": 0.72},
            "side": {"widthM": 0.55, "head": "flat"}}],
 "canopies": [{"x0": 11.0, "x1": 14.0, "zM": 3.3, "depthM": 1.5}],
 "parapetTop": null,             // or the wall's top outline as [[x, z], ...] when it is not level (stepped, gabled)
 "confidence": "high",           // high: measured cleanly; medium: some parts estimated; low: mostly guessed
 "notes": "short: what you could not see"
}

Meanings:
- family: brick, render (stucco, painted plaster), stone (cut stone or stone cladding), concrete, metal-panel,
  glass (a curtain wall or glazed spandrel: record it as a bank of kind "curtain" as well), timber (wood siding or
  boards), tile, terracotta. hex: the colour as it looks in daylight, not in shadow.
- kind: window, shopfront (ground-floor glazing), entry (a door), garage (a vehicle door), rollup (a roller shutter),
  louvre (a vent grille), curtain (floor-to-floor glazing with thin mullions).
- Doors: record leaf kind and casing separately. For a garage leaf, `door.kind` is panel, glazed, grille, or open;
  `door.construction` is sectional or roller only when visible. `door.cells` may give measured leaf-relative rectangles
  `{rect:[left,bottom,right,top],kind:"panel"|"glass"}` in 0..1. Use only observed cells: an ordinary panel is opaque,
  and no pane grid is implied by an unresolved count. An entry `door.kind:"grille"` is an open-bar gate. Use
  `door.kind:"open"` for a truly unobstructed opening, distinct from a recessed passage with a door at its rear.
- head describes the actual window/door aperture edge, separate from its casing or decoration: flat, arch (round top), pointed (two curves meeting at a Gothic peak), or segmental (shallow curved top). For arch or pointed, include springRatio (height of the jamb-to-curve springline divided by the full aperture height); for segmental include riseRatio when it is clear. A hood, pediment, or decorative lintel belongs in its own surround/hood field and must not change the aperture head.
- A bank's openingWidthM/openingHeightM is one opening's glass-plus-frame size; columns x rows openings are evenly
  spaced inside the bank rectangle. Use several banks when spacing or size changes (ground floor, top floor).
  omit lists [column, row] cells (0,0 = bottom left) that are wall, not opening.
- ribbon: true when the openings in a row are one continuous band of glass split only by mullions.
- bands: horizontal projecting lines: cornice (the top), string (between floors), sill, fascia (sign fascia above shops),
  coping (the cap on a parapet).
- bays: the parts of the wall that project out as bay windows. Record each bay's window rows and the window shape as seen on its front/side (head and springRatio/riseRatio); when front/side styles differ, put the head fields on that face. Do not use a casing, hood, or pediment to infer the aperture shape.
- Trees, cars and people are not part of the wall: read through them from the other view or say it is not visible.

Be complete: every window row and column you can see. Count carefully. Keep it short: no prose beyond `notes`.
