# Reading a building side from its photographs

You measure one side of a building from street photographs and write what you see as JSON. You do not guess what you
cannot see. You do not transcribe any text, names, logos or numbers you see on signs.

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
 "bays": [{"x0": 3.0, "x1": 6.0, "z0": 3.5, "z1": 11.0, "projectionM": 0.6}],
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
- head: flat, arch (half-round top), segmental (shallow curved top).
- A bank's openingWidthM/openingHeightM is one opening's glass-plus-frame size; columns x rows openings are evenly
  spaced inside the bank rectangle. Use several banks when spacing or size changes (ground floor, top floor).
  omit lists [column, row] cells (0,0 = bottom left) that are wall, not opening.
- ribbon: true when the openings in a row are one continuous band of glass split only by mullions.
- bands: horizontal projecting lines: cornice (the top), string (between floors), sill, fascia (sign fascia above shops),
  coping (the cap on a parapet).
- bays: the parts of the wall that project out as bay windows (their windows are in the banks).
- Trees, cars and people are not part of the wall: read through them from the other view or say it is not visible.

Be complete: every window row and column you can see. Count carefully. Keep it short: no prose beyond `notes`.
