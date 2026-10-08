# Reading a building's sides from street photographs (reading v2, the Lafayette Park ring)

You are rebuilding one building as accurately as it can be built from street photographs. You measure each of its
photographed sides and write what you see as JSON, which a generator turns into 3D (walls, windows set into the wall
with real depth, bays, cornices, rustication, fire escapes...). Accuracy is the whole point: every window, every storey,
the right sizes, the right colours. Do not guess what you cannot see, and say so in `notes`. Do not transcribe any text,
names, logos or house numbers you see.

Work from `/Users/palash/Documents/Codex/2026-09-08/i-want-to-create-a-heuristic/work` (paths below are relative to it).

## What you have, per side
- `areas/pacific-heights/regen/sides/img/<side>__v0_rect.jpg` and `__v1_rect.jpg`: the side rectified square-on from
  two camera positions. Red ticks along the top and bottom edges mark metres along the wall (labelled every 5 m, 0 at
  the left edge); blue ticks on the left and right mark metres above the ground (labelled every 5 m, 0 at the bottom).
  The white margin is not part of the wall. All your numbers are in these metres. A view may be missing.
- `<side>__v0_persp.jpg`, `__v1_persp.jpg`: the same cameras' ordinary views, for context: depth, what stands out from
  the wall, what is behind a tree, how far a cornice or bay projects.
- The tools (run them with Bash, then Read the image they write):
  - `geoenv/bin/python areas/ring/check.py grid <side> <view 0|1> <x0> <z0> <x1> <z1> <out.jpg>`: a zoomed crop of one
    rectified view with a labelled half-metre grid (red = metres along, blue = metres up). Measure with this: positions
    to about 0.1 m. Zoom on a few storeys at a time.
  - `geoenv/bin/python areas/ring/check.py <side> <reading.json> <out.jpg>`: your reading drawn over both rectified
    views, and a summary. Compare every line to the photograph and fix what is off. Bay windows you give per storey are
    drawn in orange where they will be built (see Bays), which is not where the photograph shows them.
  Put scratch images in your own scratch directory, not in the repository.

## Coordinates
x: metres from the side's left edge (0) to its right edge (`lengthM`). z: metres from the ground (0) up. A rectangle
is x0 < x1, z0 < z1.

Things that stand out from the wall (bays, cornices, balconies, fire escapes) are drawn shifted in a rectified image:
it projects everything onto the wall's plane along the camera's sight lines, so a part standing out by p metres appears
displaced away from the camera's eye point (about 2.4 m up, wherever the camera stood) by roughly p x (its distance
from the eye point) / (camera distance). Above the eye that means higher than it is, by up to half a metre for a deep
bay high on a tall building. Read heights where they meet the wall, use the storeys of the wall windows beside them,
and check against the perspective view.

## The JSON (one file per side: `areas/ring/readings/<side id>.json`)
Everything from reading v1, and more. Example (a 7-storey apartment block, abridged):
```
{
 "id": "<side id>",
 "views": [{"view": 0, "usable": true, "issue": ""}, {"view": 1, "usable": true, "issue": ""}],
 "use": 0,                         // the view you measured from
 "thisBuilding": true,             // false if the usable views show another building or nothing (then stop)
 "wallTopM": 23.1,                 // top of this building's wall at this side: parapet or cornice top (null if unseen)
 "storeys": [0, 5.0, 8.3, 11.4, 14.45, 17.3, 20.3],   // floor levels, ground first (a floor line is about a storey's sills - 0.9 m)
 "materials": [                    // wall surfaces as rectangles, the main wall first
   {"x0": 0, "x1": 21.3, "z0": 0, "z1": 5.0, "family": "stone", "hex": "#cfc6b2"},
   {"x0": 0, "x1": 21.3, "z0": 5.0, "z1": 20.7, "family": "brick", "hex": "#c2b289"}],
 "rustication": [{"x0": 0, "x1": 21.3, "z0": 0.4, "z1": 4.65, "courseM": 0.5}],   // banded courses with channels between
 "quoins": [{"xM": 0, "z0": 0, "z1": 20.5, "widthM": 0.7, "courseM": 0.45, "family": "stone", "hex": "#d8d0bd"}],
 "banks": [                        // a regular grid of like windows
   {"x0": 5.0, "x1": 14.1, "z0": 5.6, "z1": 16.6, "columns": 4, "rows": 4, "kind": "window", "head": "flat",
    "openingWidthM": 1.0, "openingHeightM": 1.65, "xCentersM": [6.05, 8.42, 10.75, 13.45], "zCentersM": [6.57, 9.72, 12.83, 15.62],
    "frame": "#ece8de", "sash": "double-hung", "meetingRail": 0.66, "lites": [1, 1], "mullions": 0, "transoms": 0, "recessM": 0.18,
    "surround": {"widthM": 0.12, "projectionM": 0.03, "hex": "#e0d8c6"},
    "hood": {"kind": "cornice", "heightM": 0.3, "projectionM": 0.12, "hex": "#e0d8c6"}, "keystone": false}],
 "openings": [                     // one-off openings: doors, arched windows, garage doors, shopfronts
   {"x0": 10.0, "x1": 11.9, "z0": 0, "z1": 4.0, "kind": "entry", "head": "arch", "springRatio": 0.72, "frame": "#3a342c", "recessM": 0.35,
    "door": {"kind": "glazed", "hex": "#2a2622"}, "transomM": 0.0, "surround": {"widthM": 0.3, "projectionM": 0.06, "hex": "#d6cdba"}, "keystone": true,
    "steps": {"count": 0, "widthM": 2.2, "family": "stone", "hex": "#c9c2b2"}}],
 "bays": [
   {"x0": 1.2, "x1": 4.6, "z0": 5.1, "z1": 20.1, "projectionM": 0.75, "shape": "angled", "sideAngleDeg": 45,
    "rows": [[5.7, 7.45], [8.9, 10.55], [12.0, 13.65], [14.85, 16.45], [17.6, 19.1]],
    "front": {"count": 1, "widthM": 1.15}, "side": {"widthM": 0.6},
    "sash": "double-hung", "meetingRail": 0.66, "lites": [1, 1], "frame": "#ece8de", "trimHex": "#e8e1d1",
    "storeyTrim": true, "corbel": false, "cap": "cornice"}],
 "bands": [
   {"zM": 4.95, "thicknessM": 0.35, "projectionM": 0.25, "hex": "#d9d0bd", "kind": "string", "profile": "moulded"},
   {"zM": 22.45, "thicknessM": 1.25, "projectionM": 1.0, "hex": "#ddd4c2", "kind": "cornice", "profile": "modillion"}],
 "piers": [{"xM": 11.1, "widthM": 0.5, "projectionM": 0.1, "hex": "#d8d6cf"}],
 "fireEscapes": [{"x0": 12.7, "x1": 18.0, "levels": [5.15, 8.2, 11.3, 14.4, 17.45], "depthM": 1.2, "hex": "#d8d2c4"}],
 "balconies": [{"x0": 3.0, "x1": 6.5, "zM": 12.1, "depthM": 0.9, "railing": "metal", "railingHeightM": 1.0, "hex": "#26282a"}],
 "ornaments": [{"x0": 5.75, "x1": 6.6, "z0": 19.5, "z1": 19.95, "hex": "#d9d0bd", "projectionM": 0.03}],
 "canopies": [{"x0": 11.0, "x1": 14.0, "zM": 3.3, "depthM": 1.5}],
 "parapetTop": null,               // or the wall's top outline [[x, z], ...] when not level (stepped, gabled, false front)
 "confidence": "high",
 "notes": "short: what you could not see"
}
```

### Meanings
- `family`: brick, render (stucco, painted plaster), stone, concrete, metal-panel, glass, timber (wood siding or
  shingles), tile, terracotta. `hex`: the surface's own colour in daylight, judged by eye as a painter would mix it:
  NOT the photograph's shaded or backlit pixel colour. The check summary prints the photograph's median colour in each
  zone (`photoMedianByView`): use it for hue only; it is darker than the material wherever the wall is in shade or
  against the light. Painted wood and stucco in this neighbourhood are often white, cream, pale grey, blue-grey, sage,
  yellow, or a deeper accent; old brick is red, brown or a buff/cream.
- Windows (`banks` and `openings` of kind window):
  - `kind`: window, shopfront, entry (a door), garage (a vehicle door), rollup, louvre, curtain.
  - `head`: flat, arch (half-round top; `springRatio` = where the curve starts, as a fraction of the opening's height
    from the bottom, default 0.68), segmental (shallow curve).
  - `sash`: double-hung (two sashes, one above the other: `meetingRail` is where they meet, a fraction of the height
    from the bottom; often 0.5, higher when the upper sash is a short transom), casement (side-hung leaves), fixed,
    sliding. `lites`: panes per sash [across, down] ([1,1] one pane, [3,2] six panes). Use `mullions`/`transoms` only
    for other fixed divisions (a shopfront's mullions, a transom light over a door): both 0 for an ordinary sash window.
  - `openingWidthM`/`openingHeightM`: one opening's glass-plus-frame size. `recessM`: how deep the glass sits behind
    the wall face (0.1 thin wall, 0.2 brick, 0.3+ thick stone base). `frame`: frame/sash colour.
  - `surround`: a flat trim (architrave, board) round the opening; `hood`: a projecting cap over it (cornice, pediment,
    segmental); `keystone`: a block at the top centre; `railing`: a railing across a French window's lower part.
  - a bank's columns x rows openings are centred on `xCentersM` x `zCentersM` (give them: measured centres beat even
    spacing); `omit` lists [column, row] cells (0,0 = bottom left) that are wall. Use several banks when size, spacing,
    sash or trim change (ground floor, top floor, a wing).
- Entries: `door` {kind: panel, glazed, double, none; hex}, `transomM` (height of a glazed transom above the door),
  `steps` {count, widthM}: the steps up to a door whose sill is above the ground (count the risers you can see).
- `bays`: parts of the wall that project as bay windows. Give `x0`/`x1` where the bay meets the wall (those points are
  on the wall's plane, so the rectified image shows them where they are), `z0`/`z1` its bottom and top, `projectionM`
  (how far its front stands out: judge from the perspective view; 0.6-0.9 m is usual), `shape` (angled: front and two
  45-degree sides; square; curved: a round bay), and its windows PER STOREY, not as image positions: `rows` (sill and
  head heights of the bay's window on each storey: the same storeys as the wall windows beside it), `front` (how many
  windows across the front face and each one's width), `side` (each side face's window width as built, or null for a
  blank side). Do not also cover the bay with a bank.
- `bands`: horizontal projecting courses: cornice (the top), string/belt (between floors), sill, fascia, coping, frieze.
  `zM` is the band's centre, `thicknessM` its height, `projectionM` how far it stands out. `profile`: flat (a plain
  slab), moulded (stepped mouldings), modillion (a cornice with a row of small brackets under its top), dentil (a row
  of small teeth), bracketed (a cornice on large widely spaced brackets, as on Italianate houses; `bracketPitchM`).
  Many cornices are a frieze band plus a cornice band: give both.
- `rustication`: zones of banded masonry (horizontal channels every `courseM`). `quoins`: corner blocks alternating
  long/short up a corner (`xM` the corner's x).
- `piers`: pilasters / vertical strips standing out of the wall. `ornaments`: low relief panels, plaques, cartouches.
- `fireEscapes`: platform `levels` (z of each platform), its x extent, depth, colour (painted or black).
- `balconies`: a slab at `zM` with a railing (metal, balustrade, solid).
- `hedges`: a clipped hedge or tall planting standing in front of the wall and hiding it: {x0, x1, heightM, depthM,
  offsetM (how far in front of the wall it stands), hex}. Give it when it hides the lower storeys, and read only what
  shows above it.
- `parapetTop`: when the top of the wall is not level (a gable, a stepped or shaped parapet, a false front).

## How to work, per side
1. Read both rectified views and both perspective views. Decide which view is clearer (`use`), which parts each hides.
2. Measure with grid zooms: the storeys (sill and head of each window row), each column of windows (left and right
   edge), each door and special opening, the bays, the bands, the wall top. Count windows twice.
3. Write the JSON. Then run the check, Read the overlay, and fix everything that does not sit on the photograph: a
   window box that is off by more than 0.15 m, a missing row or column, a band at the wrong height. Repeat until it
   matches (usually 2-4 rounds). Check the colours last. After your LAST edit, run the check once more and look at
   it: a reading is not done until you have seen its final overlay sit on the photograph.
   Windows behind a fire escape, a railing or thin branches are still windows: look between the bars.
4. A side that is mostly hidden (trees, a truck, another building in front): read what you can see, give
   `confidence` medium or low and say what is missing. A side whose views show another building or nothing:
   `{"id": ..., "thisBuilding": false, "notes": "..."}`.

Sides of one building should agree with each other: the same storeys, sash types, trims and colours where the
photographs show the same treatment wrapping a corner.
