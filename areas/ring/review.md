# Reviewing a rebuilt building against its photographs (ring review v2)

You review one building (or a few sides of one) of the Lafayette Park ring: its model, rendered from the same cameras
as its street photographs, against those photographs. You find every difference that matters to someone looking at
the building from the street, and you fix the ones that come from its readings. Be strict: the model should be
recognisably the same building, with the same parts in the same places, the same proportions and the same colours.
Earlier reviews signed buildings off from small thumbnails and missed a clay-tile roof, balconies, a second bay behind
a tree, a storey of bays above the crop, a driveway read as a building and a Victorian read as a plain box. Look closely.

Work in `/Users/palash/Documents/Codex/2026-09-08/i-want-to-create-a-heuristic/work` (paths below are relative to it).
Read `areas/ring/read-v2.md` first: it is the reading format and the reading tools (check.py, views.py), and
everything it says about seeing the whole wall and taking inventory applies to you.

## What you have
- Pairs (paths in your prompt: `areas/ring/pairs-<build>/<side>__v<view>.jpg`): left, the street photograph; right, the current model from the same
  camera position, eye height and direction, with trees, parked cars, poles and wires hidden so they cannot hide a
  building error (trees in the photograph are real; their absence in the model is not a difference). A camera's
  recorded position and heading can be a few metres or degrees off, so the model may look a little closer, further
  or turned: match parts by what they are (that corner, that bay, that door), and judge parts and proportions, not
  pixels. The model's lighting and exposure are not the photograph's: judge colour from the reading's hex against the
  photograph in daylight, not from the render's pixels. `areas/ring/pairs-<build>/full/` has the same pairs with the
  model's trees and cars shown, if you need to see what stands in front of the building.
- Close views: `geoenv/bin/python areas/ring/zoom.py <side> <view> <render prefix> <fx0> <fy0> <fx1> <fy1> <out.jpg>
  [--model mx0 my0 mx1 my1]` cuts a region (fractions of the frame, 0,0 top left) from the photograph and from the
  model (its own region with --model, where the same part sits elsewhere in the model) and enlarges both side by side.
  Every pair gets close views: at least the roofline, the middle storeys and the base, and every part you are unsure
  of. A thumbnail is never enough to pass an item.
- The readings: `areas/ring/readings/<side>.json` (one per side), and the tools in read-v2.md.

## The checklist (every item, every side)
1. Massing: overall height, width, the number of storeys, setbacks, corners (rounded, chamfered), curved walls,
   towers, and whether the building is there at all (a driveway, a garden or a lot read as a building).
2. Roofline: cornice, parapet (stepped, shaped, ornamented), false front, gable, eaves (tiled or plain), dormers,
   mansard, roof colour.
3. Windows: count per storey, size, position, shape (arched, segmental), sash and panes, frames and trim colour;
   separate windows or one band of glass a storey (`ribbon`).
4. Bays: count, the storeys each spans (bays often run up two or more storeys), shape (angled, square, curved), its
   windows on each face.
5. Balconies, railings (including railings across French windows), fire escapes.
6. Entry and base: door, stairs and stoop, porch, canopy, garage doors and how deep they sit, rustication, arcades.
7. Materials and colours: wall body, trim, accent, base; judged in daylight, not in shade.
8. Ornament: brackets, dentils, modillions, quoins, panels, hoods, keystones, pilasters, relief.
9. The street edge: planting, hedges, trellises, fences, gates, garden walls, driveways.

## How to work, per side
1. Read every pair of the side, then close views of each part. Note every difference against the checklist.
2. Whatever the given photographs do not show (a tree, a truck, the frame's edge, the top storeys or roof cut off),
   look for it from other cameras: `views.py list`, then `views.py rect ... --top 4` (more for tall buildings) and
   `views.py persp ... --up 3` from cameras standing on the other side of the obstruction. A tree in front of a wall
   means at least two other cameras tried. Stop only when every part has been seen or every camera tried.
3. Write the side's inventory (read-v2.md), top to bottom, from all the views. Compare it with the reading, field by
   field: every inventory item must be in a field of the reading or in its `unmodelled` list.
4. Fix the reading (below), run `check.py` on it, fix its warnings, and look at the final overlay.
5. Decide the side's status on each checklist item.

## What to do with each difference
- A reading error (something read wrong, missed, or not given though the format has a field for it): fix the reading
  JSON for that side. Keep everything else in it as it is; keep its `id`. Sides of one building should agree (the same
  storeys, sashes, trims and colours where the same treatment wraps a corner).
- A builder limitation: the format has no field for it, or the reading gives it and the model still does not show it.
  Add it to the reading's `unmodelled` list with `"why": "builder"` and what would be needed. Say in your report
  which ones the reading already gives (those are builder bugs, the most useful thing you can find).
- A massing error (wrong height, outline, roof form, missing or extra building): write it to
  `areas/ring/massing/<building>.json` as {"building", "problem", "evidence": ["camera: what it shows"], "suggest"}
  (add to the file if it exists).
- A difference that comes from the camera standing off its true position, or from lighting: ignore it.

## Report
Write `areas/ring/reviews/<group>.json` and reply with the same JSON:
```
{"group": "<group>", "building": "<building>",
 "sides": [{"side": "<side>", "readingChanged": true,
            "items": {"massing": "pass", "roofline": "fixed: clay-tile eave added", "windows": "pass",
                      "bays": "fixed: second bay (hidden by a tree in v0) from camera 1234", "balconies": "not visible",
                      "entry": "builder: recessed porch with columns", "materials": "fixed: body #e8dcc4 to #d9c7a1",
                      "ornament": "builder: relief panels under bay windows", "streetEdge": "pass"},
            "camerasUsed": ["<camera ids beyond the given two>"], "stillUnseen": "what no camera showed"}],
 "builderGaps": [{"side": "<side>", "what": "...", "readingGives": false}],
 "massing": ["one line per problem written to areas/ring/massing/"],
 "verdict": "recognisable | partly | not recognisable (after your fixes, as built from the readings)",
 "biggestRemaining": ["the differences that will still show after your fixes, worst first"]}
```
Each item is one of: `pass`, `fixed: <what>`, `builder: <what>`, `massing: <what>`, `not visible`. Be honest: an item
that still differs after your fixes is not a pass.
