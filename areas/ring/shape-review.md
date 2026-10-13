# Shape review of a ring building (ring16)

You compare a street photograph of a building with a render of its 3D model from the same camera, and list what is
wrong with the model's SHAPE: geometry that is missing, added, the wrong size or the wrong form. Not colour, texture,
lighting, trees, cars or the sky: shape only.

For each building you are given, open `areas/ring/review-r16/<building>.jpg` (left: the photograph; right: the model
from the same camera; paths relative to `/Users/palash/Documents/Codex/2026-09-08/i-want-to-create-a-heuristic/work`).
Look closely; zoom by cropping with Python (PIL) into your scratch directory when something is small. A model tree that
hides part of the building is not a shape fault of the building: say what it hides in `hidden`.

List every shape fault, worst first, each as:
- `category`, one of: `balcony` (balcony slabs, galleries, their railings), `cornice` (projecting cornice, brackets,
  dentils, string courses), `parapet` (parapet walls, balustrades, roof railings), `crown` (ornamental roofline: crests,
  finials, stepped or shaped tops), `entry` (porch, portico, columns, door surround, recessed entry), `stairs` (front
  steps, stoops, ramps), `bay` (bay windows: their shape, depth, number), `window-trim` (hoods, pediments, surrounds,
  sills), `awning` (awnings, canopies, marquees), `roof` (roof form, dormers, chimneys, turrets, towers), `walls` (wall
  height, set-backs, wings, recesses, a wall in the wrong place), `curve` (a curved or rounded wall or corner drawn
  straight), `openings` (windows/doors missing, extra, wrong size or arrangement), `other`.
- `severity`: 3 = it changes the building's outline or a large part of a facade (anyone would notice); 2 = clearly
  visible at street level; 1 = a detail.
- `what`: one sentence: what the photograph shows and what the model does instead.

Write `areas/ring/review-r16/<building>.json` for each building:
```
{"building": "<id>", "verdict": "close" | "recognisable" | "partly" | "not",
 "faults": [{"category": "...", "severity": 3, "what": "..."}], "hidden": "...", "notes": "..."}
```
You may write only those files. Reply with a one-line summary per building.
