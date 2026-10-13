# Setting a ring building's massing (ring massing v1)

You set the shape of one building of the Lafayette Park ring: how tall its walls stand, what roof sits on them, and
which parts of it stand at different heights. The facades (windows, colours, ornament, parapets) are read elsewhere;
you set the mass they are drawn on. Reviewers comparing the model with the street photographs found its shape wrong:
pitched, hipped and mansard roofs, gables and turrets built as flat boxes, walls metres too tall or too short, lower
wings and set-back storeys missing, a few neighbouring buildings missing. Your building's notes say what they saw.

Work in `/Users/palash/Documents/Codex/2026-09-08/i-want-to-create-a-heuristic/work` (paths below are relative to it).
Run the tools with `geoenv/bin/python`. Put every image you make in your scratch directory, not in the repository.

## What you have
- Your building's massing notes: the entry for it in `areas/ring/massing-notes.json` (what the reviewers saw).
- The street photographs and the model as it is now: `areas/ring/pairs-r7/<side>__v<view>.jpg` (left the photograph,
  right the model from the same camera). Any camera that sees a side: `areas/ring/views.py list <side>`, and
  `views.py persp <side> <camera> <out.jpg> --up 6` aims higher to see the roof (see `areas/ring/read-v2.md` for
  views.py). The far cameras see roofs best.
- `areas/ring/massing_view.py` (everything in the world's local frame: metres east and north; heights in metres above
  the part's foot, z0, where the model stands it):
  - `plan <building> <out.png>`: the LiDAR's height above ground, hill-shaded (slopes, ridges, hips and steps show as
    light and shade), the model's current parts with their wall tops and roof forms, the photographed sides (red, with
    their ids), neighbours, a 5 m grid with coordinates.
  - `stats <building>` and `fit <building>`: per current part, the LiDAR heights, and the roof forms (flat, gable
    either way, hip, mansard, shed four ways) with the wall top and ridge that fit the LiDAR best, scored.
  - `section <building> <ax> <ay> <bx> <by> <out.png>`: the returns along a line (height above the foot against
    distance), the model now (red) and your given massing (green, once it is in `areas/ring/massing-overrides.json`;
    before that, read it against the returns). Cut sections across and along each roof: they show eaves, ridges,
    steps and towers far better than numbers do.
  - `split <building> <ax> <ay> <bx> <by> ...`: the building's whole outline cut by lines (each through two points):
    the pieces' outlines, for parts. `rect <cx> <cy> <width> <depth> <angleDeg>`: a rectangle's outline.
  - `check <building> <spec.json> [--out check.png]`: your massing against the LiDAR, part by part (see below).

## How to read it
1. Read your notes and the pairs. Look at the photographs for the roof: which form (hip, gable and which way its
   ridge runs, mansard, flat behind a parapet), towers and turrets, wings, set-back storeys, dormers.
2. `plan`, `fit`, and sections across and along the building. The LiDAR sets the heights; the photographs set the
   form and confirm it. Both must agree. Trees over a roof show in the LiDAR as rough, irregular heights (no roof is
   bumpy): read around them and say so.
3. Decide the massing:
   - `wallTopM`: where the roof meets the walls: the eaves of a pitched roof (the lowest line of the roof in the
     sections, at the walls), the main roof level of a flat roof. A parapet or cornice standing above a flat roof is
     drawn from the facade readings: do not raise the walls for it.
   - `roof`: {"kind": flat | gable | hip | mansard | shed | pyramid, "ridgeM": the ridge, apex or flat top above the
     foot, "along": "long" | "short" (a gable's ridge runs along the part's longer or shorter side), "lowSide": [dx, dy]
     (the way a shed falls), "pitchDeg" (a mansard's faces, about 70), "overhangM" (0.2-0.6 where the photographs show
     eaves overhanging; 0 for none)}. A turret or a tower's cone is a `pyramid` on its own part.
   - `parts`: when the LiDAR shows areas whose roofs stand 1.5 m or more apart (a lower wing, a tower, a portico, a
     set-back top storey or penthouse, which is a part standing inside the outline), cut the whole outline into
     parts with `split` (they must cover the outline with no gaps or overlaps; each 8 m2 or more) and give each its own
     `wallTopM` and `roof`. Ids: the main part keeps the building's id; a part matching one of the current parts the
     plan lists (an id ending `_p1`, `_p2`) keeps that id; a new one is `<building>_p21`, `_p22`...
   - `outline`: only when the reviewers found the outline itself wrong (a wall running metres past the building, a
     garden read as building) and the LiDAR plan shows the building's edge elsewhere: the outline the LiDAR shows.
   - `adds`: a building the model is missing: only when the photographs show it and the LiDAR plan shows its roof
     where the model has no outline. Its outline (`rect` helps), wall top and roof. Id `add_<building number>_<word>`.
4. Write your spec, run `check`, and fix it until every part passes: a part fails when more than 15 % of it is built
   more than 1 m above what the LiDAR sees (`below1` > 0.15): a wall or roof the LiDAR has no evidence for. Its
   `within0.6` (the share of its roof within 0.6 m of the LiDAR) should beat the model's now (check a spec holding
   only the current wall top to see that). When the photographs and the LiDAR disagree and you cannot settle it,
   keep the model as it is (`"decision": "keep"`) and say why.
5. Things the massing cannot hold yet (dormers, chimneys, crowns, balustrades, a curved wall) go in `unmodelled`.

## What the first buildings taught
- A corner tower's cone or a turret is not left in `unmodelled`: cut its footprint out (the round or angled bay at a
  corner, usually 2.5-4 m across) and give it a `pyramid` part, its wall top at the main eaves, its apex from the
  photographs and the LiDAR. A tower that rises well above the roof is its own part at its own wall top.
- A gable facing the street on a hipped or long-gabled roof (a front gable, a cross gable) is a part of its own: cut
  the gabled wing out and give it a `gable` whose ridge runs out toward the street (`along` the side that does).
- The reviewers' notes are usually right about forms and often wrong about heights (they measure from photographs):
  take heights from the LiDAR, forms from the photographs. Look at every pair yourself.
- An outline that runs onto ground the LiDAR shows bare (a strip 2 m or more wide along an edge, a courtyard) built a
  wall where there is none: trim it (`outline`) to where the LiDAR shows the roof, and say so.

## Your spec
Write `areas/ring/massing/specs/<building>.json` and reply with the same JSON:
```
{"building": "<id>", "decision": "change" | "keep",
 "spec": {"wallTopM": 10.6, "roof": {"kind": "hip", "ridgeM": 14.0, "overhangM": 0.4}},
   or   {"parts": [{"id": "<id>", "outline": [[x, y], ...], "wallTopM": 9.0, "roof": {...}}, ...]},
   (add "outline" beside either when the outline itself changes)
 "adds": [{"id": "add_...", "outline": [[x, y], ...], "wallTopM": 6.5, "roof": {...}, "what": "..."}],
 "evidence": ["section a->b: eaves at 10.6, ridge 14.0 mid-depth", "camera 1234: hipped slate roof, ridge along the street"],
 "check": <the final check output>, "unmodelled": ["two dormers on the front slope", "chimney"],
 "notes": "short: what you could not see or settle"}
```
