# Dormers and chimneys on a ring building's roof (roof features v1)

You add what stands on one building's roof that its roof form does not hold: dormers (a small gabled, hipped or
shed-roofed projection with a window, standing out of a sloping roof) and chimneys (a stack of brick or stucco with a
cap). The roof form itself (hip, gable, mansard, turret cones) is already set and built; do not change it.

Work in `/Users/palash/Documents/Codex/2026-09-08/i-want-to-create-a-heuristic/work` (paths below are relative to it).
Run the tools with `geoenv/bin/python`. Put every image you make in your scratch directory, not in the repository.

## What you have
- The photographs and the model from the same registered cameras: `areas/ring/framing-r14reg/<building>.jpg`
  (photograph | old camera | registered camera, one side) and the pairs `areas/ring/pairs-r7/<side>__v<view>.jpg`
  (photograph | an older model render; good for seeing the roof in the photograph). `areas/ring/views.py persp <side>
  <camera> <out.jpg> --up 6` aims a camera higher to see the roof (see `areas/ring/read-v2.md` for views.py).
- `areas/ring/massing_view.py` (metres east and north in the world's local frame; heights above the building's foot):
  - `bumps <building>`: clusters of LiDAR returns standing more than 0.5 m above the roof as now built: centre, size
    (long and short side, metres), bearing of the long side (degrees from east), top above the foot, rise above the
    roof, number of returns, distance inside the outline. Chimneys are small (under about 2.5 x 2.5 m) and rise 1.5 m or
    more; dormers are 1.5-4 m wide and rise 1-3 m out of a slope; a tree hanging over the roof is irregular and usually
    near the edge; a penthouse or plant room is large and on a flat roof (not yours).
  - `plan <building> <out.png>` (the LiDAR heights hill-shaded, the parts, the sides' ids, a 5 m grid) and
    `section <building> <x0> <y0> <x1> <y1> <out.png>` (returns along a line against the model's roof, in red): cut a
    section through each bump, across the slope, to read a dormer's eave (its lowest edge where it leaves the slope) and
    ridge, or a chimney's top.

## How to read it
1. Look at the photographs: count the dormers on each visible slope and the chimneys; note each one's place along the
   facade (left, middle, right; which slope), its roof (gable, hip, shed), its width against the windows below it.
2. Run `bumps` and `plan`. Match each photographed dormer and chimney to a bump: same slope, same place along the
   building. A bump nobody can see in the photographs (a back slope) may still be a dormer or chimney when its size and
   rise say so clearly; a bump the photographs show as a tree is not.
3. Every feature you give must stand on a bump: the LiDAR is the evidence, the photographs say what it is. No bump, no
   feature (say so in `notes` when the photographs show one the LiDAR misses).
4. Measure from the bump and a section: `xy` the bump's centre (move it to the photographed dormer's centre when the
   bump merges two dormers: split them, each on its own half), `widthM` along the slope's eave, `depthM` back into the
   roof, `ridgeM` the bump's top (a dormer's ridge, or a chimney's `topM`), `eaveM` a dormer's eave (from the section:
   usually 0.8-1.4 m under its ridge for a gable). Chimney `material`: "brick" or "wall" (stucco or painted, the
   building's colour), from the photographs.

## Your file
Write `areas/ring/massing/features/<building>.json` and reply with the same JSON:
```
{"building": "<id>",
 "roofFeatures": [
   {"kind": "dormer", "xy": [x, y], "widthM": 2.2, "depthM": 2.6, "eaveM": 12.0, "ridgeM": 13.2, "roof": "gable",
    "window": true, "bump": 3, "seen": "camera 2274605496076843: second of two gabled dormers on the front slope"},
   {"kind": "chimney", "xy": [x, y], "widthM": 0.9, "depthM": 0.7, "topM": 14.7, "material": "brick", "bump": 0,
    "seen": "left chimney, white stucco with a cap"}],
 "notes": "short: what you could not see or settle"}
```
`bump` is the bump's index in the `bumps` output (0 = first). Leave `roofFeatures` empty when the building has none.
