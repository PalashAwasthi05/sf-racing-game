# Checking a facade reading against its photograph

You get, per building side: the rectified photograph (`rect`), the same photograph with the reading drawn over it
(`overlay`) and the camera's ordinary view (`persp`). Overlay key: cyan boxes = openings of a bank (thin blue = the bank,
labelled B0, B1...), yellow = single openings (O0...), red bars = bands (L0...), green = piers, orange = projecting bays,
magenta = material zones (M0 = main wall), white line = wall top.

Judge only what the photograph shows. For each problem, name the element label and say what is wrong in a few words.
Problems that matter:
- an opening box not on a real opening (off by more than about a third of its width), or a real opening with no box;
- a wrong number of columns or rows in a bank;
- a missing entry, shopfront, garage door, bay window, cornice or string course that the photo clearly shows;
- a wrong main material family (brick, render, stone, concrete, metal-panel, glass, timber, tile, terracotta) or a
  colour clearly different in daylight;
- parts of another building, a tree or a vehicle read as this building.
Ignore small offsets (under a third of an opening), sign text, and anything hidden in both views.

Write to the side's `out` path, valid JSON:
{"id": "<id>", "verdict": "pass" | "minor" | "major",
 "problems": [{"element": "B1", "problem": "5 columns in the photo, 4 drawn"}], "missing": ["arched entry near 12 m"]}
pass: nothing that matters is wrong. minor: 1-2 small problems. major: counts, materials or several openings wrong.
