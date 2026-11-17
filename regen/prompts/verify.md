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
- parts of another building, a tree or a vehicle read as this building;
- wrong window head shapes, mullion divisions, doors mistaken for windows, or a shadow mistaken for a recess;
- missing changes in material, setbacks, projecting caps, cornices, or ground-floor structure;
- legible sign text that is missing, invented, misspelled, or placed on a different sign.

Assess placement at the source image's resolution. A systematic shift of many openings is a registration problem,
even when each shift is less than a third of an opening's width. Report it once as a whole-reading problem, rather
than moving individual windows to disguise the alignment error. Treat every material zone separately; a correct
main wall color does not validate the rest of the facade. Do not infer the contents of areas hidden in every view.
List those areas as unverified. An unusable photograph or ambiguous building identity cannot receive a pass.

This review checks the reading against its source photograph. It does not establish that generated geometry
matches a separately held-out camera or the real building in world coordinates; those checks remain separate.

Write to the side's `out` path, valid JSON:
{"id": "<id>", "verdict": "pass" | "minor" | "major",
 "problems": [{"element": "B1", "problem": "5 columns in the photo, 4 drawn"}], "missing": ["arched entry near 12 m"],
 "unverified": ["ground-floor doors behind parked truck"]}
pass: nothing that matters is wrong. minor: 1-2 small problems. major: counts, materials or several openings wrong.
