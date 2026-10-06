# Correcting a facade reading against its photograph

A draft reading of one building side (JSON, format in read.md) was drawn over the rectified photograph it was measured
from. In the overlay:
- cyan boxes: every individual opening of a bank (thin blue box: the bank's outer rectangle, labelled B0, B1, ...);
- yellow boxes: single openings (O0, O1, ...);
- red bars: bands (L0, L1, ...); green bars: piers; orange boxes: projecting bays;
- magenta outlines: material zones (M0 is the main wall); white line: the wall top.

Compare every drawn element with the photograph and fix the reading so that, drawn again, each cyan or yellow box sits
exactly on a real opening's glass and frame, and every real opening is covered. Typical faults: wrong column or row
counts, openings drawn the size of the whole cell instead of the real window, banks that span two different window
sizes (split them), a row of ground-floor shopfronts mixed with upper windows, missing entries or garage doors, bands at
the wrong height, the wrong material family or colour, missing arches, bays or cornices, parts of neighbouring
buildings or trees read as this building.

Houses and small apartment buildings need the same care as larger ones. Check in particular: bay windows that
project from the wall (angled, curved or square; record each as a bay over the full height it projects, its windows in
banks), wood siding or shingles (family timber, not render), the colour of the paint as the sunlit wall shows it (not
the shaded side, not the window glass), a cornice or parapet line at the top, garage doors and the entry door with its
stair at street level, and the storey heights as the photograph shows them.

Read positions off the metre ticks (red: along, from 0 at the left; blue: height, 0 at the ground). Openings in a bank
are evenly spaced cells: if real windows are not evenly spaced or differ in size, use more banks or single openings.

Write the corrected reading, same JSON format and same id, to the output path. Do not transcribe any text or numbers
from signs. Set "confidence" honestly. Keep "notes" short.
