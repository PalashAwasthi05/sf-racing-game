# Fixing a facade reading that the built model gets wrong

You get, per side:
- `compare`: the model as it is now, drawn square-on, next to the photographs of the same wall (aligned to the same
  frame). Where they differ, the model is wrong.
- `overlay`: the current reading drawn over the photograph it was measured on (same key as correct.md), and `rect`, that
  photograph clean. The reading's coordinates are in this photograph's metres.
- `check`: a measure of the mismatch: windowsOnSolidWall is the share of model openings where the photographs show
  solid wall; photoDetailMissing the share of photographed openings and detail the model leaves blank.

Find what makes the model differ from the photographs (missing or extra openings, wrong rows or columns, openings too
small or too big, wrong heights, missing shopfronts, entries, garage doors, bays, bands, wrong materials) and fix the
reading so that drawn again on `rect` every box sits on a real opening and every real opening has a box. Positions are
read off the metre ticks on `rect` and `overlay`. Do not transcribe sign text. Write the whole corrected reading (same JSON
format as read.md, same id) to the `out` path.
