# Edit, check, repeat: fixing a facade reading until the model matches the photographs

You fix the reading (JSON, format in read.md) of one building side so that the model drawn from it matches the
street photographs. You can check any version yourself with the preview command:

    cd /Users/palash/Documents/Codex/2026-09-08/i-want-to-create-a-heuristic/work && geoenv/bin/python regen/preview_side.py <id> <reading.json> <image.jpg>

It writes an image (the model as the generator will draw it, then each photograph of the same wall in the same frame)
and prints scores per photograph (higher is better; falseOpenings = model windows on solid wall, missingDetail =
photographed openings the model lacks) and their mean. A photograph scores null when too little of the wall is
visible to judge; it does not count. Windows, doors and dark glass are detected in each photograph, so removing
openings the photographs show lowers the score.

For each side:
1. Preview the current reading (`reading`), then Read the image. Also Read `rect` (the photograph the reading is measured
   on, with metre ticks; the reading's coordinates are in this photograph's metres) and `overlay` (the reading drawn on it).
2. Find what makes the model differ from the photographs: windows missing across part of the wall, wrong column or row
   counts, wrong sizes or heights, missing shopfronts, entries, garage doors, bays, bands, curtain-wall glazing drawn as
   punched windows, wrong materials.
3. Write a corrected reading to `out` (the whole JSON), preview `out`, Read its image and compare the mean score.
4. Repeat steps 2-3 at most twice more. Stop early when the preview matches the photographs well.
5. The file left at `out` must be your best-scoring version: if a later edit scored worse, write the best one back.
Draw only this wall: the rect photograph may show neighbouring buildings past the wall's ends. Only run the preview command; no other commands. Do not transcribe sign text.
