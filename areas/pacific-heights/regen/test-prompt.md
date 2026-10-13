You are reading the photographed sides of buildings of a 3D model of San Francisco (Pacific Heights). Your buildings and their sides are given in the task line that sent you here (each side is [side id, length m, height m]).

Work in /Users/palash/Documents/Codex/2026-09-08/i-want-to-create-a-heuristic/work. First read areas/ring/read-v2.md (your protocol) and follow it for every side, with these differences:
- Your images are areas/pacific-heights/regen/sides-reg/img/<side>__v0_rect.jpg, __v1_rect.jpg and their _persp.jpg views (not regen/sides/img).
- Run every tool with SIDES_DIR=regen/sides-reg set, e.g. `SIDES_DIR=regen/sides-reg geoenv/bin/python areas/ring/check.py <side> <reading.json> <out.jpg>`.
- Write each reading to the OUTPUT folder the task line names, as <side>.json, with "registeredCrops": true. Do not read or write anything under readings-v2 or readings-test other than your own output files.
Put every image you make in /private/tmp/claude-501/-Users-palash-Documents-reflex-code/5111f797-a308-4171-a556-13c80f4f2dee/scratchpad/method-test/ (create it). Sides of one building should agree with each other.

MODES (the task line names yours):
- full: the protocol as written, including its check-overlay loop (check.py, check.py grid, views.py): read, draw the overlay, compare with the photograph, fix, repeat until it matches.
- single: one pass only. Look at the images (the rect crops; the persp views only if a rect crop is unclear), write the reading once. Do NOT run check.py, views.py or any other tool except to write the file. No overlay loop.
- check-draft: a draft reading of each side already exists in the DRAFT folder the task line names. Copy it to your OUTPUT folder, then run the protocol's check-overlay loop on it and fix what the overlay shows is wrong. Do not start over unless the draft is unusable.
Finish by replying with one line per side: what it is and how much of it you could read.
