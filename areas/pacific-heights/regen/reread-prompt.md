You are reading the photographed sides of one building of a 3D model of San Francisco (Pacific Heights): building {building}. Its sides to read: {sides}.

Work in /Users/palash/Documents/Codex/2026-09-08/i-want-to-create-a-heuristic/work. First read areas/ring/read-v2.md (your protocol) and follow it exactly for every side listed, with these differences:
- Your images are the crops cut from cameras placed against the LiDAR: areas/pacific-heights/regen/sides-reg/img/<side>__v0_rect.jpg, __v1_rect.jpg and their _persp.jpg views (not regen/sides/img).
- Run every tool with SIDES_DIR=regen/sides-reg set, e.g. `SIDES_DIR=regen/sides-reg geoenv/bin/python areas/ring/check.py <side> <reading.json> <out.jpg>`; the same for check.py grid and views.py.
- Write each reading to areas/pacific-heights/regen/readings-v2/<side>.json, and add "registeredCrops": true to it.
Put every image you make in /private/tmp/claude-501/-Users-palash-Documents-reflex-code/5111f797-a308-4171-a556-13c80f4f2dee/scratchpad/area-reread/{building}/ (create it). You may write only those readings-v2 files. Sides of one building should agree with each other. Finish by replying with one line per side: what it is and how much of it you could read.
