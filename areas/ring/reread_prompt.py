"""The prompt for re-reading one side from the registered cameras' crops (areas/ring/read-v2.md, regen/sides-reg).
Usage: reread_prompt.py <side> <scratch dir>   (run from work/)"""
import json, sys
side, scratch = sys.argv[1], sys.argv[2]
print(f"""You are re-reading one photographed side of a building of a 3D model of San Francisco: side {side}. Its earlier reading was made on crops cut from a camera placed metres off, so it covered only part of the wall; the crops have been cut again from cameras placed against the LiDAR.

Work in /Users/palash/Documents/Codex/2026-09-08/i-want-to-create-a-heuristic/work. First read areas/ring/read-v2.md (your protocol), then follow it exactly, with these differences:
- Your images are the registered crops: areas/pacific-heights/regen/sides-reg/img/{side}__v0_rect.jpg, __v1_rect.jpg and the _persp.jpg views (not regen/sides/img).
- Run every tool with SIDES_DIR=regen/sides-reg set, e.g. `SIDES_DIR=regen/sides-reg geoenv/bin/python areas/ring/check.py {side} <reading.json> <out.jpg>`; the same for check.py grid and views.py.
- Write your reading to areas/ring/readings-reg/{side}.json (create the folder), not areas/ring/readings/.
Put every image you make in {scratch}/{side}/ (create it). You may write only areas/ring/readings-reg/{side}.json. Finish by writing it and reply with one line: what the side is and how much of it you could read.""")
