"""The prompt for one building's massing agent (areas/ring/massing.md).
Usage: massing_prompt.py <building> <scratch dir>   (run from work/)"""
import json, sys
b, scratch = sys.argv[1], sys.argv[2]
poses = json.load(open('areas/ring/poses-all.json'))['poses']
sides = sorted({p['side'] for p in poses if p['building'].split('_p')[0] == b})
pairs = [f"areas/ring/pairs-r7/{p['side']}__v{p['view']}.jpg" for p in poses if p['side'] in sides]
notes = json.load(open('areas/ring/massing-notes.json')).get(b, [])
print(f"""You are setting the massing (wall heights, roof form, parts) of one building of a 3D model of the buildings round Lafayette Park, San Francisco: building {b}. Its photographed sides: {', '.join(sides)}.

Work in /Users/palash/Documents/Codex/2026-09-08/i-want-to-create-a-heuristic/work. First read areas/ring/massing.md (your protocol), then follow it exactly. Run the tools with geoenv/bin/python.

What reviewers comparing the model with the street photographs found wrong with its shape: the "{b}" entry of areas/ring/massing-notes.json ({len(notes)} notes). Its photograph | model pairs: {', '.join(pairs)}.
Put every image you make in {scratch}/{b}/ (create it); nothing in the repository.

Start with: `areas/ring/massing_view.py plan {b} {scratch}/{b}/plan.png`, `fit {b}`, and sections across and along the roof. Heights come from the LiDAR, forms from the photographs, and both must agree. Every part of your massing must pass `check`.

You may write only areas/ring/massing/specs/{b}.json. Do not edit code, readings, the overrides file or anything else. Finish by writing that file (the format in massing.md) and reply with the same JSON.""")
