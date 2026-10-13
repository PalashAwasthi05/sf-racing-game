"""The prompt for one building's roof-features agent (areas/ring/roof-features.md).
Usage: features_prompt.py <building> <scratch dir>   (run from work/)"""
import json, sys
b, scratch = sys.argv[1], sys.argv[2]
poses = json.load(open('areas/ring/poses-all.json'))['poses']
sides = sorted({p['side'] for p in poses if p['building'].split('_p')[0] == b})
pairs = [f"areas/ring/pairs-r7/{p['side']}__v{p['view']}.jpg" for p in poses if p['side'] in sides]
print(f"""You are adding the dormers and chimneys on the roof of one building of a 3D model of the buildings round Lafayette Park, San Francisco: building {b}. Its photographed sides: {', '.join(sides)}.

Work in /Users/palash/Documents/Codex/2026-09-08/i-want-to-create-a-heuristic/work. First read areas/ring/roof-features.md (your protocol), then follow it exactly. Run the tools with geoenv/bin/python.

The building's photograph | model sheet: areas/ring/framing-r14reg/{b}.jpg (if present). Its photograph | model pairs: {', '.join(pairs)}.
Put every image you make in {scratch}/{b}/ (create it); nothing in the repository.

Start with: `areas/ring/massing_view.py bumps {b}` and `areas/ring/massing_view.py plan {b} {scratch}/{b}/plan.png`, then look at every pair for dormers and chimneys, and cut sections through the bumps you use. Every feature must stand on a bump.

You may write only areas/ring/massing/features/{b}.json. Do not edit code, readings, the overrides file or anything else. Finish by writing that file (the format in roof-features.md) and reply with the same JSON.""")
