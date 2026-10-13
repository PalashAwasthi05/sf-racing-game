"""Review groups whose renders all exist (areas/captures/<prefix>__<side>__v<view>.png for every pose of their sides),
minus the groups already reviewed or listed as launched. Usage: ready_groups.py <prefix> [launched.txt]"""
import json, sys
from pathlib import Path
prefix = sys.argv[1]; launched = set(Path(sys.argv[2]).read_text().split()) if len(sys.argv) > 2 and Path(sys.argv[2]).exists() else set()
poses = json.load(open('areas/ring/poses-all.json'))['poses']
for g in json.load(open('areas/ring/review-groups.json')):
    if g['group'] in launched or Path(f"areas/ring/reviews/{g['group']}.json").exists(): continue
    need = [f"areas/captures/{prefix}__{p['side']}__v{p['view']}.png" for p in poses if p['side'] in g['sides']]
    if need and all(Path(n).exists() for n in need): print(g['group'])
