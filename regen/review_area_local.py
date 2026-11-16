"""Expose a focused generated geometry comparison without changing the user's area menu."""
import argparse, json, subprocess, sys
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('candidate',type=Path);ap.add_argument('label');a=ap.parse_args()
if not a.label or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in a.label):ap.error('expected label')
work=Path(__file__).resolve().parents[1];candidate=a.candidate.resolve();out=work/'king-street/public/compare'
subprocess.run([sys.executable,str(work/'preview_full_candidate_v26.py'),str(candidate),a.label],cwd=work.parent,check=True)
world=json.loads((candidate/'public/world.json').read_text());world['previewRevision']=a.label
(out/f'{a.label}-world.json').write_text(json.dumps(world,separators=(',',':')))
m=out/f'{a.label}.json';j=json.loads(m.read_text());j['world']='/compare/'+a.label+'-world.json';m.write_text(json.dumps(j))
