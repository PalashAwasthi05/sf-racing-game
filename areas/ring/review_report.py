"""The ring review in one place: every side's checklist (areas/ring/reviews/*.json), the builder gaps and massing
problems the reviewers found, and counts. Usage: review_report.py [out.json]   (run from work/)"""
import collections, json, sys
from pathlib import Path
ITEMS = ['massing', 'roofline', 'windows', 'bays', 'balconies', 'entry', 'materials', 'ornament', 'streetEdge']
rows, gaps, massing, verdicts = [], [], [], {}
for f in sorted(Path('areas/ring/reviews').glob('*.json')):
    r = json.load(open(f)); verdicts[r.get('group', f.stem)] = r.get('verdict', '')
    for s in r.get('sides', []):
        items = s.get('items', {})
        rows.append({'group': r.get('group'), 'building': r.get('building'), 'side': s['side'], 'changed': s.get('readingChanged'),
                     **{k: (items.get(k) or '').split(':')[0].strip().split(' ')[0] for k in ITEMS}, 'detail': items})
    gaps += [{**g, 'group': r.get('group')} for g in r.get('builderGaps', [])]
    massing += [{'group': r.get('group'), 'problem': m} for m in r.get('massing', [])]
count = collections.Counter(row[k] for row in rows for k in ITEMS)
out = {'groups': len(verdicts), 'sides': len(rows), 'statusCounts': dict(count), 'verdicts': dict(collections.Counter(v.split(' ')[0] for v in verdicts.values())),
       'builderGaps': gaps, 'massing': massing, 'rows': rows}
if len(sys.argv) > 1: json.dump(out, open(sys.argv[1], 'w'), indent=1)
print(json.dumps({k: out[k] for k in ('groups', 'sides', 'statusCounts', 'verdicts')}))
print(len(gaps), 'builder gaps,', sum(g.get('readingGives') is True for g in gaps), 'of them given by the reading;', len(massing), 'massing problems')
