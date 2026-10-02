"""Merge loop results into a new reading set: a side's loop reading replaces the base reading only when the preview
scores it at least 0.02 higher and it keeps at least 70% of the base reading's opening area.
Usage: merge_loop.py <base dir> <loop dir> <out dir> <report.json> [rejected id ...]"""
import collections, glob, json, os, shutil, subprocess, sys, pathlib
sys.path.insert(0, os.path.dirname(__file__)); import specs

base, loop, dst, report = sys.argv[1:5]; REJECT = set(sys.argv[5:])
root = pathlib.Path(os.environ['REGEN_ROOT']).resolve() if os.environ.get('REGEN_ROOT') else pathlib.Path(__file__).resolve().parent / 'v1'
byid = {p['id']: p for p in json.load(open(root / 'sides/sides.json'))['pieces']}
tmp = '/tmp/claude-501/merge-pv.jpg'; os.makedirs('/tmp/claude-501', exist_ok=True)

def score(i, f):
    out = subprocess.run([sys.executable, str(pathlib.Path(__file__).parent / 'preview_side.py'), i, f, tmp],
                         capture_output=True, text=True).stdout.strip().splitlines()
    return json.loads(out[-1])['mean'] if out else None

def area(i, r):
    r = specs.clean(r, byid[i])
    cells = [c for b in r['banks'] for c in specs.bank_cells(b)] + [(o['x0'], o['z0'], o['x1'], o['z1']) for o in r['openings']]
    return sum((c[2] - c[0]) * (c[3] - c[1]) for c in cells)

if os.path.exists(dst): shutil.rmtree(dst)
shutil.copytree(base, dst)
acc, rej = [], []
for f in sorted(glob.glob(f'{loop}/*.json')):
    i = os.path.basename(f)[:-5]
    if i not in byid: continue
    o = f'{base}/{i}.json'
    try:
        so, sl = score(i, o), score(i, f)
        ao, al = area(i, json.load(open(o))), area(i, json.load(open(f)))
    except Exception as e:
        rej.append((i, 'error ' + str(e)[:40])); continue
    why = None
    if i in REJECT: why = 'rejected on review'
    elif sl is None or so is None or sl < so + .02: why = 'no gain'
    elif ao > 1 and al < .7 * ao: why = f'removed openings ({al:.0f} of {ao:.0f} m2)'
    if why: rej.append((i, why, so, sl))
    else: acc.append((i, so, sl)); shutil.copy(f, f'{dst}/{i}.json')
print('accepted', len(acc), 'mean gain', round(sum(b - a for _, a, b in acc) / max(1, len(acc)), 3))
print('rejected', len(rej), collections.Counter(r[1].split(' (')[0] for r in rej))
json.dump({'accepted': acc, 'rejected': rej}, open(report, 'w'), indent=1)
