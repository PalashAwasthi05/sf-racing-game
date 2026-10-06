#!/bin/zsh
# A second photograph pass after more panoramas were fetched: pieces that already have photographs are kept as they are
# (their images are not rewritten while earlier jobs read them); the rest are redone with every panorama, older ones
# ranked lower. New draft jobs go to <prefix>-NNN.json, leaving out pieces already queued in draft-*.json.
# Usage: work/areas/sides_more.sh <area name> <job prefix>     e.g.  pacific-heights draft2
set -e
cd "$(dirname "$0")/.."
A=areas/$1; R=$A/regen; LOG=$A/sides.log
export SFW_AREA=$A REGEN_ROOT=$R PYTHONUNBUFFERED=1
py=geoenv/bin/python
cp $R/sides/sides.json $R/sides/sides-pass1.json
for k in 0 1 2; do caffeinate -i -s $py watch_mem.py 8 $LOG.$k $py regen/sides.py --evidence $A/evidence.json --out $R/sides --shard $k/3 --keep $R/sides/sides-pass1.json & done; wait
$py - <<EOF
import json
P=[]
for k in range(3): P+=json.load(open('$R/sides/sides-%d.json'%k))['pieces']
T=json.load(open('$R/sides/sides-tiers.json'))['pieces']; have={p['id'] for p in P}; P+=[p for p in T if p['id'] not in have]
json.dump({'summary':{'pieces':len(P),'seen':sum(bool(p['views']) for p in P)},'pieces':P},open('$R/sides/sides.json','w'),indent=1)
old={p['id'] for p in json.load(open('$R/sides/sides-pass1.json'))['pieces'] if p['views']}
print(json.dumps({'pieces':len(P),'withPhoto':sum(bool(p['views']) for p in P),'newlyPhotographed':sum(bool(p['views']) and p['id'] not in old for p in P)}))
EOF
$py regen/make_drafts.py --per 10 --prefix $2 --exclude 'draft-*.json'
echo SIDES-MORE-DONE $1
