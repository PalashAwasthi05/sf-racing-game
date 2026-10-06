#!/bin/zsh
# Photograph every side of an area's latest build: sides (street-framed walls), tiers (framed from their own foot), one
# merged sides file and the draft-reading jobs (the LiDAR skyline alignment runs on its own: align_area.sh).
# Usage: work/areas/sides_area.sh <area name> <build dir name>      e.g.  pacific-heights v1
set -e
cd "$(dirname "$0")/.."
A=areas/$1; B=$A/$2; R=$A/regen; LOG=$A/sides.log
export SFW_AREA=$A REGEN_ROOT=$R PYTHONUNBUFFERED=1
py=geoenv/bin/python
cp $B/public/world.json $A/world.json; cp $B/king-street-evidence.json $A/evidence.json
mkdir -p $R/sides
# three shards by building, side by side (one is single-threaded and spends half its time decoding panoramas)
for k in 0 1 2; do caffeinate -i -s $py watch_mem.py 8 $LOG.$k $py regen/sides.py --evidence $A/evidence.json --out $R/sides --shard $k/3 & done; wait
$py - <<EOF2
import json
P=[]; S={}
for k in range(3):
    d=json.load(open('$R/sides/sides-%d.json'%k)); P+=d['pieces']
json.dump({'summary':{'pieces':len(P),'seen':sum(bool(p['views']) for p in P)},'pieces':P},open('$R/sides/sides.json','w'),indent=1)
print('merged',len(P),'pieces')
EOF2
caffeinate -i -s $py watch_mem.py 14 $LOG $py regen/tier_sides.py --evidence $A/evidence.json --out $R/sides
$py - <<EOF
import json
S=json.load(open('$R/sides/sides.json')); T=json.load(open('$R/sides/sides-tiers.json'))['pieces']
have={p['id'] for p in S['pieces']}; S['pieces']+=[p for p in T if p['id'] not in have]
json.dump(S,open('$R/sides/sides.json','w'),indent=1)
seen=[p for p in S['pieces'] if p['views']]
print(json.dumps({'pieces':len(S['pieces']),'withPhoto':len(seen),'readable':sum(p['lengthM']*p['heightM']>=30 for p in seen)}))
EOF
$py regen/make_drafts.py --per 10
echo SIDES-DONE $1
