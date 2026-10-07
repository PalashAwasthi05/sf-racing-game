#!/bin/zsh
# Rebuild a few buildings of a built area (the ring round a park) and splice them into that build's model, for a local
# preview in minutes instead of a whole-area build: generator --only, textures, slim, materials on the small model,
# then areas/ring/splice.py and the usual preview. Parked cars and overhead equipment are the base build's.
# Usage: work/areas/ring/build_ring.sh <area> <base build> <label> <ids.json> <facade readings: profiles.json>
set -e
cd "$(dirname "$0")/../../.."
A=work/areas/$1; B=$A/$2; R=$A/$3-raw; D=$A/$3; LOG=$A/ring.log
export PYTHONPATH=work PYTHONUNBUFFERED=1
py=work/geoenv/bin/python
rm -rf $R $D
caffeinate -i -s $py work/watch_mem.py 20 $LOG $py work/generate_detailed_v14.py --output $R --area $A --regen-profiles $5 --only $4
caffeinate -i -s $py work/watch_mem.py 20 $LOG $py work/apply_textures_v10.py --scene $R/king-street-pilot.glb --output $R/textures
caffeinate -i -s $py work/watch_mem.py 20 $LOG $py work/areas/slim_textures.py $R/textures/candidate.glb $R/textures/candidate-slim.glb
caffeinate -i -s $py work/watch_mem.py 20 $LOG $py work/apply_materials_v13.py --scene $R/textures/candidate-slim.glb --observations work/regen/v1/no-material-observations.json --output $R/materials
$py work/areas/ring/splice.py $B $R $4 $D
$py work/preview_area_v1.py $D $1-$3
C=work/king-street/public/compare
for k in cars overhead; do cp $C/$1-$2-$k.json $C/$1-$3-$k.json; done
$py -c "import json,sys;p='$C/$1-$3.json';m=json.load(open(p));m.update(cars='/compare/$1-$3-cars.json',overhead='/compare/$1-$3-overhead.json');json.dump(m,open(p,'w'))"
echo RING-DONE $1 $3
