#!/bin/zsh
# Build one area end to end: generator (--area), textures, colour textures shrunk for the viewer, materials, local
# preview with the area's world data.
# Each heavy step runs alone under the memory watchdog.
# Usage: work/areas/build_area.sh <area name> <build label> [facade readings: the area's regen profiles.json]
set -e
cd "$(dirname "$0")/../.."
A=work/areas/$1; D=$A/$2; LOG=$A/build.log
export PYTHONPATH=work PYTHONUNBUFFERED=1
py=work/geoenv/bin/python
P=(); if [[ -n $3 ]]; then P=(--regen-profiles $3); fi   # an array: zsh does not split a plain expansion into words
M=(); if [[ -f work/areas/ring/massing-overrides.json ]]; then M=(--massing work/areas/ring/massing-overrides.json); fi   # outlines the photographs and LiDAR contradict
caffeinate -i -s $py work/watch_mem.py 28 $LOG $py work/apply_textures_v10.py --scene $D/king-street-pilot.glb --output $D/textures
caffeinate -i -s $py work/watch_mem.py 28 $LOG $py work/areas/slim_textures.py $D/textures/candidate.glb $D/textures/candidate-slim.glb
caffeinate -i -s $py work/watch_mem.py 28 $LOG $py work/apply_materials_v13.py --scene $D/textures/candidate-slim.glb --observations work/regen/v1/no-material-observations.json --output $D/materials
caffeinate -i -s $py work/preview_area_v1.py $D $1-$2
SFW_AREA=$A $py work/areas/parking.py $A $1-$2    # parked cars along the kerbs OSM marks for parking
SFW_AREA=$A $py work/areas/overhead.py $A $1-$2   # street lamps, trolleybus poles and wires on the streets OSM tags lit / trolley_wire
echo PIPELINE-DONE $1 $2
