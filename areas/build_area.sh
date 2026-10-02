#!/bin/zsh
# Build one area end to end: generator (--area), textures, materials, local preview with the area's world data.
# Each heavy step runs alone under the memory watchdog. Usage: work/areas/build_area.sh <area name> <build label>
set -e
cd "$(dirname "$0")/../.."
A=work/areas/$1; D=$A/$2; LOG=$A/build.log
export PYTHONPATH=work PYTHONUNBUFFERED=1
py=work/geoenv/bin/python
rm -rf $D
caffeinate -i -s $py work/watch_mem.py 20 $LOG $py work/generate_detailed_v14.py --output $D --area $A
caffeinate -i -s $py work/watch_mem.py 20 $LOG $py work/apply_textures_v10.py --scene $D/king-street-pilot.glb --output $D/textures
caffeinate -i -s $py work/watch_mem.py 20 $LOG $py work/apply_materials_v13.py --scene $D/textures/candidate.glb --observations work/regen/v1/no-material-observations.json --output $D/materials
caffeinate -i -s $py work/preview_area_v1.py $D $1-$2
echo PIPELINE-DONE $1 $2
