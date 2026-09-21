#!/bin/zsh
# General building rebuild; area inputs remain external data. Does not publish.
set -e -o pipefail
cd "$(dirname "$0")/.."
D=${1:-work/building-v22-release}
export PYTHONPATH=work
work/geoenv/bin/python work/generate_detailed_v14.py --output "$D"
work/geoenv/bin/python work/audit_roof_height_v22.py --scene "$D/king-street-pilot.glb" --output "$D/height-audit.json"
work/geoenv/bin/python work/apply_textures_v10.py --scene "$D/king-street-pilot.glb" --output "$D/textures"
work/geoenv/bin/python work/apply_materials_v13.py --scene "$D/textures/candidate.glb" --observations work/material-observations-v13.json --output "$D/materials"
