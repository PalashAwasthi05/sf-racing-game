#!/bin/zsh
# Re-run the inherited post-processing adapters (v0.9 roof refit, v0.10 textures, v0.11 member support,
# v0.12 orientation + roof coverage, v0.13 materials) on a freshly generated pilot scene. Every step is the
# same general adapter Codex ran on the v0.7 scene; only the input scene changes.
set -e -o pipefail   # a failing step must stop the chain even though its output is piped through cut
cd "$(dirname "$0")/.."
PY=work/geoenv/bin/python; export PYTHONPATH=work
D=${1:-work/detailed-v14}; C=$D/chain; mkdir -p $C
WORLD=$D/public/world.json; FP=work/support-footprints-v11.json; NI=work/normalized-v9/input.json
echo "[v9 roof refit]";        $PY work/repair_preview_roofs_v9.py --input $NI --scene $D/king-street-pilot.glb --world $WORLD --output $C/v9 | cut -c1-300
echo "[v10 textures]";         $PY work/apply_textures_v10.py --scene $C/v9/candidate.glb --output $C/v10 | tail -2 | cut -c1-300
echo "[v11 member support]";   $PY work/repair_members_v11.py --scene $C/v10/candidate.glb --footprints $FP --output $C/v11 | cut -c1-300
echo "[v12 orientation]";      $PY work/repair_surface_orientation_v12.py --scene $C/v11/candidate.glb --footprints $FP --normalized-input $NI --world $WORLD --output $C/v12a | cut -c1-300
echo "[v12 roof coverage]";    $PY work/repair_roof_coverage_v12.py --scene $C/v12a/candidate.glb --footprints $FP --normalized-input $NI --world $WORLD --area-job work/area-source-job.json --output $C/v12b | cut -c1-300
echo "[v13 materials]";        $PY work/apply_materials_v13.py --scene $C/v12b/candidate.glb --observations work/material-observations-v13.json --output $D/final | tail -2 | cut -c1-300
echo "chain complete: $D/final/candidate.glb"
