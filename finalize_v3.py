import json,zipfile
from pathlib import Path
root=Path('.');out=Path('outputs');v=json.load(open(out/'king-street-validation.json'));w=json.load(open('work/king-street/public/world.json'));fits=json.load(open('work/pipeline/facade-layouts.json'))['fits'];queue=json.load(open(out/'king-street-observation-queue.json'))
notes=f'''# 188 King Street — street pilot 03

Heuristic v0.4 · September 8, 2026

[Open the live viewer](https://king-street-world-pilot.reflex-4683.chatgpt.site/)

This revision prioritizes driver-height facade correspondence. It remains an incomplete reconstruction and has not passed the requested 1:1 or game-reference realism gate.

## What changed

- Sparse image landmarks now feed projective rectification and a numerical repetition fit. The loft bank fits 5 columns × 3 rows; a hotel bay bank fits 3 × 7. Missing cells are marked as inferred repeats.
- Five oriented frontages across four building identities use observation-derived region grammars. Ground floors, terrace frames, continuous glazing, projected bays, shade fins, canopies and blank material fields can have different rules on the same wall.
- Fragmented municipal features referring to the same OSM building are reconciled by an overlap rule, reducing duplicate identities. The facade phase is anchored to the full frontage instead of restarting at each footprint segment.
- Frontage height uses local high-level LiDAR support. A low courtyard slab no longer compresses the street facade into a few meters.
- Densified roads and sidewalks follow the same terrain interpolation, reducing pale ground breaking through the asphalt. This does not establish surveyed curb geometry.
- The viewer starts at 1.25 m driver height with the details panel collapsed. Five street stations come from the mapped road lines. Afternoon sun / soft daylight, camera-following shadows, GTAO contact shading, AgX tone mapping, and correct linear material factors expose depth and material differences.

## Evidence and actual limits

January 2025 Street View supports the 188/190 and 170 King street facades. Hotel operator/engineer photographs support Hyatt and VIA patterns, but their capture dates are unknown. Hotel VIA's upper facade reference is historical construction photography; completed entrance photographs provide separate evidence. Design renderings and courtyard-only photographs were excluded from street-facing counts. Source pixels are not embedded as facade textures.

Two banks use numerical fitting. The remaining profile parameters are assistant visual annotations or explicit transfers within the same photographed building, not dense automatic image extraction. Regional height fractions, many depths, tile sizes, absolute opening dimensions and unseen surfaces remain inferred. The sculptural frieze at 170 King remains unresolved. Most of the surrounding scene still uses architectural priors.

The full evidence file records 5 profile-bearing wall sections and 969 remaining inferred sections. That wall-section count includes small upper/setback sections and is not an area-weighted accuracy measure. The road-view work queue currently flags {queue['counts']['roadVisibleCandidates']} candidate wall sections under a coarse 2D visibility approximation. Independent views and metric ground truth are still absent.

## Result and validation

| Measure | Result |
|---|---:|
| Building identities | {w['counts']['buildings']} |
| Window assemblies | {w['counts']['windowAssemblies']:,} |
| Trees | {w['counts']['trees']} |
| Sign candidates | {w['counts']['signPanels']} |
| Meshes | {v['meshes']} |
| Triangles | {v['triangles']:,} |
| Raw GLB | {v['rawBytes']/1e6:.1f} MB |
| Compressed transfer | {v['transferBytes']/1e6:.1f} MB |
| Cached geometry generation, before export/compression | {w['counts']['generationSeconds']:.1f} s |

Finite geometry/UVs, mesh indices, compressed transfer round-trip, opening-region non-overlap and source compilation checks pass. Three synthetic perspective/noise/occlusion examples recover their expected grids, with withheld-center errors below one pixel. Real same-image withheld landmarks have errors of 4.6/9.4 px and 11.2/16.1 px; these are annotation consistency results, not independent reconstruction accuracy.

Browser interactions and runtime frame time were not benchmarked in this revision. The scene still contains over four million triangles and ordinary baked meshes. Contact shading adds rendering cost. Instancing, HLOD, collision, calibrated depth, current sign/paint observations, material capture and citywide imagery extraction remain work to do.

## Reuse and the next scale test

Change region adapters and observation data, not mesh code. Run the same fitter/compiler on 20 previously unseen frontages and measure image acquisition, annotation and correction minutes separately from cached generation. This prototype has not yet demonstrated the one-person-month San Francisco target. See the accompanying heuristic and pipeline README for the full workflow and reproducible inputs.

The GLB preserves meter scale and PBR materials. The browser's lighting and ambient occlusion are renderer settings, not baked into the interchange file. CC0 material maps are reusable surface examples, not scans of these walls.
'''
(out/'king-street-pilot-notes.md').write_text(notes)
# User-facing reproducible package; reference photographs are intentionally excluded.
files=[out/'king-street-pilot.glb',out/'world-generation-blueprint.md',out/'king-street-pilot-notes.md',out/'king-street-evidence.json',out/'king-street-validation.json',out/'king-street-inference-validation.json',out/'king-street-observation-queue.json']
files += [Path('work')/x for x in ['build_world.py','fetch_geodata.py','fetch_lidar.py','requirements-king-pilot.txt','validate_detail.py','validate_inference.py']]
files += [p for d in ['work/pipeline','work/material-assets'] for p in Path(d).rglob('*') if p.is_file() and '__pycache__' not in str(p)]
files += [Path('work/king-street/app')/x for x in ['world-view.ts','king-pilot.tsx','globals.css']]
with zipfile.ZipFile(out/'king-street-pilot-bundle.zip','w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for p in files:
  if p.exists():z.write(p,str(p))
print('Bundle files',len(files),'bytes',(out/'king-street-pilot-bundle.zip').stat().st_size)
