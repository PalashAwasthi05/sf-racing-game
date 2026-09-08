from pathlib import Path
import json,shutil,zipfile
root=Path('.');out=root/'outputs';site=root/'work/king-street';public=site/'public'
w=json.load(open(public/'world.json'));v=json.load(open(out/'king-street-validation.json'))
notes=f'''# 188 King Street — street pilot 05

Heuristic v0.6 · September 8, 2026

[Open the updated private viewer](https://king-street-world-pilot.reflex-4683.chatgpt.site/)

This revision treats false windows, wrong material families and missing photographed lettering as evidence problems. Every correction is compiled through general evidence and attachment rules; no per-address mesh branches are used. The scene remains an approximation and has not passed an independent street-accuracy test.

## Changes in the regenerated pilot

- 188's broad loft bank stops before the separate small-window stack. Eight individual opening boundaries in that stack were rectified from the broker photograph; the main ground floor distinguishes retail, an opaque roll-up garage and the residential entrance. Silver **188** lettering sits above that entrance, corroborated in January 2025 Street View.
- A rule audit records rejected openings, material-family corrections and sign attachment adjustments. Contradictory materials and floating signs are corrected by reusable rules tested on unfamiliar inputs. Canopy lettering derives its position from the compiled entrance support.
- Observed solid and opaque regions override inferred opening grids in both facade compilers. The Oracle clock/sign field stays brick above its lower opening; Glassworks has a separate recessed opaque metal infill below projecting glazing. Unknown and solid are distinct labels.
- Small-joint red brick, darker brick, panel/ribbed terracotta, fine-grain stucco, smooth pale trim and opaque metal panels use separate physically scaled appearance recipes. Stone no longer aliases board-form concrete. These recipes are synthesized approximations, not exact building scans.
- Nine lettering placements reproduce photographed words: 188; FLAGSHIP on both photographed corner faces; WELLS FARGO; HOTEL VIA on the alley entrance; HYATT PLACE at its entrance; ORACLE PARK on two tower faces; and 24. Exact letterforms, dimensions and placement remain estimates. No names were taken from a business directory. Unreadable 170 lettering, historical TOTO campaign copy and historical Glassworks marketing signs were excluded.
- The street route now begins at the corrected 188 frontage. Afternoon sun, soft daylight and contact shadows remain available.

## Evidence and implementation limits

January 2025 is the date displayed by the checked Google panorama; it is not a claim of current September 2026 conditions. Detailed broker, architect and operator photos mostly have unknown capture dates. Source photos are referenced but not embedded as model textures.

The new opening-boundary fitter consumes manual edge annotations. Its eight real openings have a mean same-image corner residual of 0.85 pixels. This is consistency with that annotation, not a real-world accuracy score. The facade's metric regions, depths, most window counts and sign poses remain image estimates. Most of the study still uses unverified facade priors: 13 of 746 reported wall/profile sections have image-supported layout or structure. This is not an area-weighted completeness measure.

Dense automatic facade inference, independent multiview validation, exact custom brand typography, sculpted relief, exhaustive storefront inventory and calibrated material capture remain unfinished. The one-month San Francisco goal is unproved. Reuse requires new observation packets, not new address-specific rendering branches.

## Output and validation

| Measure | Result |
|---|---:|
| Building identities | {w['counts']['buildings']} |
| Envelope nodes / frontages | {w['counts']['envelopeNodes']} / {w['counts']['envelopeFrontages']} |
| Explicit four-edge opening fits | 8 |
| Photographed lettering placements | 9 |
| Window assemblies including priors | {w['counts']['windowAssemblies']:,} |
| Triangles | {v['triangles']:,} |
| Raw GLB | {v['rawBytes']/1e6:.1f} MB |
| Compressed transfer | {v['transferBytes']/1e6:.1f} MB |
| Cached geometry generation, before export/compression | {w['counts']['generationSeconds']:.1f} s |

Compiler checks cover solid-region vetoes, preservation of unknowns, explicit-opening precedence, synthetic perspective edge recovery, hollow letter contours, distinct stone/concrete materials, and nonoverlapping real opening records. The envelope and previous inference checks pass. Mesh coordinates, UVs, indices and compressed transfer roundtrip pass. Source type checking and the production build pass. Browser performance and independent photographic accuracy have not been benchmarked.

Cached generation time excludes source research, manual image annotation, review, tokens and corrections. Those unmeasured costs dominate the month-long feasibility test. The scene still needs instancing, streaming, collision and performance profiling for game integration.

The downloadable bundle contains the general compiler, source observations, heuristic, GLB, material scans and validations. Fonts use available system fonts with a bundled Pillow fallback; no font binaries or copyrighted reference photographs are redistributed. The GLB uses meters. Viewer lighting is not baked into it.
'''
(out/'king-street-pilot-notes.md').write_text(notes)
# Keep the source ledger portable: temporary machine paths are research state only.
for name in ['lettering-observations.json']:
 p=root/'work/pipeline'/name;d=json.load(open(p))
 for src in d['sources'].values():
  for k in ['temporary_inspection_file','existing_file']:src.pop(k,None)
 p.write_text(json.dumps(d,indent=2))
# Metadata only. No geometry or source observations changed.
for p in [out/'king-street-evidence.json',public/'facade-evidence.json']:
 d=json.load(open(p))
 for src in d.get('surfaceSources',{}).values():
  for k in ['temporary_inspection_file','existing_file']:src.pop(k,None)
 p.write_text(json.dumps(d,indent=2 if p.parent==out else None,separators=None if p.parent==out else (',',':')))
repo=site/'pipeline-source';repo.mkdir(exist_ok=True)
shutil.copy2(root/'work/build_world.py',repo/'build_world.py');shutil.copytree(root/'work/pipeline',repo/'pipeline',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'));shutil.copy2(out/'world-generation-blueprint.md',repo/'heuristic.md')
files=[p for p in out.iterdir() if p.suffix in ['.md','.json','.glb','.svg']]
files += [root/'work'/x for x in ['build_world.py','fetch_geodata.py','fetch_lidar.py','requirements-king-pilot.txt','validate_detail.py','validate_inference.py','validate_envelopes.py','validate_surfaces.py']]
files += [p for d in ['work/pipeline','work/material-assets'] for p in Path(d).rglob('*') if p.is_file() and '__pycache__' not in str(p)]
files += [site/'app'/x for x in ['world-view.ts','king-pilot.tsx','globals.css']]
files += [root/'work/facade-references/v5/reference-envelope-observations.json',root/'work/facade-references/v6/signs-materials-blank-zones.json']
with zipfile.ZipFile(out/'king-street-pilot-bundle.zip','w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for p in files:
  if p.exists():z.write(p,str(p))
print('Bundle files',len(files),'bytes', (out/'king-street-pilot-bundle.zip').stat().st_size)
