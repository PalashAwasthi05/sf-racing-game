from pathlib import Path
import json,shutil,zipfile
root=Path('.');out=root/'outputs';site=root/'work/king-street';public=site/'public'
w=json.load(open(public/'world.json'));v=json.load(open(out/'king-street-validation.json'));e=json.load(open(out/'king-street-evidence.json'));m=json.load(open(out/'king-street-generation-manifest.json'));c=w['counts'];known=len({x['buildingId'] for x in e['facadeWalls'] if x['sourceIds']});fresh=e['freshness'];unknown=sum(r['needsFreshCapture'] for r in fresh)
assert m['outputSha256']==v['sha256']
notes=f'''# 188 King Street — street pilot 06

Heuristic v0.7 · September 8, 2026

[Open the regenerated private viewer](https://king-street-world-pilot.reflex-4683.chatgpt.site/)

The complete study was regenerated from the revised heuristic: {c['buildings']} building identities, all street layers and all facade geometry. The generation manifest fingerprints the compiler, observations and cached source inputs before and after the run; it rejects changed inputs. This is a structurally checked approximation, not a verified 1:1 reconstruction.

## Changes across the map

- **Ground support and ownership:** all {c['foundationChecks']} building footprints receive boundary terrain samples and a continuous foundation below the lowest sample. {c['spatialCorrections']} outline reconciliation/ownership decisions are logged. Every final pair of building footprints is checked for overlapping area. New map edits are not treated as proof of construction or demolition.
- **Wall and roof construction:** adjacent facade patches share the exposed depth difference instead of producing duplicate buried returns. Replacement regions split old wall edges. Roof tiers use nested volumes with separate vertical intervals and exposed caps. They no longer receive automatic window floors. The generator removed {c['removedDuplicateTriangles']:,} coincident triangles after geometry batching.
- **Broader facade evidence:** twelve additional frontages now use the common envelope system: 128 King; 180, 200, 224, 139, 133/135 and 101 Townsend; both China Basin buildings; 25 and 45/47 Lusk; and the Third Street low shop row. Their region records distinguish material changes, solid fascia bands, projecting bays, arched openings, small-pane industrial glazing, fire escapes and raised entrances. Shared operators generate those details from observations. Metric extents, depths and several incomplete-view counts remain estimates.
- **Literal text and freshness:** {c['facadeLetteringPlacements']} photographed text placements attach to their declared facade carriers. New words include the Garcia & Maggini belt, INSEAD, the photographed bank fascia, SAP, CLOUDFLARE, 25 LUSK and andreessen/horowitz. Former BASEBALLISM text is excluded after confirmed 2026 tenant-change evidence; no uninspected CardVault sign is invented. FIRST REPUBLIC BANK is explicitly the lettering photographed in May 2025. Undated owner photos do not become current observations.
- **Street mobility:** {c['mappedStairways']} mapped stairways, {c['cyclewayFeatures']} cycleway/lane features and {c['mappedBarrierSections']} mapped barrier sections are compiled from topology. Both sides of bike-lane and parking tags are parsed. Shared promenades remain distinct from protected lanes. Tram tracks no longer generate railings automatically. Widths, paint patterns and terrain-derived stair rises remain approximate.
- **Lighting and small geometry:** curb edges conform to the terrain. The moving sunlight region follows shadow-map texel spacing, reducing abrupt shadow changes; contact shading uses a smaller radius and thickness. This has not been independently checked for all browser/GPU combinations.

## Measured output

| Measure | Result |
|---|---:|
| Fully regenerated building identities | {c['buildings']} |
| Buildings with some direct facade evidence | {known} |
| Articulated nodes / frontages | {c['envelopeNodes']} / {c['envelopeFrontages']} |
| Wall/profile sections with visual evidence | {c['imageFittedWalls']} / {c['facadeWalls']} |
| Frontage packets needing a fresh capture | {unknown} / {len(fresh)} |
| Window assemblies, including priors | {c['windowAssemblies']:,} |
| Photographed lettering placements | {c['facadeLetteringPlacements']} |
| Triangles | {v['triangles']:,} |
| Raw GLB / compressed transfer | {v['rawBytes']/1e6:.1f} / {v['transferBytes']/1e6:.1f} MB |
| Geometry generation before export | {c['generationSeconds']:.1f} seconds |
| Full rebuild including export | {m['elapsedSecondsIncludingExport']:.1f} seconds |

The wall-section ratio is not area-weighted coverage; it is not comparable directly with the earlier count because opaque roof risers no longer produce window-grid records. A building with one photographed face is not a fully reconstructed building. The whole-area observation queue retains unobserved and independently unvalidated wall regions.

## Validation and remaining work

Physical regression tests cover unfamiliar sloping sites, competing outlines, nested roofs, adjacent facade planes, true arch apertures, coincident triangles, asymmetric bike lanes and evidence freshness. Previous envelope, four-edge fitting and material/sign admission tests pass. The exported mesh has finite coordinates and UVs, valid indices and lossless compressed-transfer reconstruction. Source type checking and the production build pass. These checks validate internal consistency, not photographic likeness or game performance.

The source packet was manually interpreted from photographs by the assistant. Dense automatic segmentation, calibrated multiview depths, independent street-view comparison, complete storefront inventories, exact custom typography, sculpted ornament, collision, streaming and performance profiling remain unfinished. Many unobserved walls still use priors. Geometry from the 2023 LiDAR and historical municipal data may still be outdated; fresh-image coverage does not independently validate every roof or footprint. No claim is made that every reported visual defect has been eliminated.

Numerical expansion uses no model tokens. The reported seconds exclude source research, manual annotation, review, tokens and correction passes. Those costs have not been measured adequately to establish one person generating all San Francisco within a month.

The bundle contains the general compiler, observation data, heuristic, GLB, CC0 material examples and checks. Reference photograph pixels and font binaries are excluded. The GLB uses meters; browser lighting is not baked into it.
'''
(out/'king-street-pilot-notes.md').write_text(notes)
repo=site/'pipeline-source';repo.mkdir(exist_ok=True)
for name in ['build_world.py','rebuild_world.py']:shutil.copy2(root/'work'/name,repo/name)
shutil.copytree(root/'work/pipeline',repo/'pipeline',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'));shutil.copy2(out/'world-generation-blueprint.md',repo/'heuristic.md')
files=[p for p in out.iterdir() if p.suffix in ['.md','.json','.glb']]
files += [root/'work'/x for x in ['build_world.py','rebuild_world.py','fetch_geodata.py','fetch_lidar.py','requirements-king-pilot.txt','validate_detail.py','validate_inference.py','validate_envelopes.py','validate_surfaces.py','validate_spatial.py']]
files += [p for d in ['work/pipeline','work/material-assets'] for p in Path(d).rglob('*') if p.is_file() and '__pycache__' not in str(p)]
files += [site/'app'/x for x in ['world-view.ts','king-pilot.tsx','globals.css']]
with zipfile.ZipFile(out/'king-street-pilot-bundle.zip','w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for p in files:
  if p.exists():z.write(p,str(p))
print('Bundle files',len(files),'bytes', (out/'king-street-pilot-bundle.zip').stat().st_size,'direct-evidence buildings',known)
