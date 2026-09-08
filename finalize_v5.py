from pathlib import Path
import json,zipfile,shutil
root=Path('.');out=Path('outputs');public=Path('work/king-street/public')
# Source corrections affect metadata only, not geometry or its hashed transfer parts.
p=Path('work/pipeline/envelopes.py');s=p.read_text().replace("if radius==0:status='documented-dimension'","if radius==0:status=a.get('status','fixed-image-estimate')").replace("r['joins'].append(event);other['joins'].append(event)","r['joins'].append(event);other['joins'].append({**event,'otherProfile':r['profile']['id']})");p.write_text(s)
p=Path('work/pipeline/envelope-observations.json');d=json.loads(p.read_text())
for prof in d['profiles']:
 for key,a in prof['levels'].items():
  if a.get('searchRadiusM')==0:a['status']='document-constrained-estimate';a['basis']='Documented total tower height37.1856m minus visually estimated pyramidal roof rise5.48m; cornice height itself is not surveyed.'
p.write_text(json.dumps(d,indent=2))
p=Path('work/build_world.py');s=p.read_text().replace("features.append({'id':id", "if er:height=max(height,max(r['heightM'] for r in er))\n features.append({'id':id");p.write_text(s)
e=json.load(open(out/'king-street-evidence.json'));w=json.load(open(public/'world.json'))
for r in e['envelopeGraphs']:
 for a in r['levels']:
  if a['status']=='documented-dimension':a['status']='document-constrained-estimate'
 for join in r['cornerJoins']:
  if join['otherProfile']==r['grammarId']:
   join['otherProfile']={'stepped-corner-king':'stepped-corner-third','mixed-height-loft-king':'double-height-loft-third'}[r['grammarId']]
for r in e['facadeWalls']:
 if r.get('status')=='envelope-graph':r.update(next(p for p in e['envelopeGraphs'] if p['grammarId']==r['grammarId']))
for b in w['buildings']:
 hs=[r['heightM'] for r in e['envelopeGraphs'] if r['buildingId']==b['id']]
 if hs:b['height']=round(max(b['height'],max(hs)),2)
e['world']=w
doc=json.load(open(public/'facade-evidence.json'));doc['walls']=e['facadeWalls'];doc['envelopes']=e['envelopeGraphs']
(public/'facade-evidence.json').write_text(json.dumps(doc,separators=(',',':')));(public/'world.json').write_text(json.dumps(w,separators=(',',':')));(out/'king-street-evidence.json').write_text(json.dumps(e,indent=2))
v=json.load(open(out/'king-street-validation.json'));q=json.load(open(out/'king-street-observation-queue.json'))
notes=f'''# 188 King Street — street pilot04

Heuristic v0.5 · September8,2026

[Open the live viewer](https://king-street-world-pilot.reflex-4683.chatgpt.site/)

The generator now reconstructs changing building form before repeating windows. This is a richer, evidence-guided approximation, not a completed1:1 city or proof of the one-month San Francisco target.

## Changes

- The Beacon has separately modeled lower wings, a higher glass corner, a raised pale spine, contrasting commercial base, inset glazing and cornices.
- Glassworks at201King has three commercial glazing bands beneath two double-height loft bands, projecting silver frames, terracotta-clad core and a detached roof blade. The south-west corner is Glassworks, not China Basin Landing.
- Oracle Park separates masonry wings and clocktowers from the open steel stand structure and light supports. Clock faces, projecting piers/cornices and pyramidal roof forms replace the previous generic extrusion on the studied frontages.
- Eight envelope profiles contain37 nested parts. Adjacent Beacon and Glassworks frontages share named roof datums and solved corner positions. Bounded local LiDAR fitting supplies17 level estimates; depth remains visually estimated.
- Openings can be polygonal. Child surfaces cut their parent skin; returns, soffits, detached shades and braces are geometry. Metric cladding joints stop at apertures. No address-specific geometry branches were added.
- Street stations include the new targets. Afternoon sun, soft daylight, camera-following shadows and contact shading remain available.

## Evidence limits

The assistant inspected a January2025 Google Street View panorama of all three buildings and actual architect/operator/partner photographs for detailed structure. The downloaded detailed photographs have unknown capture dates; brochure/PDF publication dates are not capture dates. Source photographs are not embedded as facade textures.

The hierarchy is manually image-annotated. This pass implements numerical level fitting, shared constraints and a generic compiler; it does not implement automatic dense multiview reconstruction. A total documented clocktower height constrains its silhouette, but splitting that height into wall and roof remains a visual estimate. Many window counts, recess depths, material colors, interiors and minor details are approximate. Oracle signage, sculpture, much of the seating bowl, unseen facades, exact street markings and sign poses remain unresolved.

Of746 reported wall/profile sections,13 have image-supported layout or structure and733 retain priors. These are not area-weighted completeness measures. The coarse road-view queue identifies{q['counts']['roadVisibleCandidates']} candidate sections for further observation. No independent matched-photo street-accuracy test has passed.

## Result

| Measure | Result |
|---|---:|
| Building identities | {w['counts']['buildings']} |
| Envelope parts / frontages |37 /8 |
| Window assemblies | {w['counts']['windowAssemblies']:,} |
| Triangles | {v['triangles']:,} |
| Meshes / materials | {v['meshes']} / {v['materials']} |
| Raw GLB | {v['rawBytes']/1e6:.1f}MB |
| Compressed transfer | {v['transferBytes']/1e6:.1f}MB |
| Cached geometry generation before export/compression | {w['counts']['generationSeconds']:.1f}s |

Structural tests cover rejection of a taller background roof, coordinate invariance, nested offsets, shared corners/roof datums and a nonrectangular aperture that stays open. Mesh indices, finite coordinates/UVs, transfer roundtrip and earlier numerical repetition tests pass. Source type checking and the production build are checked before publication. Browser performance and independent photographic accuracy have not been benchmarked.

The scene has fewer triangles than the previous4.22million-triangle pass, but still uses ordinary baked meshes. It needs instancing, spatial streaming, collision and runtime profiling before city-scale use. Generation time excludes image research, annotation and correction; those costs determine whether the month-long target is viable.

## Reuse

Change the geographic source selection and observation graph, then run the same operators. The bundle includes the compiler, observations, source manifests, validation scripts and material assets. Cached city LiDAR and copyrighted reference photos are excluded. The GLB uses meters; the browser lighting is not baked into it. The next test should be20 unfamiliar frontages with measured annotation/correction effort and no new building-specific compiler branches.
'''
# Plain prose spacing for the user-facing report.
for a,b in [('pilot04','pilot 04'),('September8,2026','September 8, 2026'),('completed1:1','completed 1:1'),('at201King','at 201 King'),('contain37','contain 37'),('supplies17','supplies 17'),('January2025','January 2025'),('Of746','Of 746'),('sections,13','sections, 13'),('and733','and 733'),('identifies'+str(q['counts']['roadVisibleCandidates']),'identifies '+str(q['counts']['roadVisibleCandidates'])),('before city','before city'),('previous4.22million','previous 4.22 million'),('be20','be 20')]:notes=notes.replace(a,b)
(out/'king-street-pilot-notes.md').write_text(notes)
# Keep source provenance alongside deployed assets, without restricted source imagery.
site=Path('work/king-street');dest=site/'pipeline-source';dest.mkdir(exist_ok=True)
shutil.copy2('work/build_world.py',dest/'build_world.py');shutil.copytree('work/pipeline',dest/'pipeline',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'));shutil.copy2(out/'world-generation-blueprint.md',dest/'heuristic.md')
files=[p for p in out.iterdir() if p.suffix in ['.md','.json','.glb','.svg']]
files += [Path('work')/x for x in ['build_world.py','fetch_geodata.py','fetch_lidar.py','requirements-king-pilot.txt','validate_detail.py','validate_inference.py','validate_envelopes.py']]
files += [p for d in ['work/pipeline','work/material-assets'] for p in Path(d).rglob('*') if p.is_file() and '__pycache__' not in str(p)]
files += [Path('work/king-street/app')/x for x in ['world-view.ts','king-pilot.tsx','globals.css']]
files += [Path('work/facade-references/v5/reference-envelope-observations.json')]
with zipfile.ZipFile(out/'king-street-pilot-bundle.zip','w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for p in files:
  if p.exists():z.write(p,str(p))
print('Bundle',len(files),'files', (out/'king-street-pilot-bundle.zip').stat().st_size,'bytes')
