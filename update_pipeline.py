from pathlib import Path
p=Path('work/pipeline/facades.py');s=p.read_text();s=s.replace('from shapely.geometry import Point,LineString','from shapely.geometry import Point,LineString\nfrom . import facade_grammar')
s=s.replace("default_height=main_height;g=", "default_height=main_height;g=")
s=s.replace("matches=[o for o in observations", "profiles=ctx.get('PROFILES',[]);fits=ctx.get('FITS',[])\n matches=[o for o in observations")
s=s.replace("angle=math.atan2(u[1],u[0]);style=prior(tags,height,c['full_area'])", "angle=math.atan2(u[1],u[0]);style=prior(tags,height,c['full_area'])\n    profile=next((p for p in profiles if c['osm'] and str(p['match'].get('osmWayId'))==str(c['osm']['id']) and np.dot(n,p['normal'])>.95),None)\n    if profile and c.get('elevationOffset',0)<.5:\n     fitted=facade_grammar.compile_wall(ctx,c,a,b,z0,default_height,profile,fits)\n     if fitted:\n      records.append(fitted['record']);components+=fitted['components'];continue")
p.write_text(s)
p=Path('work/build_world.py');s=p.read_text()
s=s.replace("CONFIG=json.load", "PROFILES=json.load(open(ROOT/'pipeline/facade-profiles.json'))['profiles'];FITS=json.load(open(ROOT/'pipeline/facade-layouts.json'))['fits']\nCONFIG=json.load")
# Identity consolidation uses one declared footprint per fragmented OSM building, not address-specific branches.
s=s.replace("for c in candidates:\n c['full_area']", "grouped=collections.defaultdict(list)\nfor c in candidates:\n if c['osm']:grouped[c['osm']['id']].append(c)\nconsolidations=[]\nfor oid,parts in grouped.items():\n if len(parts)<2:continue\n w=parts[0]['osm'];old_union=unary_union([p['g'] for p in parts]);coverage=old_union.intersection(w['g']).area/max(1,w['g'].area);agreement=old_union.intersection(w['g']).area/max(1,old_union.area)\n if coverage>.55 and agreement>.60:\n  candidates=[c for c in candidates if c not in parts];candidates.append({'id':'osm_'+oid,'g':w['g'],'old':{},'osm':w,'fp':'OSM; fragmented municipal identity reconciled'})\n  consolidations.append({'osmWayId':oid,'municipalParts':len(parts),'coverage':round(coverage,3),'agreement':round(agreement,3),'decision':'one OSM identity envelope; source disagreement retained'})\nfor c in candidates:\n c['full_area']")
s=s.replace("observed_count=sum(r['status']=='visual-fit' for r in records)","observed_count=sum(r['status'] in ['visual-fit','image-layout-fit'] for r in records)")
s=s.replace("'Image-informed facade components; remaining walls use unverified priors.'", "'Street-facing wall regions use observation-derived layouts; unseen walls remain inferred.'")
s=s.replace("'Partial visual fit' if observed_count", "'Image layout / visual grammar' if observed_count")
s=s.replace("r['status']=='visual-fit' for r in facade_records", "r['status'] in ['visual-fit','image-layout-fit'] for r in facade_records")
s=s.replace("'observationSources':OBS['sources']", "'observationSources':OBS['sources'],'footprintConsolidations':consolidations,'layoutFits':FITS,'profiles':PROFILES")
s=s.replace("'pipelineVersion':CONFIG['version']", "'pipelineVersion':CONFIG['version'],'streetViews':street_views")
s=s.replace("counts['generationSeconds']=", "counts['numericalFacadeFits']=len(FITS);counts['profileFrontages']=len(PROFILES);counts['inferredRepeatCells']=sum(f['completedCells'] for f in FITS)\ncounts['generationSeconds']=")
s=s.replace("'visuallyFittedWalls':counts['imageFittedWalls']", "'visuallyFittedWalls':counts['imageFittedWalls'],'numericalFits':len(FITS),'currentStreetImageryBuildings':2,'photoAgeUnknownBuildings':2")
s=s.replace("{'name':'Street-level reference','date':'January 2025','detail':'Dated frontage observations fit reusable window/frame/canopy parameters. No Google image textures are embedded; most facades remain unobserved.'", "{'name':'Street-level reference','date':'January 2025','detail':'Sparse image landmarks and visual grammars distinguish loft terraces, continuous glazing and separate ground-floor patterns. Source pixels are not embedded.'")
s=s.replace("'url':'https://ambientcg.com/'},", "'url':'https://ambientcg.com/'},\n {'name':'Hotel exterior photography','date':'Capture dates unknown','detail':'Operator and structural engineer photographs fit bay stacks, terracotta fields and external shades. Historical layouts need current independent checks.','url':'https://www.hyatt.com/hyatt-place/en-US/sfozs-hyatt-place-san-francisco-downtown/gallery'},")
# Road-space camera stations are computed from actual source lines and wall directions.
s=s.replace("# One mesh per material and feature;", "street_views=[]\nfor profile in PROFILES:\n matches=[c for c in candidates if c['osm'] and c['osm']['id']==profile['match']['osmWayId']]\n if not matches:continue\n c=matches[0];n=np.asarray(profile['normal']);n=n/np.linalg.norm(n);u=np.array([-n[1],n[0]]);coord=np.concatenate([np.array(p.exterior.coords) for p in polys(c['g'])]);d=np.max(coord@n);front=coord[(coord@n)>d-3];focus=front.mean(axis=0) if len(front) else np.array(c['g'].centroid.coords[0]);desired=Point(*(focus+n*20))\n ranked=[]\n for w,width,lanes in roads:\n  q=w['g'].interpolate(w['g'].project(desired));pos=np.array(q.coords[0]);v=pos-focus\n  if np.dot(v,n)<4 or np.linalg.norm(v)>60 or all_buildings.contains(q):continue\n  ranked.append((desired.distance(q),pos,w))\n if ranked:\n  _,pos,w=min(ranked,key=lambda v:v[0]);tangent=(u if np.dot(u,focus-pos)>0 else -u);look=focus+tangent*8\n  street_views.append({'position':np.round(pos,2).tolist(),'lookAt':[round(float(look[0]),2),round(float(look[1]),2),3.8],'label':w['tags'].get('name','Street')+' / '+profile['id']})\n# One mesh per material and feature;")
# Keep only current chunk manifest files after a changed model becomes smaller.
s=s.replace("(PUBLIC/'model.json').write_text", "for old in PUBLIC.glob('king-street-*.bin'):\n if '/'+old.name not in chunks:old.unlink()\n(PUBLIC/'model.json').write_text")
p.write_text(s)
p=Path('work/pipeline/facade_grammar.py');s=p.read_text().replace("if kind=='rollup':","if kind=='projected_bay':recess-=proj\n  if kind=='rollup':");p.write_text(s)
p=Path('work/pipeline/facade-profiles.json');import json
v=json.load(open(p))
v['profiles'][1]['groups'][0]['omit']=[[2,0]]
for pr in v['profiles']:
 if pr['id'].startswith('brick-hotel'):
  pr['groups'][2]['region']=[0,.125,1,.79];pr['groups'][2]['xCenters']=[.117,.37,.622,.91] if pr['id'].endswith('third') else [.13,.393,.627,.914];pr['groups'][2]['spacingX']=.13;pr['groups'][2]['style']['openingRatio']=.3
json.dump(v,open(p,'w'),indent=2)
p=Path('work/pipeline/config.json');v=json.load(open(p));v['version']='0.4';v['quality']['priority']='driver-view visible error';json.dump(v,open(p,'w'),indent=2)
