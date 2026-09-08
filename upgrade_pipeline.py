from pathlib import Path
p=Path(__file__).parent
s=(p/'build_world_v1.py').read_text()
s=s.replace('import numpy as np','import time\nSTART=time.perf_counter()\nimport numpy as np',1)
s=s.replace('from rasterio.transform import from_origin','from rasterio.transform import from_origin\nfrom pipeline.materials import Materials\nfrom pipeline import facades,streets,vegetation,signs\nfrom trimesh.visual.texture import TextureVisuals')
s=s.replace("OUT=ROOT.parent/'outputs';OUT.mkdir(exist_ok=True)","OUT=ROOT.parent/'outputs';OUT.mkdir(exist_ok=True)\nCONFIG=json.load(open(ROOT/'pipeline/config.json'));OBS=json.load(open(ROOT/'pipeline/observations.json'));facade_records=[];sign_records=[]")
s=s.replace("if w[0]=='149167949'","if w[0]==CONFIG['region']['addressWayId']")
s=s.replace('Point(0,0).buffer(265,quad_segs=96);study=Point(0,0).buffer(200)',"Point(0,0).buffer(CONFIG['region']['contextRadiusM'],quad_segs=96);study=Point(0,0).buffer(CONFIG['region']['studyRadiusM'])")
a=s.index('def put(');b=s.index('def cube(',a)
s=s[:a]+'''material_bank=Materials(ROOT/'material-assets',palette)
batches=collections.defaultdict(lambda:[[],[],[],[]])
def put(mesh,key,color,uv=None):
 if len(mesh.vertices)==0:return
 material,scale=material_bank.get(color);batchkey=key+'___'+material.name
 vertices,faces,uvs,normals=batches[batchkey];off=sum(len(v) for v in vertices)
 # Split face corners to preserve hard normals and per-face, metric texture coordinates.
 inds=mesh.faces.ravel();v=np.asarray(mesh.vertices)[inds];norm=np.repeat(mesh.face_normals,3,axis=0)
 if uv is None:
  axis=np.argmax(np.abs(norm),axis=1);coord=np.zeros((len(v),2));horizontal=axis==2
  coord[horizontal]=v[horizontal,:2]/scale
  for ax in [0,1]:
   mask=axis==ax;coord[mask,0]=v[mask,1-ax]/scale[0];coord[mask,1]=v[mask,2]/scale[1]
 else:coord=np.asarray(uv)[inds]
 vertices.append(v);faces.append(np.arange(len(v)).reshape(-1,3)+off);uvs.append(coord);normals.append(norm)
''' + s[b:]
a=s.index(' target_address=');b=s.index(' sample=get_points',a)
s=s[:a]+" stadium=c['full_area']>18000\n"+s[b:]
s=s.replace(" color='brick' if hero or stadium or height<20 else ['stone','brick2','stone','brick'][sum(map(ord,id))%4]", " style=facades.prior(tags,height,c['full_area']);color=style['material']")
s=s.replace('rr=median_filter(rr,3)','rr=median_filter(rr,5)')
s=s.replace("extrude(g,z0,base_roof-z0,'buildings__'+id,color)","# Facade compiler builds walls with real openings; no opaque block behind glazing.")
s=s.replace("quant=(np.round(rr-base_roof)).astype(np.int16);quant[quant<2]=0;quant[quant>max(height+3,5)]=0", "step=CONFIG['defaults']['roofRegularizationM'];quant=(np.round((rr-base_roof)/step)*step).astype(np.int16);quant[quant<3]=0;quant[quant>max(height+3,5)]=0")
s=s.replace("part=shape(geo).intersection(g)\n   if part.area<12:continue", "part=shape(geo).buffer(.8).buffer(-.8).simplify(.45).intersection(g)\n   if part.area<max(25,g.area*.012):continue")
a=s.index(' # Facade dimensions are procedural');b=s.index(' print(name,',a)
s=s[:a]+''' records,component_count=facades.compile(globals(),c,z0,base_roof-z0,height,tags,OBS['facades']);facade_records.extend(records);counts['windowAssemblies']+=component_count
 observed_count=sum(r['status']=='visual-fit' for r in records)
 features.append({'id':id,'name':name,'height':round(height,2),'heightSource':source,'footprintSource':c['fp'],'roofParts':roofparts,'center':[round(g.centroid.x,2),round(g.centroid.y,2),round(z0,2)],'area':round(g.area,1),'clippedAtBoundary':c['clipped'],'pointCount':len(roof),'facade':'Image-informed facade components; remaining walls use unverified priors.' if observed_count else 'Detailed facade generated from an unverified architectural prior.','facadeEvidence':'Partial visual fit' if observed_count else 'Unverified prior','fittedWalls':observed_count,'totalWalls':len(records),'sourceUrl':'https://data.sfgov.org/d/ynuv-fyni' if c['fp'].startswith('Data') else 'https://www.openstreetmap.org/way/'+c['osm']['id']})
''' + s[b:]
s=s.replace("width=lanes*3.15", "width=lanes*CONFIG['defaults']['laneWidthM']")
# Smooth grade is already present; surface vertices maintain world-space metric UVs.
a=s.index('# Tree inventory placement');b=s.index('# Render independently mapped',a)
s=s[:a]+"# Shared species templates replace low-detail canopy primitives.\nvegetation.compile(globals())\n\n"+s[b:]
a=s.index('# Finish batches:');b=s.index('scene.apply_transform',a)
s=s[:a]+'''# Street compiler uses source topology and ROW; every snap remains an uncertain record.
make_sign=signs.install(globals());sign_records,sidewalk_geometry=streets.compile(globals())
# One mesh per material and feature; image payloads are shared by glTF.
for key,(v,f,uv,normals) in batches.items():
 if not v:continue
 semantic,material_name=key.split('___',1);mat,_=material_bank.get(material_name)
 mesh=trimesh.Trimesh(np.concatenate(v),np.concatenate(f),vertex_normals=np.concatenate(normals),visual=TextureVisuals(uv=np.concatenate(uv),material=mat),process=False)
 mesh.merge_vertices(merge_tex=False,merge_norm=False)
 layer=semantic.split('__')[0];fid=semantic.split('__')[1] if '__' in semantic else None
 status='measured-fit' if layer in ['terrain','roofs'] else 'visual-fit' if fid and any(r['buildingId']==fid and r['status']=='visual-fit' for r in facade_records) else 'inferred'
 mesh.metadata={'layer':layer,'featureId':fid,'evidence':status,'material':material_name}
 scene.add_geometry(mesh,node_name=key,geom_name=key,metadata=mesh.metadata)
''' + s[b:]
s=s.replace("counts['unresolvedSigns']=len(coarse_signs)", "counts['unresolvedSigns']=len(coarse_signs)\ncounts['facadeWalls']=len(facade_records);counts['imageFittedWalls']=sum(r['status']=='visual-fit' for r in facade_records);counts['pbrMaterials']=len(material_bank.cache)\ncounts['generationSeconds']=round(time.perf_counter()-START,2)")
s=s.replace("world={'studyRadiusM'", "world={'pipelineVersion':CONFIG['version'],'facadeCoverage':{'walls':len(facade_records),'visuallyFittedWalls':counts['imageFittedWalls'],'independentlyVerifiedWalls':0,'accuracyGate':'NOT PASSED'},'studyRadiusM'")
s=s.replace("'facades':", "'facades':")
s=s.replace("'No Google", "'No Google")
s=s.replace("{'world':world,'unresolvedSigns':coarse_signs}", "{'world':world,'unresolvedSigns':coarse_signs,'facadeWalls':facade_records,'signPlacements':sign_records,'observationSources':OBS['sources'],'materials':material_bank.manifest}")
s=s.replace("{'name':'Mapillary coverage'", "{'name':'Reusable PBR surfaces','date':'CC0 scans','detail':'Scanned brick, asphalt and concrete at physical texture scale; material identity remains inferred on unobserved walls.','url':'https://ambientcg.com/'},\n {'name':'Mapillary coverage'")
s=s.replace("Targeted Google Street View inspection of the King Street frontage. No Google image textures are embedded.", "Dated frontage observations fit reusable window/frame/canopy parameters. No Google image textures are embedded; most facades remain unobserved.")
s += "\n(PUBLIC/'facade-evidence.json').write_text(json.dumps({'walls':facade_records,'signs':sign_records,'sources':OBS['sources']},separators=(',',':')))\nprint('GENERATION_SECONDS',round(time.perf_counter()-START,2),flush=True)\n"
(p/'build_world.py').write_text(s)
