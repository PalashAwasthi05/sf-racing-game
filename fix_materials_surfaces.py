from pathlib import Path
p=Path('work/pipeline/materials.py');s=p.read_text()
s=s.replace("'aluminum':'#b9c0c3'", "'silver':'#aab1b1','charcoal':'#333b3d','zinc':'#4c5359','stucco':'#c2c4bf','terracotta':'#b36f49','interior':'#111719','aluminum':'#829093'")
s=s.replace("if key.startswith('glass'):rough=.16;metal=.22", "if key.startswith('glass'):rough=.20;metal=0.")
s=s.replace("if key in ['aluminum','steel']:rough=.34;metal=.75", "if key in ['aluminum','steel','silver','charcoal','zinc']:rough=.38;metal=.55")
s=s.replace("m=PBRMaterial(name=key,baseColorFactor=rgba,", "rgb=np.array(rgba[:3],float)/255;linear=np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4);rgba=np.r_[linear,1.]\n   m=PBRMaterial(name=key,baseColorFactor=rgba,")
p.write_text(s)
p=Path('work/build_world.py');s=p.read_text().replace('from scipy.spatial import cKDTree','from scipy.spatial import cKDTree, Delaunay').replace('from shapely import contains_xy, make_valid','from shapely import contains_xy, make_valid, segmentize, polygons as polygon_array, covered_by')
start=s.index('def elev(x,y):');end=s.index('\nscene=trimesh.Scene()',start)
s=s[:start]+'''def elev(x,y):
 fx=np.clip((np.asarray(x)-MIN)/STEP,0,N-1.00001);fy=np.clip((np.asarray(y)-MIN)/STEP,0,N-1.00001);ix=np.floor(fx).astype(int);iy=np.floor(fy).astype(int);fx=fx-ix;fy=fy-iy
 a,b,c,d=grid[iy,ix],grid[iy,ix+1],grid[iy+1,ix],grid[iy+1,ix+1]
 return np.where(fx+fy<=1,a+(b-a)*fx+(c-a)*fy,d+(c-d)*(1-fx)+(b-d)*(1-fy))
''' + s[end:]
start=s.index('def surface(p,key,color,offset=.03):');end=s.index('\ndef cylinder(',start)
s=s[:start]+'''def surface(p,key,color,offset=.03):
 for a in polys(p):
  if a.area<.02:continue
  if a.area<4:
   v,f=trimesh.creation.triangulate_polygon(a,engine='earcut')
  else:
   dense=segmentize(a,1.5);coords=np.concatenate([np.array(r.coords)[:-1] for r in [dense.exterior]+list(dense.interiors)])
   xmin,ymin,xmax,ymax=a.bounds;xx,yy=np.meshgrid(np.arange(math.ceil((xmin-MIN)/STEP),math.floor((xmax-MIN)/STEP)+1)*STEP+MIN,np.arange(math.ceil((ymin-MIN)/STEP),math.floor((ymax-MIN)/STEP)+1)*STEP+MIN)
   inside=contains_xy(a,xx.ravel(),yy.ravel());inner=np.c_[xx.ravel()[inside],yy.ravel()[inside]];v=np.unique(np.r_[coords,inner],axis=0)
   if len(v)<3:continue
   f=Delaunay(v).simplices;triangles=polygon_array(v[f]);f=f[covered_by(triangles,a.buffer(.002))]
  vv=np.c_[v,elev(v[:,0],v[:,1])+offset];put(trimesh.Trimesh(vv,f,process=False),key,color)
''' + s[end:]
s=s.replace("status='measured-fit' if layer in ['terrain','roofs'] else 'visual-fit' if fid and any(r['buildingId']==fid and r['status'] in ['visual-fit','image-layout-fit'] for r in facade_records) else 'inferred'", "status='measured-fit' if layer in ['terrain','roofs'] else 'visual-fit' if '__observed_' in semantic else 'inferred'")
p.write_text(s)
p=Path('work/pipeline/facade_grammar.py');s=p.read_text().replace("layer='buildings__'+c['id'];detail='facades__'+c['id'];", "layer='buildings__'+c['id']+'__observed_'+profile['id'];detail='facades__'+c['id']+'__observed_'+profile['id'];")
p.write_text(s)
# Unknown parallel walls must not silently inherit a street observation.
p=Path('work/pipeline/facades.py');s=p.read_text().replace("obs=next((o for o in matches if np.dot(n,o['normal'])>.82),None)","obs=None if profile else next((o for o in matches if np.dot(n,o['normal'])>.82),None)");p.write_text(s)
