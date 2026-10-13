"""Adapter: apply general material decisions uniformly to an existing scene.
Historical material labels are unverified observations, never building-specific rules.
"""
from pathlib import Path
import json,hashlib,copy,collections,argparse
import numpy as np
from PIL import Image
import trimesh
from general_world.scanned_materials import inventory,load_asset,ROOT
from general_world.texture_selection import select_texture,recolor
from general_world.material_periodicity import measure_masonry_module,measure_board_pitch
root=Path(__file__).resolve().parents[1];work=root/'work';out=root/'outputs'
parser=argparse.ArgumentParser()
parser.add_argument('--scene',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
scene=trimesh.load(args.scene,force='scene',process=False)
catalog=json.loads((work/'texture-inventory-v10.json').read_text())['assets'];records={r['asset_id']:r for r in inventory()}
# Two scan libraries feed the catalog: the category library (loaded by its manifest record) and the small
# measured-tile set under material-assets (loaded from its map paths). Both give the same material shape.
from trimesh.visual.material import PBRMaterial
scans={m['id']:m for m in json.loads((work/'material-assets'/'manifest.json').read_text())['materials']}
def load_any(aid,tile):
 if aid in records:
  a=load_asset(records[aid],measured_scale_m=tile)
  if not a.resolved:raise RuntimeError(a.reason)
  return a.material
 m=scans[aid];root=work/'material-assets'
 color=Image.open(root/m['maps']['color']['path']).convert('RGB');normal=Image.open(root/m['maps']['normal_gl']['path']).convert('RGB');rough=Image.open(root/m['maps']['roughness']['path']).convert('L')
 mr=Image.merge('RGB',(Image.new('L',rough.size,0),rough,Image.new('L',rough.size,0)))
 return PBRMaterial(name=aid,baseColorTexture=color,normalTexture=normal,metallicRoughnessTexture=mr,metallicFactor=0.,roughnessFactor=1.,baseColorFactor=[255,255,255,255])
# Asset-level inspection corrects provider categories; no scene identity is used.
for aid,a in catalog.items():
 a['finish_class']='matte'
 if aid=='Travertine013':a['finish_class']='polished_banded'
 if aid=='Bricks088':a['pattern']='perforated_brick'
 if aid=='Wood090A':a['pattern']='timber_boards'
 if aid=='WoodFloor034':a['pattern']='parquet'
 if aid.startswith('Metal'):a['finish_class']='oxidized' if a['strong_weathering'] else 'dull'
 paths=list(ROOT.rglob(aid+'_1K-JPG_Color.jpg')) or list((work/'material-assets').rglob(aid+'_1K-JPG_Color.jpg'))
 a['mean_rgb']=np.asarray(Image.open(paths[0]).convert('RGB').resize((128,128))).mean((0,1)).tolist()
 # Plaster/render is coated; pigment is a reusable parameter, not a substrate change.
 if a['substrate_family']=='plaster':a['intrinsic_color']=False
 # Jointed scans carry their module measured from the maps themselves (course pitch and within-course joint period
 # for masonry, board pitch for boards), converted with the asset's own tile size. Colour and roughness must agree.
 a.pop('unit_size_m',None);a.pop('unit_open_axes',None);a.pop('module_measurement',None)
 if a.get('tile_size_m') and a['pattern'] in ('brick_running','timber_boards'):
  maps=[Image.open(paths[0]).convert('L'),Image.open(str(paths[0]).replace('_Color.jpg','_Roughness.jpg')).convert('L')]
  tile=np.array(a['tile_size_m'],float);size=np.array(maps[0].size,float)
  if a['pattern']=='brick_running':
   m=[measure_masonry_module(np.asarray(im)) for im in maps]
   if all(x['accepted'] for x in m):
    px=np.array([[x['module_px'],x['row_px']] for x in m],float)
    if np.allclose(px[0],px[1],rtol=.08,atol=0):
     a['unit_size_m']=(tile*px.mean(0)/size).round(4).tolist();a['module_measurement']={'module_px':px.mean(0).tolist(),'bond':m[0]['bond'],'method':'course pitch and within-course joint period, colour and roughness maps'}
  else:
   b=[measure_board_pitch(np.asarray(im)) for im in maps]
   if all(x['accepted'] for x in b) and b[0]['boards']==b[1]['boards'] and abs(b[0]['pitch_px']-b[1]['pitch_px'])<=.08*b[0]['pitch_px']:
    pitch=np.mean([x['pitch_px'] for x in b])
    if b[0]['boards']=='horizontal':a['unit_size_m']=[float(tile[0]),round(float(tile[1]*pitch/size[1]),4)];a['unit_open_axes']=[0]
    else:a['unit_size_m']=[round(float(tile[0]*pitch/size[0]),4),float(tile[1])];a['unit_open_axes']=[1]
    a['module_measurement']={'pitch_px':pitch,'boards':b[0]['boards'],'method':'board pitch, colour and roughness maps'}

# These describe generic material names exported by the earlier generator.
profiles={}
for name in ('render_white','render_gray','render_salmon','stucco','stucco_red'):
 profiles[name]=dict(substrate_family='plaster',pattern='smooth_grain' if name.startswith('render') else 'plaster_grain',finish_class='matte',old_scale=[2,2],provisional_repeat_m=[2,2])
profiles['concrete']=dict(substrate_family='concrete',pattern='concrete_grain',finish_class='matte',old_scale=[1.1,.55])
for name in ('stone','limestone'):
 profiles[name]=dict(substrate_family='stone',pattern='smooth_grain',finish_class='matte',old_scale=[2.4,1.2])
for name in ('brick_modular','brick_dark'):
 profiles[name]=dict(substrate_family='brick',pattern='brick_running',finish_class='matte',unit_size_m=[.225,.075],old_scale=[1.8,.6])
for name in ('timber_dark','timber_tan'):
 profiles[name]=dict(substrate_family='wood',pattern='timber_boards',finish_class='matte',unit_size_m=[2.4,.15] if name=='timber_dark' else [.1,1.2],old_scale=[2.4,1.2])
for name in ('terracotta_panel','terracotta_ribbed','panel_dark','panel_silver','roofing'):
 profiles[name]=dict(substrate_family='ceramic' if name.startswith('terracotta') else 'metal' if name.startswith('panel') else 'bitumen',pattern=name,finish_class='matte')

cache={};decisions={};counts=collections.Counter();updated=0;requests={};swatches=[]
for mesh in scene.geometry.values():
 old=mesh.visual.material;name=old.name;base=name.split('#')[0]   # 'recipe#rrggbb' is a colour variant of a recipe
 if base not in profiles:counts['protected_or_unclassified']+=1;continue
 if name not in decisions:
  t=copy.deepcopy(profiles[base]);img=old.baseColorTexture
  factor=np.array(old.baseColorFactor[:3])/255 if old.baseColorFactor is not None else np.ones(3)
  if img is not None:
   px=np.asarray(img.convert('RGB').resize((128,128)),float);color=px.mean((0,1))
   # For masonry the colour to match is the brick's, not the mean of brick and mortar: the same chroma mask the
   # recolour uses on the scan is used on the source.
   if t['pattern']=='brick_running':
    mask=(px.max(2)-px.min(2))/255>.12
    if mask.mean()>=.2:color=px[mask].mean(0)
  else:color=np.where(factor<=.0031308,12.92*factor,1.055*factor**(1/2.4)-.055)*255
  t['color_rgb']=color.tolist()
  d=select_texture(t,catalog);d['observation_provenance']='inherited material intent; not image verified';d['target']=t
  decisions[name]=d
  if d['action']=='request_new':requests[d['request_key']]=d['requirements']
  if d['action'] in ('reuse','recolor'):
   aid=d['asset_id'];source=load_any(aid,d['tile_size_m'])
   # A shallow copy of a trimesh material shares its property store: every variant of one asset then carries
   # the last name and recolour assigned. Each material name gets its own deep copy.
   # The images are lifted off before the copy and shared after it: a deep copy duplicated every map per colour variant,
   # and an area read in hundreds of colours outgrew the machine's memory.
   maps={k:getattr(source,k,None) for k in ('baseColorTexture','normalTexture','metallicRoughnessTexture','occlusionTexture','emissiveTexture')}
   for k,v in maps.items():
    if v is not None:setattr(source,k,None)
   mat=copy.deepcopy(source);mat.name='auto_'+name+'_'+aid
   for k,v in maps.items():
    if v is not None:setattr(source,k,v);setattr(mat,k,v)
   if d['action']=='recolor':mat.baseColorTexture=recolor(source.baseColorTexture,color,catalog[aid]['pattern']=='brick_running')
   # Source normal and roughness maps are shared unchanged across color variants.
   cache[name]=mat
   swatches.append((name,old.baseColorTexture,mat.baseColorTexture))
 d=decisions[name]
 if name in cache:
  if mesh.visual.uv is None:counts['missing_uv_retained']+=1;continue
  metric=mesh.visual.uv*np.array(d['target']['old_scale'])
  if d.get('rotate90'):metric=metric[:,::-1]*np.array([1.,-1.])   # a right-angle turn of the scan, not a stretch
  mesh.visual.uv=metric/np.array(d['tile_size_m'])
  mesh.visual.material=cache[name];updated+=1
 counts[d['action']]+=1
candidate=args.output;candidate.mkdir(parents=True,exist_ok=True)
scene.export(candidate/'candidate.glb')
# The working scene is done with: freed before the two reloads below (with it held, a large area peaked past 20 GB).
del scene,mesh;import gc;gc.collect()
# Verify exact geometry preservation through actual export/reload, not only in memory.
before=trimesh.load(args.scene,force='scene',process=False);after=trimesh.load(candidate/'candidate.glb',force='scene',process=False)
assert set(before.geometry)==set(after.geometry)
for key,m in before.geometry.items():
 n=after.geometry[key];assert np.array_equal(m.vertices,n.vertices) and np.array_equal(m.faces,n.faces),key
 if n.visual.uv is not None:assert np.isfinite(n.visual.uv).all()
assert np.allclose(before.graph.to_flattened()[next(iter(before.graph.nodes_geometry))]['transform'],after.graph.to_flattened()[next(iter(after.graph.nodes_geometry))]['transform'])
report={'revision':'0.10','geometryMeshesPreserved':len(before.geometry),'updatedMeshes':updated,'counts':dict(counts),'decisions':decisions,'newTextureRequests':[dict(request_key=k,**v) for k,v in requests.items()],'uniqueAssignedAssets':sorted({d['asset_id'] for d in decisions.values() if 'asset_id'in d}),'materialVariants':len(cache),'automaticAcquisitionImplemented':False,'imageVerifiedClassification':False,'newTextureRequestPolicy':'deduplicate by structure, exclude color; retain prior surface until acquired and validated'}
(candidate/'report.json').write_text(json.dumps(report,indent=2));(out/'texture-assignment-report.json').write_text(json.dumps(report,indent=2))
(work/'texture-inventory-v10.json').write_text(json.dumps({'schema':'reusable-material-descriptor-v10','assets':catalog},indent=2))
from PIL import ImageDraw
# A JPEG holds at most 65,500 px a side: a long sheet (a large area recolours hundreds of materials) goes in pages.
PAGE=290
for f in out.glob('automatic-material-comparison-*.jpg'):f.unlink()
for p in range(0,max(len(swatches),1),PAGE):
 page=swatches[p:p+PAGE]
 sheet=Image.new('RGB',(768,max(len(page),1)*220),'#dddddd');draw=ImageDraw.Draw(sheet)
 for i,(name,old,new) in enumerate(page):
  draw.text((10,i*220+4),name+' / previous | automatically selected',fill='black')
  if old is not None:sheet.paste(old.resize((360,190)),(10,i*220+25))
  sheet.paste(new.resize((360,190)),(394,i*220+25))
 sheet.save(out/('automatic-material-comparison.jpg' if p==0 else f'automatic-material-comparison-{p//PAGE+1}.jpg'))
print(json.dumps({k:report[k] for k in ('updatedMeshes','counts','uniqueAssignedAssets','materialVariants','geometryMeshesPreserved')}))
