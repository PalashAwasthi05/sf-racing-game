"""Merge frontage readings (content) with the questionnaire (frame and identity) into an observation file.

Readings hold only what a reader saw in the crops: material family, repeat banks, panels, bands. The frame
(normal, width, wall heights), the identity keys (building id, OSM way) and the source images with their
capture dates come from the questionnaire and the crop manifest. Output: pipeline/frontage-observations-<name>.json,
which the generator loads next to the hand-annotated profiles. Every profile is validated before writing.
"""
import argparse,json,pathlib,subprocess,sys
WORK=pathlib.Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--questionnaire',required=True);ap.add_argument('--readings',required=True);ap.add_argument('--name',required=True)
ap.add_argument('--manifest',default=str(WORK/'facade-priority-v15/crops/manifest.json'));a=ap.parse_args()
q={e['profileIdHint']:e for e in json.load(open(a.questionnaire))};readings=json.load(open(a.readings));manifest=json.load(open(a.manifest))
crops=manifest if isinstance(manifest,list) else next(v for v in manifest.values() if isinstance(v,list));by_image={c['image']:c for c in crops}
EYE_M=2.4   # camera height above the wall base used by rectify_frontages_v15.py

def to_fractions(content,W,Href,camera_x=None,pieces=None):
 """Readings measured on the rectified image (metres from its left edge and above the wall base) become
 fractions of the frontage.

 The rectified image is drawn through a camera pose that is off by metres: a pose too far from the wall shrinks
 the whole facade about the camera's foot (equally in x and z), a pose shifted along the street moves it. Two
 cues recover the truth, and their disagreement is itself a finding:
   * the building's visible corners (`visibleLeftM`, `visibleRightM`): their image distance against the
     footprint width gives a horizontal scale k_h and the shift;
   * the wall top seen in the image (`imageWallTopM`) against the height the LiDAR measured for that wall
     (`lidarWallTopM`, default the image height): a vertical scale k_v about the eye height.
 k_v is trusted first; if k_h differs from it by more than a tenth the footprint is not the facade (it runs
 over a neighbour, or stops short), which the note records and the reading covers by describing what the
 image shows across the footprint's whole frontage.
 """
 if not content.get('unitsM'):return content
 xl=content.get('visibleLeftM');xr=content.get('visibleRightM');img_top=content.get('imageWallTopM')
 # The frontage the visible corners span is the run of LiDAR wall pieces along this frame, not the frame itself:
 # a footprint whose other edges reach further along the street draws a frame wider than its frontage wall.
 ext_lo,ext_hi=0.,W
 if pieces:
  ranges=[p['xFractionRange'] for p in pieces if p.get('xFractionRange') and 0<=min(p['xFractionRange'])<=1 and max(p['xFractionRange'])<=1]
  if ranges:ext_lo=min(min(r) for r in ranges)*W;ext_hi=max(max(r) for r in ranges)*W
  # The listed pieces describe the frontage only when they cover most of the frame; a frontage whose walls were
  # partly covered by an earlier observation, or partly further from the street than the selection reaches, lists
  # a subset, and the corners then span the frame itself (one such subset compressed a 110 m reading by 0.62).
  if ext_hi-ext_lo<.8*W:ext_lo,ext_hi=0.,W
 k_h=((ext_hi-ext_lo)/(xr-xl)) if (xl is not None and xr is not None and xr>xl) else None
 lidar_top=float(content.get('lidarWallTopM',Href))
 k_v=((lidar_top-EYE_M)/(img_top-EYE_M)) if (img_top and img_top>EYE_M+1) else None
 # Each axis takes its own cue: the corners fix the facade's extent along the frontage, the roof line its heights.
 # When both agree the image was scaled uniformly by the pose distance; when they differ the image is not
 # uniformly scaled (the pose is off in heading as well as distance) and one factor must not serve both axes.
 # Whether the footprint holds more than this facade is the LiDAR's finding, not the image's: wall pieces of
 # different heights along the frontage mark it, and a frontage of one level is one facade to its corners.
 kx=k_h if k_h else (k_v if k_v else 1.);kz=k_v if k_v else (k_h if k_h else 1.)
 notes=[]
 if k_h and k_v and abs(k_h/k_v-1)>.1:notes.append(f"image scale differs by axis: corners {k_h:.3f} along the frontage, roof line {k_v:.3f} up; each axis converted by its own cue")
 if pieces:
  hs=[float(p['heightM']) for p in pieces if p.get('heightM') is not None]
  if hs and max(hs)-min(hs)>3.:notes.append("LiDAR wall pieces along this frontage differ in height by %.1f m: the footprint frontage holds more than one wall level and height-limited banks follow each piece"%(max(hs)-min(hs)))
 if xl is not None:fx=lambda x:min(1.,max(0.,(ext_lo+(float(x)-xl)*kx)/W))
 elif xr is not None:fx=lambda x:min(1.,max(0.,(ext_hi-(xr-float(x))*kx)/W))
 else:
  cx=float(camera_x) if camera_x is not None else W/2
  fx=lambda x:min(1.,max(0.,(cx+(float(x)-cx)*kx)/W))
 zt=lambda z:EYE_M+(float(z)-EYE_M)*kz
 fy=lambda z:min(1.,max(0.,zt(z)/Href))
 fyu=lambda z:max(0.,zt(z)/Href)   # unclipped: parapets and signs may rise above the image height
 rect=lambda r:[fx(r[0]),fy(r[1]),fx(r[2]),fy(r[3])]
 out={kk:vv for kk,vv in content.items() if kk not in ('unitsM','visibleLeftM','visibleRightM','imageWallTopM','lidarWallTopM','heightRefM')}
 out['groups']=[]
 for g in content.get('groups',[]):
  g=dict(g);g['region']=rect(g['region'])
  if 'xCentersM' in g:
   r=g['region'];w=max(r[2]-r[0],1e-6);g['xCenters']=[(fx(x)-r[0])/w for x in g.pop('xCentersM')];g['columns']=len(g['xCenters'])
  if 'spacingXM' in g:
   r=g['region'];g['spacingX']=float(g.pop('spacingXM'))*kx/max((r[2]-r[0])*W,1e-6)
  if 'spacingYM' in g:
   r=g['region'];g['spacingY']=float(g.pop('spacingYM'))*kz/max((r[3]-r[1])*Href,1e-6)
  if 'yCentersM' in g:
   r=g['region'];h=max(r[3]-r[1],1e-6);g['yCenters']=[(fy(y)-r[1])/h for y in g.pop('yCentersM')];g['rows']=len(g['yCenters'])
  for key,kk in (('rowPitchM',kz),('columnPitchM',kx),('pierWidthM',kx)):
   if key in g:g[key]=round(float(g[key])*kk,3)
  out['groups'].append(g)
 if 'panels' in content:out['panels']=[dict(pn,rect=rect(pn['rect'])) for pn in content['panels']]
 if 'bands' in content:
  out['bands']=[]
  for bd in content['bands']:
   bd=dict(bd,height=fy(bd['height']))
   if 'xRangeM' in bd:a,b_=bd.pop('xRangeM');bd['xRange']=[fx(a),fx(b_)]
   out['bands'].append(bd)
 if 'piers' in content:out['piers']=[dict(pi,x=fx(pi['x']),y=[fy(pi['y'][0]),fy(pi['y'][1])] if 'y' in pi else [0,1]) for pi in content['piers']]
 if 'openings' in content:out['openings']=[dict(o,rect=rect(o['rect'])) for o in content['openings']]
 if 'parapets' in content:out['parapets']=[dict(pp,top=[[fx(x),fyu(z)] for x,z in pp['top']]) for pp in content['parapets']]
 if 'ornaments' in content:
  out['ornaments']=[]
  for orn in content['ornaments']:
   orn=dict(orn,z=fy(orn['z']))
   if 'xs' in orn:orn['xs']=[fx(x) for x in orn['xs']]
   if 'xRangeM' in orn:a,b_=orn.pop('xRangeM');orn['xRange']=[fx(a),fx(b_)]
   out['ornaments'].append(orn)
 if 'signs' in content:out['signs']=[dict(sg,rect=[fx(sg['rect'][0]),fyu(sg['rect'][1]),fx(sg['rect'][2]),fyu(sg['rect'][3])]) for sg in content['signs']]
 if 'observedWallTopM' in content:out['observedWallTopM']=round(zt(content['observedWallTopM']),2)
 anchor='left corner at %.1f m'%xl if xl is not None else ('right corner at %.1f m'%xr if xr is not None else 'no corner visible; scaled about the camera foot')
 out['frameNote']=f"positions measured on the rectified frontage image ({anchor}; scale x {kx:.3f} z {kz:.3f}"+(f", corners {k_h:.3f}" if k_h else "")+(f", roof line {k_v:.3f}" if k_v else "")+f"); heights as fractions of {Href} m"
 if notes:out['frameNote']+='; '+'; '.join(notes)
 return out

profiles=[];sources={};skipped=[]
for hint,content in readings['readings'].items():
 e=q.get(hint)
 if e is None:skipped.append((hint,'not in questionnaire'));continue
 if not e['crops']:skipped.append((hint,'no crop'));continue
 images=[];[images.append(c['image']) for c in e['crops'] if c['image'] not in images]
 ids=['mapillary:'+i for i in images]
 for i in images:
  c=by_image[i];sources['mapillary:'+i]={'url':'https://www.mapillary.com/app/?pKey='+i,'captureDate':c['captureDate'],'provider':'Mapillary','type':'street panorama',
   'use':'perspective frontage crop read by the observation step; no pixels embedded','registrationGrade':c.get('registrationGrade')}
 Href=float(content.get('heightRefM') or (e['rectified'][0]['heightM'] if e.get('rectified') else max(e['wallHeightsM'])))
 camera_x=None
 if e.get('crops') and e.get('frame'):
  import numpy as _np
  crop=by_image.get(e['crops'][0]['image'])
  if crop and crop.get('cameraLocal'):
   cam=_np.asarray(crop['cameraLocal'],float);u_=_np.asarray(e['frame']['u'],float);camera_x=float(cam@u_-e['frame']['left'])
 content=to_fractions(content,float(e['frontageWidthM']),Href,camera_x,e.get('walls'))
 profile={'id':hint,'match':{'osmWayId':e['match']['osmWayId'],'featureId':e['match']['featureId']},'normal':e['normal'],'sourceIds':ids,
          'materialEvidence':{**content['materialEvidence'],'sourceIds':ids},'groups':content.get('groups',[]),
          'unresolved':[readings['method']]+content.get('unresolved',[])+[f"Frame: frontage width {e['frontageWidthM']} m from the footprint; wall heights {e['wallHeightsM']} m from LiDAR."]}
 for k in ('panels','bands','copingMaterial','openings','surfaceEvidence','openingPolicy','planeToleranceM','piers','observedWallTopM','parapets','ornaments','signs'):
  if k in content:profile[k]=content[k]
 if content.get('frameNote'):profile['heightRefM']=round(Href,2)   # fractions of a measured reading refer to this height
 if content.get('frameNote'):profile['unresolved'].append(content['frameNote'])
 profiles.append(profile)
out={'schemaVersion':1,'method':readings['method'],'street':a.name,'sources':sources,'profiles':profiles}
path=WORK/'pipeline'/f'frontage-observations-{a.name}.json';path.write_text(json.dumps(out,indent=1))
print(f'wrote {path}: {len(profiles)} profiles, {len(sources)} sources; skipped {skipped}')
r=subprocess.run([sys.executable,str(WORK/'validate_profile_v15.py'),str(path),'--no-fits','--json',str(WORK/'facade-priority-v15'/f'validate-{a.name}.json')],capture_output=True,text=True)
print(r.stdout[-3000:]);print(r.stderr[-1500:])
