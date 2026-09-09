from pathlib import Path
p=Path('work/pipeline/envelopes.py');s=p.read_text().replace('from shapely.geometry import Polygon,Point,box','from shapely.geometry import Polygon,Point,box,LineString')
pos=s.index('\ndef matches')
s=s[:pos]+'''
def aperture_shape(style):
 if style.get('shapeKind')=='arch':
  spring=style.get('springRatio',.68)
  return [[0,0],[1,0],[1,spring]]+[[.5+.5*math.cos(a),spring+(1-spring)*math.sin(a)] for a in np.linspace(0,math.pi,17)]+[[0,0]]
 return style.get('shape',[[0,0],[1,0],[1,1],[0,1]])

def return_edges(node,region,nodes,shapes):
 """Adjacent patches own only their exposed depth difference, not two buried walls."""
 for aa,bb in zip(region.exterior.coords,list(region.exterior.coords)[1:]):
  edge=LineString([aa,bb]);remaining=edge
  for other in nodes:
   if other['id']==node['id'] or other['parentId']!=node['parentId']:continue
   shared=edge.intersection(shapes[other['id']].boundary)
   if shared.length<.001:continue
   remaining=remaining.difference(shared)
   if node['depthM']>other['depthM']+.002:
    for line in _lines(shared):yield line.coords[0],line.coords[-1],other['depthM']
  for line in _lines(remaining):yield line.coords[0],line.coords[-1],node['parentDepthM']

def _lines(g):
 if g.is_empty:return []
 if g.geom_type=='LineString':return [g]
 return [p for q in getattr(g,'geoms',[]) for p in _lines(q)]
''' +s[pos:]
s=s.replace("shape=s.get('shape',[[0,0],[1,0],[1,1],[0,1]])","shape=aperture_shape(s)")
s=s.replace("a,b,cc,dd=op['rect'];aperture=box(x0+a*wid,zmin+b*H,x0+cc*wid,zmin+dd*H).intersection(visible)","a,b,cc,dd=op['rect'];shape=aperture_shape(op.get('style',{}));aperture=Polygon([(x0+(a+xx*(cc-a))*wid,zmin+(b+zz*(dd-b))*H) for xx,zz in shape]).intersection(visible)")
s=s.replace("for a,b in zip(region.exterior.coords,list(region.exterior.coords)[1:]):\n     aa=np.array(a);bb=np.array(b);v=bb-aa", "for a,b,back_depth in return_edges(node,region,nodes,node_shapes):\n     aa=np.array(a);bb=np.array(b);v=bb-aa")
s=s.replace("point(*bb,node['parentDepthM']),point(*aa,node['parentDepthM'])","point(*bb,back_depth),point(*aa,back_depth)")
s=s.replace("for frac in s.get('mullions',[.5]):block(xa+ow*frac,cz,.043,oh,.055,frame_mat,gd+.025)\n    for frac in s.get('transoms',[.76]):block(cx,za+oh*frac,ow,.043,.055,frame_mat,gd+.025)","""for frac in s.get('mullions',[.5]):
     for ln in _lines(LineString([(xa+ow*frac,za),(xa+ow*frac,zb)]).intersection(op)):beam([*ln.coords[0],gd+.025],[*ln.coords[-1],gd+.025],.043,frame_mat)
    for frac in s.get('transoms',[.76]):
     for ln in _lines(LineString([(xa,za+oh*frac),(xb,za+oh*frac)]).intersection(op)):beam([*ln.coords[0],gd+.025],[*ln.coords[-1],gd+.025],.043,frame_mat)""")
s=s.replace("elif typ=='blade':", """elif typ=='canopy':
     projection=component.get('projectionM',1);thickness=component.get('thicknessM',.12)
     block((xa+xb)/2,zz,xb-xa,thickness,projection,cm,dep+projection/2)
     if component.get('braces'):
      for xx in [xa+.15,xb-.15]:beam([xx,zz+.75,dep],[xx,zz,dep+projection*.9],.035,cm)
    elif typ=='pilasters':
     for xx in np.linspace(xa,xb,component.get('count',6)):
      block(xx,(zmin+zmax)/2,component.get('widthM',.25),H,component.get('projectionM',.25),cm,dep+.10)
    elif typ=='dentils':
     for xx in np.arange(xa,xb,component.get('pitchM',.32)):block(xx,zz,.14,.22,.24,cm,dep+.1)
    elif typ=='fireEscape':
     # Observed platforms and alternating open stair flights are anchored to floor datums.
     zs=component['levelsM'];projection=component.get('projectionM',1.1)
     for hh in zs:
      block((xa+xb)/2,hh,xb-xa,.1,projection,cm,dep+projection/2)
      for xx in np.arange(xa,xb,.15):beam([xx,hh,dep+projection],[xx,hh+1.0,dep+projection],.022,cm)
      beam([xa,hh+1,dep+projection],[xb,hh+1,dep+projection],.035,cm)
     for i,(lo,hi) in enumerate(zip(zs,zs[1:])):
      xx1,xx2=(xa,xb) if i%2==0 else (xb,xa);steps=max(5,round((hi-lo)/.19))
      for j in range(steps):block(xx1+(xx2-xx1)*(j+.5)/steps,lo+(hi-lo)*(j+1)/steps,abs(xx2-xx1)/steps+.02,.045,projection*.75,cm,dep+projection/2)
      for d in [dep+.1,dep+projection-.1]:beam([xx1,lo+1,d],[xx2,hi+1,d],.035,cm)
    elif typ=='entrySteps':
     # A photographed raised entrance supplies rise. Ground supplies the lower support.
     top=component.get('topM',.65);run=component.get('runM',1.4);steps=component.get('count',4)
     for j in range(steps):
      h=top*(j+1)/steps;d=run*(1-(j+.5)/steps);block((xa+xb)/2,h/2,xb-xa,h+.05,run/steps+.02,cm,dep+d)
     if component.get('handrail'):
      for xx in [xa,xb]:beam([xx,1,dep+run],[xx,top+1,dep],.035,'charcoal')
    elif typ=='blade':""")
p.write_text(s)
# Keep roof volume nesting despite independent contour simplification.
p=Path('work/pipeline/spatial_rules.py');s=p.read_text().replace("if g.area>=minimum_area:result.append({'bottom':last,'top':level,'geometry':g})\n  last=level", "if result:g=g.intersection(result[-1]['geometry'])\n  if g.area>=minimum_area:\n   result.append({'bottom':last,'top':level,'geometry':g});last=level")
p.write_text(s)
# Do not propagate one frontage's material to an entire building's unknown surfaces.
p=Path('work/pipeline/facades.py');s=p.read_text();a=s.index('    material_profiles=');b=s.index("    mat=style",a);s=s[:a]+s[b:];s=s.replace('complete facade has no direct image fit; material may use same-building family prior','complete facade has no direct image fit; material and openings remain unverified priors')
a=s.index('       # Plinth, wall light');b=s.index('      components+=1',a);s=s[:a]+s[b:]
a=s.index('    # Downpipes:');b=s.index(' return records',a);s=s[:a]+s[b:];p.write_text(s)
# Curbs and paving seams follow the same terrain function as road surfaces.
p=Path('work/pipeline/streets.py');s=p.read_text();a=s.index(' for line in ctx[\'lines\'](curb_lines):');b=s.index(' # Score joints',a)
s=s[:a]+''' for line in ctx['lines'](curb_lines):
  # Short terrain-conforming strips avoid metre-long floating curb slabs.
  for ss in np.arange(0,line.length,.65):
   seg=LineString([line.interpolate(ss),line.interpolate(min(ss+.65,line.length))])
   surface(seg.buffer(.09,cap_style=2),'sidewalks','concrete',.26)
''' +s[b:]
s=s.replace("c,d=np.array(joint.coords[0]),np.array(joint.coords[-1]);vv=d-c;cube([*(c+d)/2,float(elev(*(c+d)/2))+.265],[np.linalg.norm(vv),.015,.008],'sidewalks','joint',math.atan2(vv[1],vv[0]))", "surface(joint.buffer(.0075,cap_style=2),'sidewalks','joint',.269)")
p.write_text(s)
