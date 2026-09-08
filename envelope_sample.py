exec(open('work/inspect_geometry.py').read().split('for c in candidates:\n if np.linalg')[0])
queries=[('149253142',[.707,.707]),('149253142',[.707,-.707]),('679290110',[.707,.707]),('679290110',[-.707,.707]),('sf_201006_0000003',[-.707,.707]),('sf_201006_0000003',[-.707,-.707])]
res=[]
for id,normal in queries:
 c=next(c for c in candidates if c['id']==id or (c['osm'] and c['osm']['id']==id));g=c['g'];n=np.array(normal);n/=np.linalg.norm(n);u=np.array([-n[1],n[0]])
 edges=[]
 for p in polys(g):
  coords=np.array(p.exterior.coords)
  for a,b in zip(coords,coords[1:]):
   v=b-a;le=np.linalg.norm(v)
   if le>3 and abs(v/le@u)>.94:edges.append((le,(a+b)/2@n,min(a@u,b@u),max(a@u,b@u)))
 out=max(e[1] for e in edges);sel=[e for e in edges if e[1]>out-3];left=min(e[2] for e in sel);right=max(e[3] for e in sel)
 sample=get_points(g);x=(sample[:,:2]@u-left)/(right-left);d=sample[:,:2]@n-out;zs=[]
 for a,b in zip(np.arange(0,1,.1),np.arange(.1,1.1,.1)):
  vals=sample[(x>=a)&(x<b)&(d>-9)&(d<1)&(sample[:,2]>6),2]
  hs,es=np.histogram(vals,np.arange(6,60,.5));zs.append({'range':[round(a,2),round(b,2)],'points':len(vals),'modes':sorted([(int(h),float(e)) for h,e in zip(hs,es[:-1])],reverse=True)[:4]})
 r={'match':id,'normal':normal,'left':left,'right':right,'plane':out,'width':right-left,'edges':[list(e) for e in sorted(edges,reverse=True)[:12]],'heightSamples':zs};res.append(r)
 print(json.dumps(r))
Path('work/envelope-samples.json').write_text(json.dumps(res,indent=2))
