from pathlib import Path
s=Path('work/build_world.py').read_text().split('for c in candidates:\n id,g=c')[0]
s=s.replace("ROOT=pathlib.Path(__file__).parent", "ROOT=pathlib.Path('work').resolve()")
exec(compile(s,'work/build_world.py','exec'))
for c in candidates:
 if np.linalg.norm(c['g'].centroid.coords[0])>120:continue
 t=c['osm']['tags'] if c['osm'] else {};g=c['g'];v=get_points(g.buffer(-1));v=v[v[:,2]>4];p95=np.quantile(v[:,2],.95) if len(v) else 0
 print(c['id'],c['osm']['id'] if c['osm'] else '',t.get('name',t.get('addr:housenumber','')),round(g.area),round(p95,1),np.round(g.centroid.coords[0],1).tolist(), list(g.exterior.coords)[:3] if g.geom_type=='Polygon' else '')
for w in ways:
 if w['tags'].get('name') in ['King Street','3rd Street','Townsend Street'] and 'highway' in w['tags'] and w['g'].distance(Point(0,0))<130:
  print(w['tags']['name'],w['id'],np.round(list(w['g'].coords),1).tolist())
for c in candidates:
 if c['osm'] and c['osm']['id']=='149167949':
  g=c['g'];p=get_points(g.buffer(-2));v=p[p[:,2]>4,2];h,e=np.histogram(v,np.arange(4,46,.5));print('HEIGHT MODES',c['id'],sorted(zip(h,e[:-1]),reverse=True)[:12])
