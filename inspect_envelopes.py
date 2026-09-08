exec(open('work/inspect_geometry.py').read().split('for c in candidates:\n if np.linalg')[0])
report=[]
for c in candidates:
 t=c['osm']['tags'] if c['osm'] else {};g=c['g']
 if g.distance(Point(15,-60))>155:continue
 record={'id':c['id'],'osmId':c['osm']['id'] if c['osm'] else None,'tags':t,'area':round(g.area),'centroid':np.round(g.centroid.coords[0],1).tolist(),'rings':[np.round(p.exterior.coords,2).tolist() for p in polys(g)]}
 report.append(record)
Path('work/envelope-footprints.json').write_text(json.dumps(report,indent=2))
for r in report:print(r['id'],r['osmId'],r['tags'].get('name',r['tags'].get('addr:housenumber','')),r['area'],r['centroid'])
