"""Fetch one configured corridor's public map inputs sequentially, then derive and clip its AOI.
Run with the project's geospatial Python environment. Network access required.
"""
import argparse,json,pathlib,sys,subprocess,shutil
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
import area_fetch
p=argparse.ArgumentParser();p.add_argument('area');a=p.parse_args();area=pathlib.Path(a.area);data=area/'source-cache';data.mkdir(parents=True,exist_ok=True);source=json.load(open(area/'corridor-source.json'));box=tuple(source['fetchBoxSouthWestNorthEast'])
for row in area_fetch.CITY:print(area_fetch.city(data,*row,box),flush=True)
for name,dataset,_ in area_fetch.CITY:
 if name in area_fetch.META:area_fetch.meta(data,name,dataset)
print(area_fetch.city(data,'city-blocks','p2fw-hsrv','multipolygon',box),flush=True)
print('osm bytes',area_fetch.osm(data,box),flush=True)
target=area/'data';target.mkdir(exist_ok=True)
for f in data.iterdir():
 if f.is_file():shutil.copy2(f,target/f.name)
for script in ['build_corridor_route.py','trim_route_to_frontage.py','prepare_corridor.py','clip_corridor_inputs.py']:subprocess.run([sys.executable,str(pathlib.Path(__file__).parent/script),str(area)],check=True)
