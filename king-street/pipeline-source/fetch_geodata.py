import urllib.request, urllib.parse, json, pathlib, concurrent.futures
ROOT=pathlib.Path(__file__).parent/'data'; ROOT.mkdir(exist_ok=True)
LAT,LON=37.7786154,-122.3912719
BBOX=(LAT-.003,LON-.004,LAT+.003,LON+.004)
def fetch(name,url,data=None):
    dest=ROOT/(name+'.json')
    if dest.exists(): return name, 'cached',len(dest.read_bytes())
    try:
        req=urllib.request.Request(url,data=data,headers={'User-Agent':'KingStreetResearchPilot/0.1'})
        raw=urllib.request.urlopen(req,timeout=90).read()
        parsed=json.loads(raw); dest.write_bytes(raw)
        return name, len(parsed) if isinstance(parsed,list) else len(parsed.get('elements',[])),len(raw)
    except Exception as e:return name,'ERROR',str(e)
def city(name,dataset,field):
    south,west,north,east=BBOX
    params={'$where':f'within_box({field},{north},{west},{south},{east})','$limit':'5000'}
    return fetch(name,'https://data.sfgov.org/resource/'+dataset+'.json?'+urllib.parse.urlencode(params))
jobs=[('buildings','ynuv-fyni','shape'),('trees','tkzw-k3nq','location'),('streets','3psu-pn9h','line'),('signs','m48z-6ji4','shape'),('row','h8n7-e4ns','the_geom')]
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
    pending=[pool.submit(city,*j) for j in jobs]
    for future in concurrent.futures.as_completed(pending):print(future.result(),flush=True)
print(fetch('ept','https://s3-us-west-2.amazonaws.com/usgs-lidar-public/CA_SanFrancisco_1_B23/ept.json'),flush=True)

osm_dest=ROOT/'osm.xml'
if not osm_dest.exists():
    req=urllib.request.Request('https://api.openstreetmap.org/api/0.6/map?bbox=-122.3953,37.7756,-122.3872,37.7817',headers={'User-Agent':'KingStreetResearchPilot/0.1'})
    with urllib.request.urlopen(req,timeout=90) as response:
        osm_dest.write_bytes(response.read())
print('OSM XML',osm_dest.stat().st_size,flush=True)
