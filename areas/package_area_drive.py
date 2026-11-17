"""Attach spatial streaming for an area's connected driving route, preserving other published areas."""
import argparse
import json
from pathlib import Path
import re
import sys
import tempfile

from pyproj import Transformer

PIPELINE = Path(__file__).resolve().parents[1]/"king-street/pipeline-source"
sys.path.insert(0,str(PIPELINE))
import package_neighborhood
import package_startup
from package_drive_stream_gpu import package as pack_gpu
from package_facade_middle import publish as pack_windows
from package_route_block import write_json

def atomic_json(path, document):
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as output:
        json.dump(document,output,separators=(",",":"))
        tmp=Path(output.name)
    tmp.chmod(path.stat().st_mode & 0o777 if path.exists() else 0o644)
    tmp.replace(path)

def install(public, area, label, gpu):
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*",label):
        raise ValueError("Expected an area name")
    alias_url=f"/compare/{label}.json"
    archive_path=public/alias_url.lstrip("/")
    archive=json.loads(archive_path.read_text())
    world=json.loads((public/archive["world"].lstrip("/")).read_text())
    src=json.loads((area/"area-source-job.json").read_text())
    center=world["center"]
    proj=Transformer.from_crs(4326,src["projectedCrs"],always_xy=True)
    ox,oy=proj.transform(center["lon"],center["lat"])
    line=json.loads((area/"route-drive.geojson").read_text())
    if line["type"]!="LineString":
        raise ValueError("A connected route is required")
    coords=[]
    for lon,lat in line["coordinates"]:
        x,y=proj.transform(lon,lat)
        point=[round(x-ox,3),round(y-oy,3)]
        if not coords or coords[-1]!=point:
            coords.append(point)
    seed={"sourceWorldHash":archive["contentHash"],"routeEN":coords,"driveFromRouteStart":True}
    seed_url=write_json(public, label+"-drive-route",seed)
    if not package_startup.package(archive_path,gpu):
        raise ValueError("Unsupported startup asset")
    alias=label+"-stream"
    report=package_neighborhood.package(public,alias_url,seed_url,55,alias)
    if gpu:
        pack_gpu(public,area/"gpu-package-report.json","neighborhood-stream",alias)
    result=pack_windows(public,report["manifest"])
    latest=json.loads(archive_path.read_text())
    if latest["contentHash"]!=archive["contentHash"]:
        raise ValueError("Archive changed during stream build")
    latest["spatialIndex"]=result["immutableManifest"]
    atomic_json(archive_path,latest)
    menu_path=public/"compare/areas.json"
    menu=json.loads(menu_path.read_text()) if menu_path.exists() else {"areas":[]}
    entries=menu["areas"]
    entry={"id":label,"label":world["label"],"detail":"Driving corridor","manifest":alias_url}
    entries[:]=[item for item in entries if item.get("id")!=label]+[entry]
    atomic_json(menu_path,menu)
    report.update(result)
    (area/"stream-report.json").write_text(json.dumps(report,indent=2))

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("area",type=Path)
    parser.add_argument("public",type=Path)
    parser.add_argument("label")
    parser.add_argument("--gpu-textures",action="store_true")
    args=parser.parse_args()
    install(args.public.resolve(),args.area.resolve(),args.label,args.gpu_textures)
