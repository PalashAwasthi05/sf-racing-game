"""Normalize provider evidence for an external area job; never claim registration.

Camera poses are checked against the matching reconstruction shot when possible.
This checks provider consistency, not agreement with independently measured walls.
"""
import argparse, json, hashlib, urllib.request, zlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import numpy as np
from scipy.spatial.transform import Rotation
from pyproj import Transformer


def enu_basis(lat,lon):
    lat,lon=np.radians([lat,lon])
    return np.array([[-np.sin(lon),np.cos(lon),0],[-np.sin(lat)*np.cos(lon),-np.sin(lat)*np.sin(lon),np.cos(lat)],[np.cos(lat)*np.cos(lon),np.cos(lat)*np.sin(lon),np.sin(lat)]])


def main():
    p=argparse.ArgumentParser();p.add_argument('--imagery',type=Path,required=True);p.add_argument('--normalized-input',type=Path,required=True);p.add_argument('--as-of',required=True);args=p.parse_args()
    root=args.imagery;d=json.loads((root/'selected.json').read_text());records=d['images'];cache=root/'clusters';cache.mkdir(exist_ok=True)
    recent=[r for r in records if (datetime.fromisoformat(args.as_of).replace(tzinfo=timezone.utc)-datetime.fromtimestamp(r['captured_at']/1000,timezone.utc)).days<=730]
    clusters={r['sfm_cluster']['id']:r['sfm_cluster']['url'] for r in recent if r.get('sfm_cluster',{}).get('url')}
    def download(pair):
        cid,url=pair;dest=cache/(cid+'.json')
        try:
            if not dest.exists():
                with urllib.request.urlopen(url,timeout=50) as r:raw=r.read(80_000_001)
                if len(raw)>80_000_000:return cid,'oversize'
                content=json.loads(zlib.decompress(raw));dest.write_text(json.dumps(content,separators=(',',':')))
            return cid,'ok'
        except Exception:return cid,'download-or-decode-failed'
    with ThreadPoolExecutor(max_workers=4) as pool:
        results=list(pool.map(download,clusters.items()))
    print(json.dumps({'recentImages':len(recent),'clusters':len(results),'clustersDownloaded':sum(status=='ok' for _,status in results)}),flush=True)
    loaded={cid:json.loads((cache/(cid+'.json')).read_text())[0] for cid,status in results if status=='ok'}
    source={};geocentric=Transformer.from_crs(4979,4978,always_xy=True)
    for rec in records:
        image=Path(rec['local_image']);cid=rec.get('sfm_cluster',{}).get('id');cluster=loaded.get(cid)
        capture=datetime.fromtimestamp(rec['captured_at']/1000,timezone.utc).date().isoformat()
        camera={'model':rec.get('camera_type'),'parameters':rec.get('camera_parameters',[]),'rotationAngleAxis':rec.get('computed_rotation'),'locationWgs84':rec.get('computed_geometry',{}).get('coordinates'),'altitudeM':rec.get('computed_altitude'),'worldRegistrationValidated':False,'providerPoseConsistent':False}
        if cluster and cluster.get('shots'):
            # Numeric image ID and internal reconstruction shot key differ.
            # Match capture timestamp + rotation, then independently check center.
            shots=[s for s in cluster['shots'].values() if abs(s.get('capture_time',-1)-rec['captured_at']/1000)<.002 and np.linalg.norm(np.asarray(s['rotation'])-rec['computed_rotation'])<1e-5]
            if len(shots)==1:
                shot=shots[0];R=Rotation.from_rotvec(shot['rotation']).as_matrix();center=-R.T@shot['translation'];ref=cluster['reference_lla'];ecef0=np.array(geocentric.transform(ref['longitude'],ref['latitude'],ref['altitude']));basis=enu_basis(ref['latitude'],ref['longitude']);lon,lat=camera['locationWgs84'];ecef=np.array(geocentric.transform(lon,lat,camera['altitudeM']));given=basis@(ecef-ecef0);err=float(np.linalg.norm(given-center))
                camera.update(providerPoseConsistent=err<1.,providerCenterResidualM=err,rotationAngleAxis=shot['rotation'],translation=shot['translation'],referenceLla=ref,orientation=shot.get('orientation'),clusterId=cid)
        source['mapillary:'+rec['id']]={'provider':'Mapillary','sourceUrl':'https://www.mapillary.com/app/?pKey='+rec['id'],'imagePath':str(image.resolve()),'imageSha256':hashlib.sha256(image.read_bytes()).hexdigest(),'captureDate':capture,'viewGroup':str(rec.get('sequence')),'camera':camera,'admission':'pose-registration-required' if camera['providerPoseConsistent'] else 'provider-pose-unverified'}
    (root/'source-index.json').write_text(json.dumps(source,indent=2))
    normalized=json.loads(args.normalized_input.read_text());normalized['images']=source
    # Resolve existing file paths before writing alongside the imagery manifest.
    for k in ('surfacePointsFile','groundFile'):
        if normalized.get(k):normalized[k]=str((args.normalized_input.parent/normalized[k]).resolve())
    (root/'input-with-imagery.json').write_text(json.dumps(normalized,separators=(',',':')))
    report={'discoveredImages':len(json.loads((root/'coverage.json').read_text())['images']),'selectedImages':len(source),'recentImagesWithin730Days':len(recent),'panoramas':sum(r.get('is_pano',False) for r in records),'providerPosesConsistent':sum(r['camera']['providerPoseConsistent'] for r in source.values()),'worldRegistrationsValidated':0,'facadeMeasurementsGenerated':0,'oldestSelectedCapture':min(r['captureDate'] for r in source.values()),'newestSelectedCapture':max(r['captureDate'] for r in source.values()),'asOf':args.as_of,'limitations':['Spatial selection includes historical-only cells; old imagery is not admitted as current evidence.','Provider pose checks do not validate alignment to the independently measured world.','No facade geometry has been derived from these images yet.']}
    (root/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':main()
