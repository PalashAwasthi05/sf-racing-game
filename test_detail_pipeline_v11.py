import json
import numpy as np
import trimesh
from general_world.run import execute

def test_accepted_detail_reaches_scene_without_overlay(tmp_path,monkeypatch):
 import general_world.run as module
 def host(*args):
  m=trimesh.creation.box([6,4,4]);m.apply_translation([3,2,2]);return m,{'watertight':True,'status':'closed-proxy-solid'}
 monkeypatch.setattr(module,'reconstruct_building',host)
 np.savez(tmp_path/'ground.npz',points=np.array([[0,0,0],[6,0,0],[0,4,0],[6,4,0]]));np.savez(tmp_path/'roof.npz',points=np.array([[1,1,4]]))
 images={str(i):dict(imageSha256=str(i)*64,captureDate='2026-09-01',camera={'position':[i,-3,2],'worldRegistrationValidated':True},viewGroup=str(i)) for i in range(2)}
 detail=dict(hostKey='opaque',producer={'mode':'automatic','inputHash':'test-input'},sourceIds=['0','1'],confidence=.99,label='wall',wall_polygon=[[0,0],[6,0],[6,4],[0,4]],wall_thickness_m=.3,coordinate_frame={'origin':[0,0,0],'horizontal':[1,0,0],'inward':[0,1,0],'vertical':[0,0,1]},apertures=[{'kind':'window','boundary_polygon':[[1,1],[3,1],[3,3],[1,3]],'frame_width_m':.1}])
 data=dict(schema='normalized-world-input/v1',snapshotDate='2026-09-09',groundFile='ground.npz',surfacePointsFile='roof.npz',buildings=[{'key':'opaque','geometry':{'type':'Polygon','coordinates':[[[0,0],[6,0],[6,4],[0,4],[0,0]]]},'heightM':4}],images=images,derivedDetails=[detail])
 path=tmp_path/'input.json';path.write_text(json.dumps(data));r=execute(path,tmp_path/'result')
 assert r['facadesReconstructedFromImages']==1 and r['detailVerdicts'][0]['compiled']
 assert r['publicationApprovedByQualityGate'] is False
 scene=trimesh.load(tmp_path/'result/candidate.glb',force='scene',process=False)
 assert scene.geometry['solid_opaque'].volume<96
 assert any(m.metadata.get('component','').endswith('glazing') for m in scene.geometry.values())
