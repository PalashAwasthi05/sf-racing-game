import json
from types import SimpleNamespace
import numpy as np
from regen.photo_inputs import photo_context,facade_walls

def test_compact_context_and_wall_fields(tmp_path):
    world=tmp_path/'world.json'
    world.write_text(json.dumps({'center':{'lon':3,'lat':4},'grid':{'min':-1,'step':2,'n':2},'ground':[1,2,3,4],
      'buildings':[{'id':'a','center':[1,2,3],'unused':[99]*100}],'other':[1]*100}))
    fc=SimpleNamespace(WORLD_PATH=world,LocalFrame=lambda lon,lat:(lon,lat))
    buildings,frame=photo_context(fc)
    assert frame==(3,4) and buildings=={'a':{'id':'a','center':[1,2,3]}}
    np.testing.assert_array_equal(fc._GROUND[3],[[1,2],[3,4]])
    evidence=tmp_path/'evidence.json'
    evidence.write_text(json.dumps({'facadeWalls':[{'buildingId':'a','zBaseM':0,'lengthM':4,'unused':[1]*100},
       {'buildingId':'a','zBaseM':10,'lengthM':3}, {'buildingId':'other','zBaseM':0}]}))
    assert list(facade_walls(evidence,buildings))==[{'buildingId':'a','zBaseM':0,'lengthM':4}]
    assert list(facade_walls(evidence,buildings,tiers=True))==[{'buildingId':'a','zBaseM':10,'lengthM':3}]

def test_isolated_decode_and_balance_preserve_pixels(tmp_path,monkeypatch):
    from PIL import Image
    from panorama_balance import load_balanced_pano, road_gain
    monkeypatch.setenv('SFW_AREA',str(tmp_path))
    source=tmp_path/'photo.jpg'
    pixels=np.random.default_rng(8).integers(0,256,(180,360,3),dtype=np.uint8)
    Image.fromarray(pixels).save(source,quality=92)
    with Image.open(source) as p:
        p.draft('RGB',(160,80));p.thumbnail((160,80));eager=np.asarray(p.convert('RGB'),dtype=np.float32)
    expected=np.clip(eager*road_gain(eager),0,255).astype(np.uint8)
    np.testing.assert_array_equal(load_balanced_pano(source,(160,80)),expected)
    np.testing.assert_array_equal(load_balanced_pano(source,(160,80)),expected)
