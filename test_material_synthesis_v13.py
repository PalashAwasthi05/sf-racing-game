import numpy as np
import pytest
from general_world.material_synthesis import synthesize,structure_key

def example():return dict(substrate_family='brick',pattern='brick_running',finish_class='matte',tile_size_m=[1.8,.6],unit_size_m=[.225,.075],joint_width_m=.006,relief_m=.002,color_rgb=[145,70,49],joint_rgb=[139,128,108],roughness=.82)

def test_recolor_preserves_structure_and_all_relief_maps():
 a=example();b=dict(a,color_rgb=[110,65,48]);ma,_=synthesize(a,128);mb,_=synthesize(b,128)
 assert structure_key(a)==structure_key(b)
 assert np.array_equal(ma.normalTexture,mb.normalTexture)
 assert np.array_equal(ma.metallicRoughnessTexture,mb.metallicRoughnessTexture)
 assert not np.array_equal(ma.baseColorTexture,mb.baseColorTexture)

def test_structure_change_creates_new_asset():
 a=example();assert structure_key(a)!=structure_key(dict(a,unit_size_m=[.45,.075]))

@pytest.mark.parametrize('change',[{'tile_size_m':[0,1]},{'unit_size_m':[.22,.075]},{'joint_width_m':.1},{'color_rgb':[float('nan'),2,3]},{'relief_m':.3}])
def test_invalid_physical_observation_does_not_generate(change):
 with pytest.raises(ValueError):synthesize(dict(example(),**change),128)

def test_joints_have_separate_reflectance_and_periodic_edges():
 a=example();ma,_=synthesize(a,512);normal=np.asarray(ma.normalTexture,float)/255*2-1
 assert np.allclose(np.linalg.norm(normal,axis=2),1,atol=.014)
 rgb=np.asarray(ma.baseColorTexture,float)
 assert np.abs(rgb[:,0]-rgb[:,-1]).mean()<8
