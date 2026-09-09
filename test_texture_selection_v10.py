import numpy as np
from PIL import Image
from general_world.texture_selection import select_texture,recolor

def asset(**kw):return dict(substrate_family='plaster',pattern='plaster_grain',finish_class='matte',strong_weathering=False,intrinsic_color=False,tile_size_m=[2,2],mean_rgb=[180,180,180],**kw)
def target():return dict(substrate_family='plaster',pattern='plaster_grain',finish_class='matte',color_rgb=[180,180,180])
def test_reuse():assert select_texture(target(),{'a':asset()})['action']=='reuse'
def test_recolor():
 t=target();t['color_rgb']=[120,60,50];assert select_texture(t,{'a':asset()})['action']=='recolor'
def test_structure_beats_color():
 a=asset();a['pattern']='stone_blocks';assert select_texture(target(),{'a':a})['action']=='request_new'
def test_weathering_not_color():
 a=asset();a['strong_weathering']=True;assert select_texture(target(),{'a':a})['action']=='request_new'
def test_requests_deduplicate_colors():
 t=target();x=select_texture(t,{})['request_key'];t['color_rgb']=[0,0,0];assert select_texture(t,{})['request_key']==x

def test_unknown_scale():
 a=asset();a['tile_size_m']=None;assert select_texture(target(),{'a':a})['action']=='request_new'
 t=target();t['provisional_repeat_m']=[2,2];assert 'provisional' in select_texture(t,{'a':a})['scale_provenance']
def test_joints_need_unit_scale():
 t=target();t['unit_size_m']=[.2,.1];assert select_texture(t,{'a':asset()})['action']=='request_new'
def test_recolor_keeps_detail_and_target():
 x=np.tile(np.arange(100,201,dtype=np.uint8)[None,:,None],(20,1,3));y=np.asarray(recolor(Image.fromarray(x),[100,70,50]));assert (np.diff(y[0,:,0].astype(float))>=0).all();assert y[:,:,0].std()>10
 assert np.max(abs(y.mean((0,1))-[100,70,50]))<8

def test_recolor_preserves_mortar():
 x=np.array([[[180,170,160],[160,70,40],[140,60,30]]],np.uint8);y=np.asarray(recolor(Image.fromarray(x),[100,50,30],True));assert np.array_equal(x[0,0],y[0,0])

def test_bad_observation():
 t=target();t['color_rgb']=[float('nan'),0,0];assert select_texture(t,{'a':asset()})['action']=='retain_unresolved'

def test_order_independent():
 assert select_texture(target(),{'b':asset(),'a':asset()})==select_texture(target(),{'a':asset(),'b':asset()})
