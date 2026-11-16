import numpy as np
import pytest
from shapely.geometry import box
from regen.specs import to_profile
from pipeline import facade_detail


def profile(depth=.1, height=.45, **extra):
    piece={'id':'test-side','building':'test-building','lengthM':10.,'heightM':12.,
           'normal':[0.,1.],'frameU':[0.,10.],'planeV':0.,'views':[{'image':'test-image'}]}
    reading={'balconies':[{'x0':2.,'x1':3.4,'zM':5.,'depthM':depth,
                          'railingHeightM':height,'railing':'balustrade',**extra}]}
    return to_profile(reading,piece)


def render(p):
    blocks=[];beams=[]
    ctx={'block':lambda *a,**kw:blocks.append(a),'beam':lambda *a:beams.append(a),
         'prism':lambda *a:None,'clip':box(0,0,10,12),'low':0.,'high':10.,
         'W':10.,'Hs':12.,'H':12.,'mat':'wall','wall_mat':lambda x,z:'wall'}
    facade_detail.wall(ctx,p,[])
    return blocks,beams


def test_measured_guard_does_not_become_a_walkable_balcony():
    p=profile();guard=p['balconies'][0]
    assert guard['depthM']==.1 and guard['railingHeightM']==.45
    blocks,beams=render(p)
    assert beams and blocks
    assert all(b[5]!='wall' for b in blocks)  # no invented deck
    # Every infill member stays within the measured shallow projection.
    assert all(b[6]+b[4]/2 <= .1+1e-9 for b in blocks)
    assert all(b[1]-b[3]/2 >= 5.-1e-4 for b in blocks)


def test_real_balcony_retains_deck_and_return_rails():
    blocks,beams=render(profile(.85,1.05))
    assert any(b[5]=='wall' and b[4]==.85 for b in blocks)
    assert any(abs(a[2]-b[2])>.5 for a,b,*_ in beams)


def test_explicit_juliet_without_deck_preserves_wider_projection():
    blocks,beams=render(profile(.4,.75,hasDeck=False))
    assert all(b[5]!='wall' for b in blocks)
    assert all(abs(a[2]-b[2])<1e-9 for a,b,*_ in beams)


@pytest.mark.parametrize('depth,height',[(float('nan'),1.),(.1,float('inf')),(-.1,.5)])
def test_invalid_measurement_does_not_silently_become_default_balcony(depth,height):
    assert not profile(depth,height).get('balconies')
