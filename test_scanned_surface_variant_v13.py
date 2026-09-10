import numpy as np
import pytest
from PIL import Image
from general_world.scanned_surface_variant import running_bond_scale

def scan():
 rng=np.random.default_rng(8); a=np.full((192,256),.72)
 for y in range(0,192,24):
  a[y:y+24,(y//24%2)*32::64]=.1;a[y:y+2,:]=.1
 return Image.fromarray(np.uint8(np.clip(a+rng.normal(0,.02,a.shape),0,1)*255))

def test_measured_modules_rescale_without_changing_bond():
 image=scan();scale,e=running_bond_scale(image,image,[1.0,.6],[.2375,.075])
 assert np.allclose(scale,[.95,.6]);assert np.allclose(e['moduleMeasurementM'],[.25,.075])

def test_unmeasurable_scan_cannot_supply_target_modules():
 with pytest.raises(ValueError):running_bond_scale(Image.new('L',(256,192),128),Image.new('L',(256,192),128),[1,.6],[.225,.075])
