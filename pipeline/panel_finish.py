"""Separate a facade panel's substrate from its visible finish.
An explicitly bare/anodized/brushed metal keeps its metallic response. When
only a strongly pigmented or near-white swatch is known, an opaque coating
is a provisional optical prior; never label the paint type as measured.
"""
import re

def panel_recipe(color,finish=None,coating=None):
    if coating in ('exposed','bare','uncoated') or finish in ('brushed','polished','anodized'):return 'panel_silver'
    if coating in ('painted','powder-coated','opaque'):return 'panel_coated'
    if not isinstance(color,str) or not re.fullmatch(r'#[0-9a-fA-F]{6}',color):return 'panel_silver'
    rgb=bytes.fromhex(color[1:]);lo=min(rgb);hi=max(rgb)
    return 'panel_coated' if lo>=205 or (hi-lo>=44 and (hi-lo)/max(1,hi)>.22) else 'panel_silver'
