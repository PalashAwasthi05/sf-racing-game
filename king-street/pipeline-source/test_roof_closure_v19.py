"""Roof continuity rules: a skin box that rises above the host cap gets a lid, a skin that meets the cap keeps it,
and tier outlines snap to their parent so no cell-wide cap slivers or pole-thin tier parts survive."""
import numpy as np
from shapely.geometry import box
from rasterio.transform import from_origin
from pipeline import envelopes
from pipeline.spatial_rules import roof_bands
from test_envelope_closure_v14 import compile_profile, HOST_HEIGHT


def test_cap_kept_under_a_skin_that_meets_it_and_a_lid_closes_a_skin_above_it():
    resolved, _ = compile_profile()
    tall = [(slot, z1) for slot, z1 in envelopes.lids(resolved, HOST_HEIGHT)]
    assert len(tall) == 1 and tall[0][1] > HOST_HEIGHT, tall              # the 12 m end rises above the 9 m cap: it gets a lid
    assert not envelopes.roof_mask(resolved, 12.0).contains(tall[0][0].representative_point())   # at a 12 m host the same skin only meets the cap: the cap stays


def test_tier_edges_snap_to_the_parent_and_thin_parts_are_dropped():
    # A 45 x 30 m roof (1.5 m cells) with a raised block whose outline drifts by one cell against the parent tier,
    # plus a single-cell spike and a one-cell-wide finger attached to the block.
    quant = np.zeros((20, 30), np.int16)
    quant[4:12, 6:18] = 4            # block, 12 x 18 m
    quant[6:10, 8:16] = 8            # upper tier inside it
    quant[12, 6:18] = 4              # a one-cell strip along the block edge (a drifted outline)
    quant[15, 25] = 6                # a lone spike
    quant[4:12, 18] = 4;             # one-cell-wide finger
    bands = roof_bands(quant, from_origin(0, 30, 1.5, 1.5), box(0, 0, 45, 30), minimum_area=25, simplify_m=1.125, cell_m=1.5)
    assert [b['top'] for b in bands] == [4, 8], [b['top'] for b in bands]
    for b in bands:
        for part in getattr(b['cap'], 'geoms', [b['cap']]):
            if part.is_empty: continue
            assert part.buffer(-.7).area > 0 or part.area >= 9, ('cap sliver', part.area, part.bounds)   # nothing narrower than a cell
        for part in getattr(b['geometry'], 'geoms', [b['geometry']]):
            assert part.area >= 9, ('pole-thin tier part', part.area)
