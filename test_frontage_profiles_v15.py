"""Observed frontage profiles: repeat banks by pitch in metres follow each wall piece's measured height."""
import numpy as np
from pipeline import facade_grammar

PROFILE={'id':'pitch-test','match':{'osmWayId':None,'featureId':'x'},'normal':[0.,1.],'sourceIds':['s1'],'material':'render_white',
         'groups':[{'region':[0.,.2,1.,1.],'columnPitchM':4.,'rowPitchM':3.,'style':{'kind':'window','openingRatio':.6,'openingHeightRatio':.55,'mullionFractions':[],'transomFractions':[]}},
                   {'region':[0.,0.,1.,.15],'columns':3,'rows':1,'maxHeightM':12.,'style':{'kind':'shopfront'}},
                   {'region':[0.,0.,1.,.1],'columns':2,'rows':1,'minHeightM':30.,'style':{'kind':'rollup'}}]}

def rows_cols(ops):
    return len({o['iy'] for o in ops}),len({o['ix'] for o in ops})

def test_rows_follow_the_measured_height_and_columns_the_width():
    ops=[o for o in facade_grammar.openings(PROFILE,[],W=40.,H=9.) if o['style'].get('kind')=='window']
    assert rows_cols(ops)==(2,10),rows_cols(ops)          # .8*9/3 = 2.4 -> 2 rows; 40/4 = 10 columns
    ops=[o for o in facade_grammar.openings(PROFILE,[],W=40.,H=48.) if o['style'].get('kind')=='window']
    assert rows_cols(ops)==(13,10),rows_cols(ops)         # .8*48/3 = 12.8 -> 13 rows

def test_height_limited_banks_exist_only_on_pieces_of_that_height():
    kinds=lambda H:{o['style'].get('kind') for o in facade_grammar.openings(PROFILE,[],W=40.,H=H)}
    assert kinds(9.)=={'window','shopfront'}
    assert kinds(48.)=={'window','rollup'}

def test_height_limits_gate_on_the_wall_piece_not_the_frame():
    # a 48 m frame read on a tower and its 9 m podium: the podium piece keeps its shopfront bank and loses the rollups
    kinds=lambda H,wallH:{o['style'].get('kind') for o in facade_grammar.openings(PROFILE,[],W=40.,H=H,wallH=wallH)}
    assert kinds(48.,9.)=={'window','shopfront'}
    assert kinds(48.,48.)=={'window','rollup'}

def test_counts_without_pitch_are_unchanged():
    plain={'id':'plain','match':{'osmWayId':None},'normal':[0.,1.],'sourceIds':[],'groups':[{'region':[0,0,1,1],'columns':4,'rows':3}]}
    assert rows_cols(facade_grammar.openings(plain,[],W=10.,H=10.))==(3,4)
    assert rows_cols(facade_grammar.openings(plain,[]))==(3,4)


def _ctx(records):
    from shapely.geometry import Polygon
    def cube(center, size, key, material, angle=0): records.append(('cube', key, material, tuple(round(float(v), 3) for v in size), tuple(round(float(v), 3) for v in center)))
    def cylinder(a, b, r, key, material, sections=8): records.append(('cyl', key, material))
    def put(mesh, key, material, uv=None): records.append(('put', key, material, int(len(mesh.faces))))
    def polys(g): return [g] if g.geom_type == 'Polygon' else list(g.geoms)
    return {'cube': cube, 'cylinder': cylinder, 'put': put, 'polys': polys}

def _wall():
    from shapely.geometry import box
    g = box(0, 0, 20, 10)   # frontage along x at y=0, normal -y (facing south)
    return {'g': g, 'id': 'x', 'osm': None}, np.array([0., 0.]), np.array([20., 0.])

def test_arched_heads_fill_the_corners_with_wall_and_run_the_frame_along_the_arc():
    records = []; c, a, b = _wall()
    profile = {'id': 'arch', 'match': {'osmWayId': None}, 'normal': [0., -1.], 'sourceIds': ['s'], 'material': 'brick_modular',
               'groups': [{'region': [.1, .1, .9, .9], 'columns': 2, 'rows': 1, 'style': {'head': 'arch', 'springRatio': .7, 'mullionFractions': [], 'transomFractions': []}}]}
    out = facade_grammar.compile_wall(_ctx(records), c, a, b, 0., 12., profile, [])
    assert out and out['record']['openingCount'] == 2
    prisms = [r for r in records if r[0] == 'put' and r[1].startswith('buildings__') and r[2] == 'brick_modular']
    beams = [r for r in records if r[0] == 'put' and r[1].startswith('facades__') and r[2] == 'charcoal']
    assert len(prisms) >= 2, prisms          # one corner infill per arched opening
    rings = [r for r in beams if r[3] >= 40]
    assert len(rings) >= 2, beams           # each arched opening has one frame ring that follows the arc (many faces), since v22

def test_piers_stand_at_bay_boundaries_and_spandrels_between_rows():
    records = []; c, a, b = _wall()
    profile = {'id': 'piers', 'match': {'osmWayId': None}, 'normal': [0., -1.], 'sourceIds': ['s'], 'material': 'render_gray',
               'groups': [{'region': [0., .3, 1., 1.], 'columns': 4, 'rows': 2, 'pierWidthM': .5, 'pierProjectionM': .12, 'pierMaterial': 'stone',
                           'spandrelMaterial': 'panel_dark', 'style': {'openingRatio': .6, 'openingHeightRatio': .5, 'mullionFractions': [], 'transomFractions': []}}]}
    out = facade_grammar.compile_wall(_ctx(records), c, a, b, 0., 12., profile, [])
    piers = [r for r in records if r[0] == 'cube' and r[2] == 'stone' and abs(r[3][0] - .5) < 1e-6]
    spandrels = [r for r in records if r[0] == 'cube' and r[2] == 'panel_dark']
    assert len(piers) == 5, len(piers)        # 4 bays -> 5 boundaries
    assert len(spandrels) == 4, len(spandrels)  # one strip per column between the two rows

def test_band_section_sweeps_a_profile_and_rollup_takes_its_colour():
    records = []; c, a, b = _wall()
    profile = {'id': 'band', 'match': {'osmWayId': None}, 'normal': [0., -1.], 'sourceIds': ['s'], 'material': 'render_gray',
               'bands': [{'height': .95, 'material': 'stone', 'section': [[0, .12, .2], [.12, .1, .35], [.22, .08, .5]]}],
               'groups': [{'region': [.1, 0., .5, .4], 'columns': 1, 'rows': 1, 'style': {'kind': 'rollup', 'material': 'greenpaint'}}]}
    facade_grammar.compile_wall(_ctx(records), c, a, b, 0., 12., profile, [])
    steps = [r for r in records if r[0] == 'cube' and r[2] == 'stone' and abs(r[3][0] - 20.) < 1e-3]
    assert len(steps) >= 3, steps
    assert any(r[0] == 'cube' and r[2] == 'greenpaint' for r in records)

def test_observed_colour_makes_a_named_variant_of_the_recipe():
    from pipeline import surface_evidence, material_recipes
    name = surface_evidence.resolve_material({'material': 'stucco', 'materialEvidence': {'family': 'render', 'finish': 'gray', 'colorRgb': [70, 74, 78], 'sourceIds': ['s']}})
    assert name == 'render_gray#464a4e' and material_recipes.RECIPES[name]['color'] == [70, 74, 78]
    assert material_recipes.RECIPES[name]['rough'] == material_recipes.RECIPES['render_gray']['rough']

def test_unobserved_walls_take_the_observed_frontage_material():
    from pipeline import facades
    profiles = [{'id': 'p', 'match': {'osmWayId': None, 'featureId': 'b1'}, 'normal': [0., 1.], 'sourceIds': ['s'], 'materialEvidence': {'family': 'metal-panel', 'finish': 'dark', 'sourceIds': ['s']}}]
    assert facades.observed_material(profiles, {'id': 'b1', 'osm': None}) == 'panel_dark'
    assert facades.observed_material(profiles, {'id': 'b2', 'osm': {'id': '5'}}) is None

def test_any_material_name_with_a_hex_suffix_is_a_colour_variant_of_its_recipe():
    import pathlib
    from pipeline import materials, material_recipes
    bank = materials.Materials(pathlib.Path(__file__).resolve().parent / 'material-assets', {'glass': '#4b7686'})
    mat, scale = bank.get('brick_modular#b45a3c')
    assert mat.name == 'brick_modular#b45a3c' and material_recipes.RECIPES['brick_modular#b45a3c']['color'] == [180, 90, 60]
    assert list(scale) == list(material_recipes.RECIPES['brick_modular']['scale'])

def test_measured_readings_are_anchored_to_the_visible_corners():
    import importlib.util, pathlib, json
    spec = importlib.util.spec_from_file_location('wfo', pathlib.Path(__file__).resolve().parent / 'write_frontage_observations_v15.py')
    src = open(spec.origin).read().split('profiles=[];sources={};skipped=[]')[0]
    ns = {}; exec(src.split('WORK=')[0] + src[src.index('EYE_M='):], ns)
    content = {'unitsM': True, 'visibleLeftM': 2.0, 'groups': [{'region': [3.0, 1.0, 7.0, 4.0], 'xCentersM': [4.0, 6.0]}], 'bands': [{'height': 8.0}], 'piers': [{'x': 12.0}]}
    out = ns['to_fractions'](content, 20.0, 10.0)
    r = out['groups'][0]['region']
    assert [round(v, 3) for v in r] == [0.05, 0.1, 0.25, 0.4], r
    assert [round(v, 3) for v in out['groups'][0]['xCenters']] == [0.25, 0.75]
    assert round(out['bands'][0]['height'], 3) == 0.8 and round(out['piers'][0]['x'], 3) == 0.5


def test_measured_fractions_land_at_their_metres_whatever_the_wall_height():
    """A profile read against an 8 m image places an opening whose top was measured at 4.4 m at 4.4 m even when
    the pipeline's wall came out 6.7 m tall; an opening measured above the wall is clipped, not rescaled."""
    records = []; c, a, b = _wall()
    profile = {'id': 'ref', 'match': {'osmWayId': None}, 'normal': [0., -1.], 'sourceIds': ['s'], 'material': 'render_gray', 'heightRefM': 8.0,
               'groups': [{'region': [.1, 0., .3, .55], 'columns': 1, 'rows': 1, 'style': {'kind': 'entry', 'openingHeightRatio': 1., 'mullionFractions': [], 'transomFractions': []}},
                          {'region': [.5, .9, .7, 1.], 'columns': 1, 'rows': 1, 'style': {'mullionFractions': [], 'transomFractions': []}}],
               'bands': [{'height': .98, 'material': 'stone'}]}
    facade_grammar.compile_wall(_ctx(records), c, a, b, 0., 6.7, profile, [])
    glass = [r for r in records if r[0] == 'cube' and r[2] == 'glass']
    tops = sorted(round(r[4][2] + r[3][2] / 2, 2) for r in glass)
    assert tops and abs(tops[0] - 4.4) < .05, tops            # 0.55 * 8 m, not 0.55 * 6.7 m
    assert len(glass) == 1                                      # the opening measured at 7.2-8 m lies above the 6.7 m wall
    assert not any(r[0] == 'cube' and r[2] == 'stone' and r[4][2] > 7 for r in records)   # band at 7.84 m is above the wall


def _builder_ns():
    import importlib.util, pathlib
    spec = importlib.util.spec_from_file_location('wfo', pathlib.Path(__file__).resolve().parent / 'write_frontage_observations_v15.py')
    src = open(spec.origin).read().split('profiles=[];sources={};skipped=[]')[0]
    ns = {}; exec(src.split('WORK=')[0] + src[src.index('EYE_M='):], ns); return ns

def test_each_axis_takes_its_own_cue_when_corners_and_roof_line_disagree():
    ns = _builder_ns()
    # Camera too far: the 20 m wide, 12 m tall facade appears 16 m wide with its top at 10.4 m (k = 1.25 about the 2.4 m eye).
    content = {'unitsM': True, 'visibleLeftM': 1.0, 'visibleRightM': 17.0, 'imageWallTopM': 10.08,
               'groups': [{'region': [1.0, 2.4, 9.0, 6.4]}]}
    out = ns['to_fractions'](content, 20.0, 12.0)
    r = out['groups'][0]['region']
    assert [round(v, 3) for v in r] == [0.0, 0.2, 0.5, round((2.4 + 4.0 * 1.25) / 12, 3)], r
    assert 'over its neighbour' not in out['frameNote']
    # Footprint 14.5 m, facade 9.6 m in the image, roof line says k = 1.06: the cues disagree. The image is then not
    # uniformly scaled, so each axis takes its own cue (x from the corners, z from the roof line) and the note says
    # so; whether the footprint holds more than this facade is the LiDAR's finding, not the image's.
    content = {'unitsM': True, 'visibleLeftM': 0.3, 'visibleRightM': 9.9, 'imageWallTopM': 12.6,
               'groups': [{'region': [0.3, 2.4, 5.1, 7.4]}]}
    pieces = [{'xFractionRange': [0.018, 0.897], 'heightM': 13.24}]
    out = ns['to_fractions'](content, 14.51, 13.24, None, pieces)
    kx = (0.897 - 0.018) * 14.51 / 9.6; kz = (13.24 - 2.4) / (12.6 - 2.4)
    r = out['groups'][0]['region']
    expected = [0.018, 2.4 / 13.24, (0.018 * 14.51 + 4.8 * kx) / 14.51, (2.4 + 5.0 * kz) / 13.24]
    assert [round(v, 3) for v in r] == [round(v, 3) for v in expected], (r, expected)
    assert 'differs by axis' in out['frameNote'] and 'over its neighbour' not in out['frameNote'], out['frameNote']
    # Wall pieces of different height along the frontage are the LiDAR's word that the footprint holds more than one level.
    out = ns['to_fractions'](dict(content), 14.51, 13.24, None, pieces + [{'xFractionRange': [0.9, 1.0], 'heightM': 22.6}])
    assert 'differ in height' in out['frameNote'], out['frameNote']

def test_parapet_outline_stands_on_the_wall_top_and_a_band_follows_it():
    records = []; c, a, b = _wall()
    profile = {'id': 'pp', 'match': {'osmWayId': None}, 'normal': [0., -1.], 'sourceIds': ['s'], 'material': 'brick_modular', 'heightRefM': 10.,
               'parapets': [{'top': [[.1, 1.0], [.5, 1.2], [.9, 1.0]]}, {'top': [[.2, 1.3], [.5, 1.45], [.8, 1.3]], 'bandM': .4, 'material': 'stone'}]}
    facade_grammar.compile_wall(_ctx(records), c, a, b, 0., 10., profile, [])
    prisms = [r for r in records if r[0] == 'put' and r[1].startswith('buildings__')]
    assert len(prisms) == 2 and {r[2] for r in prisms} == {'brick_modular', 'stone'}, prisms

def test_relief_arch_ornaments_louvre_sign_and_band_range():
    records = []; c, a, b = _wall()
    profile = {'id': 'orn', 'match': {'osmWayId': None}, 'normal': [0., -1.], 'sourceIds': ['s'], 'material': 'brick_modular',
               'groups': [{'region': [.1, 0., .3, .35], 'columns': 1, 'rows': 1, 'style': {'kind': 'entry', 'reliefArch': {'bandM': .4}, 'mullionFractions': [], 'transomFractions': []}},
                          {'region': [.5, .1, .7, .3], 'columns': 1, 'rows': 1, 'style': {'kind': 'louvre'}}],
               'ornaments': [{'kind': 'diamond', 'z': .8, 'xs': [.2, .4, .6], 'sizeM': .5, 'material': 'redpaint'}, {'kind': 'dentil', 'z': .9, 'xRange': [0., 1.], 'pitchM': 1., 'sizeM': .25}],
               'bands': [{'height': .5, 'xRange': [.5, 1.], 'material': 'terracotta'}],
               'signs': [{'rect': [.4, .55, .8, .65], 'boardMaterial': 'white', 'text': 'OPEN', 'textMaterial': 'charcoal'}]}
    facade_grammar.compile_wall(_ctx(records), c, a, b, 0., 12., profile, [])
    assert sum(1 for r in records if r[0] == 'put' and r[2] == 'brick_modular' and r[1].startswith('buildings__')) >= 1   # relief arch ring
    assert sum(1 for r in records if r[0] == 'put' and r[2] == 'redpaint') == 3                                          # diamonds
    assert sum(1 for r in records if r[0] == 'cube' and r[2] == 'brick_modular' and abs(r[3][0] - .25) < 1e-6) == 20    # dentils at 1 m pitch over 20 m
    slats = [r for r in records if r[0] == 'cube' and r[2] == 'silver' and abs(r[3][2] - .02) < 1e-6]
    assert len(slats) >= 15, len(slats)                                                                                  # louvre slats
    band = [r for r in records if r[0] == 'cube' and r[2] == 'terracotta']
    assert len(band) == 1 and abs(band[0][3][0] - 10.) < 1e-3, band                                                       # band clipped to its x range
    assert any(r[0] == 'cube' and r[2] == 'white' and abs(r[3][0] - 8.) < 1e-3 for r in records)                       # sign board
    assert sum(1 for r in records if r[0] == 'put' and r[2] == 'charcoal' and r[1].startswith('facades__')) >= 4        # letters
