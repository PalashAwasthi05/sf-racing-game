import numpy as np
import trimesh
from shapely.geometry import Point, Polygon, box

from pipeline import facade_grammar, facade_wall_domain
from pipeline.facade_wall_domain import clip_owned_skirt, project_profile_domain


def test_boundary_samples_project_only_when_they_match_and_cover_the_segment():
    profile = {'id': 'gabled', 'normal': [0., -1.], 'frameU': [0., 10.]}
    samples = np.array([
        [0., 0., 4.], [5., 0., 8.], [10., 0., 4.],
        [5., 3., 20.],  # a nearby roof return is outside the plane tolerance
    ])
    domain = project_profile_domain(samples, profile, (0., 10.), 0., 0.)
    assert domain is not None
    assert domain.covers(Point(5., 7.)) and not domain.covers(Point(5., 8.1))
    assert project_profile_domain(samples[1:3], profile, (0., 10.), 0., 0.) is None


def test_chain_selector_rejects_nearby_parallel_recessed_roof_edge():
    profile = {'id': 'gable', 'normal': [0., -1.], 'frameU': [0., 10.]}
    xs = np.linspace(0., 10., 21)
    chains = [
        {'ring': 'exterior', 'points': np.c_[xs, np.zeros_like(xs), 4. + 4. * (1. - np.abs(xs - 5.) / 5.)]},
        {'ring': 'exterior', 'points': np.c_[xs, np.full_like(xs, .38), 9. + 6. * (1. - np.abs(xs - 5.) / 5.)]},
    ]
    domain = facade_wall_domain.project_profile_chain_domain(
        chains, profile, (0., 10.), 0., 0., [0., 0.], [10., 0.], domain_base_z=0.)
    assert domain is not None
    assert abs(domain.bounds[3] - 8.) < 1e-8
    assert domain.covers(Point(5., 7.)) and not domain.covers(Point(5., 9.))


def _wall_context():
    records = []
    def cube(*args): records.append(('cube', args))
    def cylinder(*args): records.append(('cylinder', args))
    def put(mesh, key, material, uv=None): records.append(('mesh', key, material, mesh.copy()))
    return records, {
        'polys': lambda g: [g] if g.geom_type == 'Polygon' else list(g.geoms),
        'cube': cube, 'cylinder': cylinder, 'put': put,
        'wallTopBoundaryChainsByProfile': {'synthetic-gable': [
            {'ring': 'exterior', 'points': np.c_[
                np.linspace(0., 10., 21), np.zeros(21),
                4. + 4. * (1. - np.abs(np.linspace(0., 10., 21) - 5.) / 5.)]},
            # Nearby recessed edge: exact chain matching must not merge its
            # much taller samples into the gable's wall-top trace.
            {'ring': 'exterior', 'points': np.c_[
                np.linspace(0., 10., 21), np.full(21, .38),
                9. + 6. * (1. - np.abs(np.linspace(0., 10., 21) - 5.) / 5.)]},
        ]},
        'roofSkirtOwners': [],
    }


def test_gable_wall_domain_keeps_attic_aperture_and_owns_only_matching_skirt():
    records, ctx = _wall_context()
    footprint = box(0., 0., 10., 2.)
    c = {'id': 'synthetic', 'g': footprint}
    profile = {
        'id': 'synthetic-gable', 'match': {'osmWayId': None}, 'normal': [0., -1.],
        'frameU': [0., 10.], 'planeV': 0., 'heightRefM': 8., 'sourceIds': ['photo'],
        'material': 'brick_modular', 'gableEnd': True,
        'openings': [{'rect': [.45, .65, .55, .85], 'style': {
            'kind': 'window', 'sash': 'double-hung', 'lites': [1, 1], 'mullionFractions': [],
            'transomFractions': [], 'frameMaterial': '#eeeeee',
        }}],
    }
    result = facade_grammar.compile_wall(ctx, c, np.array([0., 0.]), np.array([10., 0.]), 0., 4., profile, [])
    assert result and result['record']['openingCount'] == 1
    assert result['record']['heightM'] == 8.  # extends above the 4 m eave to the sampled gable ridge

    wall_front = []
    for rec in records:
        if rec[0] != 'mesh' or not rec[1].startswith('buildings__'): continue
        _, key, material, mesh = rec
        for tri in mesh.triangles:
            if np.all(np.abs(tri[:, 1]) < 1e-7):
                wall_front.append(Polygon(tri[:, [0, 2]]))
    assert not any(poly.covers(Point(5., 6.)) for poly in wall_front)  # real aperture in the gable wall

    owner = ctx['roofSkirtOwners'][0]
    full_gable_skirt = trimesh.Trimesh(
        vertices=[[0., 0., 0.], [10., 0., 0.], [5., 0., 8.]], faces=[[0, 1, 2]], process=False)
    clipped = clip_owned_skirt(full_gable_skirt, [owner])
    assert len(clipped.faces) == 0  # no opaque roof skirt remains over the window aperture

    unmatched = trimesh.Trimesh(
        vertices=[[0., 5., 0.], [10., 5., 0.], [5., 5., 8.]], faces=[[0, 1, 2]], process=False)
    kept = clip_owned_skirt(unmatched, [owner])
    assert len(kept.faces) == 1  # a different, recessed facade is never cut


def test_observed_gable_outline_extends_exact_eave_chain_without_parapet_slab():
    records, ctx = _wall_context()
    xs = np.linspace(0., 10., 21)
    ctx['wallTopBoundaryChainsByProfile']['synthetic-gable'][0]['points'] = np.c_[xs, np.zeros(21), np.full(21, 4.)]
    # Retain the recessed parallel chain as a red herring; it must not own the facade.
    profile = {
        'id': 'synthetic-gable', 'match': {'osmWayId': None}, 'normal': [0., -1.],
        'frameU': [0., 10.], 'planeV': 0., 'heightRefM': 8., 'sourceIds': ['photo'],
        'material': 'brick_modular',
        'parapets': [{'top': [[0., .5], [.5, 1.], [1., .5]], 'bandM': .1,
                      'thicknessM': .3, 'material': 'timber_dark'}],
        'openings': [{'rect': [.45, .65, .55, .85], 'style': {
            'kind': 'window', 'sash': 'double-hung', 'lites': [1, 1], 'mullionFractions': [],
            'transomFractions': [], 'frameMaterial': '#eeeeee',
        }}],
    }
    result = facade_grammar.compile_wall(
        ctx, {'id': 'synthetic', 'g': box(0., 0., 10., 2.)},
        np.array([0., 0.]), np.array([10., 0.]), 0., 4., profile, [])
    assert result and result['record']['heightM'] == 8.
    wall_front = []
    for rec in records:
        if rec[0] != 'mesh' or not rec[1].startswith('buildings__'): continue
        for tri in rec[3].triangles:
            if np.all(np.abs(tri[:, 1]) < 1e-7): wall_front.append(Polygon(tri[:, [0, 2]]))
    assert not any(poly.covers(Point(5., 6.)) for poly in wall_front)  # observed attic window remains open
    assert any(rec[0] == 'mesh' and rec[2] == 'timber_dark' for rec in records)  # thin outline trim is preserved


def test_wrong_parapet_triangle_cannot_replace_rising_shell_domain():
    records, ctx = _wall_context()
    xs=np.linspace(0.,10.,21)
    ctx['wallTopBoundaryChainsByProfile']['synthetic-gable'][0]['points']=np.c_[
        xs,np.zeros_like(xs),7.+5.*(1.-np.abs(xs-5.)/5.)]
    profile={
        'id':'synthetic-gable','match':{'osmWayId':None},'normal':[0.,-1.],
        'frameU':[0.,10.],'planeV':0.,'heightRefM':8.,'sourceIds':['photo'],
        'material':'brick_modular',
        # This stale hand-read triangle has matching eaves but a fabricated 7.9 m apex.
        'parapets':[{'top':[[0.,.5],[.5,.9875],[1.,.5]],'bandM':.1,
                     'thicknessM':.3,'material':'stale-trim'}],
        'openings':[{'rect':[.45,.65,.55,.85],'style':{
            'kind':'window','sash':'double-hung','lites':[1,1],
            'mullionFractions':[],'transomFractions':[],'frameMaterial':'#eeeeee'}}],
    }
    result=facade_grammar.compile_wall(ctx,{'id':'synthetic','g':box(0.,0.,10.,2.)},
        np.array([0.,0.]),np.array([10.,0.]),3.,4.,profile,[])
    assert result and result['record']['heightM']==9.
    assert any('disagrees with variable shell boundary' in note for note in result['record']['unresolved'])
    # The real shell remains the domain and the attic opening is still a void.
    wall_front=[]
    for rec in records:
        if rec[0]!='mesh' or not rec[1].startswith('buildings__'):continue
        for tri in rec[3].triangles:
            if np.all(np.abs(tri[:,1])<1e-7):wall_front.append(Polygon(tri[:,[0,2]]))
    assert not any(poly.covers(Point(5.,9.)) for poly in wall_front)
    # Misaligned legacy parapet fill/trim is not re-emitted above the actual shell.
    assert not any(rec[0]=='mesh' and rec[2]=='stale-trim' for rec in records)


def test_gable_match_checks_shell_vertices_away_from_source_apex():
    from shapely.geometry import Polygon
    # At the source parapet's only interior x (x=5), the shell agrees exactly.
    # Two real shell ridges at x=3 and x=7 are visible only if the entire shell
    # top trace, rather than the parapet's sample locations, is checked.
    shell=Polygon([(0.,0.),(0.,4.),(3.,9.),(5.,7.9),(7.,9.),(10.,4.),(10.,0.)])
    profile={'parapets':[{'top':[[0.,.5],[.5,.9875],[1.,.5]]}]}
    domain, ignored, unresolved=facade_wall_domain.observed_gable_domain(
        profile,(0.,10.),10.,8.,0.,shell)
    assert domain is None
    assert ignored=={0}
    assert unresolved and 'disagrees with variable shell boundary' in unresolved[0]


def test_variable_roof_boundary_suppresses_only_synthetic_ridge_coping():
    records, ctx = _wall_context()
    profile = {
        'id': 'synthetic-gable', 'match': {'osmWayId': None}, 'normal': [0., -1.],
        'frameU': [0., 10.], 'planeV': 0., 'heightRefM': 8., 'sourceIds': ['photo'],
        'material': 'brick_modular', 'gableEnd': False,
        'bands': [{'height': .75, 'thicknessM': .2, 'depthM': .2, 'material': 'observed-band'}],
    }
    result = facade_grammar.compile_wall(
        ctx, {'id': 'synthetic', 'g': box(0., 0., 10., 2.)},
        np.array([0., 0.]), np.array([10., 0.]), 0., 4., profile, [])
    assert result and result['record']['heightM'] == 8.
    cubes = [rec[1] for rec in records if rec[0] == 'cube']
    bands = [rec[3] for rec in records if rec[0] == 'mesh' and rec[2] == 'observed-band']
    assert bands  # a swept band, not necessarily a box, remains at the source elevation
    assert all(np.allclose(mesh.bounds[:, 2], [5.9, 6.1]) for mesh in bands)
    assert not any(args[3] == 'charcoal' and abs(args[1][2] - .1) < 1e-7
                   and args[0][2] > 7.5 for args in cubes)  # no full-width horizontal bar at the sampled ridge


def test_skirt_clip_triangulates_only_the_remainder_and_preserves_face_winding():
    skirt = trimesh.Trimesh(
        vertices=[[0., 0., 0.], [10., 0., 0.], [5., 0., 8.]],
        faces=[[0, 1, 2]], process=False)
    owner = {
        'normal': [0., -1.], 'tangent': [1., 0.], 'plane': 0., 'left': 0., 'zBaseM': 0.,
        'domain': box(0., 0., 5., 8.),
    }
    remainder = clip_owned_skirt(skirt, [owner])
    assert len(remainder.faces) >= 1
    assert abs(float(remainder.area) - 20.) < 1e-6
    source_normal = skirt.face_normals[0]
    assert np.all(remainder.face_normals @ source_normal > 0.)


def test_skirt_union_transforms_adjacent_owner_frames_before_subtracting():
    skirt = trimesh.Trimesh(
        vertices=[[0., 0., 0.], [10., 0., 0.], [5., 0., 8.]],
        faces=[[0, 1, 2]], process=False)
    common = {'normal': [0., -1.], 'tangent': [1., 0.], 'plane': 0., 'zBaseM': 0.}
    owners = [
        {**common, 'left': 0., 'domain': box(0., 0., 5., 8.)},
        {**common, 'left': 5., 'domain': box(0., 0., 5., 8.)},
    ]
    clipped = clip_owned_skirt(skirt, owners)
    assert len(clipped.faces) == 0  # the two local halves cover the whole triangle


def test_skirt_remainder_constrained_triangulation_preserves_concave_notch():
    skirt = trimesh.Trimesh(
        vertices=[[0., 0., 0.], [10., 0., 0.], [5., 0., 8.]],
        faces=[[0, 1, 2]], process=False)
    owner = {
        'normal': [0., -1.], 'tangent': [1., 0.], 'plane': 0., 'left': 0., 'zBaseM': 0.,
        'domain': box(4., 0., 6., 4.),
    }
    remainder = clip_owned_skirt(skirt, [owner])
    expected_cut = Polygon([[0., 0.], [10., 0.], [5., 8.]]).intersection(owner['domain']).area
    assert abs(float(remainder.area) - (40. - expected_cut)) < 1e-6
    assert all(abs(float(Polygon(face[:, [0, 2]]).intersection(owner['domain']).area)) < 1e-8
               for face in remainder.triangles)
