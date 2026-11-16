import json
import struct
from pathlib import Path

from regen.audit_aperture_repair_targets import resolve_exact_source, _ray_box_intersects
from regen.reading_provenance import reading_sha256, sha256_file
from regen.glb_stream import GltfStream, GlbAppendWriter
from regen.patch_apertures_glb import patch_world
import numpy as np
import trimesh


def test_cached_ray_blocks_match_full_mesh_across_instances_and_short_last_block(tmp_path):
    from regen.host_shell_apertures import TriangleIndex
    shape = trimesh.creation.icosphere(subdivisions=3)
    scene = trimesh.Scene()
    scene.add_geometry(shape, geom_name='sphere', node_name='first')
    shifted = np.eye(4)
    shifted[0, 3] = 3.
    scene.graph.update(frame_to='second', matrix=shifted, geometry='sphere')
    path = tmp_path / 'two.glb'
    scene.export(path)
    with GltfStream(path) as stream:
        for row in [*stream.rows, *reversed(stream.rows)]:
            full = TriangleIndex.from_scene(stream.scene([row]))
            cx = row['transform'][0, 3]
            for offset in (-1.2, -.72, .03, .81):
                start = [cx + offset, -4., .11]
                stop = [cx + offset + .01, 4., -.27]
                hits = stream.segment_mesh_hits(row, start, stop)
                target = full.ray_hits(start, stop)
                assert [h['node'] for h in hits] == [h['node'] for h in target]
                assert [h['triangleIndex'] for h in hits] == [h['triangleIndex'] for h in target]
                np.testing.assert_allclose([h['distanceM'] for h in hits],
                                           [h['distanceM'] for h in target], atol=1e-7)


def _fixture(tmp_path):
    area=tmp_path/'area';side_dir=area/'regen/sides-reg';img=side_dir/'img'
    img.mkdir(parents=True)
    for name in ('fixture__v0_rect.jpg','fixture__v0_persp.jpg'):
        (img/name).write_bytes(b'fixture')
    reading={'id':'fixture__s0p0','use':0,'confidence':'high'}
    reading_path=area/'readings/fixture.json';reading_path.parent.mkdir(parents=True)
    reading_path.write_text(json.dumps(reading))
    view={'image':'image-7','registered':'skyline','rect':'fixture__v0_rect.jpg',
          'persp':'fixture__v0_persp.jpg','ppm':45.}
    side={'id':'fixture__s0p0','frameU':[0.,4.],'normal':[0.,1.],
          'planeV':2.,'lengthM':4.,'heightM':4.,'views':[view]}
    (side_dir/'sides-0.json').write_text(json.dumps({'pieces':[side]}))
    profile={'id':'regen-fixture__s0p0','sourceIds':['mapillary:image-7'],
        'frameU':[0.,4.],'normal':[0.,1.],'planeV':2.,'heightRefM':4.,
        'readingProvenance':{'status':'ok','accepted':True,
          'registeredCropSource':True,'pathBase':'area',
          'sourceReading':{'path':'readings/fixture.json',
                           'sha256':sha256_file(reading_path)},
          'resultingReadingSha256':reading_sha256(reading),
          'sourceSide':'regen/sides-reg/sides-0.json','image':'image-7',
          'rect':view['rect'],'persp':view['persp']}}
    return area,profile


def test_exact_source_join_checks_hash_frame_image_and_crop_files(tmp_path):
    area,profile=_fixture(tmp_path)
    from regen.reading_provenance import load_registered_sides
    joined=load_registered_sides(area/'regen/sides-reg')
    source=resolve_exact_source(profile,area,joined)
    assert source['status']=='ok'
    assert source['image']=='image-7'
    assert 'exact registered side/frame/image/crops' in source['frameCheck']


def test_source_join_abstains_on_image_or_frame_mismatch(tmp_path):
    area,profile=_fixture(tmp_path)
    from regen.reading_provenance import load_registered_sides
    joined=load_registered_sides(area/'regen/sides-reg')
    profile['readingProvenance']['image']='different-image'
    assert resolve_exact_source(profile,area,joined)['reason']=='registered-image-does-not-match-provenance'
    profile['readingProvenance']['image']='image-7'
    profile['planeV']=2.2
    assert 'frame-planeV-mismatch' in resolve_exact_source(profile,area,joined)['reason']


def test_ray_neighborhood_keeps_only_bounds_crossed_by_source_camera_segment():
    assert _ray_box_intersects([0.,5.,2.],[0.,-1.,2.],[-.5,-.1,1.5],[.5,.1,2.5])
    assert not _ray_box_intersects([0.,5.,2.],[0.,-1.,2.],[2.,-.1,1.5],[3.,.1,2.5])


def test_glb_stream_decodes_only_requested_primitive(tmp_path):
    scene=trimesh.Scene()
    mesh=trimesh.creation.box(extents=[2.,3.,4.])
    scene.add_geometry(mesh,node_name='buildings__fixture___wall',
                       geom_name='fixture-wall')
    path=tmp_path/'fixture.glb';path.write_bytes(scene.export(file_type='glb'))
    with GltfStream(path) as stream:
        rows=stream.rows_for_feature('fixture')
        assert len(rows)==1
        selected=stream.scene(rows)
        node=selected.graph.nodes_geometry[0]
        transform,name=selected.graph.get(node)
        decoded=selected.geometry[name]
        assert len(decoded.faces)==12
        assert np.allclose(decoded.bounds,mesh.bounds)
        assert np.allclose(transform,np.eye(4))
        hits=stream.segment_mesh_hits(rows[0],[0.,5.,0.],[0.,-5.,0.])
        assert hits and hits[0]['node']=='buildings__fixture___wall'


def test_feature_rays_include_observed_structural_walls(tmp_path):
    scene=trimesh.Scene()
    for node in ('buildings__fixture___shell',
                 'buildings__fixture__observed_reading_wall___brick',
                 'facades__fixture__observed_reading_window___glass',
                 'buildings__other___shell'):
        scene.add_geometry(trimesh.creation.box(),node_name=node,geom_name=node)
    path=tmp_path/'observed.glb';path.write_bytes(scene.export(file_type='glb'))
    with GltfStream(path) as stream:
        names={row['node'] for row in stream.rows_for_feature('fixture')}
        assert 'buildings__fixture__observed_reading_wall___brick' in names
        assert 'facades__fixture__observed_reading_window___glass' in names
        assert 'buildings__other___shell' not in names


def _bin_chunk(path):
    data=Path(path).read_bytes()
    offset=12
    while offset<len(data):
        size=int.from_bytes(data[offset:offset+4],'little')
        kind=data[offset+4:offset+8]
        offset+=8
        if kind==b'BIN\0':return data[offset:offset+size]
        offset+=size
    raise AssertionError('missing BIN chunk')


def _duplicate_first_primitive(source,destination):
    data=Path(source).read_bytes();offset=12;chunks=[];document=None
    while offset<len(data):
        size=int.from_bytes(data[offset:offset+4],'little')
        kind=data[offset+4:offset+8];payload=data[offset+8:offset+8+size]
        if kind==b'JSON':document=json.loads(payload)
        else:chunks.append((kind,payload))
        offset+=8+size
    document['meshes'][0]['primitives'].append(
        json.loads(json.dumps(document['meshes'][0]['primitives'][0])))
    encoded=json.dumps(document,separators=(',',':')).encode()+b' '
    encoded+=b' '*((-len(encoded))%4)
    total=12+8+len(encoded)+sum(8+len(payload) for _,payload in chunks)
    with open(destination,'wb') as out:
        out.write(struct.pack('<4sII',b'glTF',2,total))
        out.write(struct.pack('<I4s',len(encoded),b'JSON'));out.write(encoded)
        for kind,payload in chunks:
            out.write(struct.pack('<I4s',len(payload),kind));out.write(payload)


def _append_inactive_node(source,destination):
    data=Path(source).read_bytes();offset=12;chunks=[];document=None
    while offset<len(data):
        size=int.from_bytes(data[offset:offset+4],'little')
        kind=data[offset+4:offset+8];payload=data[offset+8:offset+8+size]
        if kind==b'JSON':document=json.loads(payload)
        else:chunks.append((kind,payload))
        offset+=8+size
    duplicate=json.loads(json.dumps(document['nodes'][0]))
    duplicate['translation']=[1000.,0.,0.]
    document['nodes'].append(duplicate)  # deliberately absent from active scene roots
    encoded=json.dumps(document,separators=(',',':')).encode()
    encoded+=b' '*((-len(encoded))%4)
    total=12+8+len(encoded)+sum(8+len(payload) for _,payload in chunks)
    with open(destination,'wb') as out:
        out.write(struct.pack('<4sII',b'glTF',2,total))
        out.write(struct.pack('<I4s',len(encoded),b'JSON'));out.write(encoded)
        for kind,payload in chunks:
            out.write(struct.pack('<I4s',len(payload),kind));out.write(payload)


def test_glb_stream_ignores_detached_nodes_even_when_names_duplicate(tmp_path):
    scene=trimesh.Scene()
    scene.add_geometry(trimesh.creation.box(),node_name='buildings__fixture___wall',
                       geom_name='wall')
    source=tmp_path/'active.glb';source.write_bytes(scene.export(file_type='glb'))
    detached=tmp_path/'detached.glb';_append_inactive_node(source,detached)
    with GltfStream(detached) as stream:
        rows=stream.rows_for_feature('fixture')
        assert len(rows)==1
        assert len(stream.by_node)==1
        assert np.allclose(rows[0]['transform'],np.eye(4))


def test_append_writer_preserves_original_bin_and_unmodified_nodes(tmp_path):
    scene=trimesh.Scene()
    scene.add_geometry(trimesh.creation.box(extents=[2.,1.,1.]),
                       node_name='buildings__fixture___wall',geom_name='wall')
    scene.add_geometry(trimesh.creation.box(extents=[1.,1.,1.]),
                       node_name='buildings__fixture___roof',geom_name='roof')
    source=tmp_path/'source.glb';source.write_bytes(scene.export(file_type='glb'))
    original_bin=_bin_chunk(source)
    output=tmp_path/'patched.glb'
    with GltfStream(source) as stream:
        writer=GlbAppendWriter(stream)
        wall=stream.by_node['buildings__fixture___wall']
        roof=stream.by_node['buildings__fixture___roof']
        roof_mesh_index=roof['meshIndex']
        changed=stream.mesh(wall).copy()
        changed.vertices[:,2]*=.5
        writer.replace_node_mesh(wall['node'],changed,wall['material'],
                                 {'featureId':'fixture','layer':'building-facade'})
        added=trimesh.creation.box(extents=[.2,.2,.2])
        writer.add_node('facades__fixture__observed_test_window___glass',added,
                        writer.material_for_name('glass'),np.eye(4),
                        {'featureId':'fixture','layer':'facade'})
        writer.write(output)
        writer.close()
    new_bin=_bin_chunk(output)
    assert new_bin[:len(original_bin)]==original_bin
    with GltfStream(output) as stream:
        assert stream.by_node['buildings__fixture___roof']['meshIndex']==roof_mesh_index
        assert stream.by_node['buildings__fixture___wall']['meshIndex']!=wall['meshIndex']
        assert 'facades__fixture__observed_test_window___glass' in stream.node_index_by_name
        assert len(stream.mesh(stream.by_node['buildings__fixture___wall']).faces)==12


def test_patch_refuses_to_overwrite_input_or_existing_output(tmp_path):
    source=tmp_path/'release.glb';source.write_bytes(b'original-release')
    with np.testing.assert_raises_regex(ValueError,'must not overwrite'):
        patch_world(source,tmp_path/'profiles.json',tmp_path,tmp_path/'audit.json',
                    source,tmp_path/'report.json')
    existing=tmp_path/'candidate.glb';existing.write_bytes(b'keep-me')
    with np.testing.assert_raises_regex(FileExistsError,'refusing to overwrite'):
        patch_world(source,tmp_path/'profiles.json',tmp_path,tmp_path/'audit.json',
                    existing,tmp_path/'report.json')
    assert existing.read_bytes()==b'keep-me'


def test_append_writer_replaces_one_primitive_without_dropping_siblings(tmp_path):
    scene=trimesh.Scene()
    scene.add_geometry(trimesh.creation.box(),node_name='buildings__fixture___wall',
                       geom_name='wall')
    ordinary=tmp_path/'ordinary.glb';ordinary.write_bytes(scene.export(file_type='glb'))
    source=tmp_path/'multi.glb';_duplicate_first_primitive(ordinary,source)
    output=tmp_path/'multi-patched.glb'
    with GltfStream(source) as stream:
        rows=[row for row in stream.rows_for_feature('fixture')
              if row['node'].startswith('buildings__fixture___wall')]
        assert len(rows)==2
        original_sibling=rows[1]['primitive']
        changed=stream.mesh(rows[0]).copy();changed.vertices[:,2]*=.5
        writer=GlbAppendWriter(stream)
        writer.replace_node_mesh(rows[0]['node'],changed,rows[0]['material'])
        writer.write(output);writer.close()
        sibling_indices=original_sibling['indices']
    with GltfStream(output) as stream:
        node_index=stream.document_node_index('buildings__fixture___wall')
        mesh=stream.document['meshes'][stream.document['nodes'][node_index]['mesh']]
        assert len(mesh['primitives'])==2
        assert mesh['primitives'][1]['indices']==sibling_indices
        assert len(stream.rows_for_feature('fixture'))==2
def test_camera_segment_rejects_disjoint_slab_intervals_and_hits_axis_aligned_faces():
    from regen.host_shell_apertures import segment_boxes
    lo=np.array([[1.,8.,-.1],[4.,4.,-.1]])
    hi=lo+np.array([1.,1.,.2])
    assert segment_boxes([0,0,0],[10,10,0],lo,hi).tolist()==[False,True]
    assert segment_boxes([4.1,0,0],[4.1,10,0],lo,hi).tolist()==[False,True]


def test_serialized_window_receivers_obey_measured_opening_and_role():
    from regen.prototype_host_shell_apertures import _receiver_node_expected
    profile={'id':'reading','match':{'featureId':'fixture'}}
    opening={'style':{'kind':'window'}}
    assert _receiver_node_expected('facades__fixture__observed_reading_aperture_3_part_1_muntin___#aaa',profile,opening,3)
    assert not _receiver_node_expected('facades__fixture__observed_reading_aperture_8_part_0_glazing___glass',profile,opening,3)
    assert not _receiver_node_expected('facades__fixture__observed_reading_aperture_3_reveal___wall',profile,opening,3)
