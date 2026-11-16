import io
import hashlib
import unittest
from types import SimpleNamespace
from package_drive_pieces import subdivide_plan

import numpy as np
from pyproj import Transformer
from package_drive_stream import Plan, compile_part, select_route, station
from package_route_block import Source, building_id
from test_package_route_block import fixture


class DriveStreamTests(unittest.TestCase):
    def test_generic_route_follows_connected_active_roads_and_intersections(self):
        center={'lon':-122.4,'lat':37.8}
        project=Transformer.from_crs('EPSG:4326','EPSG:26910',always_xy=True)
        inverse=Transformer.from_crs('EPSG:26910','EPSG:4326',always_xy=True)
        ox,oy=project.transform(center['lon'],center['lat']);records=[]
        for i in range(4):
            a,b=-100+i*100,i*100
            records.append({'cnn':str(i),'active':True,'classcode':'5','oneway':'F',
                'streetname':'Unfamiliar Avenue','f_st':'Cross '+str(i),'t_st':'Cross '+str(i+1),
                'line':{'coordinates':[inverse.transform(ox+a,oy),inverse.transform(ox+b,oy)]}})
        # An inactive record could offer the same heading but must never be selected.
        records.insert(0,{**records[1],'cnn':'retired','active':False})
        route,cells=select_route(records,center,[[20,0],[80,0]],500,4,42)
        self.assertEqual([c['id'] for c in cells],['road-0','road-1','road-2','road-3'])
        self.assertAlmostEqual(cells[-1]['endM'],400,places=5)
        np.testing.assert_allclose(route[0],[-100,0],atol=1e-6)
        for a,b in zip(cells,cells[1:]):self.assertEqual(a['endM'],b['startM'])
        # Data-provided one-way restrictions are respected.
        records[2]['oneway']='T'
        with self.assertRaises(ValueError):select_route(records,center,[[20,0],[80,0]],500,4,42)

    def test_complete_features_have_one_owner_and_support_only_in_base(self):
        raw,_=fixture();source=Source(io.BytesIO(raw))
        source.model['nodes'][1]['rotation']=[float(np.sin(np.pi/8)),0,0,float(np.cos(np.pi/8))]
        source.model['nodes'].append({'name':'foundations__A___stone','mesh':0,'rotation':source.model['nodes'][1]['rotation']})
        source.model['nodes'][0]['children'].append(len(source.model['nodes'])-1)
        cells=[{'id':'left','endM':25},{'id':'right','endM':150}]
        plan=Plan(source,[[0,0],[150,0]],cells,2)
        self.assertEqual(plan.owners,{'A':1})
        outputs=[]
        for cell in (None,0,1):
            packed=io.BytesIO();compile_part(source,io.BytesIO(),packed,plan,cell)
            out=Source(io.BytesIO(packed.getvalue()));outputs.append(out)
            self.assertTrue(all('children' not in n for n in out.model['nodes']))
            for image in out.model.get('images',[]):
                start,length=out.view_range(image['bufferView']);out.stream.seek(out.offset+start)
                self.assertEqual(image['extras']['viewerImageSha256'],hashlib.sha256(out.stream.read(length)).hexdigest())
        base,left,right=outputs
        self.assertIn('roads__shared',[n.get('name') for n in base.model['nodes']])
        self.assertTrue(any(n.get('name','').startswith('foundations') for n in base.model['nodes']))
        for out in (left,right):
            self.assertFalse(any(n.get('name','').startswith(('roads','foundations')) for n in out.model['nodes']))
        ids=[building_id(n,right.model['meshes'][n['mesh']]) for n in right.model['nodes']]
        self.assertEqual(ids.count('A'),2) # full facade + far roof remain together
        self.assertFalse(any(building_id(n,left.model['meshes'][n['mesh']]) for n in left.model['nodes']))
        hull=next(n for n in base.model['nodes'] if n.get('extras',{}).get('class')=='collision_hull')
        self.assertEqual(hull['extras']['streamCell'],'right')
        self.assertTrue(any(m.get('name')=='drive_stream_collision_hull_right' for m in base.model['materials']))
        # Two foliage centres owned on left, one on right; no centre duplication/loss.
        counts=[]
        for out in (left,right):
            p=next(p for m in out.model['meshes'] for p in m['primitives'])
            counts.append(len(out.array(p['attributes']['POSITION'])))
        self.assertEqual(counts[0],2)
        # Right's first primitive is A's facade, so inspect fan by its metadata.
        fan_node=next(n for n in right.model['nodes'] if n.get('extras',{}).get('foliageFanCentres'))
        p=right.model['meshes'][fan_node['mesh']]['primitives'][0]
        self.assertEqual(len(right.array(p['attributes']['POSITION'])),1)

    def test_subpieces_preserve_semantic_parents_and_complete_feature_owners(self):
        parents=[{'id':'a','startM':0.,'endM':150.,'roadId':'alpha'},
                 {'id':'b','startM':150.,'endM':300.,'roadId':'beta'}]
        owners={str(i):i//15 for i in range(30)}
        stations={str(i):(i*10.,i*10.+8,i*10.+4) for i in range(30)}
        plan=SimpleNamespace(route=np.array([[0.,0.],[300.,0.]]),owners=owners.copy(),feature_stations=stations)
        pieces=subdivide_plan(plan,parents,[])
        self.assertEqual(len(pieces),6)
        self.assertEqual(set(plan.owners),set(owners))
        for bid,index in plan.owners.items():
            self.assertEqual(pieces[index]['parentId'],parents[owners[bid]]['id'])
        for i,piece in enumerate(pieces):
            self.assertTrue(45<=piece['endM']-piece['startM']<=60)
            if i:self.assertEqual(pieces[i-1]['endM'],piece['startM'])
            self.assertEqual(plan.materials[plan.hull_materials[i]]['extras']['streamCell'],piece['id'])
        self.assertEqual(pieces[0]['startM'],0)
        self.assertEqual(pieces[-1]['endM'],300)

    def test_surface_crossing_cell_edge_is_retained_once(self):
        raw,_=fixture();source=Source(io.BytesIO(raw))
        plan=Plan(source,[[0,0],[150,0]],[{'id':'left','endM':25},{'id':'right','endM':150}],2)
        positions=np.array([[24,0,0],[26,0,0],[25,0,-1]],dtype=float)
        indices=np.arange(3,dtype=np.uint32)
        # Nonbuilding streamed triangle chooses one centroid owner, retaining full surface.
        selections=[plan.select(positions,np.eye(4),indices,False,4,c) for c in (0,1)]
        self.assertEqual(sum(len(order)//3 for _,order in selections),1)
        used,order=next(v for v in selections if len(v[0]))
        np.testing.assert_array_equal(used,[0,1,2])
        a,d=station([[50,3]],[[0,0],[150,0]])
        self.assertAlmostEqual(a[0],50);self.assertAlmostEqual(d[0],3)


if __name__=='__main__':unittest.main()
