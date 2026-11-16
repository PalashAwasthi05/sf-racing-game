import io
import types
import unittest
import numpy as np
from scipy.ndimage import label
from neighborhood_canopies import collect_canopies,crown_records,pack_crowns,render_crown,append_canopies,MIB
from package_route_block import Writer


def fixture():
    rng=np.random.default_rng(4)
    clouds=[]
    for x in (0,20):
        d=rng.normal(size=(500,3));d/=np.linalg.norm(d,axis=1)[:,None]
        clouds.append(d*rng.random((500,1))**(1/3)*[2.5,2.2,2.5]+[x,6,0])
    leaves=np.concatenate(clouds).astype('f4')
    verts=[];faces=[]
    for x in (0,20):
        p=np.array([[x+dx,y,dz]for y in (0,6)for dx,dz in [(-.2,-.2),(.2,-.2),(.2,.2),(-.2,.2)]],dtype='f4')
        f=np.array([[0,1,2],[0,2,3],[4,6,5],[4,7,6],[0,4,5],[0,5,1],[1,5,6],[1,6,2],[2,6,7],[2,7,3],[3,7,4],[3,4,0]])
        faces.append(f+len(verts)*8);verts.append(p)
    arrays=[leaves,np.concatenate(verts),np.concatenate(faces).astype('u4')]
    model={'materials':[{'pbrMetallicRoughness':{'baseColorFactor':[.04,.12,.02,1]}},{'pbrMetallicRoughness':{'baseColorFactor':[.2,.13,.08,1]}}],
           'meshes':[{'primitives':[{'attributes':{'POSITION':0},'material':0}]},{'primitives':[{'attributes':{'POSITION':1},'indices':2,'material':1}]}],
           'accessors':[{'componentType':5126,'type':'VEC3'}]*3}
    source=types.SimpleNamespace(model=model,array=lambda i:arrays[i].copy())
    nodes=[{'mesh':0,'name':'vegetation___leaf1','extras':{'layer':'vegetation','material':'leaf1','foliageFanCentres':True}},
           {'mesh':1,'name':'vegetation___bark','extras':{'layer':'vegetation','material':'bark'}}]
    matrix=np.eye(4);matrix[:3,3]=[30,2,-20]
    tiles=[(e,n)for n in range(-10,11)for e in range(-10,11)]
    plan=types.SimpleNamespace(nodes=[(n,matrix)for n in nodes],node_groups={id(n):None for n in nodes},owner_tiles={},tile_size=10,tile_index={t:i for i,t in enumerate(tiles)},cells=[{'id':f'grid-{e}-{n}'}for e,n in tiles],materials=[])
    return source,plan,leaves+matrix[:3,3]


class CanopyTests(unittest.TestCase):
    def test_all_source_samples_and_bark_are_transformed_and_assigned(self):
        source,plan,expected=fixture();data=collect_canopies(source,plan)
        np.testing.assert_allclose(data['positions'],expected,atol=2e-6)
        self.assertEqual(len(data['positions']),1000)
        self.assertEqual(len(data['bark']),24)
        self.assertEqual(len(data['anchors']),2)
        self.assertEqual(float(data['bark'][:,:,1].min()),2)
        np.testing.assert_allclose(data['colors'][0],[.04,.12,.02])
        records=crown_records(data)
        self.assertEqual(sum(len(r['leaf'])for r in records),2000)
        self.assertEqual(sum(len(r['bark'])for r in records),48)
        for r in records:
            self.assertTrue(np.all(data['owners'][r['leaf']]==r['owner']))
            self.assertTrue(np.all(data['bark_owners'][r['bark']]==r['owner']))

    def test_full_source_crown_alpha_is_connected_and_trunk_reaches_source_base(self):
        source,plan,_=fixture();data=collect_canopies(source,plan);records=crown_records(data)
        parts=[r for r in records if r['tree']==0 and r['axis']==0]
        record=dict(parts[0],lo=np.min([r['lo']for r in parts],axis=0),hi=np.max([r['hi']for r in parts],axis=0),leaf=np.concatenate([r['leaf']for r in parts]),bark=np.concatenate([r['bark']for r in parts]))
        image=render_crown(data,record,(96,160));alpha=np.asarray(image)[:,:,3]>128
        labels,count=label(alpha,np.ones((3,3)))
        sizes=np.bincount(labels.ravel())[1:]
        self.assertGreater(sizes.max()/alpha.sum(),.97)
        self.assertTrue(alpha[-1].any(),'source-ground trunk was lost')
        self.assertLess(alpha.mean(),.75,'alpha projection became a solid rectangle')
        self.assertGreater(count,0)
        np.testing.assert_array_equal(np.asarray(image),np.asarray(render_crown(data,record,(96,160))))

    def test_generated_atlas_uses_writer_resource_path_exclusive_nodes_and_vram_budget(self):
        source,plan,_=fixture();writer=Writer(io.BytesIO());nodes=[];meshes=[]
        stats=append_canopies(source,writer,nodes,meshes,plan,max_bytes=7*MIB)
        self.assertEqual(stats['sourceSamples'],stats['retainedSamples'])
        self.assertEqual(stats['crowns'],2)
        self.assertLessEqual(stats['estimatedResidentBytes'],7*MIB)
        self.assertTrue(source.model['images'][0]['extras']['neighborhoodWriterView'])
        self.assertEqual(len(source.model['textures']),1)
        for n in nodes:
            self.assertEqual(n['extras']['class'],'distant_exterior')
            self.assertEqual(n['extras']['layer'],'vegetation')
            self.assertTrue(n['extras']['bakedAppearance'])
            self.assertTrue(n['extras']['canopyImpostor'])
            self.assertIn(n['extras']['streamCell'],{c['id']for c in plan.cells})
            self.assertNotIn('foliageFanCentres',n['extras'])
            self.assertFalse(n['name'].split('___')[-1].startswith('leaf'))
        self.assertEqual({p['material']for m in meshes for p in m['primitives']},{0})
        self.assertEqual(plan.materials[0]['alphaMode'],'MASK')
        self.assertEqual(plan.materials[0]['pbrMetallicRoughness']['baseColorFactor'],[1,1,1,1])

    def test_packing_is_deterministic_bounded_and_rejects_impossible_budget(self):
        source,plan,_=fixture();records=crown_records(collect_canopies(source,plan))
        pack_crowns(records,128,128);one=[r['atlas']for r in records]
        pack_crowns(records,128,128);self.assertEqual(one,[r['atlas']for r in records])
        for x,y,w,h in one:self.assertLessEqual(x+w,128);self.assertLessEqual(y+h,128)
        with self.assertRaises(ValueError):append_canopies(source,Writer(io.BytesIO()),[],[],plan,max_bytes=4096)

    def test_atlas_resolution_does_not_depend_on_world_origin(self):
        source,plan,_=fixture();records=crown_records(collect_canopies(source,plan))
        pack_crowns(records,256,256);sizes=[r['size']for r in records]
        for r in records:
            r['lo']=r['lo']+[10000,0,-20000];r['hi']=r['hi']+[10000,0,-20000]
        pack_crowns(records,256,256)
        self.assertEqual(sizes,[r['size']for r in records])

if __name__=='__main__':unittest.main()
