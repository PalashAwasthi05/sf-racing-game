import io
import hashlib
import json
from pathlib import Path
import unittest
import tempfile
import numpy as np
from package_facade_middle import SolidSimplifier, packed_vertices, split_batches, write_batch, publish
from package_route_block import json_bytes


def boundary(p,ix):
    edges=np.sort(np.concatenate([ix.reshape(-1,3)[:,[0,1]],ix.reshape(-1,3)[:,[1,2]],ix.reshape(-1,3)[:,[2,0]]]),axis=1)
    unique,count=np.unique(edges,axis=0,return_counts=True)
    return {tuple(tuple(v) for v in sorted(map(tuple,p[e]))) for e in unique[count==1]}


class FacadeMiddleTests(unittest.TestCase):
    def test_real_window_hole_and_outer_boundary_survive_solid_reduction(self):
        # Planar facade tessellation around a real opening, with internal edges.
        p=np.array([[x,y,0] for y in range(9) for x in range(9)],dtype='<f4');ix=[]
        for y in range(8):
            for x in range(8):
                if 2<=x<6 and 2<=y<6:continue
                a=y*9+x;ix.extend([a,a+1,a+10,a,a+10,a+9])
        ix=np.array(ix,dtype='<u4');simplifier=SolidSimplifier()
        try:q,used,reduced=simplifier.reduce(p,ix)
        finally:simplifier.close()
        self.assertLess(len(reduced),len(ix))
        self.assertEqual(boundary(p,ix),boundary(q,reduced))
        # Every simplifier location is an actual source position.
        np.testing.assert_array_equal(q,p[used])

    def test_hard_edges_quantization_and_bounded_batches_preserve_all_faces(self):
        p=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],dtype='<f4');ix=np.array([0,1,2,0,3,1],dtype='<u4');c=np.tile([40,80,120],(4,1)).astype('u1')
        packed=packed_vertices(p,ix,c)
        self.assertEqual(len(packed[0]),6,'hard corner normals must split')
        batches=list(split_batches([packed],limit=3));self.assertEqual(sum(len(b[3]) for b in batches),len(ix))
        for b in batches:
            out=io.BytesIO();record=write_batch(out,b,True)
            self.assertEqual(record['residentBytes'],len(b[0])*18+len(b[3])*2)
            self.assertLessEqual(record['quantizationErrorM'],max(record['step'])/2+1e-7)
            self.assertEqual(record['indices'],3)

    def test_production_all_owners_and_every_glazing_triangle_are_present(self):
        public=Path(__file__).parents[1]/'public'
        source=json.loads((public/'compare/neighborhood-stream.json').read_text())
        if not source.get('facadeMiddle'):self.skipTest('Production facade asset not generated')
        path=public/source['facadeMiddle'].lstrip('/')
        data=json.loads(path.read_text())
        self.assertTrue(data['complete']);self.assertEqual(data['sourceWorldHash'],source['sourceWorldHash'])
        owners={c['id']:c for c in source['cells'] if c['stats']['buildings']}
        self.assertEqual({c['id'] for c in data['cells']},set(owners))
        self.assertEqual(data['stats']['buildings'],sum(c['stats']['buildings'] for c in owners.values()))
        for cell in data['cells']:
            self.assertEqual(cell['sourceHash'],owners[cell['id']]['contentHash'])
            self.assertEqual(cell['stats']['glazingTriangles'],cell['stats']['sourceGlazingTriangles'])
            self.assertTrue((public/cell['url'].lstrip('/')).exists())
            self.assertLess(cell['stats']['quantizationErrorM'],.001)
        self.assertLess(data['stats']['triangles'],data['stats']['sourceTriangles']*.5)

    def test_publication_changes_only_optional_metadata_and_rejects_stale_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            public=Path(directory);(public/'compare').mkdir()
            binary=b'pinned facade example';digest=hashlib.sha256(binary).hexdigest();binary_url='/compare/facade-cell-'+digest[:16]+'.bin.gz'
            (public/binary_url.lstrip('/')).write_bytes(binary)
            owner={'id':'cell','contentHash':'abc','boundsEN':[0,0,1,1],'stats':{'buildings':1}}
            source={'scope':'neighborhood-stream','version':1,'sourceWorldHash':'world','cells':[owner],'base':{'opaque':'unchanged'}}
            path=public/'compare/neighborhood-stream.json';path.write_bytes(json_bytes(source))
            asset={'version':1,'format':'facade-middle-v1','complete':True,'sourceWorldHash':'world','sourceCells':[{'id':'cell','contentHash':'abc'}],
                   'cells':[{'id':'cell','sourceHash':'abc','boundsEN':[0,0,1,1],'url':binary_url,'sha256':digest,'compressedBytes':len(binary),
                             'stats':{'sourceGlazingTriangles':2,'glazingTriangles':2,'buildings':1}}],'stats':{'buildings':1}}
            data=json_bytes(asset);url='/compare/neighborhood-facades-'+hashlib.sha256(data).hexdigest()[:16]+'.json';(public/url.lstrip('/')).write_bytes(data)
            result=publish(public,asset_url=url);after=json.loads(path.read_text())
            self.assertEqual(after.pop('facadeMiddle'),url);self.assertEqual(after,source);self.assertTrue(result['baseAndDetailUnchanged'])
            before=path.read_bytes();asset['sourceCells'][0]['contentHash']='stale';data=json_bytes(asset)
            stale_url='/compare/neighborhood-facades-'+hashlib.sha256(data).hexdigest()[:16]+'.json';(public/stale_url.lstrip('/')).write_bytes(data)
            with self.assertRaises(ValueError):publish(public,asset_url=stale_url)
            self.assertEqual(path.read_bytes(),before)


if __name__=='__main__':unittest.main()
