import copy
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
from PIL import Image
from package_route_block import Source
from scene_residency import estimate,image_bytes,publish


def fixture(image=None,tree=False,dds=False):
 if image is None:
  file=io.BytesIO();Image.new('RGBA',(8,4),(100,120,140,255)).save(file,format='PNG');image=file.getvalue()
 parts=[bytes(36),struct.pack('<3H',0,1,2),image];views=[];binary=bytearray()
 for part in parts:
  binary.extend(bytes(-len(binary)%4));views.append({'buffer':0,'byteOffset':len(binary),'byteLength':len(part)});binary.extend(part)
 binary.extend(bytes(-len(binary)%4));matrix=[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]
 model={'asset':{'version':'2.0'},'scene':0,'scenes':[{'nodes':[0]}],'nodes':[{'name':'facades__A___glass_dark','matrix':matrix,'mesh':0}],
  'meshes':[{'primitives':[{'attributes':{'POSITION':0},'indices':1,'material':0}]}],
  'accessors':[{'bufferView':0,'componentType':5126,'count':3,'type':'VEC3','min':[0,0,0],'max':[1,1,1]},{'bufferView':1,'componentType':5123,'count':3,'type':'SCALAR'}],
  'bufferViews':views,'buffers':[{'byteLength':len(binary)}],'materials':[{'pbrMetallicRoughness':{'baseColorTexture':{'index':0}}}],
  'textures':[{'source':0}],'images':[{'bufferView':2,'mimeType':'image/vnd-ms.dds' if dds else 'image/png'}]}
 if tree:model['nodes'][0].update(name='vegetation___leaf1',extras={'layer':'vegetation'})
 text=json.dumps(model,separators=(',',':')).encode();text+=b' '*(-len(text)%4)
 data=struct.pack('<5I',0x46546c67,2,28+len(text)+len(binary),len(text),0x4e4f534a)+text+struct.pack('<II',len(binary),0x004e4942)+binary
 return Source(io.BytesIO(data))


def dds(width=8,height=4,levels=4):
 header=bytearray(128);header[:4]=b'DDS ';struct.pack_into('<II',header,12,height,width);struct.pack_into('<I',header,28,levels);header[84:88]=b'DXT1'
 payload=sum(((max(1,width>>i)+3)//4)*((max(1,height>>i)+3)//4)*8 for i in range(levels))
 return bytes(header)+bytes(payload)


class ResidencyTests(unittest.TestCase):
 def test_png_header_reservation_never_decodes_pixels(self):
  source=fixture()
  with patch.object(Image.Image,'load',side_effect=AssertionError('Pixel decoding forbidden')):result=estimate(source)
  self.assertEqual(result['mipBytes'],(8*4+4*2+2*1+1)*4);self.assertEqual(result['sourceGeometryBytes'],42)
  self.assertEqual(result['geometryBytes'],42+3*12+3*28+3*(4+12+12)+3*8)
  source.model['samplers']=[{'minFilter':9729}];source.model['textures'][0]['sampler']=0
  self.assertEqual(estimate(source)['mipBytes'],8*4*4)

 def test_dds_uses_exact_full_payload_and_rejects_truncation(self):
  self.assertEqual(image_bytes(fixture(dds(),dds=True),0),40)
  with self.assertRaisesRegex(ValueError,'payload length'):image_bytes(fixture(dds()[:-1],dds=True),0)
  with self.assertRaisesRegex(ValueError,'dimensions'):image_bytes(fixture(dds(width=65536),dds=True),0)

 def test_only_known_flat_tree_classes_are_omitted(self):
  source=fixture(tree=True);self.assertGreater(estimate(source)['geometryBytes'],0)
  self.assertEqual(estimate(source,{'leaf1'})['geometryBytes'],0);self.assertEqual(estimate(source,{'leaf1'})['mipBytes'],0)
  self.assertGreater(estimate(source,{'bark'})['geometryBytes'],0)
  source.model['nodes'][0]['children']=[];self.assertGreater(estimate(source,{'leaf1'})['geometryBytes'],0)

 def test_uncovered_shared_material_and_unknown_flat_features_remain_reserved(self):
  source=fixture(tree=True);node=copy.deepcopy(source.model['nodes'][0]);node['name']='facades__A___glass_dark';node.pop('extras');source.model['nodes'].append(node);source.model['scenes'][0]['nodes'].append(1)
  self.assertGreater(estimate(source,{'leaf1'})['mipBytes'],0)
  source=fixture(tree=True);source.model['accessors'][0]['normalized']=True
  self.assertGreater(estimate(source,{'leaf1'})['geometryBytes'],0)

 def test_accessor_outside_view_is_rejected(self):
  source=fixture();source.model['accessors'][0]['count']=100000
  with self.assertRaisesRegex(ValueError,'outside view'):estimate(source)

 def test_publish_merges_fresh_alias_metadata_without_overwriting_other_fields(self):
  with tempfile.TemporaryDirectory() as temp:
   public=Path(temp);(public/'compare').mkdir();path=public/'compare/neighborhood-stream.json'
   initial={'sourceWorldHash':'hash','base':{'contentHash':'base','stats':{'other':1}},'cells':[{'id':'A','contentHash':'a','stats':{'vertices':10}}],'facadeMiddle':'first'};path.write_text(json.dumps(initial))
   def compute(_public,prepared):
    for entry in [prepared['base']]+prepared['cells']:entry['stats']['residency']={'geometryBytes':3,'portableMipBytes':4,'gpuMipBytes':1}
    fresh=copy.deepcopy(initial);fresh['facadeMiddle']='changed-concurrently';fresh['newField']={'preserve':True};path.write_text(json.dumps(fresh));return []
   with patch('scene_residency.annotate_manifest',compute):publish(public)
   result=json.loads(path.read_text());self.assertEqual(result['facadeMiddle'],'changed-concurrently');self.assertEqual(result['newField'],{'preserve':True})
   self.assertEqual(result['base']['contentHash'],'base');self.assertEqual(result['cells'][0]['stats']['vertices'],10);self.assertEqual(result['cells'][0]['stats']['residency']['gpuMipBytes'],1)

if __name__=='__main__':unittest.main()
