"""Header-only per-package resident-byte estimates; never decode image pixels.

Offline metadata only. One packed cell at a time; no global image identity table.
GPU bytes count active DDS payloads and portable fallbacks, never CPU companions.
Geometry reserves source arrays plus known preparation arrays and conservative
color/depth/index assembly storage. Cross-cell texture sharing can reduce these
per-cell upper bounds; transfer byte counts must never substitute for them.
"""
import argparse
import copy
import hashlib
import io
import json
from pathlib import Path
import re
import struct
import tempfile

from PIL import Image
from package_route_block import Source, local_path, unpack_portable, write_json, json_bytes

SIZES={5120:1,5121:1,5122:2,5123:2,5125:4,5126:4}
WIDTHS={'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4,'MAT2':4,'MAT3':9,'MAT4':16}
MAX_IMAGE_EDGE=32768
POLICY={'version':1,'geometry':'Referenced accessor bytes + missing normals + 28B/glazing vertex + 4B/stream-tier vertex + 12B/color vertex + 12B/depth vertex + 8B/index assembly reserve. No pixel decode. Native trees omitted only for exact classes in a complete matching tree binding.','textures':'Sum referenced per-cell image upload roles/samplers; portable RGBA mips from header dimensions; GPU DDS exact encoded mip payloads, with RGBA for uncompressed fallbacks. CPU image companions excluded from gpuMipBytes. Per-cell upper bounds do not subtract sharing with other cells.'}

class ImageView(io.RawIOBase):
 def __init__(self,source,index):
  self.stream=source.stream;start,self.length=source.view_range(index);self.start=source.offset+start;self.position=0
 def readable(self):return True
 def seekable(self):return True
 def tell(self):return self.position
 def seek(self,offset,whence=0):
  value=offset if whence==0 else self.position+offset if whence==1 else self.length+offset
  if value<0 or value>self.length:raise ValueError('Image header outside view')
  self.position=value;return value
 def read(self,size=-1):
  count=self.length-self.position if size<0 else min(size,self.length-self.position)
  self.stream.seek(self.start+self.position);data=self.stream.read(count);self.position+=len(data);return data


def image_bytes(source,index,mips=True):
 image=source.model['images'][index];view=ImageView(source,image['bufferView']);header=view.read(148)
 if header[:4]==b'DDS ':
  if len(header)<128:raise ValueError('Incomplete DDS header')
  height,width=struct.unpack_from('<II',header,12);levels=struct.unpack_from('<I',header,28)[0] or 1;fourcc=header[84:88];offset=128
  if fourcc==b'DXT1':block=8
  elif fourcc in (b'DXT3',b'DXT5'):block=16
  elif fourcc==b'DX10' and len(header)>=148 and struct.unpack_from('<I',header,128)[0] in (98,99):block=16;offset=148
  else:raise ValueError('Unsupported DDS footprint format')
  if not 0<width<=MAX_IMAGE_EDGE or not 0<height<=MAX_IMAGE_EDGE or levels>max(width,height).bit_length():raise ValueError('Invalid DDS dimensions/mips')
  total=sum(((max(1,width>>i)+3)//4)*((max(1,height>>i)+3)//4)*block for i in range(levels))
  if offset+total!=view.length:raise ValueError('DDS mip payload length mismatch')
  return total
 view.seek(0)
 with Image.open(view) as decoded_header:
  width,height=decoded_header.size
 if not 0<width<=MAX_IMAGE_EDGE or not 0<height<=MAX_IMAGE_EDGE:raise ValueError('Image dimensions exceed footprint bounds')
 levels=max(width,height).bit_length() if mips else 1
 return sum(max(1,width>>i)*max(1,height>>i)*4 for i in range(levels))


def consumers(model,omit_classes=()):
 roots=model.get('scenes',[{}])[model.get('scene',0)].get('nodes',[])
 # Only the same static flat layouts as the runtime's pruning path may omit roots.
 flat=(len(model.get('scenes',[]))==1 and len(roots)==len(model.get('nodes',[])) and not model.get('animations') and not model.get('skins') and not model.get('cameras') and
  all(set(n)<= {'mesh','name','extras','matrix'} and 'mesh' in n and len(n.get('matrix',[]))==16 for n in model.get('nodes',[])) and
  len({n['mesh'] for n in model.get('nodes',[])})==len(model.get('nodes',[])) and
  all(not a.get('normalized') and not a.get('sparse') and a['componentType'] in (5121,5123,5125,5126) and not model['bufferViews'][a['bufferView']].get('byteStride') for a in model.get('accessors',[])))
 if flat:
  for n in model['nodes']:
   primitives=model['meshes'][n['mesh']]['primitives']
   if len(primitives)!=1 or primitives[0].get('mode',4)!=4 or set(primitives[0])-{'mode','indices','attributes','material'} or set(primitives[0]['attributes'])-{'POSITION','NORMAL','TEXCOORD_0','COLOR_0'}:
    flat=False;break
 nodes=[];seen=set()
 def visit(i):
  if i in seen:raise ValueError('Repeated/cyclic scene root')
  seen.add(i);node=model['nodes'][i]
  name=node.get('name','');layer=node.get('extras',{}).get('layer',name.split('___')[0])
  if flat and layer=='vegetation' and name.startswith('vegetation___') and name[len('vegetation___'):] in omit_classes:return
  if 'mesh' in node:nodes.append(node)
  for child in node.get('children',[]):visit(child)
 for i in roots:visit(i)
 return nodes


def estimate(source,omit_classes=()):
 model=source.model;accessors=set();positions=set();indices=set();missing_normals=set();glazing=set();materials=set()
 for node in consumers(model,omit_classes):
  for primitive in model['meshes'][node['mesh']]['primitives']:
   accessors.update(primitive['attributes'].values());position=primitive['attributes']['POSITION'];positions.add(position)
   if 'indices' in primitive:accessors.add(primitive['indices']);indices.add(primitive['indices'])
   if 'NORMAL' not in primitive['attributes']:missing_normals.add(position)
   material=primitive.get('material')
   if material is not None:
    materials.add(material);name=node.get('name','').split('___')[-1].split('#')[0].split('~obs1')[0].split('__')[0]
    if re.match(r'^(glass|curtain)',name):glazing.add(position)
 def count(ids):return sum(model['accessors'][i]['count'] for i in ids)
 def accessor_bytes(i):
  a=model['accessors'][i];count=a['count']
  if not isinstance(count,int) or count<0:raise ValueError('Invalid accessor count')
  size=count*SIZES[a['componentType']]*WIDTHS[a['type']];view=model['bufferViews'][a['bufferView']];offset=a.get('byteOffset',0)
  stride=view.get('byteStride',SIZES[a['componentType']]*WIDTHS[a['type']]);span=(count-1)*stride+SIZES[a['componentType']]*WIDTHS[a['type']] if count else 0
  if offset<0 or span+offset>view['byteLength']:raise ValueError('Accessor footprint outside view')
  return size
 source_bytes=sum(accessor_bytes(i) for i in accessors);vertices=count(positions);index_count=count(indices)
 geometry=source_bytes+count(missing_normals)*12+count(glazing)*28+vertices*(4+12+12)+index_count*8
 uploads=set()
 def texture_refs(value,key=''):
  if not isinstance(value,dict):return
  if key.endswith('Texture') and 'index' in value:
   texture=model.get('textures',[])[value['index']];image=texture['source'];sampler=model.get('samplers',[])[texture['sampler']] if 'sampler' in texture else {}
   color=key in ('baseColorTexture','emissiveTexture');need_mips=sampler.get('minFilter',9987) not in (9728,9729)
   state=(sampler.get('magFilter',9729),sampler.get('minFilter',9987),sampler.get('wrapS',10497),sampler.get('wrapT',10497))
   uploads.add((image,color,state,need_mips))
  for k,v in value.items():texture_refs(v,k)
 for i in materials:texture_refs(model['materials'][i])
 texture=sum(image_bytes(source,image,mips) for image,color,state,mips in uploads)
 return {'geometryBytes':int(geometry),'mipBytes':int(texture),'vertices':vertices,'indices':index_count,'sourceGeometryBytes':source_bytes,'imageUploads':len(uploads)}


def tree_classes(public,manifest,tree_url=None):
 if manifest.get('scope')!='neighborhood-stream':return {}
 # Keep the runtime tree format binding lightweight; importing the compiler
 # would pull projection/geometry dependencies into this header-only pass.
 expected={'version':3,'sourceWorldHash':manifest['sourceWorldHash'],'cells':[{'id':c['id'],'contentHash':c['contentHash']} for c in manifest['cells']]}
 if tree_url is None:tree_url='/compare/tree-detail-'+hashlib.sha256(json.dumps(expected,separators=(',',':')).encode()).hexdigest()[:16]+'.json'
 path=local_path(public,tree_url)
 if not path.is_file():return {}
 tree=json.loads(path.read_bytes())
 if any(tree.get(k)!=v for k,v in expected.items()):return {}
 result={}
 for asset in tree.get('treeCells',[]):
  if not any(c['id']==asset.get('id') and c['contentHash']==asset.get('contentHash') for c in manifest['cells']):return {}
  classes=asset.get('classes',[])
  binary=local_path(public,asset['binary'])
  if not binary.is_file() or binary.stat().st_size!=asset.get('compressedBytes') or any(not re.fullmatch(r'(leaf\w*|frond|bark|wood|trunk)',c) for c in classes):return {}
  result[asset['id']]=set(classes)
 return result


def annotate_manifest(public,manifest,tree_url=None,progress=None):
 public=Path(public);covered=tree_classes(public,manifest,tree_url);rows=[]
 with tempfile.TemporaryDirectory(prefix='scene-residency-') as temp:
  for entry in [manifest['base']]+manifest['cells']+manifest.get('pieces',[]):
   omit=covered.get(entry.get('id'),())
   values=[]
   for descriptor in [entry,entry.get('gpu')]:
    if descriptor is None:values.append(None);continue
    if descriptor is not entry and descriptor.get('portableHash')!=entry['contentHash']:values.append(None);continue
    with (Path(temp)/'cell.glb').open('w+b') as file:
     unpack_portable(public,descriptor,file);values.append(estimate(Source(file),omit))
   portable,gpu=values;entry.setdefault('stats',{})['residency']={'geometryBytes':portable['geometryBytes'],'portableMipBytes':portable['mipBytes'],'gpuMipBytes':gpu['mipBytes'] if gpu else portable['mipBytes']}
   rows.append({'id':entry.get('id','base'),'contentHash':entry['contentHash'],**entry['stats']['residency'],'omittedTreeClasses':sorted(omit),'vertices':portable['vertices'],'indices':portable['indices']})
   if progress:progress(rows[-1])
 return rows


def publish(public,index_url='/compare/neighborhood-stream.json',report=None):
 public=Path(public);path=local_path(public,index_url);snapshot=json.loads(path.read_bytes());prepared=copy.deepcopy(snapshot);rows=annotate_manifest(public,prepared)
 def identities(value):return [(c.get('id','base'),c['contentHash'],c.get('gpu',{}).get('contentHash')) for c in [value['base']]+value['cells']+value.get('pieces',[])]
 for attempt in range(5):
  before=path.read_bytes();current=json.loads(before)
  if current.get('sourceWorldHash')!=snapshot.get('sourceWorldHash') or identities(current)!=identities(snapshot):raise RuntimeError('Source/GPU identities changed; residency not published')
  patches={c.get('id','base'):c['stats']['residency'] for c in [prepared['base']]+prepared['cells']+prepared.get('pieces',[])}
  for c in [current['base']]+current['cells']+current.get('pieces',[]):c.setdefault('stats',{})['residency']=patches[c.get('id','base')]
  immutable=write_json(public,Path(index_url).stem,current)
  with tempfile.NamedTemporaryFile(dir=path.parent,prefix='.residency-',delete=False) as file:temporary=Path(file.name);file.write(json_bytes(current))
  if path.read_bytes()!=before:temporary.unlink();continue
  temporary.replace(path);break
 else:raise RuntimeError('Neighborhood alias kept changing; residency not published')
 result={'version':1,'manifest':index_url,'immutableManifest':immutable,'sourceWorldHash':snapshot['sourceWorldHash'],'policy':POLICY,'parts':rows}
 if report:Path(report).write_text(json.dumps(result,indent=2)+'\n')
 return result

if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('public',type=Path);parser.add_argument('--index',default='/compare/neighborhood-stream.json');parser.add_argument('--report',type=Path,required=True);args=parser.parse_args()
 result=publish(args.public,args.index,args.report);print(json.dumps({'manifest':result['manifest'],'immutableManifest':result['immutableManifest'],'parts':len(result['parts'])}))
