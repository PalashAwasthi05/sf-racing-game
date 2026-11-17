"""Combine static generator GLBs on disk while keeping identical material resources shared."""
import hashlib
import json
from pathlib import Path
import shutil
import struct
import tempfile

HEADER = struct.Struct("<5I")
CHUNK = struct.Struct("<2I")
JSON_CHUNK = 0x4E4F534A
BIN_CHUNK = 0x004E4942


def copy_view(stream, start, length, output=None):
    stream.seek(start)
    digest = hashlib.sha256() if output is None else None
    while length:
        block = stream.read(min(length, 1024 * 1024))
        if not block:
            raise ValueError("Truncated GLB buffer view")
        if output is None:
            digest.update(block)
        else:
            output.write(block)
        length -= len(block)
    return digest.digest() if digest else None


def key(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def remap_material(value, textures):
    for name, child in value.items():
        if isinstance(child, dict):
            if name.lower().endswith("texture") and "index" in child:
                child["index"] = textures[child["index"]]
            remap_material(child, textures)


def merge_static_glbs(paths, destination):
    destination = Path(destination)
    result = {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0]}],
              "nodes": [{"name":"world", "children":[]}], "buffers": [{"byteLength":0}],
              "bufferViews":[], "accessors":[], "materials":[], "meshes":[],
              "images":[], "textures":[], "samplers":[]}
    shared = {field:{} for field in ("images","textures","materials","samplers")}
    image_views = {}
    extensions = {name:set() for name in ("extensionsUsed","extensionsRequired")}

    def intern(field, item):
        identity = key(item)
        if identity not in shared[field]:
            shared[field][identity] = len(result[field])
            result[field].append(item)
        return shared[field][identity]

    with tempfile.TemporaryFile(dir=destination.parent) as binary:
        for part in paths:
            with Path(part).open("rb") as source:
                magic, version, size, text_size, kind = HEADER.unpack(source.read(20))
                if magic != 0x46546C67 or version != 2 or kind != JSON_CHUNK:
                    raise ValueError(f"Invalid generator GLB: {part}")
                model = json.loads(source.read(text_size))
                bin_len, bin_kind = CHUNK.unpack(source.read(8))
                base = source.tell()
                if base+bin_len!=size or bin_kind!=BIN_CHUNK or len(model["buffers"])!=1:
                    raise ValueError(f"Unexpected generator GLB layout: {part}")
                if any(model.get(k) for k in ("animations","skins","cameras","extensions")):
                    raise ValueError("Only static generator geometry/materials are supported")
                result["asset"] = model["asset"]
                for field in extensions:
                    extensions[field].update(model.get(field, []))

                images = {im["bufferView"] for im in model.get("images",[]) if "bufferView" in im}
                views = []
                for i, view in enumerate(model.get("bufferViews",[])):
                    length=view["byteLength"]; offset=view.get("byteOffset",0)
                    if view["buffer"]!=0 or length<0 or offset<0 or offset+length>bin_len:
                        raise ValueError("Buffer view outside source")
                    identity=None
                    if i in images:
                        metadata={k:v for k,v in view.items() if k not in ("buffer","byteOffset")}
                        identity=(key(metadata),copy_view(source,base+offset,length))
                        if identity in image_views:
                            views.append(image_views[identity]);continue
                    view_id=len(result["bufferViews"])
                    result["bufferViews"].append({**view,"buffer":0,"byteOffset":binary.tell()})
                    views.append(view_id)
                    if identity is not None:
                        image_views[identity]=view_id
                    copy_view(source,base+offset,length,binary)
                    binary.write(bytes(-binary.tell()%4))

                accessors=[]
                for accessor in model.get("accessors",[]):
                    if "bufferView" in accessor:
                        accessor["bufferView"]=views[accessor["bufferView"]]
                    if "sparse" in accessor:
                        for field in ("indices","values"):
                            sub=accessor["sparse"][field]
                            sub["bufferView"]=views[sub["bufferView"]]
                    accessors.append(len(result["accessors"]));result["accessors"].append(accessor)
                image_map=[]
                for im in model.get("images",[]):
                    if "bufferView" in im:
                        im["bufferView"]=views[im["bufferView"]]
                    elif not im.get("uri","").startswith("data:"):
                        raise ValueError("Expected embedded image")
                    image_map.append(intern("images",im))
                samplers=[intern("samplers",s) for s in model.get("samplers",[])]
                textures=[]
                for tex in model.get("textures",[]):
                    if "source" in tex:tex["source"]=image_map[tex["source"]]
                    if "sampler" in tex:tex["sampler"]=samplers[tex["sampler"]]
                    for ext in tex.get("extensions",{}).values():
                        if "source" in ext:ext["source"]=image_map[ext["source"]]
                    textures.append(intern("textures",tex))
                materials=[]
                for mat in model.get("materials",[]):
                    remap_material(mat,textures);materials.append(intern("materials",mat))

                mesh_offset=len(result["meshes"]);node_offset=len(result["nodes"])
                for mesh in model.get("meshes",[]):
                    for prim in mesh["primitives"]:
                        if prim.get("extensions"):
                            raise ValueError("Unsupported extended mesh primitive")
                        if "indices" in prim:prim["indices"]=accessors[prim["indices"]]
                        if "material" in prim:prim["material"]=materials[prim["material"]]
                        for attr in [prim["attributes"], *prim.get("targets",[])]:
                            for semantic in attr:attr[semantic]=accessors[attr[semantic]]
                    result["meshes"].append(mesh)
                for node in model.get("nodes",[]):
                    if "mesh" in node:node["mesh"]+=mesh_offset
                    if "children" in node:node["children"]=[n+node_offset for n in node["children"]]
                    if "skin" in node or "camera" in node or node.get("extensions"):
                        raise ValueError("Unsupported generator node")
                    if node.get("name")=="world" and "mesh" not in node:
                        node.pop("name")
                    result["nodes"].append(node)
                roots=model["scenes"][model.get("scene",0)]["nodes"]
                result["nodes"][0]["children"].extend(n+node_offset for n in roots)
        if not result["meshes"]:
            raise ValueError("No meshes to publish")
        for k in ("images","samplers","textures","materials"):
            if not result[k]:result.pop(k)
        for k,v in extensions.items():
            if v:result[k]=sorted(v)
        bin_len=binary.tell();result["buffers"][0]["byteLength"]=bin_len
        encoded=json.dumps(result,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()
        encoded+=b" "*(-len(encoded)%4)
        size=28+len(encoded)+bin_len
        if size>=2**32:raise ValueError("Merged generator GLB exceeds 32-bit GLB size")
        with tempfile.NamedTemporaryFile(dir=destination.parent,delete=False) as stage:
            target=Path(stage.name)
            try:
                stage.write(HEADER.pack(0x46546C67,2,size,len(encoded),JSON_CHUNK))
                stage.write(encoded);stage.write(CHUNK.pack(bin_len,BIN_CHUNK))
                binary.seek(0);shutil.copyfileobj(binary,stage,1024*1024)
                stage.close();target.replace(destination)
            finally:
                target.unlink(missing_ok=True)
    return size
