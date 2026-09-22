"""Replace selected glTF scene layers without loading geometry into a mesh library.

The files produced by the world generator are intentionally simple: a flat list
of scene nodes, with a semantic layer stored in ``mesh.extras.layer``.  This
module keeps that property useful while still doing real glTF reference
rewriting.  Only data reachable from the scene is copied to the resulting GLB;
buffer views are copied as byte ranges, so vertex, index, UV, and image bytes
are not decoded or re-encoded.

The implementation deliberately rejects features for which silently retaining a
stale reference would be worse than stopping (skins, animations, sparse
accessors, and compressed mesh payloads).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import struct
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


GLB_MAGIC = 0x46546C67
GLB_VERSION = 2
JSON_CHUNK = 0x4E4F534A
BIN_CHUNK = 0x004E4942


class GLBError(ValueError):
    """Raised for malformed GLB or glTF data."""


class UnsupportedGLTFError(GLBError):
    """Raised when a feature cannot be copied without risking corruption."""


class GLBDocument:
    def __init__(self, json_doc: dict[str, Any], binary: bytes | memoryview, *, path: str = ""):
        self.json = json_doc
        self.binary = memoryview(binary)
        self.path = path


def _read_glb(path: str | Path) -> GLBDocument:
    """Read the JSON and BIN chunks from a version 2 GLB using ``struct``."""

    source = Path(path)
    raw = source.read_bytes()
    if len(raw) < 12:
        raise GLBError(f"{source}: file is shorter than a GLB header")
    magic, version, declared_length = struct.unpack_from("<III", raw, 0)
    if magic != GLB_MAGIC or version != GLB_VERSION:
        raise GLBError(f"{source}: expected GLB version 2")
    if declared_length != len(raw):
        raise GLBError(f"{source}: header length {declared_length} does not match {len(raw)}")

    json_chunk: bytes | None = None
    binary_chunks: list[memoryview] = []
    offset = 12
    while offset < declared_length:
        if offset + 8 > declared_length:
            raise GLBError(f"{source}: truncated chunk header")
        chunk_length, chunk_type = struct.unpack_from("<II", raw, offset)
        offset += 8
        end = offset + chunk_length
        if end > declared_length:
            raise GLBError(f"{source}: chunk extends beyond file")
        chunk = memoryview(raw)[offset:end]
        if chunk_type == JSON_CHUNK:
            if json_chunk is not None:
                raise GLBError(f"{source}: duplicate JSON chunk")
            json_chunk = bytes(chunk).rstrip(b" \t\r\n\x00")
        elif chunk_type == BIN_CHUNK:
            binary_chunks.append(chunk)
        offset = end
    if json_chunk is None:
        raise GLBError(f"{source}: missing JSON chunk")
    if len(binary_chunks) > 1:
        raise UnsupportedGLTFError(f"{source}: multiple BIN chunks are unsupported")
    try:
        document = json.loads(json_chunk.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GLBError(f"{source}: invalid JSON chunk") from exc
    if not isinstance(document, dict):
        raise GLBError(f"{source}: glTF root must be an object")
    return GLBDocument(document, binary_chunks[0] if binary_chunks else b"", path=str(source))


def _pack_glb(document: dict[str, Any], binary: bytes | bytearray) -> bytes:
    """Encode a glTF document and one embedded BIN chunk as a GLB."""

    try:
        json_bytes = json.dumps(
            document, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise GLBError("glTF JSON cannot be encoded as strict JSON") from exc
    json_bytes += b" " * ((-len(json_bytes)) % 4)
    bin_bytes = bytes(binary)
    bin_bytes += b"\x00" * ((-len(bin_bytes)) % 4)
    total = 12 + 8 + len(json_bytes) + (8 + len(bin_bytes) if bin_bytes else 0)
    header = struct.pack("<III", GLB_MAGIC, GLB_VERSION, total)
    result = bytearray(header)
    result += struct.pack("<II", len(json_bytes), JSON_CHUNK)
    result += json_bytes
    if bin_bytes:
        result += struct.pack("<II", len(bin_bytes), BIN_CHUNK)
        result += bin_bytes
    return bytes(result)


def _walk(value: Any) -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield key, child
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def _validate_document(document: GLBDocument, *, label: str) -> None:
    root = document.json
    if root.get("asset", {}).get("version") != "2.0":
        raise GLBError(f"{label}: glTF asset version must be 2.0")

    # URI-backed buffers cannot be represented by the compact single-BIN writer.
    buffers = root.get("buffers", [])
    if len(buffers) > 1:
        raise UnsupportedGLTFError(f"{label}: multiple buffers are unsupported")
    for buffer in buffers:
        if buffer.get("uri"):
            raise UnsupportedGLTFError(f"{label}: external buffers are unsupported")
        declared = buffer.get("byteLength")
        if declared is not None and declared > len(document.binary):
            raise GLBError(f"{label}: buffer byteLength exceeds BIN chunk")

    for key, value in _walk(root):
        if key in {"skins", "animations"} and value:
            raise UnsupportedGLTFError(f"{label}: skins and animations are unsupported")
        if key in {"skin", "animation"} and value is not None:
            raise UnsupportedGLTFError(f"{label}: skins and animations are unsupported")
        if key == "sparse" and value:
            raise UnsupportedGLTFError(f"{label}: sparse accessors are unsupported")
        if key in {"KHR_draco_mesh_compression", "EXT_meshopt_compression"}:
            raise UnsupportedGLTFError(f"{label}: compressed mesh payloads are unsupported")
    for extension in root.get("extensionsUsed", []) + root.get("extensionsRequired", []):
        if extension in {"KHR_draco_mesh_compression", "EXT_meshopt_compression"}:
            raise UnsupportedGLTFError(f"{label}: compressed mesh payloads are unsupported")

    # The replacement operation intentionally leaves node transforms on the base
    # nodes.  Rejecting child links makes that rule unambiguous and prevents a
    # copied hierarchy from referring to an omitted replacement node.
    for index, node in enumerate(root.get("nodes", [])):
        if node.get("children"):
            raise UnsupportedGLTFError(
                f"{label}: hierarchical nodes are unsupported (node {index} has children)"
            )


def _array(document: GLBDocument, name: str) -> list[dict[str, Any]]:
    value = document.json.get(name, [])
    if not isinstance(value, list):
        raise GLBError(f"{document.path}: {name} must be an array")
    return value


def _index(value: Any, *, label: str, size: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0 or value >= size:
        raise GLBError(f"{label}: invalid index {value!r}")
    return value


def _node_layer(document: GLBDocument, node_index: int) -> str | None:
    nodes = _array(document, "nodes")
    node = nodes[node_index]
    mesh_index = node.get("mesh")
    if mesh_index is None:
        return None
    meshes = _array(document, "meshes")
    mesh_index = _index(mesh_index, label=f"{document.path} node {node_index} mesh", size=len(meshes))
    extras = meshes[mesh_index].get("extras")
    if not isinstance(extras, dict) or extras.get("layer") is None:
        return None
    return str(extras["layer"])


def _reachable_node_indices(document: GLBDocument) -> list[int]:
    """Return scene-reachable node indices for the flat-scene writer."""

    nodes = _array(document, "nodes")
    scenes = _array(document, "scenes")
    if not scenes:
        return list(range(len(nodes)))
    reachable: set[int] = set()
    for scene in scenes:
        for node_index in scene.get("nodes", []):
            reachable.add(_index(node_index, label=f"{document.path} scene node", size=len(nodes)))
    return sorted(reachable)


def _layer_nodes(document: GLBDocument, node_indices: Iterable[int] | None = None) -> dict[str, list[int]]:
    result: dict[str, list[int]] = defaultdict(list)
    if node_indices is None:
        node_indices = _reachable_node_indices(document)
    for index in node_indices:
        layer = _node_layer(document, index)
        if layer is not None:
            result[layer].append(index)
    return dict(result)


class _ResourceCopier:
    """Copy resources from base/replacement documents while rewriting indices."""

    def __init__(self, base: GLBDocument, replacement: GLBDocument):
        self.documents = {"base": base, "replacement": replacement}
        self.output: dict[str, list[dict[str, Any]]] = {
            name: []
            for name in (
                "meshes",
                "materials",
                "textures",
                "images",
                "samplers",
                "accessors",
                "bufferViews",
                "cameras",
            )
        }
        self.maps: dict[tuple[str, str], dict[int, int]] = defaultdict(dict)
        self.binary = bytearray()

    def _source(self, source: str, kind: str, index: int) -> dict[str, Any]:
        values = _array(self.documents[source], kind)
        return values[_index(index, label=f"{source} {kind}", size=len(values))]

    def _copy_plain(self, source: str, kind: str, index: int) -> int:
        key = (source, kind)
        if index in self.maps[key]:
            return self.maps[key][index]
        value = copy.deepcopy(self._source(source, kind, index))
        output_index = len(self.output[kind])
        self.maps[key][index] = output_index
        self.output[kind].append(value)
        return output_index

    def copy_sampler(self, source: str, index: int) -> int:
        return self._copy_plain(source, "samplers", index)

    def copy_camera(self, source: str, index: int) -> int:
        return self._copy_plain(source, "cameras", index)

    def copy_buffer_view(self, source: str, index: int) -> int:
        key = (source, "bufferViews")
        if index in self.maps[key]:
            return self.maps[key][index]
        view = self._source(source, "bufferViews", index)
        buffer_index = view.get("buffer", 0)
        buffers = _array(self.documents[source], "buffers")
        _index(buffer_index, label=f"{source} bufferView buffer", size=max(1, len(buffers)))
        if buffer_index != 0 and buffers:
            raise UnsupportedGLTFError(f"{source}: bufferView references a nonzero buffer")
        start = view.get("byteOffset", 0)
        length = view.get("byteLength")
        if not isinstance(start, int) or start < 0 or not isinstance(length, int) or length < 0:
            raise GLBError(f"{source}: invalid bufferView byte range")
        end = start + length
        if end > len(self.documents[source].binary):
            raise GLBError(f"{source}: bufferView extends beyond BIN chunk")
        # GLB BIN data and glTF buffer views are 4-byte aligned.  Preserve the
        # exact view bytes; alignment bytes between views are new padding.
        padding = (-len(self.binary)) % 4
        if padding:
            self.binary += b"\x00" * padding
        output_start = len(self.binary)
        self.binary += self.documents[source].binary[start:end]
        copied = copy.deepcopy(view)
        copied["buffer"] = 0
        copied["byteOffset"] = output_start
        copied["byteLength"] = length
        output_index = len(self.output["bufferViews"])
        self.maps[key][index] = output_index
        self.output["bufferViews"].append(copied)
        return output_index

    def copy_accessor(self, source: str, index: int) -> int:
        key = (source, "accessors")
        if index in self.maps[key]:
            return self.maps[key][index]
        accessor = copy.deepcopy(self._source(source, "accessors", index))
        if accessor.get("sparse"):
            raise UnsupportedGLTFError(f"{source}: sparse accessors are unsupported")
        if "bufferView" in accessor:
            accessor["bufferView"] = self.copy_buffer_view(source, accessor["bufferView"])
        output_index = len(self.output["accessors"])
        self.maps[key][index] = output_index
        self.output["accessors"].append(accessor)
        return output_index

    def copy_image(self, source: str, index: int) -> int:
        key = (source, "images")
        if index in self.maps[key]:
            return self.maps[key][index]
        image = copy.deepcopy(self._source(source, "images", index))
        if "bufferView" in image:
            image["bufferView"] = self.copy_buffer_view(source, image["bufferView"])
        output_index = len(self.output["images"])
        self.maps[key][index] = output_index
        self.output["images"].append(image)
        return output_index

    def copy_texture(self, source: str, index: int) -> int:
        key = (source, "textures")
        if index in self.maps[key]:
            return self.maps[key][index]
        texture = copy.deepcopy(self._source(source, "textures", index))
        if "sampler" in texture:
            texture["sampler"] = self.copy_sampler(source, texture["sampler"])
        if "source" in texture:
            texture["source"] = self.copy_image(source, texture["source"])
        output_index = len(self.output["textures"])
        self.maps[key][index] = output_index
        self.output["textures"].append(texture)
        return output_index

    def _rewrite_material_textures(self, source: str, value: Any) -> Any:
        if isinstance(value, list):
            return [self._rewrite_material_textures(source, item) for item in value]
        if not isinstance(value, dict):
            return value
        result: dict[str, Any] = {}
        for key, child in value.items():
            if key.lower().endswith("texture") and isinstance(child, dict) and "index" in child:
                child = copy.deepcopy(child)
                child["index"] = self.copy_texture(source, child["index"])
                result[key] = child
            else:
                result[key] = self._rewrite_material_textures(source, child)
        return result

    def copy_material(self, source: str, index: int) -> int:
        key = (source, "materials")
        if index in self.maps[key]:
            return self.maps[key][index]
        material = self._source(source, "materials", index)
        rewritten = self._rewrite_material_textures(source, copy.deepcopy(material))
        output_index = len(self.output["materials"])
        self.maps[key][index] = output_index
        self.output["materials"].append(rewritten)
        return output_index

    def copy_mesh(self, source: str, index: int) -> int:
        key = (source, "meshes")
        if index in self.maps[key]:
            return self.maps[key][index]
        mesh = self._source(source, "meshes", index)
        rewritten = copy.deepcopy(mesh)
        output_index = len(self.output["meshes"])
        self.maps[key][index] = output_index
        # Reserve the slot before walking children, so a malformed self-ref can
        # never recurse forever and the resulting error remains deterministic.
        self.output["meshes"].append(rewritten)
        primitives = rewritten.get("primitives", [])
        if not isinstance(primitives, list):
            raise GLBError(f"{source}: mesh primitives must be an array")
        for primitive in primitives:
            if not isinstance(primitive, dict):
                raise GLBError(f"{source}: mesh primitive must be an object")
            attributes = primitive.get("attributes", {})
            if not isinstance(attributes, dict):
                raise GLBError(f"{source}: primitive attributes must be an object")
            primitive["attributes"] = {
                semantic: self.copy_accessor(source, accessor_index)
                for semantic, accessor_index in attributes.items()
            }
            if "indices" in primitive:
                primitive["indices"] = self.copy_accessor(source, primitive["indices"])
            if "material" in primitive:
                primitive["material"] = self.copy_material(source, primitive["material"])
            if "targets" in primitive:
                primitive["targets"] = [
                    {
                        semantic: self.copy_accessor(source, accessor_index)
                        for semantic, accessor_index in target.items()
                    }
                    for target in primitive["targets"]
                ]
        return output_index


def _rewrite_top_level_extensions(base: dict[str, Any], replacement: dict[str, Any]) -> None:
    for key in ("extensionsUsed", "extensionsRequired"):
        merged: list[Any] = []
        for document in (base, replacement):
            for extension in document.get(key, []):
                if extension not in merged:
                    merged.append(extension)
        if merged:
            base[key] = merged


def _mesh_view_bytes(document: GLBDocument, mesh_index: int) -> bytes:
    """Hash the full raw buffer-view bytes used by a mesh's accessors."""

    mesh = _array(document, "meshes")[_index(mesh_index, label="mesh", size=len(_array(document, "meshes")))]
    accessors = _array(document, "accessors")
    views = _array(document, "bufferViews")
    result = bytearray()
    seen: set[int] = set()
    for primitive in mesh.get("primitives", []):
        accessor_indices: list[int] = list(primitive.get("attributes", {}).values())
        if "indices" in primitive:
            accessor_indices.append(primitive["indices"])
        for target in primitive.get("targets", []):
            accessor_indices.extend(target.values())
        for accessor_index in accessor_indices:
            accessor = accessors[_index(accessor_index, label="accessor", size=len(accessors))]
            if "bufferView" not in accessor:
                continue
            view_index = _index(accessor["bufferView"], label="accessor bufferView", size=len(views))
            if view_index in seen:
                continue
            seen.add(view_index)
            view = views[view_index]
            start = view.get("byteOffset", 0)
            length = view["byteLength"]
            result += document.binary[start : start + length]
    return bytes(result)


def _sha256(data: bytes | bytearray) -> str:
    return hashlib.sha256(data).hexdigest()


def replace_scene_layers(
    base_path: str | Path,
    replacement_path: str | Path,
    layers: Iterable[str] | str,
    output_path: str | Path,
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Replace every base node in ``layers`` with replacement geometry.

    Node objects come from the base document, so their matrices/TRS, names, and
    extras remain byte-for-byte equivalent at the semantic JSON level.  The
    replacement node's mesh is copied with all dependent glTF indices remapped.
    """

    base = _read_glb(base_path)
    replacement = _read_glb(replacement_path)
    _validate_document(base, label="base")
    _validate_document(replacement, label="replacement")
    if isinstance(layers, str):
        selected = [item.strip() for item in layers.split(",") if item.strip()]
    else:
        selected = [str(item).strip() for item in layers if str(item).strip()]
    selected = list(dict.fromkeys(selected))
    if not selected:
        raise GLBError("at least one layer must be selected")

    base_reachable_nodes = _reachable_node_indices(base)
    replacement_reachable_nodes = _reachable_node_indices(replacement)
    base_layers = _layer_nodes(base, base_reachable_nodes)
    replacement_layers = _layer_nodes(replacement, replacement_reachable_nodes)
    replacement_mesh_for_base_node: dict[int, int] = {}
    replacement_node_for_base_node: dict[int, int] = {}
    unmatched_replacement_nodes: list[int] = []
    dropped_base_nodes: set[int] = set()
    for layer in selected:
        base_nodes = base_layers.get(layer, [])
        replacement_nodes = replacement_layers.get(layer, [])
        if not base_nodes:
            raise GLBError(f"base has no nodes for layer {layer!r}")
        if not replacement_nodes:
            raise GLBError(f"replacement has no nodes for layer {layer!r}")
        # Names are useful when a layer has multiple pieces; ordinal matching is
        replacement_by_name: dict[str, list[int]] = defaultdict(list)
        for node_index in replacement_nodes:
            name = replacement.json["nodes"][node_index].get("name")
            if name is not None:
                replacement_by_name[str(name)].append(node_index)
        used: set[int] = set()
        for ordinal, base_node_index in enumerate(base_nodes):
            base_name = base.json["nodes"][base_node_index].get("name")
            candidates = replacement_by_name.get(str(base_name), []) if base_name is not None else []
            replacement_node_index = next((i for i in candidates if i not in used), None)
            if replacement_node_index is None:
                # Pair unnamed or renamed pieces by ordinal while both sides
                # still have an unclaimed node.  Any remaining base nodes are
                # removed and remaining replacement nodes are appended below;
                # this is what makes a layer replacement useful when a new
                # material split adds or removes pieces.
                ordinal_candidate = next((i for i in replacement_nodes if i not in used), None)
                if ordinal_candidate is None:
                    dropped_base_nodes.add(base_node_index)
                    continue
                replacement_node_index = ordinal_candidate
            used.add(replacement_node_index)
            replacement_node_for_base_node[base_node_index] = replacement_node_index
            replacement_mesh_for_base_node[base_node_index] = replacement.json["nodes"][replacement_node_index]["mesh"]
        unmatched_replacement_nodes.extend(i for i in replacement_nodes if i not in used)

    copier = _ResourceCopier(base, replacement)
    output = copy.deepcopy(base.json)
    base_nodes = _array(base, "nodes")
    output_node_index: dict[int, int] = {}
    output_nodes: list[dict[str, Any]] = []
    for node_index in base_reachable_nodes:
        node = base_nodes[node_index]
        if node_index in dropped_base_nodes:
            continue
        output_node_index[node_index] = len(output_nodes)
        rewritten = copy.deepcopy(node)
        if "mesh" in node:
            if node_index in replacement_mesh_for_base_node:
                rewritten["mesh"] = copier.copy_mesh("replacement", replacement_mesh_for_base_node[node_index])
                # A node paired by ordinal (no name match) is a different piece of geometry: it must carry the
                # replacement's own name, extras and transform. Keeping the base node's name mislabelled most
                # building nodes, and the viewer derives layer, identity and surface finish from that name.
                source = replacement.json["nodes"][replacement_node_for_base_node[node_index]]
                if source.get("name") != node.get("name"):
                    for key in ("name", "extras", "matrix", "translation", "rotation", "scale"):
                        rewritten.pop(key, None)
                        if key in source: rewritten[key] = copy.deepcopy(source[key])
            else:
                rewritten["mesh"] = copier.copy_mesh("base", node["mesh"])
        if "camera" in node:
            rewritten["camera"] = copier.copy_camera("base", node["camera"])
        output_nodes.append(rewritten)

    # New replacement pieces have no base transform to retain.  Their own flat
    # node metadata is copied, while pieces with a name match above used the
    # base node and therefore retained its matrix/TRS.
    unmatched_output_nodes: list[int] = []
    for replacement_node_index in unmatched_replacement_nodes:
        node = replacement.json["nodes"][replacement_node_index]
        rewritten = copy.deepcopy(node)
        rewritten["mesh"] = copier.copy_mesh("replacement", node["mesh"])
        if "camera" in node:
            rewritten["camera"] = copier.copy_camera("replacement", node["camera"])
        unmatched_output_nodes.append(len(output_nodes))
        output_nodes.append(rewritten)

    output_scenes: list[dict[str, Any]] = []
    for scene_index, scene in enumerate(_array(base, "scenes")):
        rewritten = copy.deepcopy(scene)
        if "nodes" in scene:
            rewritten["nodes"] = [
                output_node_index[node_index]
                for node_index in scene["nodes"]
                if _index(node_index, label="scene node", size=len(base_nodes)) in output_node_index
            ]
            if scene_index == output.get("scene", 0):
                rewritten["nodes"].extend(unmatched_output_nodes)
        output_scenes.append(rewritten)

    output["nodes"] = output_nodes
    if output_scenes or "scenes" in base.json:
        output["scenes"] = output_scenes
    for resource_name in copier.output:
        if copier.output[resource_name] or resource_name in base.json:
            output[resource_name] = copier.output[resource_name]
    if copier.binary or "buffers" in base.json or "buffers" in replacement.json:
        buffer_metadata = copy.deepcopy((_array(base, "buffers") or _array(replacement, "buffers") or [{}])[0])
        buffer_metadata.pop("uri", None)
        buffer_metadata["byteLength"] = len(copier.binary)
        output["buffers"] = [buffer_metadata]
    _rewrite_top_level_extensions(output, replacement.json)

    packed = _pack_glb(output, copier.binary)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(packed)

    # Read the just-written representation for its preservation check.  This
    # also catches a bad offset before a report claims success.
    written = _read_glb(output_path)
    untouched_records: list[dict[str, Any]] = []
    preservation_ok = True
    for node_index, layer in ((i, _node_layer(base, i)) for i in base_reachable_nodes):
        if layer is None or layer in selected:
            continue
        base_mesh_index = base_nodes[node_index].get("mesh")
        output_mesh_index = written.json["nodes"][output_node_index[node_index]].get("mesh")
        if base_mesh_index is None or output_mesh_index is None:
            continue
        old_hash = _sha256(_mesh_view_bytes(base, base_mesh_index))
        new_hash = _sha256(_mesh_view_bytes(written, output_mesh_index))
        match = old_hash == new_hash
        preservation_ok = preservation_ok and match
        untouched_records.append({"node": node_index, "layer": layer, "sha256": old_hash, "outputSha256": new_hash, "match": match})

    base_sha = _sha256(Path(base_path).read_bytes())
    replacement_sha = _sha256(Path(replacement_path).read_bytes())
    output_sha = _sha256(packed)
    preserved_counts = {
        layer: count
        for layer, count in sorted((layer, len(nodes)) for layer, nodes in base_layers.items())
        if layer not in selected
    }
    replaced_counts = {layer: len(base_layers.get(layer, [])) for layer in selected}
    report: dict[str, Any] = {
        "base": {"path": str(base_path), "sha256": base_sha},
        "replacement": {"path": str(replacement_path), "sha256": replacement_sha},
        "output": {"path": str(output_path), "sha256": output_sha},
        "layers": selected,
        "baseLayerCounts": {layer: len(nodes) for layer, nodes in sorted(base_layers.items())},
        "replacementLayerCounts": {layer: len(nodes) for layer, nodes in sorted(replacement_layers.items())},
        "preservedLayerCounts": preserved_counts,
        "replacedLayerCounts": replaced_counts,
        "preserved_layer_counts": preserved_counts,
        "replaced_layer_counts": replaced_counts,
        "preservationHashes": untouched_records,
        "preservationHashCheck": preservation_ok,
        "preservation_hash_check": {"ok": preservation_ok, "items": untouched_records},
    }
    if report_path is not None:
        report_file = Path(report_path)
        report_file.parent.mkdir(parents=True, exist_ok=True)
        report_file.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="existing detailed GLB")
    parser.add_argument("--replacement", required=True, help="new GLB containing selected layers")
    parser.add_argument("--layers", required=True, help="comma-separated mesh.extras.layer values")
    parser.add_argument("--output", required=True, help="output GLB")
    parser.add_argument("--report", help="optional JSON report path")
    args = parser.parse_args(argv)
    try:
        report = replace_scene_layers(args.base, args.replacement, args.layers, args.output, args.report)
    except GLBError as exc:
        parser.error(str(exc))
    if args.report is None:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by CLI smoke tests
    raise SystemExit(main())
