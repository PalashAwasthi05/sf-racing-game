import json
import struct

import pytest

from replace_scene_layers_v21 import (
    GLBError,
    UnsupportedGLTFError,
    _mesh_view_bytes,
    _pack_glb,
    _read_glb,
    replace_scene_layers,
)


def _write_glb(path, document, binary):
    path.write_bytes(_pack_glb(document, binary))


def _document(*, street_accessor_offset, street_layer="street", hierarchy=False, sparse=False):
    nodes = [
        {"name": "Building", "mesh": 0, "matrix": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 4, 5, 6, 1]},
        {"name": "Street", "mesh": 1, "matrix": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 9, 8, 7, 1]},
    ]
    if hierarchy:
        nodes[0]["children"] = [1]
    accessors = [
        {"bufferView": 0, "componentType": 5126, "count": 1, "type": "VEC3"},
        {"bufferView": 1, "componentType": 5126, "count": 1, "type": "VEC3"},
    ]
    if sparse:
        accessors[0]["sparse"] = {"count": 1}
    return {
        "asset": {"version": "2.0", "generator": "synthetic-test"},
        "scene": 0,
        "scenes": [{"nodes": [0, 1]}],
        "nodes": nodes,
        "meshes": [
            {"name": "old-building", "extras": {"layer": "buildings"}, "primitives": [{"attributes": {"POSITION": 0}, "material": 0}]},
            {"name": "old-street", "extras": {"layer": "street"}, "primitives": [{"attributes": {"POSITION": 1}, "material": 1}]},
        ],
        "materials": [
            {"name": "brick", "pbrMetallicRoughness": {"baseColorFactor": [0.8, 0.2, 0.1, 1]}},
            {"name": "old-asphalt", "pbrMetallicRoughness": {"baseColorFactor": [0.1, 0.1, 0.1, 1]}},
        ],
        "accessors": accessors,
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": 12},
            {"buffer": 0, "byteOffset": street_accessor_offset, "byteLength": 12},
        ],
        "buffers": [{"byteLength": 24}],
    }


def _replacement_document():
    # The replacement has a deliberately different local transform and resource
    # indices.  The utility must take its mesh but keep the base node matrix.
    return {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"name": "Street", "mesh": 0, "matrix": [2] * 16}],
        "meshes": [
            {"name": "new-street", "extras": {"layer": "street"}, "primitives": [{"attributes": {"POSITION": 0}, "indices": 1, "material": 0}]}
        ],
        "materials": [{"name": "new-asphalt", "pbrMetallicRoughness": {"baseColorFactor": [0.3, 0.4, 0.5, 1]}}],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 1, "type": "VEC3"},
            {"bufferView": 1, "componentType": 5123, "count": 1, "type": "SCALAR"},
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": 12},
            {"buffer": 0, "byteOffset": 12, "byteLength": 2},
        ],
        "buffers": [{"byteLength": 16}],
    }


def test_replaces_layer_remaps_accessors_and_preserves_base_transform(tmp_path):
    base_path = tmp_path / "base.glb"
    replacement_path = tmp_path / "replacement.glb"
    output_path = tmp_path / "output.glb"
    report_path = tmp_path / "report.json"
    old_building_bytes = b"BUILDINGVERT"
    old_street_bytes = b"OLDSTREETBYT"
    _write_glb(base_path, _document(street_accessor_offset=12), old_building_bytes + old_street_bytes + b"padding")
    _write_glb(replacement_path, _replacement_document(), b"NEWSTREETPOS" + struct.pack("<H", 0) + b"pad")

    report = replace_scene_layers(base_path, replacement_path, "street", output_path, report_path)
    result = _read_glb(output_path)
    assert result.json["nodes"][0]["matrix"] == _document(street_accessor_offset=12)["nodes"][0]["matrix"]
    assert result.json["nodes"][1]["matrix"] == _document(street_accessor_offset=12)["nodes"][1]["matrix"]

    building_mesh = result.json["nodes"][0]["mesh"]
    street_mesh = result.json["nodes"][1]["mesh"]
    assert _mesh_view_bytes(result, building_mesh) == old_building_bytes
    assert _mesh_view_bytes(result, street_mesh).startswith(b"NEWSTREETPOS")
    street_primitive = result.json["meshes"][street_mesh]["primitives"][0]
    assert street_primitive["indices"] != 1  # source replacement index was remapped
    assert street_primitive["material"] < len(result.json["materials"])
    assert report["preservationHashCheck"] is True
    assert json.loads(report_path.read_text())["replacedLayerCounts"] == {"street": 1}


def test_untouched_material_properties_and_binary_are_preserved(tmp_path):
    base_path = tmp_path / "base.glb"
    replacement_path = tmp_path / "replacement.glb"
    output_path = tmp_path / "output.glb"
    base = _document(street_accessor_offset=12)
    replacement = _replacement_document()
    _write_glb(base_path, base, b"BUILDINGVERTOLDSTREETBYTES")
    _write_glb(replacement_path, replacement, b"NEWSTREETPOS" + struct.pack("<H", 0) + b"pad")
    replace_scene_layers(base_path, replacement_path, ["street"], output_path)
    result = _read_glb(output_path)
    building_material = result.json["materials"][result.json["meshes"][result.json["nodes"][0]["mesh"]]["primitives"][0]["material"]]
    assert building_material == base["materials"][0]
    assert _mesh_view_bytes(result, result.json["nodes"][0]["mesh"]) == b"BUILDINGVERT"


@pytest.mark.parametrize(
    "document, exception",
    [(_document(street_accessor_offset=12, hierarchy=True), UnsupportedGLTFError), (_document(street_accessor_offset=12, sparse=True), UnsupportedGLTFError)],
)
def test_unsupported_features_fail_loudly(tmp_path, document, exception):
    base_path = tmp_path / "base.glb"
    replacement_path = tmp_path / "replacement.glb"
    _write_glb(base_path, document, b"BUILDINGVERTOLDSTREETBYTES")
    _write_glb(replacement_path, _replacement_document(), b"NEWSTREETPOS" + struct.pack("<H", 0) + b"pad")
    with pytest.raises(exception):
        replace_scene_layers(base_path, replacement_path, "street", tmp_path / "output.glb")
