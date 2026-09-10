import struct

from mapillary_mesh_probe_v12 import decode_mesh_pbf, select_spatial


def _varint(value):
    out = bytearray()
    while value >= 0x80:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    out.append(value)
    return bytes(out)


def _mesh_bytes():
    points = (0.0, 0.0, 1.0, 2.0, 0.0, 3.0, 0.0, 4.0, 5.0, 2.0, 4.0, 5.0)
    data = bytearray()
    for value in points:
        data.extend(_varint((1 << 3) | 5))
        data.extend(struct.pack("<f", value))
    for value in (0, 1, 2, 0, 2, 3):
        data.extend(_varint((2 << 3) | 0))
        data.extend(_varint(value))
    return bytes(data)


def test_decode_documented_unpacked_mesh_fields():
    decoded = decode_mesh_pbf(_mesh_bytes())
    assert decoded["vertexCount"] == 4
    assert decoded["triangleCount"] == 2
    assert decoded["invalidFaceIndexCount"] == 0
    assert decoded["axisExtentsMeters"]["max"] == [2.0, 4.0, 5.0]
    assert decoded["depthResolutionMeters"] is None


def test_spatial_selection_is_deterministic_and_spread():
    records = [
        {"id": str(i), "computed_geometry": {"coordinates": [float(i), float(i % 2)]}, "mesh": {"url": "https://example.invalid/" + str(i)}}
        for i in range(10)
    ]
    selected = select_spatial(records, 3)
    assert selected[0]["id"] == "0"
    assert selected[1]["id"] == "9"
    assert selected[2]["id"] in {"3", "4", "5", "6"}
