"""Sample the drivable centerline on the road triangles shipped in the spatial base."""
import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import shapely
from shapely.geometry import LineString

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "king-street/pipeline-source"))
from package_drive_stream import layer_of
from package_route_block import Source, en_positions, flat_nodes, local_path, unpack_portable


def inspect(public, archive_url):
    archive = json.loads(local_path(public, archive_url).read_text())
    index = json.loads(local_path(public, archive["spatialIndex"]).read_text())
    if archive["contentHash"] != index["sourceWorldHash"]:
        raise ValueError("Source and road package mismatch")
    route = LineString(index["routeEN"])
    stations = np.linspace(0, route.length, int(np.ceil(route.length)) + 1)
    points = shapely.line_interpolate_point(route, stations)
    point_tree = shapely.STRtree(points)
    covered = np.zeros(len(stations), dtype=bool)
    with tempfile.TemporaryFile() as binary:
        unpack_portable(public, index["base"], binary)
        source = Source(binary)
        for node, matrix in flat_nodes(source.model):
            mesh = source.model["meshes"][node["mesh"]]
            if layer_of(node, mesh) != "roads":
                continue
            for primitive in mesh["primitives"]:
                if primitive.get("mode", 4) != 4:
                    continue
                positions = en_positions(source.array(primitive["attributes"]["POSITION"]), matrix)
                indices = source.array(primitive["indices"]).reshape(-1) if "indices" in primitive else np.arange(len(positions))
                for start in range(0, len(indices), 12000):
                    triangles = positions[indices[start:start + 12000]].reshape(-1, 3, 2)
                    hits = point_tree.query(shapely.polygons(triangles), predicate="intersects")
                    covered[hits[1]] = True
    breaks = np.flatnonzero(np.diff(np.r_[False, ~covered, False])).reshape(-1, 2)
    return {"source": archive["contentHash"], "lengthM": route.length, "stationCount": len(points),
            "onRoadMesh": int(covered.sum()), "uncoveredRangesM":
            [[round(stations[a], 2), round(stations[b-1], 2)] for a, b in breaks]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("public", type=Path)
    parser.add_argument("archive_url")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = inspect(args.public.resolve(), args.archive_url)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(result)
