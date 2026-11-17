"""Choose recent panoramas in metric cells within the supplied area's actual outline."""
import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from pyproj import Transformer
from shapely import contains_xy
from shapely.geometry import shape

def choose(area, cell_m, since):
    config = json.loads((area / "config.json").read_text())
    region = config["region"]
    polygon = shape(region["contextGeometry"])
    coverage = json.loads((area / "imagery/coverage.json").read_text())["images"]
    rows = [r for r in coverage if r.get("is_pano") and r.get("captured_at", 0) >= since]
    if not rows:
        return []
    xy = np.array([r["tile_location"] for r in rows])
    admitted = contains_xy(polygon, xy[:, 0], xy[:, 1])
    lon = region["center"]["lon"]
    lat = region["center"]["lat"]
    projection = Transformer.from_crs(4326, (32600 if lat >= 0 else 32700) + int((lon+180)//6)+1, always_xy=True)
    east, north = projection.transform(xy[:, 0], xy[:, 1])
    candidates = defaultdict(list)
    for i in np.flatnonzero(admitted):
        k = (int(east[i] // cell_m), int(north[i] // cell_m))
        candidates[k].append(rows[i])
    def rank(r):
        return (int(r["captured_at"]) // (30*86400000), r.get("quality_score") or 0, int(r["captured_at"]), str(r["id"]))
    return [str(max(candidates[k], key=rank)["id"]) for k in sorted(candidates)]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("area", type=Path)
    parser.add_argument("--cell-m", type=float, default=38.)
    parser.add_argument("--since", default="2023-01-01")
    args = parser.parse_args()
    if args.cell_m <= 0:
        parser.error("--cell-m must be positive")
    ids = choose(args.area, args.cell_m, datetime.fromisoformat(args.since).replace(tzinfo=timezone.utc).timestamp()*1000)
    (args.area/"imagery/selected-ids.json").write_text(json.dumps(ids))
    print(f"Selected {len(ids)} spatially distributed panoramas")
