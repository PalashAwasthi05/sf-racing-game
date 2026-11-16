"""Replay just the generator's source-owned street-inventory tree prefix.

The area generator batches all geometries at the end, but calls
``vegetation.compile`` before front-garden and open-ground trees. This replay
uses the normal ``--streets-only`` context and stops immediately after that
call, exporting its unmaterialized material batches. It is a staged audit /
replacement candidate and does not edit production rules or the source world.
"""
from __future__ import annotations

import argparse
import json
import runpy
import sys
from pathlib import Path

import numpy as np

WORK_ROOT = Path(__file__).resolve().parents[1]
if str(WORK_ROOT) not in sys.path:
    sys.path.insert(0, str(WORK_ROOT))

from regen.tree_size_correction import assess_lidar_height, deduplicate_inventory_records


class _StopAfterInventoryTrees(Exception):
    pass


def _merge_inventory_candidates(items):
    """Conservatively merge repeated municipal rows before tree generation."""
    records = []
    for item in items:
        r = dict(item["record"])
        r["position"] = np.asarray(item["position"], dtype=float).tolist()
        r["radiusM"] = float(item["radius"])
        records.append(r)
    merged, audit = deduplicate_inventory_records(records, max_distance_m=1.5)
    by_source = {str(item["record"].get("treeid")): item for item in items}
    result = []
    for audit_row, row in zip(audit, merged):
        refs = [str(ref) for ref in row.get("sourceRecordRefs", [])]
        original = by_source.get(refs[0]) if refs else None
        if original is None:
            raise ValueError("merged inventory row lost its source row reference")
        # Production tree_candidates may already have consolidated repeated
        # source rows. A second audit pass sees that as one row and would
        # otherwise replace its original refs with only the representative ID.
        source_refs = original["record"].get("sourceRecordRefs") or refs
        source_refs = [str(ref) for ref in source_refs]
        source_count = int(original["record"].get("sourceRecordCount", len(source_refs)))
        audit_row["sourceRecordRefs"] = source_refs
        audit_row["sourceRecordCount"] = source_count
        if source_count > 1:
            audit_row["decision"] = "already-merged-by-production-identity"
        item = dict(original)
        item["record"] = dict(original["record"], dbh=row.get("dbh"),
                               sourceRecordRefs=source_refs, sourceRecordCount=source_count)
        item["position"] = np.asarray(row["position"], dtype=float)
        item["radius"] = float(row.get("radiusM", original["radius"]))
        item["sourceRecordRefs"] = source_refs
        result.append(item)
    return result, audit


def _vegetation_arrays(ctx):
    out = {}
    for batch_key, (vertices, faces, uvs, normals) in ctx["batches"].items():
        if not batch_key.startswith("vegetation___"):
            continue
        if not vertices:
            continue
        out[batch_key] = {
            "vertices": np.concatenate(vertices).astype(np.float32, copy=False),
            "faces": np.concatenate(faces).astype(np.int32, copy=False),
            "uvs": np.concatenate(uvs).astype(np.float32, copy=False),
            "normals": np.concatenate(normals).astype(np.float32, copy=False),
        }
    return out


def compare_material_prefix(replay_npz: Path, full_glb: Path, *, tolerance=1e-5):
    """Compare staged corner streams against the beginning of each full GLB batch.

    Positions/normals are compared in source coordinates after the GLB scene
    transform; UVs are compared unchanged. A positive result is required
    before an inventory-only batch may replace a full-scene buffer prefix.
    """
    from regen.glb_stream import GltfStream
    from regen.host_shell_apertures import GLB_TO_SOURCE

    stored = np.load(replay_npz)
    result = {}
    with GltfStream(full_glb) as glb:
        rows = {row["node"]: row for row in glb.rows if row["node"].startswith("vegetation___")}
        for batch in sorted(key[:-len("__vertices")] for key in stored.files if key.endswith("__vertices")):
            if batch not in rows:
                result[batch] = {"status": "missing-full-scene-material"}
                continue
            row = rows[batch]
            prim = row["primitive"]
            idx = (np.asarray(glb._accessor(prim["indices"]), dtype=np.uint32).reshape(-1, 3).copy()
                   if "indices" in prim else np.arange(len(glb._accessor(prim["attributes"]["POSITION"])), dtype=np.uint32).reshape(-1, 3))
            attrs = prim["attributes"]
            p = np.asarray(glb._accessor(attrs["POSITION"]), dtype=np.float32).copy()
            world = (np.c_[p, np.ones(len(p))] @ row["transform"].T)[:, :3]
            source = (GLB_TO_SOURCE[:3, :3] @ world.T).T.astype(np.float32)
            full_pos = source[idx].reshape(-1, 3)
            full_uv = np.asarray(glb._accessor(attrs["TEXCOORD_0"]), dtype=np.float32).copy()[idx].reshape(-1, 2)
            channels = {}
            channel_values = [("vertices", full_pos), ("uvs", full_uv)]
            if "NORMAL" in attrs:
                full_norm = np.asarray(glb._accessor(attrs["NORMAL"]), dtype=np.float32).copy()
                full_norm = (GLB_TO_SOURCE[:3, :3] @ full_norm.T).T.astype(np.float32)[idx].reshape(-1, 3)
                channel_values.append(("normals", full_norm))
            else:
                channels["normals"] = {"status": "no-authored-normal-accessor-in-full-glb"}
            for suffix, actual in channel_values:
                expected = np.asarray(stored[f"{batch}__{suffix}"])
                n = min(len(expected), len(actual))
                diff = np.abs(actual[:n] - expected[:n])
                mismatch = np.any(diff > tolerance, axis=1) if n else np.zeros(0, dtype=bool)
                first = int(np.argmax(mismatch)) if mismatch.any() else n
                channels[suffix] = {
                    "stagedCorners": int(len(expected)), "fullCorners": int(len(actual)),
                    "matchingPrefixCorners": first,
                    "maxPrefixAbsDifference": float(np.max(diff)) if diff.size else 0.0,
                    "matchesEntireStagedStream": first == len(expected) and len(actual) >= len(expected),
                }
            result[batch] = {
                "status": "exact-prefix" if all(v.get("matchesEntireStagedStream", True) for v in channels.values()) else "prefix-mismatch",
                "channels": channels,
            }
    stored.close()
    return result


def _raw_measurement(ctx, original_measure, canopy, x, y, others):
    measured = original_measure(ctx, x, y, others)
    cached = canopy._TREE_INDEX
    if cached is None:
        return measured, {"status": "no-measurement-index"}
    tree, xy, heights = cached
    near = np.asarray(tree.query_ball_point([x, y], 7.0), dtype=int)
    if not len(near):
        return measured, {"status": "no-nearby-returns"}
    d = np.hypot(xy[near, 0] - x, xy[near, 1] - y)
    h = heights[near]
    sample_xy = xy[near]
    nearby_others = [p for p in others if 0 < np.hypot(p[0] - x, p[1] - y) < 14.0]
    if nearby_others:
        o = np.asarray(nearby_others, dtype=float)
        other_d = np.min(np.hypot(xy[near, 0, None] - o[:, 0], xy[near, 1, None] - o[:, 1]), axis=1)
        keep = d <= other_d
        d, h, sample_xy = d[keep], h[keep], sample_xy[keep]
    core = h[d <= 3.0]
    if len(core) < 5:
        return measured, {"status": "too-few-core-returns", "coreReturnCount": int(len(core))}
    bld = ctx.get("all_buildings")
    if bld is None or bld.is_empty:
        return measured, {"status": "no-building-geometry-for-wall-check"}
    from shapely.geometry import Point
    from shapely.ops import nearest_points
    center = Point(float(x), float(y))
    edge = nearest_points(center, bld)[1]
    wall = np.asarray([edge.x - x, edge.y - y], dtype=float)
    wall_distance = float(np.linalg.norm(wall))
    record = next((i["record"] for i in ctx["tree_placements"]
                   if np.linalg.norm(np.asarray(i["position"]) - [x, y]) < 1e-4), {})
    decision = assess_lidar_height(
        np.column_stack((sample_xy, h)),
        [x, y], wall, wall_distance,
        fallback_seed=record.get("treeid"),
    )
    # Production measure_tree already applies this gate. Recompute the normal
    # size only for accepted candidates so this audit can explain its decision.
    if measured is None and decision["status"] != "rejected-wall-facing-upper-tail":
        height = float(np.percentile(core, 95))
        wide = d[h >= .5 * height]
        radius = float(np.clip(np.percentile(wide, 90) if len(wide) >= 5 else 2., 1.5, 7.))
        radius = min(radius, max(1., float(bld.distance(center)) - .3))
        measured = (height, radius)
    return (None if decision["status"] == "rejected-wall-facing-upper-tail" else measured), decision


def replay(area: Path, output: Path, *, corrected: bool):
    """Run the normal area setup and save only inventory vegetation batches."""
    import pipeline.canopy as canopy
    import pipeline.vegetation as vegetation

    generator = WORK_ROOT / "generate_detailed_v14.py"
    output.mkdir(parents=True, exist_ok=True)
    original_compile = vegetation.compile
    original_draw = canopy.draw_tree
    original_measure = canopy.measure_tree
    audit = {"mode": "corrected" if corrected else "baseline", "duplicateGroups": [],
             "heightDecisions": [], "trunkRadiusOverrides": []}

    def stop_after_compile(ctx):
        if corrected:
            ctx["tree_placements"], audit["duplicateGroups"] = _merge_inventory_candidates(ctx["tree_placements"])

            def measured_with_wall_gate(local_ctx, x, y, others=None):
                measured, evidence = _raw_measurement(local_ctx, original_measure, canopy, x, y, others or [])
                record = next((i["record"] for i in local_ctx["tree_placements"]
                               if np.linalg.norm(np.asarray(i["position"]) - [x, y]) < 1e-4), {})
                audit["heightDecisions"].append({
                    "sourceRefs": record.get("sourceRecordRefs", [record.get("treeid")]),
                    "selectedHeightM": float(measured[0]) if measured is not None else evidence.get("fallbackHeightM"),
                    "decision": "accepted-lidar" if measured is not None else "existing-fallback-after-wall-filter",
                    "evidence": evidence,
                })
                return measured

            canopy.measure_tree = measured_with_wall_gate

            def dbh_first_draw(put, cyl, x, y, z, height, crown_radius, *, trunk_radius_m=None):
                item = min(ctx["tree_placements"], key=lambda i: np.linalg.norm(np.asarray(i["position"]) - [x, y]))
                delta = float(np.linalg.norm(np.asarray(item["position"]) - [x, y]))
                if delta > 1e-3:
                    return original_draw(put, cyl, x, y, z, height, crown_radius,
                                         trunk_radius_m=trunk_radius_m)
                audit["trunkRadiusOverrides"].append({
                    "sourceRefs": item.get("sourceRecordRefs", [item["record"].get("treeid")]),
                    "oldRuleRadiusM": min(.45, .06 + .018 * float(height)),
                    "dbHFirstRadiusM": trunk_radius_m,
                })
                return original_draw(put, cyl, x, y, z, height, crown_radius,
                                     trunk_radius_m=trunk_radius_m)

            canopy.draw_tree = dbh_first_draw

        original_compile(ctx)
        arrays = _vegetation_arrays(ctx)
        if not arrays:
            raise RuntimeError("inventory replay emitted no vegetation batches")
        np.savez_compressed(output / "inventory-vegetation.npz",
                            **{f"{batch}__{channel}": value
                               for batch, channels in arrays.items()
                               for channel, value in channels.items()})
        audit["batches"] = {
            batch: {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in channels.items()}
            for batch, channels in arrays.items()
        }
        audit["sourceTreeCount"] = len(ctx["tree_placements"])
        (output / "inventory-vegetation.json").write_text(json.dumps(audit, indent=2) + "\n")
        raise _StopAfterInventoryTrees

    vegetation.compile = stop_after_compile
    sys.argv = [str(generator), "--output", str(output / "unused-generator-output"),
                "--area", str(area), "--streets-only"]
    try:
        runpy.run_path(str(generator), run_name="__main__")
    except _StopAfterInventoryTrees:
        pass
    finally:
        vegetation.compile = original_compile
        canopy.draw_tree = original_draw
        canopy.measure_tree = original_measure
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--area", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--corrected", action="store_true")
    parser.add_argument("--compare-glb", type=Path)
    parser.add_argument("--compare-npz", type=Path)
    args = parser.parse_args()
    if args.compare_glb:
        if not args.compare_npz:
            parser.error("--compare-glb requires --compare-npz")
        print(json.dumps(compare_material_prefix(args.compare_npz, args.compare_glb), indent=2))
        return
    replay(args.area.resolve(), args.output.resolve(), corrected=args.corrected)


if __name__ == "__main__":
    main()
