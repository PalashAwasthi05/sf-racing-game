"""Pure helpers for tracing a facade reading to its source crop and camera."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reading_sha256(reading):
    payload = json.dumps(reading, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _records(document):
    if isinstance(document, dict):
        return document.get("pieces", [])
    return document if isinstance(document, list) else []


def load_registered_sides(directory):
    """Load every sides*.json, preferring the duplicate with more usable views.

    Tied records that disagree are retained as an ambiguity instead of choosing
    a camera file based on incidental filename ordering.
    """
    candidates = {}
    for path in sorted(Path(directory).glob("sides*.json")):
        try:
            document = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for record in _records(document):
            if not isinstance(record, dict) or not isinstance(record.get("id"), str):
                continue
            views = record.get("views") if isinstance(record.get("views"), list) else []
            candidates.setdefault(record["id"], []).append((len(views), path, record))

    result = {}
    for side_id, records in candidates.items():
        best_count = max(q[0] for q in records)
        best = [q for q in records if q[0] == best_count]
        distinct = {json.dumps(q[2], sort_keys=True, separators=(",", ":")) for q in best}
        if len(distinct) > 1:
            result[side_id] = {"status": "ambiguous", "candidates": [q[1] for q in best]}
        else:
            _, path, record = sorted(best, key=lambda q: str(q[1]))[0]
            result[side_id] = {"status": "ok", "path": path, "piece": record}
    return result


def select_source_view(nominal_piece, registered_sides, use, *, registered_required):
    """Select a crop view while keeping the nominal wall frame authoritative."""
    side_id = nominal_piece.get("id")
    if registered_required:
        source = registered_sides.get(side_id)
        if source is None:
            return {"status": "unresolved", "reason": "registered side record missing"}
        if source.get("status") != "ok":
            return {"status": "unresolved", "reason": "registered side record ambiguous"}
        piece = source["piece"]
        for dimension in ("lengthM", "heightM"):
            a, b = nominal_piece.get(dimension), piece.get(dimension)
            if not isinstance(a, (int, float)) or not isinstance(b, (int, float)) or abs(float(a) - float(b)) > 1e-4:
                return {"status": "unresolved", "reason": f"registered frame {dimension} mismatch"}
        for field in ("frameU", "normal", "planeV"):
            a, b = nominal_piece.get(field), piece.get(field)
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                matches = abs(float(a) - float(b)) <= 1e-4
            elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)) and len(a) == len(b):
                matches = all(isinstance(x, (int, float)) and isinstance(y, (int, float)) and abs(float(x) - float(y)) <= 1e-4 for x, y in zip(a, b))
            else:
                matches = False
            if not matches:
                return {"status": "unresolved", "reason": f"registered frame {field} mismatch"}
        views = piece.get("views")
        if not isinstance(views, list) or isinstance(use, bool) or not isinstance(use, int) or not 0 <= use < len(views):
            return {"status": "unresolved", "reason": "registered crop view missing"}
        view = views[use]
        if not isinstance(view, dict) or not view.get("image"):
            return {"status": "unresolved", "reason": "registered crop source image missing"}
        missing = [field for field in ("rect", "persp") if not isinstance(view.get(field), str) or not view[field]]
        if missing:
            return {"status": "unresolved", "reason": "registered crop missing " + ", ".join(missing) + " filename"}
        return {"status": "ok", "registered": True, "view": view, "side_path": source["path"],
                "sourceIds": ["mapillary:" + str(view["image"])]}

    views = nominal_piece.get("views")
    if not isinstance(views, list) or isinstance(use, bool) or not isinstance(use, int) or not 0 <= use < len(views):
        return {"status": "unresolved", "reason": "nominal source view missing"}
    view = views[use]
    if not isinstance(view, dict) or not view.get("image"):
        return {"status": "unresolved", "reason": "nominal source image missing"}
    missing = [field for field in ("rect", "persp") if not isinstance(view.get(field), str) or not view[field]]
    if missing:
        return {"status": "unresolved", "reason": "nominal crop missing " + ", ".join(missing) + " filename"}
    return {"status": "ok", "registered": False, "view": view, "side_path": None,
            "sourceIds": ["mapillary:" + str(view["image"])]}


def relative_path(path, root):
    try:
        return Path(os.path.relpath(Path(path).resolve(), Path(root).resolve())).as_posix()
    except (OSError, TypeError, ValueError):
        return None


def missing_crop_files(selection, image_directory):
    """Return named missing rect/persp files for a selected crop view."""
    view = selection.get("view") or {}
    missing = []
    for field in ("rect", "persp"):
        name = view.get(field)
        if not isinstance(name, str) or not name:
            missing.append(f"{field} filename")
        elif not (Path(image_directory) / name).is_file():
            missing.append(f"{field} file {name}")
    return missing
