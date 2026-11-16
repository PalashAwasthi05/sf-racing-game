#!/usr/bin/env python3
"""Measure repeated stepped contours in generic rectified street images.

The input images should be rectified crops with the facade upright. This is a
bounded evidence extractor: it finds a large dark connected mass, samples its
upper contour, and measures short horizontal runs separated by vertical rises.
It makes no building or feature classification and does not write to inputs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def extract(path: Path, pixels_per_metre: float | None = None,
            rect_pad_px: float | None = None,
            reference_height_m: float | None = None) -> dict:
    rgb = np.asarray(Image.open(path).convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    # Choose the largest substantial dark mass whose top enters the upper
    # quarter of the rectified image.
    dark = (gray < 80).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
    candidates = [i for i in range(1, n)
                  if stats[i, cv2.CC_STAT_AREA] >= .05 * w * h
                  and stats[i, cv2.CC_STAT_TOP] < .25 * h
                  and stats[i, cv2.CC_STAT_WIDTH] >= .20 * w]
    if not candidates:
        return {"image": path.name, "status": "abstain-no-large-upper-mass"}
    component = max(candidates, key=lambda i: stats[i, cv2.CC_STAT_AREA])
    x0, y0, cw, ch, area = map(int, stats[component])

    # First dark pixel per column is a raster approximation of the upper edge.
    profile = [None] * w
    for x in range(x0, x0 + cw):
        ys = np.flatnonzero(labels[:, x] == component)
        if len(ys):
            profile[x] = int(ys[0])
    # A one- or two-column hole can be caused by JPEG thresholding on an
    # otherwise continuous edge; interpolate only those short gaps.
    known = [x for x, y in enumerate(profile) if y is not None]
    for a, b in zip(known, known[1:]):
        if 1 < b - a <= 3:
            for x in range(a + 1, b):
                profile[x] = round(profile[a] + (profile[b] - profile[a]) * (x - a) / (b - a))
    peak_y = min(v for v in profile if v is not None)
    peak_x = profile.index(peak_y)
    # Limit measurement to the upper contour close to its highest point.
    selected = [x for x, y in enumerate(profile)
                if y is not None and y <= peak_y + .32 * w]

    def side_metrics(xs: list[int]) -> dict:
        if len(xs) < 8:
            return {"columns": len(xs), "stepTransitions": 0,
                    "stepEvents": [],
                    "contrastCoverage": 0.0}
        raw = np.asarray([profile[x] for x in xs], dtype=np.float32).reshape(1, -1)
        # Suppress one-pixel JPEG/raster wobble, then quantize to three-pixel
        # bands so a single flat tread remains one run.
        smooth = cv2.medianBlur(raw, 5)[0]
        levels = np.rint(smooth / 3.0) * 3.0
        runs: list[list[int]] = []
        for x, level in zip(xs, levels):
            y = int(level)
            if not runs or runs[-1][2] != y or runs[-1][1] != x - 1:
                runs.append([x, x, int(y)])
            else:
                runs[-1][1] = x
        # Collapse isolated one- or two-column bevel/alias runs toward the
        # closer neighboring tread before measuring physical step dimensions.
        j = 1
        while j < len(runs) - 1:
            if runs[j][1] - runs[j][0] + 1 <= 2:
                prev, cur, nxt = runs[j - 1], runs[j], runs[j + 1]
                if abs(cur[2] - prev[2]) <= abs(cur[2] - nxt[2]):
                    prev[1] = cur[1]
                else:
                    nxt[0] = cur[0]
                runs.pop(j)
            else:
                j += 1
        events = []
        for a, b in zip(runs, runs[1:]):
            run_w = a[1] - a[0] + 1
            next_w = b[1] - b[0] + 1
            rise = abs(b[2] - a[2])
            if 4 <= run_w <= 18 and 4 <= next_w <= 18 and 5 <= rise <= 18:
                events.append({"xPx": round((a[0] + a[1]) / 2, 1),
                               "runPx": run_w, "risePx": rise,
                               "yBeforePx": a[2], "yAfterPx": b[2]})

        # A contrasting bright/neutral edge is measured near the contour,
        # independently of the stair count. Report coverage, not a decision.
        covered = 0
        for x in xs:
            y = int(profile[x])
            xa, xb = max(0, x - 2), min(w, x + 3)
            ya, yb = max(0, y - 9), min(h, y + 10)
            patch = rgb[ya:yb, xa:xb].astype(np.int16)
            neutral = (patch.min(axis=2) >= 125) & ((patch.max(axis=2) - patch.min(axis=2)) <= 55)
            covered += int(neutral.any())
        if pixels_per_metre and rect_pad_px is not None and reference_height_m is not None:
            for event in events:
                event["xM"] = round((event["xPx"] - rect_pad_px) / pixels_per_metre, 3)
                z0 = reference_height_m - (event["yBeforePx"] - rect_pad_px) / pixels_per_metre
                z1 = reference_height_m - (event["yAfterPx"] - rect_pad_px) / pixels_per_metre
                event["zM"] = round((z0 + z1) / 2, 3)
                event["runM"] = round(event["runPx"] / pixels_per_metre, 3)
                event["riseM"] = round(event["risePx"] / pixels_per_metre, 3)
        return {"columns": len(xs), "stepTransitions": len(events),
                "stepEvents": events,
                "contrastCoverage": round(covered / len(xs), 3)}

    left = [x for x in selected if x < peak_x]
    right = [x for x in selected if x > peak_x]
    return {
        "image": path.name,
        "status": "measured",
        "imageSizePx": [w, h],
        "massAreaPx": area,
        "upperContourPeakPx": [peak_x, peak_y],
        "left": side_metrics(left),
        "right": side_metrics(right),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--pixels-per-metre", type=float)
    parser.add_argument("--rect-pad-px", type=float)
    parser.add_argument("--reference-height-m", type=float)
    args = parser.parse_args()
    print(json.dumps({"method": "dark-upper-contour-runs-v2",
                      "measurements": [extract(p, args.pixels_per_metre,
                                               args.rect_pad_px,
                                               args.reference_height_m)
                                       for p in args.images]}, indent=2))


if __name__ == "__main__":
    main()
