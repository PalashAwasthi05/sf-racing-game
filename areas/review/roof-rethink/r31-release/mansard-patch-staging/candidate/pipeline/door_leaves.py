"""Geometry helpers for observed door leaf construction.

Door cells are leaf-relative normalized rectangles.  An omitted cell layout
means one plain leaf; it never means a pane grid.  The callbacks use the
facade grammar's metric wall frame and therefore keep casing and leaf finishes
independent.
"""
from __future__ import annotations

import math


def validate_cells(cells):
    """Validate observed leaf-relative [left,bottom,right,top] cells."""
    if not isinstance(cells, list):
        raise ValueError('door cells must be a list of leaf-relative rectangles')
    result = []
    for cell in cells:
        if not isinstance(cell, dict) or cell.get('kind') not in ('panel', 'glass'):
            raise ValueError('door cells need kind panel or glass and a normalized rect')
        rect = cell.get('rect')
        if (not isinstance(rect, (list, tuple)) or len(rect) != 4 or
                any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in rect) or
                not (0 <= rect[0] < rect[2] <= 1 and 0 <= rect[1] < rect[3] <= 1)):
            raise ValueError('door cell rect must be an observed normalized [left,bottom,right,top]')
        result.append({'rect': [float(v) for v in rect], 'kind': cell['kind']})
    return result


def emit_cells(block, bounds, cells, *, depth, panel_material, glass_material='glass', inset=.035):
    """Draw only the typed cells supplied by the source; the opaque leaf is separate."""
    x0, z0, x1, z1 = bounds
    width, height = x1-x0, z1-z0
    for cell in validate_cells(cells):
        a, b, c, d = cell['rect']
        xlo, xhi = x0+width*a, x0+width*c
        zlo, zhi = z0+height*b, z0+height*d
        if cell['kind'] == 'panel':
            # A subtle raised panel is contained within its own measured cell.
            pad = min(inset, (xhi-xlo)*.12, (zhi-zlo)*.12)
            if xhi-xlo > 2*pad+.03 and zhi-zlo > 2*pad+.03:
                block((xlo+xhi)/2, (zlo+zhi)/2, xhi-xlo-2*pad, zhi-zlo-2*pad,
                      .02, panel_material, depth+.035)
        else:
            # Glass sits forward of the opaque leaf. Its perimeter remains framed.
            pad = min(inset, (xhi-xlo)*.08, (zhi-zlo)*.08)
            if xhi-xlo > 2*pad+.03 and zhi-zlo > 2*pad+.03:
                block((xlo+xhi)/2, (zlo+zhi)/2, xhi-xlo-2*pad, zhi-zlo-2*pad,
                      .015, glass_material, depth+.065)


def emit_opaque_leaf(block, bounds, material, *, depth, glass_cells=(), thickness=.08):
    """Draw an opaque leaf around only its explicitly observed glass apertures."""
    x0, z0, x1, z1 = bounds
    panes = []
    for cell in validate_cells(list(glass_cells)) if glass_cells else []:
        if cell['kind'] != 'glass':
            continue
        a, b, c, d = cell['rect']
        pad = min(.035/(x1-x0), (c-a)*.08, .035/(z1-z0), (d-b)*.08)
        panes.append((a+pad, b+pad, c-pad, d-pad))
    if not panes:
        block((x0+x1)/2, (z0+z1)/2, x1-x0, z1-z0, thickness, material, depth)
        return
    xs = sorted({0., 1., *(v for a, _, c, _ in panes for v in (a, c))})
    zs = sorted({0., 1., *(v for _, b, _, d in panes for v in (b, d))})
    for xa, xb in zip(xs, xs[1:]):
        for za, zb in zip(zs, zs[1:]):
            mx, mz = (xa+xb)/2, (za+zb)/2
            if any(a <= mx <= c and b <= mz <= d for a, b, c, d in panes):
                continue
            if xb-xa < .002 or zb-za < .002:
                continue
            block(x0+(xa+xb)*(x1-x0)/2, z0+(za+zb)*(z1-z0)/2,
                  (xb-xa)*(x1-x0), (zb-za)*(z1-z0), thickness, material, depth)


def emit_grille(block, bounds, *, depth, material='charcoal', pitch=.14, rail_height=.035):
    """Draw an open gate: dark cavity backing, then independent rails and bars."""
    x0, z0, x1, z1 = bounds
    width, height = x1-x0, z1-z0
    if width <= .08 or height <= .08:
        return
    # The backing is recessed far enough that clear openings show depth behind bars.
    block((x0+x1)/2, (z0+z1)/2, width, height, .025, 'charcoal', depth-.10)
    for z in (z0+rail_height/2, z1-rail_height/2):
        block((x0+x1)/2, z, width, rail_height, .035, material, depth)
    count = max(1, int(math.ceil(width/max(.04, pitch))))
    for i in range(count+1):
        x = x0 + width*i/count
        block(x, (z0+z1)/2, .018, height, .025, material, depth)


def draw_garage_leaf(block, bounds, style, *, depth):
    """Emit one garage leaf, keeping openings distinct from solid panels."""
    kind = style.get('doorKind', 'panel')
    material = style.get('material', 'zinc')
    if kind == 'open':
        return
    if kind == 'grille':
        emit_grille(block, bounds, depth=depth, material=style.get('slatMaterial', material))
        return
    x0, z0, x1, z1 = bounds
    cells = style.get('doorCells')
    if not cells and kind == 'glazed':
        lites = style.get('lites')
        if isinstance(lites, (list, tuple)) and len(lites) == 2:
            nx, nz = max(1, int(lites[0])), max(1, int(lites[1]))
            cells = [{'kind': 'glass', 'rect': [i/nx, j/nz, (i+1)/nx, (j+1)/nz]}
                     for i in range(nx) for j in range(nz)]
        else:
            cells = [{'kind': 'glass', 'rect': [.04, .04, .96, .96]}]
    cells = validate_cells(cells) if cells else []
    # Segment the opaque leaf around glass openings; unspecified regions stay solid.
    emit_opaque_leaf(block, bounds, material, depth=depth, glass_cells=cells)
    if cells:
        emit_cells(block, bounds, cells, depth=depth, panel_material=material)
    if not cells and style.get('doorConstruction') == 'roller':
        # A roller shutter has continuous horizontal slats; this is distinct
        # from a sectional door, whose joint positions must come from cells.
        pitch = float(style.get('slatPitchM', .13))
        z = z0 + pitch/2
        while z < z1:
            block((x0+x1)/2, z, x1-x0, .018, .025, style.get('slatMaterial', material), depth+.05)
            z += pitch
