"""Source-observed wall signs: preserve lettering layout across host segments.

Boards and lettering are separate observations. Unread letters and unsupported
mounts remain unresolved; a material label never supplies business text.
"""
import math
import re

from shapely import affinity
from shapely.geometry import box


def compile_reading(reading, piece, material):
    signs, unresolved = [], []
    images = {str(v.get('image')) for v in piece.get('views', [])}
    width, height = float(piece['lengthM']), float(piece['heightM'])
    color = lambda x: isinstance(x, str) and re.fullmatch(r'#[0-9a-fA-F]{6}', x)
    finite = lambda x: isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
    for i, raw in enumerate(reading.get('signs') or []):
        def gap(reason):
            unresolved.append({'kind': 'sign-reading', 'index': i, 'reason': reason})
        if not isinstance(raw, dict):
            gap('Sign must be a structured observation.'); continue
        if raw.get('mount') != 'wall':
            gap('Projecting, rooftop and freestanding signs require a separately measured support.'); continue
        sources = raw.get('sourceImages')
        if not isinstance(sources, list) or not sources or not set(map(str, sources)) <= images:
            gap('Sign source must identify an inspected camera of this facade.'); continue
        bounds = [raw.get(k) for k in ('x0', 'z0', 'x1', 'z1')]
        if not all(finite(v) for v in bounds) or not (bounds[0] < bounds[2] and bounds[1] < bounds[3]):
            gap('Sign needs ordered metric bounds.'); continue
        # Do not clamp/re-fit text to a cropped frame. The renderer clips the
        # complete layout to each physical host segment without resizing it.
        sg = {'rect': [bounds[0]/width, bounds[1]/height, bounds[2]/width, bounds[3]/height],
              'sourceIds': ['mapillary:' + str(s) for s in sources], 'mount': 'wall'}
        offset = raw.get('offsetM', .006)
        if not finite(offset) or not 0 <= offset <= .5:
            gap('Wall-sign offset must be finite and between zero and half a metre.'); continue
        sg['offsetM'] = offset
        board = raw.get('board')
        if isinstance(board, dict):
            depth = board.get('thicknessM')
            if board.get('family') and color(board.get('hex')) and finite(depth) and 0 < depth <= .5:
                sg.update(boardMaterial=material(board), boardThicknessM=depth)
            else:
                gap('Board requires observed finish and thickness; no guessed panel was emitted.')
        text = raw.get('text')
        if isinstance(text, str) and text.strip():
            if raw.get('textLegible') is True and color(raw.get('textHex')):
                sg.update(text=text, textMaterial=raw['textHex'],
                          fontStatus='approximate letterforms; literal text observed')
                # Flat paint is the conservative surface representation. Raised
                # letters require an explicit measured thickness.
                depth = raw.get('letterDepthM', .001)
                if not finite(depth) or not 0 <= depth <= .5:
                    gap('Invalid lettering depth.'); sg.pop('text')
                else:
                    sg['letterDepthM'] = max(.001, depth)
                font = raw.get('font')
                if isinstance(font, str) and '/' not in font and '\\' not in font and font.lower().endswith(('.ttf', '.otf')):
                    sg['font'] = font
            else:
                gap('Literal text requires a legibility decision and observed color.')
        elif raw.get('textLegible') is not True:
            gap('Letters unread; retain the observed board without invented text.')
        if sg.get('text') or sg.get('boardMaterial'):
            signs.append(sg)
    return signs, unresolved


def parts(sign, width, height, support, contours):
    """Return clipped elevation polygons without re-centering a split sign."""
    x0, z0, x1, z1 = [float(v)*s for v, s in zip(sign['rect'], (width, height, width, height))]
    if x1 <= x0 or z1 <= z0:
        return []
    domain = box(x0, z0, x1, z1).intersection(support)
    if domain.is_empty:
        return []
    offset = float(sign.get('offsetM', 0.))
    board_depth = float(sign.get('boardThicknessM', sign.get('depthM', .05))) if sign.get('boardMaterial') else 0.
    out = []
    if sign.get('boardMaterial'):
        out.append((domain, offset + board_depth, board_depth, sign['boardMaterial']))
    if sign.get('text'):
        glyphs, ww, hh = contours(sign['text'], sign.get('font', 'Arial Narrow Bold.ttf'))
        if ww <= 0 or hh <= 0 or glyphs.is_empty:
            return out
        scale = min((x1-x0)*.92/ww, (z1-z0)*.78/hh)
        # Font bitmap coordinates point downward; the facade points upward.
        glyphs = affinity.affine_transform(glyphs, [scale, 0., 0., -scale,
                    (x0+x1-ww*scale)/2, (z0+z1+hh*scale)/2]).intersection(domain)
        if not glyphs.is_empty:
            depth = max(.001, float(sign.get('letterDepthM', .001)))
            out.append((glyphs, offset + board_depth + depth + .001, depth,
                        sign.get('textMaterial', 'white')))
    return out
