"""Landmark exceptions: authored builders for the few structures the general rules cannot represent.

Each entry in landmarks.json is keyed by a footprint id (a join key) and owns that footprint: the builder emits its
whole envelope (surfaces, walls, playing field, props) and the observed street frontages are compiled on top of it.
The general heuristic never consults this registry for its own rules; the generator only asks whether a footprint is
owned and hands it over. Builders measure what they can (LiDAR surface and intensity, mapped outlines) and name the
parts they author.
"""
import copy, importlib, json, pathlib

REGISTRY = json.loads((pathlib.Path(__file__).parent / 'landmarks.json').read_text())


def entry(c):
    """The landmark entry owning this footprint candidate, or None. Split parts join through their source id."""
    return REGISTRY['landmarks'].get(c.get('sourceId', c['id']))


def build(ctx, c, z0, resolved, lm):
    """Run the entry's builder. Returns {'height', 'roofParts', 'site', 'record'}."""
    return importlib.import_module('.' + lm['builder'], __name__).build(ctx, c, z0, resolved, lm)


def frontage_view(resolved, lm):
    """Observed frontages as the landmark keeps them: facade planes, towers and trim stay; the depth they replace
    shrinks to the frontage itself (the builder owns everything behind), open decks and light banks go (the builder
    takes both from the LiDAR surface), and braced frames keep only their frame."""
    out = []
    for r in resolved:
        p = copy.deepcopy(r['profile']); p['replaceDepthM'] = lm.get('frontageDepthM', 3)
        kept = []
        for a in p.get('assemblies', []):
            if a['op'] == 'lightBank': continue
            if a['op'] == 'bracedFrame': a.pop('rearDepthM', None)
            kept.append(a)
        p['assemblies'] = kept
        # Root returns were sized from the full replaced depth when the nodes were resolved.
        nodes = copy.deepcopy(r['nodes'])
        for d in nodes:
            if d.get('parentId') is None: d['parentDepthM'] = -p['replaceDepthM']
        out.append({**r, 'profile': p, 'nodes': nodes})
    return out
