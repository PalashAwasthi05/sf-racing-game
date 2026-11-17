"""Only the source fields needed by photo preparation; never a full world DOM."""
from pathlib import Path
import numpy as np
import ijson


def items(path, prefix):
    with Path(path).open('rb') as source:
        yield from ijson.items(source, prefix, use_float=True)


def photo_context(fc):
    center = next(items(fc.WORLD_PATH, 'center'))
    grid = next(items(fc.WORLD_PATH, 'grid'), None)
    if grid:
        n = int(grid['n'])
        values = np.fromiter(items(fc.WORLD_PATH, 'ground.item'), dtype=np.float64)
        if len(values) != n*n:
            raise ValueError('Terrain sample count does not match grid dimensions')
        fc._GROUND = (float(grid['min']), float(grid['step']), n, values.reshape(n,n))
    else:
        fc._GROUND = False
    buildings = {b['id']: {'id':b['id'], 'center':b['center']}
                 for b in items(fc.WORLD_PATH, 'buildings.item')}
    return buildings, fc.LocalFrame(center['lon'], center['lat'])


def facade_walls(path, buildings, *, tiers=False):
    fields = ('buildingId','midpoint','normal','lengthM','heightM','zBaseM','status','streetFacing')
    for wall in items(path, 'facadeWalls.item'):
        if wall['buildingId'] in buildings and ((wall.get('zBaseM') or 0) > .5) == tiers:
            yield {key:wall[key] for key in fields if key in wall}
