"""Bounded source-facade orthographic atlases; no world-space geometry reduction.

Two 2048-square pages maximum. Only one page and one building's original
envelope/facade triangles and material thumbnails are resident during baking.
Transparent pixels preserve the actual projected source silhouette and voids.
"""
import hashlib
import io
import math

import numpy as np
from PIL import Image
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union

from package_drive_stream import layer_of

PAGE = 2048
MAX_PAGES = 2
PIXEL_TARGET = 5_500_000
PADDING = 2


def linear(rgb):
    return np.where(rgb <= .04045, rgb/12.92, ((rgb+.055)/1.055)**2.4)


def srgb(rgb):
    rgb = np.clip(rgb, 0, 1)
    return np.where(rgb <= .0031308, rgb*12.92, 1.055*rgb**(1/2.4)-.055)


def footprint(source, members):
    polygons = []
    for node, matrix in members:
        mesh = source.model['meshes'][node['mesh']]
        if layer_of(node, mesh) != 'foundations':
            continue
        for primitive in mesh['primitives']:
            p = source.array(primitive['attributes']['POSITION'])
            p = p @ matrix[:3, :3].T + matrix[:3, 3]
            indices = source.array(primitive['indices']).reshape(-1, 3) if 'indices' in primitive else np.arange(len(p)).reshape(-1, 3)
            for face in p[indices][:, :, [0, 2]]:
                polygon = Polygon(face)
                if polygon.area > 1e-5:
                    polygons.append(polygon)
    if not polygons:
        return None
    return unary_union(polygons).buffer(0).simplify(.12, preserve_topology=True)


def wall_records(source, plan):
    result = []
    for key, (_, _, members) in sorted(plan.groups.items()):
        if key[0] != 'building':
            continue
        bid = key[1]
        shape = footprint(source, members)
        if shape is None or shape.is_empty:
            continue
        bottom, top = plan.hulls[bid][1:]
        if top-bottom < .5:
            continue
        parts = [shape] if shape.geom_type == 'Polygon' else list(shape.geoms)
        for part in parts:
            for ring in [part.exterior, *part.interiors]:
                points = np.asarray(ring.coords)
                for a, b in zip(points, points[1:]):
                    length = float(np.linalg.norm(b-a))
                    if length < .3:
                        continue
                    u = (b-a)/length
                    normal = np.array([u[1], -u[0]])
                    middle = (a+b)/2
                    if shape.contains(Point(middle+normal*.10)):
                        normal = -normal
                    pad = min(.55, length*.10)
                    result.append({'building': bid, 'owner': plan.owners[bid], 'a': a,
                        'u': u, 'normal': normal, 'length': length, 'pad': pad,
                        'bottom': bottom, 'top': top, 'widthM': length+2*pad, 'heightM': top-bottom})
    return result


def pack(records):
    area = sum(r['widthM']*r['heightM'] for r in records)
    density = min(3., math.sqrt(PIXEL_TARGET/max(area, 1)))
    for attempt in range(40):
        dimensions = [(min(384, max(4, math.ceil(r['widthM']*density))),
                       min(384, max(4, math.ceil(r['heightM']*density)))) for r in records]
        order = sorted(range(len(records)), key=lambda i: (-dimensions[i][1], -dimensions[i][0], i))
        x = y = row = page = 0
        placements = {}
        for i in order:
            width, height = dimensions[i]
            w, h = width+2*PADDING, height+2*PADDING
            if x+w > PAGE:
                x = 0; y += row; row = 0
            if y+h > PAGE:
                page += 1; x = y = row = 0
            if page >= MAX_PAGES:
                break
            placements[i] = (page, x+PADDING, y+PADDING, width, height)
            x += w; row = max(row, h)
        if len(placements) == len(records):
            for i, location in placements.items():
                records[i]['atlas'] = location
            return {'pages': page+1, 'densityPxM': density, 'texels': (page+1)*PAGE*PAGE,
                    'rgbaMipBytes': (page+1)*PAGE*PAGE*4*4//3, 'walls': len(records)}
        density *= .90
    raise ValueError('Facade atlas cannot fit the explicit two-page budget')


class BuildingSource:
    def __init__(self, source, members):
        self.source = source
        self.parts = []
        self.images = {}
        for node, matrix in members:
            mesh = source.model['meshes'][node['mesh']]
            if layer_of(node, mesh) not in ('buildings', 'facades', 'facade'):
                continue
            for primitive in mesh['primitives']:
                if primitive.get('mode', 4) != 4:
                    continue
                attrs = primitive['attributes']
                p = source.array(attrs['POSITION'])
                p = (p @ matrix[:3, :3].T + matrix[:3, 3]).astype('<f4')
                index = source.array(primitive['indices']).reshape(-1, 3) if 'indices' in primitive else np.arange(len(p)).reshape(-1, 3)
                faces = p[index]
                color = self.colors(primitive, index)
                self.parts.append((faces, color))

    def image(self, index):
        if index in self.images:
            return self.images[index]
        image = self.source.model['images'][index]
        start, length = self.source.view_range(image['bufferView'])
        self.source.stream.seek(self.source.offset+start)
        encoded = self.source.stream.read(length)
        with Image.open(io.BytesIO(encoded)) as im:
            im = im.convert('RGBA'); im.thumbnail((256, 256), Image.Resampling.BOX)
            pixels = np.asarray(im, dtype=float)/255
        pixels[:, :, :3] = linear(pixels[:, :, :3])
        self.images[index] = pixels
        return pixels

    def colors(self, primitive, index):
        model = self.source.model
        material = model.get('materials', [])[primitive.get('material', 0)]
        pbr = material.get('pbrMetallicRoughness', {})
        color = np.tile(pbr.get('baseColorFactor', [1, 1, 1, 1]), (len(index), 1)).astype(float)
        attrs = primitive['attributes']
        if 'COLOR_0' in attrs:
            aid = attrs['COLOR_0']; tint = self.source.array(aid)
            meta = model['accessors'][aid]; values = tint.astype(float)
            if meta.get('normalized') and tint.dtype.kind in 'ui':
                values /= np.iinfo(tint.dtype).max
            color[:, :values.shape[1]] *= values[index].mean(axis=1)
        texture = pbr.get('baseColorTexture')
        uv_key = 'TEXCOORD_' + str((texture or {}).get('texCoord', 0))
        if texture and uv_key in attrs:
            entry = model['textures'][texture['index']]
            pixels = self.image(entry['source'])
            uv = self.source.array(attrs[uv_key])[index].mean(axis=1)
            sampler = model.get('samplers', [])[entry['sampler']] if 'sampler' in entry else {}
            for axis, wrap in enumerate(('wrapS', 'wrapT')):
                mode = sampler.get(wrap, 10497)
                if mode == 33071:
                    uv[:, axis] = np.clip(uv[:, axis], 0, 1-1e-9)
                elif mode == 33648:
                    uv[:, axis] = 1-np.abs(np.mod(uv[:, axis], 2)-1)
                else:
                    uv[:, axis] = np.mod(uv[:, axis], 1)
            xy = np.floor(uv*[pixels.shape[1], pixels.shape[0]]).astype(int)
            xy[:, 0] = np.clip(xy[:, 0], 0, pixels.shape[1]-1)
            xy[:, 1] = np.clip(xy[:, 1], 0, pixels.shape[0]-1)
            color *= pixels[xy[:, 1], xy[:, 0]]
        rgb = np.rint(srgb(color[:, :3])*255).astype('u1')
        # Glass occupies a real facade surface; alpha is a silhouette mask, not
        # a second transparency compositing pass over unrelated distant walls.
        alpha = np.where(color[:, 3] > .02, 255, 0).astype('u1')
        return np.column_stack((rgb, alpha))

    def render(self, record):
        _, _, _, width, height = record['atlas']
        pixels = np.zeros((height, width, 4), dtype='u1')
        zbuffer = np.full((height, width), -np.inf, dtype='<f4')
        triangles, colors, depths = [], [], []
        a, u, n = record['a'], record['u'], record['normal']
        for faces, color in self.parts:
            horizontal = faces[:, :, [0, 2]]-a
            along = horizontal @ u
            depth = horizontal @ n
            cross = np.cross(faces[:, 1]-faces[:, 0], faces[:, 2]-faces[:, 0])
            lengths = np.linalg.norm(cross, axis=1)
            facing = np.abs(cross[:, 0]*n[0]+cross[:, 2]*n[1])/np.maximum(lengths, 1e-20)
            keep = ((along.max(axis=1) >= -record['pad']) & (along.min(axis=1) <= record['length']+record['pad'])
                    & (facing > .02)
                    & (color[:, 3] > 0))
            if not keep.any():
                continue
            selected = faces[keep]
            xy = np.stack(((along[keep]+record['pad'])/record['widthM']*width,
                           (record['top']-selected[:, :, 1])/record['heightM']*height), axis=2)
            v, w = xy[:, 1]-xy[:, 0], xy[:, 2]-xy[:, 0]
            projected = np.abs(v[:, 0]*w[:, 1]-v[:, 1]*w[:, 0])
            valid = projected >= .10
            triangles.append(xy[valid]); colors.append(color[keep][valid]); depths.append(depth[keep][valid])
        if triangles:
            triangles, colors, depths = map(np.concatenate, (triangles, colors, depths))
            for triangle, color, z in zip(triangles, colors, depths):
                low = np.maximum(np.floor(triangle.min(axis=0)).astype(int), [0, 0])
                high = np.minimum(np.ceil(triangle.max(axis=0)).astype(int), [width, height])
                x0, y0 = low; x1, y1 = high
                if x0 >= x1 or y0 >= y1:
                    continue
                a, b, c = triangle
                denominator = (b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1])
                if abs(denominator) < 1e-8:
                    continue
                xx = np.arange(x0, x1)[None, :]+.5
                yy = np.arange(y0, y1)[:, None]+.5
                u = ((b[1]-c[1])*(xx-c[0])+(c[0]-b[0])*(yy-c[1]))/denominator
                v = ((c[1]-a[1])*(xx-c[0])+(a[0]-c[0])*(yy-c[1]))/denominator
                w = 1-u-v
                depth = u*z[0]+v*z[1]+w*z[2]
                target = zbuffer[y0:y1, x0:x1]
                keep = (u >= -1e-6) & (v >= -1e-6) & (w >= -1e-6) & (depth >= target-1e-5)
                target[keep] = depth[keep]
                pixels[y0:y1, x0:x1][keep] = color
        return Image.fromarray(pixels, 'RGBA')


def build_facade_impostors(source, plan, writer, nodes, meshes):
    records = wall_records(source, plan)
    if not records:
        return {'pages': 0, 'texels': 0, 'walls': 0, 'triangles': 0, 'buildings': 0}
    stats = pack(records)
    stats.update(triangles=2*len(records), buildings=len({r['building'] for r in records}), encodedBytes=0)
    model = source.model
    samplers = model.setdefault('samplers', [])
    sampler = len(samplers); samplers.append({'magFilter': 9729, 'minFilter': 9987, 'wrapS': 33071, 'wrapT': 33071})
    for page in range(stats['pages']):
        atlas = Image.new('RGBA', (PAGE, PAGE), (0, 0, 0, 0))
        selected = [r for r in records if r['atlas'][0] == page]
        by_building = {}
        for record in selected:
            by_building.setdefault(record['building'], []).append(record)
        for bid, walls in sorted(by_building.items()):
            building = BuildingSource(source, plan.groups[('building', bid)][2])
            for record in walls:
                image = building.render(record)
                _, x, y, _, _ = record['atlas']
                atlas.paste(image, (x, y))
            del building
        encoded = io.BytesIO(); atlas.save(encoded, format='PNG', optimize=False)
        png = encoded.getvalue(); stats['encodedBytes'] += len(png)
        images = model.setdefault('images', [])
        image_id = len(images)
        images.append({'bufferView': writer.append(png), 'mimeType': 'image/png',
            'extras': {'neighborhoodWriterView': True, 'viewerImageSha256': hashlib.sha256(png).hexdigest()}})
        textures = model.setdefault('textures', [])
        texture = len(textures); textures.append({'sampler': sampler, 'source': image_id})
        material = len(plan.materials)
        plan.materials.append({'name': 'neighborhood_facade_impostor_page_'+str(page),
            'pbrMetallicRoughness': {'baseColorFactor': [1, 1, 1, 1], 'baseColorTexture': {'index': texture},
                'roughnessFactor': .85, 'metallicFactor': 0},
            'alphaMode': 'MASK', 'alphaCutoff': .25, 'doubleSided': True,
            'extras': {'class': 'distant_exterior', 'bakedAppearance': True, 'layer': 'facades'}})
        by_owner = {}
        for record in selected:
            by_owner.setdefault((record['owner'],record['building']), []).append(record)
        for (owner, feature), walls in sorted(by_owner.items()):
            cell = plan.cells[owner]['id']
            positions, normals, uvs, indices = [], [], [], []
            for record in walls:
                _, x, y, w, h = record['atlas']
                a, u, n = record['a'], record['u'], record['normal']
                # Source-derived projecting relief stays within its own wall.
                left = a-u*record['pad']+n*.035
                right = a+u*(record['length']+record['pad'])+n*.035
                bottom, top = record['bottom'], record['top']
                p = np.array([[left[0], bottom, left[1]], [right[0], bottom, right[1]],
                              [right[0], top, right[1]], [left[0], top, left[1]]], dtype='<f4')
                normal = np.array([n[0], 0, n[1]])
                offset = 4*len(positions)
                index = np.array([0, 1, 2, 0, 2, 3], dtype='<u4')
                if np.dot(np.cross(p[1]-p[0], p[2]-p[0]), normal) < 0:
                    index = index.reshape(-1, 3)[:, ::-1].reshape(-1)
                positions.append(p); normals.append(np.tile(normal, (4, 1)))
                uvs.append(np.array([[x/PAGE, (y+h)/PAGE], [(x+w)/PAGE, (y+h)/PAGE],
                                     [(x+w)/PAGE, y/PAGE], [x/PAGE, y/PAGE]], dtype='<f4'))
                indices.append(index+offset)
            p, normal, uv, index = map(np.concatenate, (positions, normals, uvs, indices))
            extra = {'class': 'distant_exterior', 'streamCell': cell, 'featureId':feature, 'bakedAppearance': True, 'layer': 'facades'}
            attrs = {'POSITION': writer.accessor(p, {'componentType': 5126, 'type': 'VEC3', 'min': [], 'max': []}),
                     'NORMAL': writer.accessor(normal.astype('<f4'), {'componentType': 5126, 'type': 'VEC3'}),
                     'TEXCOORD_0': writer.accessor(uv, {'componentType': 5126, 'type': 'VEC2'})}
            primitive = {'attributes': attrs, 'material': material,
                         'indices': writer.accessor(index.reshape(-1, 1), {'componentType': 5125, 'type': 'SCALAR'})}
            nodes.append({'name': 'facades__'+cell+'___neighborhood_facade_impostor', 'mesh': len(meshes), 'extras': extra})
            meshes.append({'primitives': [primitive], 'extras': extra})
    return stats
