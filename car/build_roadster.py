"""Build the roadster model: python build_roadster.py <out.glb>"""
import sys, time
import numpy as np
import roadster as R
import wheels as Wh
from mesher import mesh_solid, mesh_solid_sharp, Mesh, split, merge, refine_near, autosmooth
from glb import GLB

# finish names shared with the viewer (app/car.ts `finishes`), which gives each its material
FINISHES = ['paint', 'hoop', 'black_gloss', 'black_satin', 'black_matte', 'gap', 'glass', 'lens', 'lamp_chrome', 'lamp_black',
            'drl', 'tail_lens', 'tail_led', 'reflector', 'tyre', 'rim', 'disc', 'hat', 'caliper', 'cabin', 'cabin_soft',
            'cabin_trim', 'steel', 'mirror']


def build(out, h=.005, mirror_to=None):
    t0 = time.time()
    g = GLB()
    for f in FINISHES:
        g.material(f)
    V, T = mesh_solid_sharp(R.body, *R.BOUNDS, h, tolerance=.00025)
    body = Mesh(V, T).with_normals(R.body)
    print('body %d tris %.1fs' % (len(T), time.time() - t0))
    body = refine_near(body, [region for _, region in R.FINISH_REGIONS], R.body, h_target=.012,
                       lines=[R.gap_centre(line) for line in R.GAP_LINES], h_line=.02)
    print('refined %d tris %.1fs' % (len(body.T), time.time() - t0))
    parts = {}
    rest = body
    for finish, region in R.FINISH_REGIONS:
        inside, rest = split(rest, region, R.body)
        parts.setdefault(finish, []).append(inside)
    # panel gaps: a band either side of each centre line, cut exactly on both edges
    for line in R.GAP_LINES:
        near_f, band_f = R.gap_cuts(line)
        near, rest = split(rest, near_f, R.body)          # g < +GAP
        band, beyond = split(near, band_f, R.body)        # g > -GAP
        rest = merge([rest, beyond])
        parts.setdefault('gap', []).append(band)
    # normals: smooth from the field on curved surfaces, split along creases (lamp and intake rims, lips, recesses)
    g.add('body__paint', autosmooth(rest, R.body), 'paint')
    for finish, meshes in parts.items():
        g.add('body__' + finish, autosmooth(merge(meshes), R.body), finish)
    # lenses: the original skin inside each lamp outline
    for name, box, outline, finish in (('headlamp', ((1.45, .45, .45), (2.2, .85, .9)), R.lamp_outline, 'lens'),
                                        ('taillamp', ((-2.2, .62, .4), (-1.4, .9, .95)), R.tail_outline, 'tail_lens')):
        for sgn in (1, -1):
            lo, hi = np.array(box[0]), np.array(box[1])
            if sgn < 0:
                lo[2], hi[2] = -hi[2], -lo[2]
            V, T = mesh_solid(R.skin, lo, hi, .008)
            skin = Mesh(V, T).with_normals(R.skin)
            # the lens ends just inside the housing's dark rim, so it never overlaps paint (paint pierced a lens laid
            # over it wherever the coarser body mesh chorded above the true surface: a ragged dark edge)
            lens, _ = split(skin, lambda x, y, z, o=outline: o(x, y, z) + .001, R.skin)
            lens.V = lens.V + .001 * lens.N
            # the lens's sealed edge: an opaque dark border round the clear part
            clear, border = split(lens, lambda x, y, z, o=outline: o(x, y, z) + .007, R.skin)
            side = '_right' if sgn > 0 else '_left'
            g.add(name + side + '__' + finish, clear, finish)
            g.add(name + side + '_border__black_gloss', border, 'black_gloss')
    print('body parts %.1fs' % (time.time() - t0))
    import parts as P
    for part, finish, mesh in P.all_parts():
        g.add(part + '__' + finish, mesh, finish)
    print('parts %.1fs' % (time.time() - t0))
    for axle in ('front', 'rear'):
        for finish, mesh, rotates in Wh.assembly(axle):
            g.add(('wheel_' if rotates else 'hub_') + axle + '__' + finish, mesh, finish)
    n = g.write(out)
    print('wrote %s %.1f MB %.1fs' % (out, n / 1e6, time.time() - t0))
    if mirror_to:
        import shutil
        shutil.copyfile(out, mirror_to)


if __name__ == '__main__':
    build(sys.argv[1], mirror_to=sys.argv[2] if len(sys.argv) > 2 else None)
