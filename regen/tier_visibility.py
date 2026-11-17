"""Upper-wall visibility in a shared world datum, independent of location."""
import numpy as np


def seen_share(grid, x0, y0, camera_xy, piece, base, wall_ground_z,
               camera_ground_z, *, cell=.5, eye_height=2.5):
    """Do not derive camera elevation from the building's podium roof.

    Camera terrain and wall datum are distinct on sloped streets. Missing
    terrain is unknown visibility, not permission to invent an eye position.
    """
    if camera_ground_z is None or not np.isfinite(camera_ground_z):
        return 0.
    normal = np.asarray(piece['normal'], float)
    normal /= np.linalg.norm(normal)
    tangent = np.array([-normal[1], normal[0]])
    midpoint = np.asarray(piece['midpoint'], float)
    a = midpoint - tangent * piece['lengthM'] / 2
    b = midpoint + tangent * piece['lengthM'] / 2
    eye = np.r_[camera_xy, float(camera_ground_z) + eye_height]
    su, sz = np.meshgrid(np.linspace(.1, .9, 7), np.linspace(.1, .9, 5))
    su, sz = su.ravel(), sz.ravel()
    points = np.c_[a[None] + (b-a)[None] * su[:, None] + normal*.3,
                   float(wall_ground_z) + base + sz * piece['heightM']]
    d = points-eye
    length = np.linalg.norm(d[:, :2], axis=1)
    fraction = np.linspace(0, 1, 400)[None, :]
    q = eye[None, None] + d[:, None, :] * fraction[..., None]
    near = (fraction*length[:, None] < 1) | (fraction*length[:, None] > length[:, None]-1.5)
    # floor, not truncation: coordinates just outside the DSM are not cell zero.
    i = np.floor((q[..., 0]-x0)/cell).astype(int)
    j = np.floor((q[..., 1]-y0)/cell).astype(int)
    inside = (i>=0) & (j>=0) & (i<grid.shape[0]) & (j<grid.shape[1])
    heights = np.where(inside, grid[np.clip(i,0,grid.shape[0]-1), np.clip(j,0,grid.shape[1]-1)], -np.inf)
    blocked = ((heights>q[..., 2]) & ~near).any(1)
    valid = length>=2
    return float((~blocked)[valid].mean()) if valid.any() else 0.
