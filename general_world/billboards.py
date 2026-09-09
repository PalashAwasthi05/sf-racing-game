"""Measured sign carriers are independent occluders, never facade classifications."""
import hashlib
from pathlib import Path
import numpy as np
import trimesh
from shapely.geometry import Polygon
from .opening_mesh import _solid

def build_carrier(record,host,obstacles=()):
    """Local X horizontal, Y into host, Z up. Return a board without editing host.

    Hosted signs require an emitted host surface and two measured support anchors.
    Freestanding boards require separately supplied support solids. Image presence
    verifies provenance, not semantic sign accuracy; OCR is not synthesized here.
    """
    if record.get('kind') not in {'wall_sign','billboard','freestanding_sign'}:raise ValueError('Unresolved sign kind')
    p=Polygon(record['boundaryXZ'])
    if not p.is_valid or p.is_empty or p.area<=0:raise ValueError('Invalid carrier boundary')
    thickness=float(record['thicknessM']);offset=float(record['standOffM'])
    if not np.isfinite([thickness,offset]).all() or thickness<=0 or offset<.005:raise ValueError('Invalid carrier thickness or clearance')
    T=np.asarray(record['localToWorld'],float)
    if T.shape!=(4,4) or not np.isfinite(T).all() or not np.allclose(T[3],[0,0,0,1]):raise ValueError('Invalid carrier frame')
    if not np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-5) or not np.isclose(np.linalg.det(T[:3,:3]),1,atol=1e-5):raise ValueError('Carrier frame must preserve metric scale and handedness')
    if record.get('hostKey')!=host.get('key') or host.get('status')!='emitted':raise ValueError('Missing or ambiguous emitted host')
    if 'localToWorld' not in host or not np.allclose(T,np.asarray(host['localToWorld']),atol=1e-5):raise ValueError('Carrier frame detached from the emitted mounting surface')
    surface=Polygon(host['boundaryXZ'])
    if not surface.covers(p) and record['kind']=='wall_sign':raise ValueError('Wall sign lies outside its carrier surface')
    anchors=np.asarray(record.get('supportAnchorsXZ',[]),float)
    from shapely.geometry import Point
    if anchors.ndim!=2 or anchors.shape[1]!=2 or len(anchors)<2 or not np.isfinite(anchors).all():raise ValueError('Missing measured attachment anchors')
    if any(not surface.covers(Point(*a)) for a in anchors):raise ValueError('Detached attachment anchor')
    if np.linalg.norm(anchors[0]-anchors[1])<.05:raise ValueError('Coincident anchors do not establish support')
    if record['kind']=='freestanding_sign' and not host.get('supportSolids'):raise ValueError('Freestanding sign requires reconstructed supports')
    # Positive Y points into the building: panels sit on the negative side.
    board=_solid(p,-offset-thickness,-offset,'carrier');board.apply_transform(T)
    for obstacle in obstacles:
        if not obstacle.is_volume:raise ValueError('Collision geometry is unresolved')
        if (board.bounds[1]<obstacle.bounds[0]).any() or (obstacle.bounds[1]<board.bounds[0]).any():continue
        intersection=trimesh.boolean.intersection([board,obstacle],engine='manifold')
        if intersection.volume>1e-6:raise ValueError('Carrier penetrates another solid')
    artwork=record.get('artwork')
    if artwork:
        path=Path(artwork['path'])
        if hashlib.sha256(path.read_bytes()).hexdigest()!=artwork['sha256']:raise ValueError('Carrier artwork fingerprint mismatch')
    board.metadata={'layer':'signs','role':'occluder','hostKey':host['key'],'artworkStatus':'source-image' if artwork else 'unresolved'}
    return board,{'hostModified':False,'cutsFacade':False,'establishesHiddenOpenings':False,'topologyClosed':board.is_volume,'artworkResolved':bool(artwork)}
