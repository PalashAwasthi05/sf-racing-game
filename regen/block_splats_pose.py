"""Train-only relative pose refinement for original spherical camera observations."""
from pathlib import Path
from functools import lru_cache
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from scipy.sparse import lil_matrix


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1.e-10)


def basis(forward):
    f = unit(forward)
    r = unit(np.cross(f, [0., 0., 1.]))
    return np.array([r, np.cross(f, r), f])


def stereo_basis(right):
    r = unit(right)
    f = unit(np.cross([0., 0., 1.], r))
    return np.array([r, np.cross(f, r), f])


@lru_cache(maxsize=4)
def pano(path):
    a = cv2.imread(path)
    if a is None:
        raise FileNotFoundError(path)
    return a


def perspective(cam, view, width, height, focal):
    image = pano(str(cam['imagePath']))
    h, w = image.shape[:2]
    y, x = np.indices((height, width), dtype=np.float32)
    rays = np.stack(((x - (width-1)/2)/focal, (y - (height-1)/2)/focal,
                     np.ones_like(x)), axis=-1)
    rays = rays @ np.asarray(view @ cam['R'].T, np.float32)
    u = ((np.arctan2(rays[..., 0], rays[..., 2]) / (2*np.pi) + .5) * w - .5).astype('f')
    v = (np.arctan2(rays[..., 1], np.hypot(rays[..., 0], rays[..., 2])) / np.pi + .5) * h - .5
    return cv2.remap(image, u, np.clip(v, 0, h-1).astype('f'), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_WRAP)


def load_cameras(rows, path):
    cameras = []
    for row in rows:
        image_path = Path(row['imagePath'])
        if not image_path.is_absolute() and not image_path.exists():
            image_path = next((p/image_path for p in [path.parent, *path.parents] if (p/image_path).exists()), image_path)
        cameras.append({**row, 'imagePath': str(image_path.resolve()), 'eye': np.array(row['eyeEnu']),
                        'R': np.array(row['rotationWorldToCamera'])})
    return cameras


def candidate_pairs(cameras, along, max_per_camera=3):
    groups = {}
    for i, c in enumerate(cameras):
        groups.setdefault(c['driveId'], []).append(i)
    pairs = set()
    for group in groups.values():
        group.sort(key=lambda i: cameras[i]['eye'][:2] @ along)
        for k, i in enumerate(group):
            ahead = [(np.linalg.norm(cameras[i]['eye']-cameras[j]['eye']), j) for j in group[k+1:]]
            pairs.update((i,j) for d,j in sorted(ahead)[:max_per_camera] if .55 < d < 18)
    return sorted(pairs)


def camera_refinement(cameras, along, pairs):
    width, height, focal = 720, 504, 310.
    perpendicular = np.r_[-along[1], along[0], 0.]
    views = [basis(perpendicular), basis(-perpendicular)]
    features = {}
    detector = cv2.SIFT_create(nfeatures=1250, contrastThreshold=.028)
    matcher = cv2.BFMatcher()
    for i,c in enumerate(cameras):
        for side, view in enumerate(views):
            im = perspective(c, view, width, height, focal)
            key, desc = detector.detectAndCompute(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), None)
            xy = np.array([k.pt for k in key], float).reshape(-1,2)
            rays = unit(np.c_[(xy-np.array([(width-1)/2,(height-1)/2]))/focal, np.ones(len(xy))] @ view)
            features[i,side] = (rays,desc)
    eye = np.stack([c['eye'] for c in cameras])
    rng = np.random.default_rng(68)
    blocks = []
    for i,j in pairs:
        for side, view in enumerate(views):
            a,da = features[i,side]
            b,db = features[j,side]
            if da is None or db is None or len(da)<6 or len(db)<6:
                continue
            ab = {m.queryIdx:m.trainIdx for m,n in matcher.knnMatch(da, db, k=2) if m.distance < .75*n.distance}
            ba = {m.queryIdx:m.trainIdx for m,n in matcher.knnMatch(db, da, k=2) if m.distance < .75*n.distance}
            matches = [(k,v) for k,v in ab.items() if ba.get(v)==k]
            if len(matches)<12:
                continue
            ia,ib=np.asarray(matches).T
            ra,rb=a[ia],b[ib]
            plausible=np.abs(np.einsum('ij,ij->i',unit(np.cross(ra,eye[j]-eye[i])),rb))*focal<28
            ra,rb=ra[plausible],rb[plausible]
            if len(ra)<12:
                continue
            # The fundamental-matrix filter excludes motion, mismatches and duplicated patterns.
            pa,pb=ra@view.T,rb@view.T
            _,inlier=cv2.findFundamentalMat((pa[:,:2]/pa[:,2:])*focal,(pb[:,:2]/pb[:,2:])*focal,
                                             cv2.USAC_MAGSAC,1.8,.999,1800)
            if inlier is None:
                continue
            sel=np.flatnonzero(inlier[:,0])
            if len(sel)<11:
                continue
            sel = rng.permutation(sel)[:100]
            blocks.append((i,j,ra[sel],rb[sel]))
    if not blocks:
        return {'used':False,'reason':'no geometrically consistent feature matches'}
    fixed_eyes = eye.copy()
    ai=np.concatenate([np.full(len(a),i) for i,j,a,b in blocks])
    bi=np.concatenate([np.full(len(b),j) for i,j,a,b in blocks])
    ar=np.concatenate([a for i,j,a,b in blocks])
    br=np.concatenate([b for i,j,a,b in blocks])
    keep=np.arange(len(ai))%6!=0
    def errors(params):
        p=params.reshape(-1,6)
        rot=Rotation.from_rotvec(p[:,:3]).as_matrix()
        eyes=fixed_eyes + p[:,3:]
        r1=np.einsum('nij,nj->ni',rot[ai],ar)
        r2=np.einsum('nij,nj->ni',rot[bi],br)
        n=unit(np.cross(r1,eyes[bi]-eyes[ai]))
        return np.einsum('ij,ij->i',n,r2)*focal
    def residual(params):
        p=params.reshape(-1,6)
        # Regularized metric gauge. Poses remain tied to the georeferenced drive.
        return np.r_[errors(params)[keep],(p[:,:3]/.025).ravel(),(p[:,3:]/.5).ravel()]
    z=np.zeros(len(cameras)*6)
    sparsity=lil_matrix((sum(keep)+len(z),len(z)),dtype=int)
    for k,(i,j) in enumerate(zip(ai[keep],bi[keep])):
        sparsity[k,6*i:6*i+6]=1; sparsity[k,6*j:6*j+6]=1
    for k in range(len(cameras)):
        for d in range(3):
            sparsity[sum(keep)+3*k+d,6*k+d]=1
            sparsity[sum(keep)+3*len(cameras)+3*k+d,6*k+3+d]=1
    bound=np.tile([.075,.075,.075,1.7,1.7,1.7],len(cameras))
    opt=least_squares(residual,z,bounds=(-bound,bound),jac_sparsity=sparsity.tocsr(),loss='soft_l1',f_scale=1.4,
                      max_nfev=40,ftol=2.e-4,xtol=1.e-5)
    p=opt.x.reshape(-1,6)
    original_error=np.abs(errors(z)[~keep])
    fitted_error=np.abs(errors(opt.x)[~keep])
    ok=(np.median(fitted_error)<np.median(original_error) and
        np.max(np.linalg.norm(p[:,3:],axis=1))<4. and np.max(np.linalg.norm(p[:,:3],axis=1))<.19)
    for i,c in enumerate(cameras):
        if ok:
            c['eye']=eye[i]+p[i,3:]
            c['R']=c['R'] @ Rotation.from_rotvec(p[i,:3]).as_matrix().T
    return {'used':bool(ok),'tieObservations':len(ai),'validationCorrespondences':int(sum(~keep)),
            'beforeMedianEpipolarPx':float(np.median(original_error)),'afterMedianEpipolarPx':float(np.median(fitted_error)),
            'maxTranslationM':float(np.max(np.linalg.norm(p[:,3:],axis=1))),
            'maxRotationDeg':float(np.degrees(np.max(np.linalg.norm(p[:,:3],axis=1)))),
            'cameras':{c['id']:{'eyeEnu':c['eye'].tolist(),'rotationWorldToCamera':c['R'].tolist()} for c in cameras}}
