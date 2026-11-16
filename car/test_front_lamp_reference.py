"""Check the modeled front lens against measured press-photo edge picks.

The camera is registered to tyres and the roll hoop, independently of this edge.
Photos are external references, never included in the game model.
"""
import numpy as np
import roadster as R
from fit_reference_views import fit, VIEWS
from parts import _onto, headlamp_internals


def test_front_lens_registered_edge():
    # Near headlamp, cropped P19_0916 (749 x 418). Measurements include both
    # its pointed front end and the rounded rear shoulder above the wheel.
    observed = np.array([(278,150), (290,139), (313,129), (338,122),
                         (356,121), (365,124), (362,134), (349,145),
                         (327,152), (302,154), (285,153)])
    plan = R.HEADLAMP_PLAN[::8]
    points = _onto(R.skin, np.column_stack((plan[:,0], np.full(len(plan), 1.2), -plan[:,1])),
                   0., (0, -1, 0))
    camera = fit(VIEWS[1])
    eye, aim, up = (np.array(camera[k]) for k in ('eye', 'aim', 'up'))
    forward = aim-eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    focal = 418/(2*np.tan(np.deg2rad(camera['verticalFov'])/2))
    rays = points-eye
    predicted = np.column_stack((374.5+focal*(rays@right)/(rays@forward),
                                 209-focal*(rays@up)/(rays@forward)))
    errors = np.linalg.norm(predicted-observed, axis=1)
    assert np.sqrt(np.mean(errors**2)) < 3.2, errors
    assert errors.max() < 6, errors


def test_front_reflector_mesh_faces_outward():
    name, finish, bowl = headlamp_internals()[0]
    assert (name, finish) == ('headlamp', 'lamp_chrome')
    vertices = bowl.V[bowl.T]
    face_normals = np.cross(vertices[:,1]-vertices[:,0], vertices[:,2]-vertices[:,0])
    facing = np.sum(face_normals*bowl.N[bowl.T].mean(1), axis=1)
    assert np.mean(facing > 0) > .98
    assert np.isfinite(bowl.V).all()
