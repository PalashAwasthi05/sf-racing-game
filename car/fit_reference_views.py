"""Register the original player car to two press views using wheel and roll-bar landmarks.

Measurements are in cropped image pixels. Photos remain external and are not game assets.
Prints the camera fits and the error at a windshield landmark held out of registration.
"""
import json
import numpy as np
from scipy.optimize import least_squares
from parts import screen_point
from roadster import AXLE, R_FRONT, R_REAR

# Porsche Newsroom, US 718 Boxster GTS 4.0 gallery, January 15 2020. Normalised
# crop rectangles precede the working image size. The final pick is the near
# windshield header; only the preceding landmarks participate in camera fitting.
VIEWS = (
    ('P19_0914', (.26, .27, .80, .73), (716, 460), -140,
     [(123,282), (122,228), (124,334), (376,295), (374,225), (378,366), (385,128), (478,129), (273,135)]),
    ('P19_0916', (.18, .39, .70, .83), (749, 418), 140,
     [(388,211), (388,153), (389,268), (593,212), (593,165), (590,265), (492,93), None, (451,76)]),
)
# Outer tyre edges and visible cap centres differ slightly across the wheel width.
ANCHORS = np.array([
    (AXLE,R_FRONT,-.845), (AXLE,2*R_FRONT,-.848), (AXLE,0,-.848),
    (-AXLE,R_REAR,-.86), (-AXLE,2*R_REAR,-.86), (-AXLE,0,-.86),
    (-.70,1.17,-.345), (-.70,1.17,.345),
])


def fit(view):
    image, crop, size, bearing, picks = view
    width, height = size
    present = [i for i, p in enumerate(picks[:-1]) if p is not None]
    observed = np.array([picks[i] for i in present])
    theta = np.radians(bearing)

    def project(params, points):
        focal, azimuth, radius, eye_y, aim_x, aim_y, aim_z, roll = params
        eye = np.array([radius*np.sin(azimuth), eye_y, radius*np.cos(azimuth)])
        aim = np.array([aim_x, aim_y, aim_z])
        forward = aim - eye
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, [0,1,0])
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        right, up = (np.cos(roll)*right + np.sin(roll)*up,
                     np.cos(roll)*up - np.sin(roll)*right)
        ray = points - eye
        xy = np.column_stack((ray @ right, -(ray @ up))) * (focal / (ray @ forward))[:, None]
        return xy + [width/2,height/2], eye, aim, up

    def residual(params):
        predicted, *_ = project(params, ANCHORS[present])
        return (predicted - observed).ravel()

    result = least_squares(residual, [2500,theta,14,1.1,0,.60,0,0],
        bounds=([500,theta-1,3,-.5,-2,0,-1,-.2], [15000,theta+1,90,6,2,1.5,1,.2]),
        loss='soft_l1', f_scale=3)
    header, eye, aim, up = project(result.x, np.array([screen_point(-1, 1)]))
    return dict(photo=image, crop=crop, size=size, eye=eye.tolist(), aim=aim.tolist(), up=up.tolist(),
        verticalFov=float(np.degrees(2*np.arctan(height/(2*result.x[0])))),
        registrationRmsPx=float(np.sqrt(np.mean(np.sum(residual(result.x).reshape(-1,2)**2, axis=1)))),
        headerErrorPx=float(np.linalg.norm(header[0]-picks[-1])))


if __name__ == '__main__':
    print(json.dumps([fit(v) for v in VIEWS], indent=2))
