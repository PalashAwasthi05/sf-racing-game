"""Whether two photographed colours are one material under different light.

A wall photographed in sun and in shade, or by cameras with different exposure and white balance, keeps its hue and
roughly its chroma while its lightness moves a long way: a brick wall read L* 67 in sun and L* 38 in shade (dE 15 with
lightness at half weight called them different, and 18 facings had no two photographs agreeing). Shade is not simply
bluer either: the darker photographs of that wall were redder, which is exposure and white balance, not skylight.
So the test is on hue and chroma, with lightness free within the range sun and shade span:
- lightness within 35 L*;
- two near-neutral colours (chroma under 10) are one material (grey render, concrete, white paint);
- a near-neutral colour and a faint tint (chroma under 16) are one material; a neutral and a clear colour are not;
- two clear colours are one material when their hues are within 20 degrees and their chromas within a factor 1.8.
A neighbour of another material (grey render against red brick, tan against brick) differs in hue or chroma and stays
apart."""
import math
import numpy as np

MAX_DL, GREY_C, TINT_C, MAX_DH, MAX_C_RATIO = 35., 10., 16., 20., 1.8


def lab(rgb):
    c = np.asarray(rgb, float) / 255.; c = np.where(c <= .04045, c / 12.92, ((c + .055) / 1.055) ** 2.4)
    X = c @ np.array([[.4124, .2126, .0193], [.3576, .7152, .1192], [.1805, .0722, .9505]]) / np.array([.9505, 1., 1.089])
    f = np.where(X > .008856, np.cbrt(X), 7.787 * X + 16 / 116)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def same_material(a, b):
    """True when RGB colours a and b can be one material seen under different light."""
    la, lb = lab(a), lab(b)
    if abs(la[0] - lb[0]) > MAX_DL: return False
    ca, cb = math.hypot(la[1], la[2]), math.hypot(lb[1], lb[2]); lo, hi = min(ca, cb), max(ca, cb)
    if hi < GREY_C: return True
    if lo < GREY_C: return hi < TINT_C
    dh = abs(math.degrees(math.atan2(la[2], la[1]) - math.atan2(lb[2], lb[1]))) % 360.
    return min(dh, 360. - dh) <= MAX_DH and hi / lo <= MAX_C_RATIO


SHADE_L, SHADE_C, SHADE_DH = 22., 16., 25.


def shaded_wall(rgb, wall_rgb):
    """True when a component read near-black (L* under 22, the readers' shade line) is the wall's own coloured
    material in shade: the wall is clearly coloured (chroma 16 or more) and the component keeps its hue within 25
    degrees. A brick pilaster's shaded side read L* 18 in the brick's hue and was drawn as a glossy black panel; a
    black metal pier on a grey wall has no hue to keep and stays what it is."""
    a, w = lab(rgb), lab(wall_rgb)
    if a[0] >= SHADE_L or math.hypot(w[1], w[2]) < SHADE_C or math.hypot(a[1], a[2]) < 4: return False
    dh = abs((math.degrees(math.atan2(a[2], a[1])) - math.degrees(math.atan2(w[2], w[1])) + 180) % 360 - 180)
    return dh <= SHADE_DH
