"""Badge sheet for a "718 Boxster GTS" rear badge.

    python -I make_gts_badge.py <boxster dir> <cayman dir> <out.png>

The Boxster S model's badge sheet has the rear lettering "718 Boxster S" as one strip; the Cayman GTS sheet has a
standalone "GTS" in the same script. This writes the Boxster sheet at twice the size with the strip redrawn as
"718 Boxster GTS". The new lettering is WIDTH_RATIO times as long as the old, squeezed into the same strip, so the
converter widens the badge quad by the same ratio (convert.js, option gtsBadge) and the letters keep their shape.
"""
import sys
import numpy as np
import scipy.ndimage as nd
from PIL import Image, ImageFilter

WIDTH_RATIO = 1.245
UP = 2                       # output sheet scale
STRIP = (3.6, 44.0, 510.0, 108.0)   # x0, y0, x1, y1 of the badge quad's texture region, in source pixels
GTS_BOX = (232, 91, 375, 128)       # the standalone GTS in the Cayman sheet
GTS_SCALE = 1.25             # GTS cap height relative to the digits, as on the Cayman's own "718 Cayman GTS" strip
GTS_AT = (454, 56)           # top-left of the scaled GTS, in Boxster sheet pixels
INK = 34

box_dir, cay_dir, out = sys.argv[1:4]
box = Image.open(f"{box_dir}/Badge_DiffuseAOSO.png").convert("RGBA")
cay = Image.open(f"{cay_dir}/Badge_D.png").convert("RGBA")
alpha = np.array(box)[..., 3].copy()
# remove the "S": the right-most connected piece of the strip
band = alpha[44:110] > 128
labels, _ = nd.label(band)
last = labels[:, 500:505].max()
assert last > 0, "S not found"
piece = nd.binary_dilation(labels == last, iterations=2)
alpha[44:110][piece] = 0

K = 4                        # working scale
x0, y0, x1, y1 = STRIP
width = (x1 - x0) * WIDTH_RATIO
canvas = Image.new("L", (round(width * K), round((y1 - y0) * K)), 0)
old = Image.fromarray(alpha).crop((0, int(y0), 445, int(y1))).resize((445 * K, int(y1 - y0) * K), Image.LANCZOS)
canvas.paste(old, (round((0 - x0) * K), 0))
gts = cay.getchannel("A").crop(GTS_BOX)
gts = gts.resize((round(gts.width * GTS_SCALE * K), round(gts.height * GTS_SCALE * K)), Image.LANCZOS)
# the donor lettering is small: blur and re-threshold it so its edges are curves, not enlarged pixels
gts = gts.filter(ImageFilter.GaussianBlur(K * 0.9)).point(lambda v: max(0, min(255, (v - 128) * 4 + 128)))
canvas.paste(gts, (round((GTS_AT[0] - x0) * K), round((GTS_AT[1] - y0) * K)), gts)
assert (GTS_AT[0] - x0) * K + gts.width <= canvas.width, "GTS overruns the strip: raise WIDTH_RATIO"

sheet = box.resize((box.width * UP, box.height * UP), Image.LANCZOS)
dst = (round(x0 * UP), round(y0 * UP), round(x1 * UP), round(y1 * UP))
strip = canvas.resize((dst[2] - dst[0], dst[3] - dst[1]), Image.LANCZOS)
ink = Image.new("RGBA", strip.size, (INK, INK, INK, 0))
ink.putalpha(strip)
sheet.paste(ink, dst[:2])
sheet.save(out)
print("wrote", out, sheet.size, "width ratio", WIDTH_RATIO)
