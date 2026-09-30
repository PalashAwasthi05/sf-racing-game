"""White balance of a street panorama against the road just below the horizon (shared by every photo reader, the judge
and the review renderer).

Until 2026-09-29 each reader balanced against rows 74-90 % down the equirectangular frame. On the roof-mounted rig's
panoramas those rows are the capture vehicle (its roof rack and decal, about RGB 85, 101, 98 in every frame), so every
such panorama took a gain of about R x1.12, G x0.94, B x0.96: a magenta cast added, not removed. Rows 55-61 % down
(9-20 degrees below the horizon) are road, pavement and kerb in these captures. The balance takes the near-neutral
pixels there (asphalt, concrete, markings) and of those the brighter half: shaded asphalt is blue from the sky and a
shaded band pushed the gain warm. Near-clipped pixels are left out: at the sensor's limit a lilac-cast capture's white
markings read white, and the sunlit half then corrected too little."""
import numpy as np

BAND = (.55, .61)          # fraction of the frame height: 9-20 degrees below the horizon
NEUTRAL_SPREAD = 40        # max - min of a road pixel's channels (0-255)
MIN_PIXELS = 500
GAIN_RANGE = (.85, 1.15)
CLIPPED = 235              # a channel this bright is at or near the sensor's limit: a tenth of a sunny band


def road_gain(pano):
    """Per-channel gain (3,) that makes the panorama's road neutral grey; ones when too little road shows."""
    p = np.asarray(pano)
    road = p[int(p.shape[0] * BAND[0]):int(p.shape[0] * BAND[1]):4, ::4].reshape(-1, 3).astype(np.float32)
    road = road[((road.max(1) - road.min(1)) < NEUTRAL_SPREAD) & (road.max(1) < CLIPPED)]   # clipped pixels have lost their cast
    if len(road) < MIN_PIXELS: return np.ones(3, np.float32)
    lum = road @ np.array([.299, .587, .114], np.float32)
    road = road[lum >= np.median(lum)]                    # sunlit rather than sky-lit asphalt
    mean = road.mean(0)
    return np.clip(mean.mean() / np.maximum(mean, 1.), *GAIN_RANGE).astype(np.float32)


def balanced(pano):
    """The panorama (uint8) balanced by road_gain."""
    p = np.asarray(pano, np.float32)
    return np.clip(p * road_gain(p), 0, 255).astype(np.uint8)
