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


def load_balanced_pano(path, max_size=(4096, 2048), band_rows=32):
    """Pixel-identical decode in a short-lived child; bounded disk cache, row balance."""
    import hashlib, json, os, pathlib, subprocess, sys, tempfile
    from PIL import __version__ as pillow_version
    path = pathlib.Path(path)
    cache = pathlib.Path(os.environ.get('SFW_AREA', tempfile.gettempdir())) / 'regen/photo-cache'
    cache.mkdir(parents=True, exist_ok=True)
    with path.open('rb') as source:
        digest = hashlib.file_digest(source, 'sha256').hexdigest()
    key = hashlib.sha256((digest + repr(max_size) + pillow_version).encode()).hexdigest()[:24]
    raw, meta = cache / (key+'.rgb'), cache / (key+'.json')
    if not raw.exists() or not meta.exists():
        with tempfile.TemporaryDirectory(dir=cache) as tmp:
            temp_raw, temp_meta = pathlib.Path(tmp)/'pixels.rgb', pathlib.Path(tmp)/'pixels.json'
            subprocess.run([sys.executable, str(pathlib.Path(__file__).with_name('pano_decode_worker.py')),
                            str(path), str(temp_raw), str(temp_meta), *map(str,max_size)], check=True)
            temp_raw.replace(raw); temp_meta.replace(meta)
    info = json.loads(meta.read_text()); raw.touch()
    shape = (info['height'], info['width'], 3)
    mapped = np.memmap(raw, dtype=np.uint8, mode='r', shape=shape)
    gain = road_gain(mapped)
    pixels = np.empty(shape, dtype=np.uint8)
    for start in range(0, len(pixels), band_rows):
        band = mapped[start:start+band_rows].astype(np.float32)
        band *= gain; np.clip(band, 0, 255, out=band)
        pixels[start:start+band_rows] = band.astype(np.uint8)
    del mapped
    # Cache is acceleration only, not source evidence. Bound disk use as well.
    files = sorted(cache.glob('*.rgb'), key=lambda p:p.stat().st_mtime, reverse=True)
    total = 0
    for f in files:
        total += f.stat().st_size
        if total > 256*1024**2 and f != raw:
            f.unlink(missing_ok=True); f.with_suffix('.json').unlink(missing_ok=True)
    return pixels
