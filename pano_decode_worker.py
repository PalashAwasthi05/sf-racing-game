"""Isolated JPEG decode: release the decoder's full-frame buffers on process exit."""
import json
import sys
from PIL import Image
path, raw, meta, width, height = sys.argv[1:]
with Image.open(path) as source:
    source.draft('RGB', (int(width), int(height)))
    source.thumbnail((int(width), int(height)))
    pixels = source if source.mode == 'RGB' else source.convert('RGB')
    with open(raw, 'wb') as target:
        for y in range(0, pixels.height, 32):
            target.write(pixels.crop((0,y,pixels.width,min(y+32,pixels.height))).tobytes())
    with open(meta,'w') as target:
        json.dump({'width':pixels.width,'height':pixels.height},target)
