"""Pack each CC0 material's maps into one detail texture for the viewer (king-street/public/materials/<id>.png).

R, G  tangent-space normal x, y (OpenGL convention), as in the source normal map
B     roughness, as in the source
A     the colour map's linear luminance over its mean, times 0.5 (0.5 = the material's average tone, 1.0 = twice it)

The viewer multiplies a surface's own colour by A/0.5, so a wall keeps the colour it was given and gains the scan's
tonal detail; the scan's hue is not used. Run: uv run --python 3.11 --with pillow --with numpy python pack_detail.py
"""
from pathlib import Path
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / 'king-street' / 'public' / 'materials'
# material id -> packed size (px)
SETS = {'Plaster001': 2048, 'PaintedPlaster017': 1024, 'WoodSiding008': 1024, 'Bark016': 1024, 'Grass004': 1024, 'Ground110': 1024}


def linear(srgb):
    c = srgb.astype(np.float64) / 255
    return np.where(c <= .04045, c / 12.92, ((c + .055) / 1.055) ** 2.4)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, size in SETS.items():
        load = lambda kind: Image.open(HERE / name / f'{name}_2K-JPG_{kind}.jpg').convert('RGB').resize((size, size), Image.LANCZOS)
        colour, normal, rough = (np.asarray(load(k)) for k in ('Color', 'NormalGL', 'Roughness'))
        lum = linear(colour) @ np.array([.2126, .7152, .0722])
        ratio = np.clip(lum / lum.mean() * .5, 0, 1)
        packed = np.dstack([normal[..., 0], normal[..., 1], rough[..., 0], np.round(ratio * 255).astype(np.uint8)])
        Image.fromarray(packed, 'RGBA').save(OUT / f'{name}.png', optimize=True)
        print(name, size, 'mean roughness %.3f' % (rough[..., 0].mean() / 255), 'tone range %.2f..%.2f' % (np.percentile(lum / lum.mean(), 1), np.percentile(lum / lum.mean(), 99)))


if __name__ == '__main__':
    main()
