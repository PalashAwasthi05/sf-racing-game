import sys, numpy as np
from PIL import Image
a_dir, b_dir = sys.argv[1], sys.argv[2]
names = sys.argv[3:] or ['street','street_corner','crossing','aerial_a','aerial_b','car_idle']
for n in names:
    try:
        A = np.asarray(Image.open(f'{a_dir}/{n}.png').convert('RGB')).astype(np.int16)
        B = np.asarray(Image.open(f'{b_dir}/{n}.png').convert('RGB')).astype(np.int16)
    except Exception as e:
        print(f'{n:14s} missing ({e})'); continue
    d = np.abs(A - B).max(axis=2)
    total = d.size
    print(f'{n:14s} max {d.max():3d}  mean {d.mean():.4f}  px>0 {np.count_nonzero(d)/total*100:6.3f}%  px>2 {np.count_nonzero(d>2)/total*100:6.3f}%  px>8 {np.count_nonzero(d>8)/total*100:6.3f}%  px>24 {np.count_nonzero(d>24)}')
    if d.max() > 0:
        im = np.clip(d * 8, 0, 255).astype(np.uint8)
        Image.fromarray(im).save(f'{b_dir}/{n}.diff.png')
