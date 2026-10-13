"""Semantic labels for rectified facade photographs, from the locally cached SegFormer-B2 (ADE20K) weights.

The model runs on the photograph as it is (no network: local files only), tiled in overlapping squares so a long
frontage keeps its proportions. Labels are grouped for facade judging: openings (windowpane, door, screen door),
signs and awnings (the fascia and shopfront furniture a facade carries), occluders that hide the wall (trees, plants,
people, cars, poles, lights, fences), sky, and the rest (wall, building, columns, railings).
"""
import glob, pathlib
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
GROUPS = {'opening': {'windowpane', 'door', 'screen door'}, 'sign': {'signboard', 'awning', 'canopy'},
          'occluder': {'tree', 'palm', 'plant', 'flower', 'person', 'car', 'pole', 'streetlight', 'traffic light', 'fence', 'flag', 'truck', 'van', 'bus', 'bicycle', 'minibike', 'booth', 'tent', 'bench', 'ashcan'},
          'sky': {'sky'}}
CODES = {'other': 0, 'opening': 1, 'sign': 2, 'occluder': 3, 'sky': 4}


def release(torch):
    """Return the MPS allocator's cached blocks after each image: across a long run of differently sized walls the
    cache otherwise grows without bound (25 GB after 18 frontages on 2026-09-25, and the machine swapped)."""
    if torch.backends.mps.is_available(): torch.mps.empty_cache()


class Segmenter:
    def __init__(self, model_glob='.models/models--nvidia--segformer-b2-finetuned-ade-512-512/snapshots/*'):
        import torch
        from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
        path = sorted(glob.glob(str(ROOT / model_glob)))[0]
        self.torch = torch
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.processor = AutoImageProcessor.from_pretrained(path, local_files_only=True)
        self.model = AutoModelForSemanticSegmentation.from_pretrained(path, local_files_only=True, use_safetensors=False).to(self.device).eval()
        labels = self.model.config.id2label
        self.lut = np.zeros(len(labels), np.uint8)
        for i, name in labels.items():
            for g, names in GROUPS.items():
                if name in names: self.lut[int(i)] = CODES[g]

    def labels(self, img, tile=512, overlap=128):
        """Per-pixel group codes (CODES) for an RGB uint8 image, from class logits averaged over overlapping tiles."""
        torch = self.torch; H, W = img.shape[:2]
        scale = tile / H if H < tile else 1.
        from PIL import Image
        big = np.asarray(Image.fromarray(img).resize((max(tile, int(round(W * scale))), max(tile, int(round(H * scale)))), Image.BILINEAR)) if scale != 1. else img
        h, w = big.shape[:2]; n = len(self.lut)
        acc = np.zeros((n, h, w), np.float32); cnt = np.zeros((h, w), np.float32)
        ys = list(range(0, max(1, h - tile), tile - overlap)) + [max(0, h - tile)]; xs = list(range(0, max(1, w - tile), tile - overlap)) + [max(0, w - tile)]
        for y in sorted(set(ys)):
            for x in sorted(set(xs)):
                crop = big[y:y + tile, x:x + tile]
                inp = self.processor(images=crop, return_tensors='pt').to(self.device)
                with torch.no_grad():
                    logits = self.model(**inp).logits
                    logits = torch.nn.functional.interpolate(logits, size=crop.shape[:2], mode='bilinear', align_corners=False)[0]
                acc[:, y:y + crop.shape[0], x:x + crop.shape[1]] += logits.float().cpu().numpy(); cnt[y:y + crop.shape[0], x:x + crop.shape[1]] += 1
        cls = (acc / np.maximum(cnt, 1)).argmax(0)
        codes = self.lut[cls]
        if scale != 1.: codes = np.asarray(Image.fromarray(codes).resize((W, H), Image.NEAREST))
        release(torch)
        return codes


class Detector:
    """Openings, signs and awnings as boxes, from the locally cached Grounding DINO (tiny) weights, run over zoomed,
    overlapping tiles of a rectified photograph (small windows are found at street scale, not in a whole-frontage
    image). A box as large as its tile is dropped (that is a neighbouring building, not an opening)."""
    PROMPT = 'a window. a door. a shop sign. an awning.'

    def __init__(self, model_glob='.models/models--IDEA-Research--grounding-dino-tiny/snapshots/*'):
        import torch
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        path = sorted(glob.glob(str(ROOT / model_glob)))[0]
        self.torch = torch; self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.processor = AutoProcessor.from_pretrained(path, local_files_only=True)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(path, local_files_only=True).to(self.device).eval()

    def boxes(self, img, tile_px=150, overlap=50, zoom=3, threshold=.28):
        """[(x0, y0, x1, y1, kind, score)] in pixels of img; kind is window, door, sign or awning."""
        from PIL import Image
        from torchvision.ops import nms
        torch = self.torch; im = Image.fromarray(img); W, H = im.size; found = []
        xs = sorted(set(list(range(0, max(1, W - tile_px), tile_px - overlap)) + [max(0, W - tile_px)]))
        ys = sorted(set(list(range(0, max(1, H - tile_px), tile_px - overlap)) + [max(0, H - tile_px)]))
        tiles = []
        for y in ys:
            for x in xs:
                t = im.crop((x, y, min(W, x + tile_px), min(H, y + tile_px))); tiles.append((x, y, t.size, t.resize((t.size[0] * zoom, t.size[1] * zoom), Image.BICUBIC)))
        for k in range(0, len(tiles), 8):                     # tiles of one wall go through the model in batches
            chunk = tiles[k:k + 8]
            inp = self.processor(images=[c[3] for c in chunk], text=[self.PROMPT] * len(chunk), return_tensors='pt', padding=True).to(self.device)
            with torch.no_grad(): out = self.model(**inp)
            res = self.processor.post_process_grounded_object_detection(out, inp.input_ids, threshold=threshold, text_threshold=threshold - .06, target_sizes=[c[3].size[::-1] for c in chunk])
            for (x, y, (tw, th), _), r in zip(chunk, res):
                for b, sc, l in zip(r['boxes'].cpu().numpy(), r['scores'].cpu().numpy(), r.get('text_labels', r.get('labels'))):
                    b = b / zoom + np.array([x, y, x, y])
                    if (b[2] - b[0]) * (b[3] - b[1]) > .8 * tw * th: continue
                    kind = 'window' if 'window' in l else 'door' if 'door' in l else 'awning' if 'awning' in l else 'sign' if 'sign' in l else None
                    if kind: found.append((*b, kind, float(sc)))
            del inp, out, res; release(torch)                # per batch: the allocator otherwise grows to 16 GB on a long wall
        if not found: return []
        B = torch.tensor([f[:4] for f in found], dtype=torch.float32); S = torch.tensor([f[5] for f in found], dtype=torch.float32)
        return [found[i] for i in nms(B, S, .4).numpy()]


def glazed_boxes(det, img, threshold=.3, min_share=.25):
    """Curtain-wall glazing is one large object, found at whole-wall scale (the tiled pass drops boxes as large as
    a tile): [(x0, y0, x1, y1, score)] in pixels, each covering at least `min_share` of the image."""
    from PIL import Image
    torch = det.torch; im = Image.fromarray(img); t = im.resize((im.width * 2, im.height * 2), Image.BICUBIC)
    inp = det.processor(images=t, text='a glass facade. a glass curtain wall.', return_tensors='pt').to(det.device)
    with torch.no_grad(): out = det.model(**inp)
    r = det.processor.post_process_grounded_object_detection(out, inp.input_ids, threshold=threshold, text_threshold=threshold - .05, target_sizes=[t.size[::-1]])[0]
    area = im.width * im.height; res = []; release(torch)
    for b, sc in zip(r['boxes'].cpu().numpy() / 2, r['scores'].cpu().numpy()):
        if (b[2] - b[0]) * (b[3] - b[1]) >= min_share * area: res.append((*b, float(sc)))
    return res
